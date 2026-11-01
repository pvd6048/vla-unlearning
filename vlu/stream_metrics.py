"""Stream-level metrics (PLAN.md, "Evaluation metrics"): from the success rate of every task after
every op of a stream, how well kept tasks survive and how well unlearned tasks are gone.
Pure Python; no GPU or GR00T imports.

A stream's success matrix is one dict per op: sr[k][task] = success rate of `task` after op k.
Tasks not evaluated at that op are absent and count as NaN, so missing evals show up as NaN.
"""

import json
import math

from vlu.nodes import Node, expand_stream, oracle_path
from vlu.tasks import STREAMS

NAN = float("nan")


def success_rates(node: Node, weights: str = "final", seed: int = 0) -> dict[str, float]:
    """A node's evaluated success rates. Purge nodes are not evaluated (their weights are their
    parent's), so they take their parent's results."""
    while not node.is_base and node.last_op.method == "purge":
        node = node.parent
    eval_dir = node.eval_dir(weights, seed)
    if not eval_dir.exists():
        return {}
    return {f.stem: json.loads(f.read_text())["success_rate"] for f in sorted(eval_dir.glob("*.json"))}


def stream_matrix(run: str, stream: str, method: str, weights: str = "final", seed: int = 0):
    """(ops, sr, oracle) for one method's branch of a stream. oracle[k] holds the success rates of the
    retrain oracle of U op k (a model that never saw the tasks unlearned so far)."""
    paths = expand_stream(STREAMS[stream], method)
    nodes = [Node.from_path(run, p) for p in paths]
    sr = [success_rates(n, weights, seed) for n in nodes]
    oracle = {
        k: success_rates(Node.from_path(run, oracle_path(run, p)), weights, seed)
        for k, (p, n) in enumerate(zip(paths, nodes))
        if n.last_op.kind == "U"
    }
    return [str(n.last_op) for n in nodes], sr, oracle


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else NAN


def _max(xs) -> float:
    xs = list(xs)
    return NAN if not xs or any(math.isnan(x) for x in xs) else max(xs)


def summarize(ops: list[str], sr: list[dict], oracle: dict[int, dict] | None = None) -> dict[str, float]:
    """All metrics of one stream. ops are tokens like "LA", "UB-ga"; see PLAN.md for definitions."""
    last = len(ops) - 1
    task = [op[1] for op in ops]
    c = lambda t, k: sr[k].get(t, NAN)  # noqa: E731  success rate of task t after op k

    def next_touch(t, k):  # first op after k that learns or unlearns t again
        return next((j for j in range(k + 1, last + 1) if task[j] == t), last + 1)

    kept, gone, retained, removed = [], [], [], []  # per op: tasks that should / should not be known
    for op, t in zip(ops, task):
        retained = [x for x in retained if x != t] + ([t] if op[0] == "L" else [])
        removed = [x for x in removed if x != t] + ([t] if op[0] == "U" else [])
        kept.append(retained)
        gone.append(removed)
    learns = [(t, k) for k, (op, t) in enumerate(zip(ops, task)) if op[0] == "L"]
    unlearns = [(t, j) for j, (op, t) in enumerate(zip(ops, task)) if op[0] == "U"]

    # Continual learning: is the kept knowledge safe?
    retain_curve = [_mean(c(t, k) for t in kept[k]) for k in range(last + 1)]
    drops = [[c(t, k) - c(t, j) for j in range(k + 1, next_touch(t, k))] for t, k in learns]
    out = {
        "retain_sr": retain_curve[-1],
        "retain_auc": _mean(retain_curve),
        "nbt": _mean(_mean(d) for d in drops if d),
        "max_forgetting": _max(x for d in drops for x in d),
        "retain_stability": _mean(abs(b - a) for a, b in zip(retain_curve, retain_curve[1:])),
    }

    # Unlearning: is the removed task really gone, and what did removing it cost?
    if unlearns:
        out["forget_sr"] = _mean(c(t, j) for t, j in unlearns)
        out["collateral"] = _mean(c(t, j - 1) - c(t, j) for _, j in unlearns for t in kept[j])
        out["relapse"] = _mean(_max(c(t, k) for k in range(j, next_touch(t, j))) - c(t, j) for t, j in unlearns)
        if oracle:
            out["oracle_gap"] = _mean(abs(c(t, j) - oracle[j].get(t, NAN)) for _, j in unlearns for t in gone[j])
            a, b = 1 - out["oracle_gap"], out["retain_sr"]  # harmonic mean: punishes one-sided methods
            out["score"] = NAN if math.isnan(a + b) else (2 * a * b / (a + b) if a + b > 0 else 0.0)
    return out
