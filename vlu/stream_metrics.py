"""Stream-level metrics (PLAN.md, "Evaluation metrics"): from the success rate of every task after
every op of a stream, how well kept tasks survive and how well unlearned tasks are gone.
Pure Python; no GPU or GR00T imports.

A stream's success matrix is one dict per op: sr[k][task] = success rate of `task` after op k.
Tasks not evaluated at that op are absent and count as NaN, so a missing eval shows up as NaN. A
metric is also NaN when it is undefined for the stream (e.g. relapse when no op follows any U op,
retain SR when nothing is kept at the end).
"""

from functools import lru_cache
import json
import math

from vlu.nodes import Node, Op, expand_stream, oracle_path
from vlu.tasks import STREAMS

NAN = float("nan")


@lru_cache(maxsize=None)  # shared-prefix and oracle nodes are read once, not once per method and run
def success_rates(node: Node, weights: str = "final", seeds: tuple[int, ...] = (0,)) -> dict[str, float]:
    """A node's evaluated success rates, averaged over the eval seeds (each seed is another set of
    rollouts of the same model). A task counts only if it was evaluated with every seed. Purge nodes
    are not evaluated (their weights are their parent's), so they take their parent's results."""
    while not node.is_base and node.last_op.method == "purge":
        node = node.parent
    per_seed = []
    for seed in seeds:
        eval_dir = node.eval_dir(weights, seed)
        files = sorted(eval_dir.glob("*.json")) if eval_dir.exists() else []
        per_seed.append({f.stem: json.loads(f.read_text())["success_rate"] for f in files})
    tasks = set.intersection(*(set(d) for d in per_seed))
    return {t: sum(d[t] for d in per_seed) / len(per_seed) for t in sorted(tasks)}


def stream_matrix(run: str, stream: str, method: str, weights: str = "final", seeds: tuple[int, ...] = (0,)):
    """(ops, sr, oracle) for one method's branch of a stream. oracle[k] holds the success rates of the
    retrain oracle of U op k (a model that never saw the tasks unlearned so far)."""
    paths = expand_stream(STREAMS[stream], method)
    nodes = [Node.from_path(run, p) for p in paths]
    sr = [success_rates(n, weights, tuple(seeds)) for n in nodes]
    oracle = {
        k: success_rates(Node.from_path(run, oracle_path(run, p)), weights, tuple(seeds))
        for k, (p, n) in enumerate(zip(paths, nodes))
        if n.last_op.kind == "U"
    }
    return [str(n.last_op) for n in nodes], sr, oracle


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval of a success rate from n rollouts (PLAN.md, reporting rules)."""
    if n == 0:
        return NAN, NAN
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else NAN


def _max(xs) -> float:
    xs = list(xs)
    return NAN if not xs or any(math.isnan(x) for x in xs) else max(xs)


def summarize(ops: list[str], sr: list[dict], oracle: dict[int, dict] | None = None) -> dict[str, float]:
    """All metrics of one stream. ops are tokens like "LA", "UB-ga"; see PLAN.md for definitions."""
    parsed = [Op.parse(op) for op in ops]
    last = len(ops) - 1
    task = [op.task for op in parsed]
    c = lambda t, k: sr[k].get(t, NAN)  # noqa: E731  success rate of task t after op k

    def next_touch(t, k):  # first op after k that learns or unlearns t again
        return next((j for j in range(k + 1, last + 1) if task[j] == t), last + 1)

    # Per op: tasks that should still be known (vlu/nodes.py's retained tasks) and tasks that should be gone.
    kept = [Node("", tuple(parsed[: k + 1])).retained_tasks() for k in range(last + 1)]
    gone, removed = [], []
    for op in parsed:
        removed = [x for x in removed if x != op.task] + ([op.task] if op.kind == "U" else [])
        gone.append(removed)
    learns = [(op.task, k) for k, op in enumerate(parsed) if op.kind == "L"]
    unlearns = [(op.task, j) for j, op in enumerate(parsed) if op.kind == "U"]

    # Continual learning: is the kept knowledge safe? The retain curve is undefined (None) after ops
    # that leave nothing kept (e.g. L_A U_A), which is not the same as a missing eval.
    retain_curve = [_mean(c(t, k) for t in kept[k]) if kept[k] else None for k in range(last + 1)]
    drops, nbt_l, nbt_u = [], [], []
    for t, k in learns:
        later = range(k + 1, next_touch(t, k))
        if not later:
            continue
        drops.append([c(t, k) - c(t, j) for j in later])
        # Split by cause: c[t,k] - c[t,j] is the sum of the per-op drops c[t,m-1] - c[t,m] over m in (k, j];
        # attribute each to the kind of op m (L: ordinary CL forgetting, U: collateral damage of unlearning).
        step = {m: c(t, m - 1) - c(t, m) for m in later}
        nbt_l.append(_mean(sum(step[m] for m in range(k + 1, j + 1) if parsed[m].kind == "L") for j in later))
        nbt_u.append(_mean(sum(step[m] for m in range(k + 1, j + 1) if parsed[m].kind == "U") for j in later))
    defined = [x for x in retain_curve if x is not None]
    out = {
        "retain_sr": retain_curve[-1] if retain_curve[-1] is not None else NAN,
        "retain_auc": _mean(defined),
        "nbt": _mean(_mean(d) for d in drops),
        "nbt_from_l_ops": _mean(nbt_l),
        "nbt_from_u_ops": _mean(nbt_u),
        "max_forgetting": _max(x for d in drops for x in d),
        "retain_stability": _mean(abs(b - a) for a, b in zip(retain_curve, retain_curve[1:]) if a is not None and b is not None),
    }

    # Unlearning: is the removed task really gone, and what did removing it cost?
    if unlearns:
        out["forget_sr"] = _mean(c(t, j) for t, j in unlearns)
        out["collateral"] = _mean(c(t, j - 1) - c(t, j) for _, j in unlearns for t in kept[j])
        # PLAN: max over k > u_X of c[X,k] - c[X,u_X], up to the op that relearns X (if any). A U op
        # with no later op before that has no relapse and is left out.
        out["relapse"] = _mean(_max(c(t, k) for k in range(j + 1, next_touch(t, j))) - c(t, j)
                               for t, j in unlearns if next_touch(t, j) > j + 1)
        if oracle:
            out["oracle_gap"] = _mean(abs(c(t, j) - oracle.get(j, {}).get(t, NAN)) for _, j in unlearns for t in gone[j])
            a, b = 1 - out["oracle_gap"], out["retain_sr"]  # harmonic mean: punishes one-sided methods
            out["score"] = NAN if math.isnan(a + b) else (2 * a * b / (a + b) if a + b > 0 else 0.0)
    return out
