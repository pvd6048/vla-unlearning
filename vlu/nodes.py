"""Ops, nodes and the on-disk run tree.

An op is a token:
    L<task>            learn a task, e.g. "LB"
    U<task>-<method>   unlearn a task with a method, e.g. "UB-ga"
optionally followed by recipe overrides for that op, e.g. "UB-ga@unlearn_lr=3e-5+unlearn_steps=500"
(hyperparameter variants: separate nodes sharing their parent).

A node is the model obtained by applying a sequence of ops to the base GR00T checkpoint. It lives at
RUNS_DIR/<run>/<op1>/<op2>/..., so shared prefixes are trained once and branches sit side by side:

    runs/goal_s0/LA/LB/LC/                          shared prefix
    runs/goal_s0/LA/LB/LC/UB-ga/LD/LE/UE-ga/        GA tail of stream S1
    runs/goal_s0/LA/LC/LD/                          retrain oracle (never saw B or E)

Files inside a node directory:
    weights.safetensors   trainable parameters after the op; its existence means "node is done"
    op.json               what was run (op, recipe, batch composition, git commit)
    train_log.jsonl       training curve and probe losses
    resume.pt             latest resumable state while the op is running; deleted when done
    step_XXXXXX.safetensors  extra weight snapshots requested with --save-at
    eval/<weights>_seed<k>/<task>.json   LIBERO rollout results
"""

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path

from vlu.config import RUNS_DIR, Recipe, VLAForgetConfig
from vlu.tasks import TASKS

_RECIPE_KEYS = ({f.name for f in fields(Recipe)} - {"vlaforget"}) | {f"vlaforget.{f.name}" for f in fields(VLAForgetConfig)}


@dataclass(frozen=True)
class Op:
    kind: str  # "L" or "U"
    task: str  # task letter
    method: str | None = None  # unlearning method, only for U ops
    # Recipe overrides for this op only, e.g. (("unlearn_lr", "3e-5"),): hyperparameter variants are
    # separate nodes that share their parent (used for calibration sweeps). Applied by
    # recipe_for (below); keys are Recipe fields or "vlaforget.<field>".
    overrides: tuple[tuple[str, str], ...] = ()

    @classmethod
    def parse(cls, token: str) -> "Op":
        """L<task>, U<task>-<method>, each optionally followed by @key=value[+key=value...]."""
        head, _, extra = token.partition("@")
        kind, rest = head[0], head[1:]
        task, _, method = rest.partition("-")
        if kind not in ("L", "U") or task not in TASKS:
            raise ValueError(f"Bad op {token!r}: expected L<task> or U<task>-<method>, tasks {list(TASKS)}")
        if kind == "U" and not method:
            raise ValueError(f"U op {token!r} needs a method, e.g. U{task}-ga")
        if kind == "L" and method:
            raise ValueError(f"L op {token!r} takes no method")
        overrides = []
        for item in filter(None, extra.split("+")):
            key, eq, value = item.partition("=")
            if not eq or not value or key not in _RECIPE_KEYS:
                raise ValueError(f"Bad override {item!r} in {token!r}: expected <recipe field>=<value>")
            overrides.append((key, value))
        return cls(kind, task, method or None, tuple(overrides))

    def __str__(self) -> str:
        base = f"{self.kind}{self.task}" + (f"-{self.method}" if self.method else "")
        return base + ("@" + "+".join(f"{k}={v}" for k, v in self.overrides) if self.overrides else "")


@dataclass(frozen=True)
class Node:
    run: str
    ops: tuple[Op, ...]

    @classmethod
    def from_path(cls, run: str, path: str) -> "Node":
        """path is "LA/LB/UB-ga" (slashes) or "" for the base model."""
        tokens = [t for t in path.strip("/").split("/") if t]
        return cls(run, tuple(Op.parse(t) for t in tokens))

    @property
    def path(self) -> str:
        return "/".join(str(op) for op in self.ops)

    def __str__(self) -> str:
        return f"{self.run}:{self.path or '<base>'}"

    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.run / self.path

    @property
    def is_base(self) -> bool:
        return not self.ops

    @property
    def parent(self) -> "Node":
        assert not self.is_base, "the base model has no parent"
        return Node(self.run, self.ops[:-1])

    @property
    def last_op(self) -> Op:
        return self.ops[-1]

    @property
    def depth(self) -> int:
        return len(self.ops)

    def child(self, token: str) -> "Node":
        return Node(self.run, self.ops + (Op.parse(token),))

    # --- Task bookkeeping -------------------------------------------------------------------------

    def retained_tasks(self) -> list[str]:
        """Tasks learned and not unlearned since, in the order they were (last) learned.
        These form the replay buffer of the next L op and the retain set of the next U op."""
        retained: list[str] = []
        for op in self.ops:
            if op.task in retained:
                retained.remove(op.task)
            if op.kind == "L":
                retained.append(op.task)
        return retained

    def seen_tasks(self) -> list[str]:
        """Every task that appears in the history (learned or unlearned), in first-seen order."""
        return list(dict.fromkeys(op.task for op in self.ops))

    # --- Files ------------------------------------------------------------------------------------

    @property
    def weights_path(self) -> Path:
        return self.dir / "weights.safetensors"

    @property
    def done(self) -> bool:
        return self.is_base or self.weights_path.exists()

    def weights_file(self, name: str = "final") -> Path | None:
        """Weights to load for this node: "final" or a snapshot name like "step_002000".
        None means the untouched base checkpoint."""
        if self.is_base:
            return None
        path = self.weights_path if name == "final" else self.dir / f"{name}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist; has {self} finished training?")
        return path

    def eval_dir(self, weights: str, seed: int) -> Path:
        return self.dir / "eval" / f"{weights}_seed{seed}"

    def write_json(self, name: str, data: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / name).write_text(json.dumps(data, indent=2, default=str) + "\n")


def run_seed(run: str, seed: int | None) -> int:
    """Each run has one seed, fixed in RUNS_DIR/<run>/run.json when the run is created."""
    path = RUNS_DIR / run / "run.json"
    if path.exists():
        stored = json.loads(path.read_text())["seed"]
        if seed is not None and seed != stored:
            raise ValueError(f"run {run} was created with seed {stored}, not {seed}")
        return stored
    if seed is None:
        raise ValueError(f"run {run} does not exist yet; pass --seed to create it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "tasks": TASKS}, indent=2) + "\n")
    return seed


def expand_stream(stream: list[str], method: str) -> list[str]:
    """Node paths after each op of a stream, with `method` attached to every U op.
    expand_stream(["LA", "UA"], "ga") -> ["LA", "LA/UA-ga"]"""
    tokens = [t if t[0] == "L" else f"{t}-{method}" for t in stream]
    return ["/".join(tokens[: i + 1]) for i in range(len(tokens))]


def oracle_path(run: str, path: str) -> str:
    """The retrain oracle of a node: its L ops for the tasks it still retains, in order, so it has
    never seen any data of the tasks that were unlearned. E.g. LA/LB/LC/UB-ga/LD -> LA/LC/LD."""
    return "/".join(f"L{task}" for task in Node.from_path(run, path).retained_tasks())


# --- Recipes of ops ----------------------------------------------------------------------------------


def _convert(current, text: str):
    """An override's text as the type of the recipe value it replaces."""
    if text == "None":
        return None
    if isinstance(current, bool):
        return text.lower() in ("1", "true", "yes")
    if isinstance(current, int) or current is None:
        value = float(text)
        if value != int(value):
            raise ValueError(f"{text!r} is not an integer")
        return int(value)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, tuple):
        return tuple(type(current[0])(x) for x in text.split(":"))
    return text


def recipe_for(op: Op, recipe: Recipe) -> Recipe:
    """The recipe of one op: the frozen recipe with the op's overrides applied."""
    changes, vlaforget = {}, {}
    for key, text in op.overrides:
        if key.startswith("vlaforget."):
            name = key.split(".", 1)[1]
            vlaforget[name] = _convert(getattr(recipe.vlaforget, name), text)
        else:
            changes[key] = _convert(getattr(recipe, key), text)
    if vlaforget:
        changes["vlaforget"] = replace(recipe.vlaforget, **vlaforget)
    return replace(recipe, **changes) if changes else recipe


# Unlearning methods that train on, or compare against, the retained tasks: they cannot unlearn a
# parent's only task (vlu/ops.py's plans rely on this check, see plan_op).
NEEDS_RETAINED_TASK = frozenset({"graddiff", "npo_rt", "ssd", "salun", "vlaforget"})


def check_op(node: Node) -> None:
    """Raise ValueError if the node's last op cannot be applied to its parent. scripts/submit.py calls
    this when planning, so an impossible op fails at once rather than in a worker job."""
    op = node.last_op
    if op.kind != "U":
        return
    kept = node.parent.retained_tasks()
    if op.task not in kept:
        raise ValueError(f"{node}: cannot unlearn {op.task}, it is not retained by the parent")
    if op.method in NEEDS_RETAINED_TASK and kept == [op.task]:
        raise ValueError(f"{node}: {op.method} needs a retained task besides {op.task}")


# The recipe fields each kind of op depends on (bookkeeping such as logging, checkpoints and probes
# is left out: it does not change the result).
_TRAIN_KEYS = ("batch_size", "weight_decay", "warmup_ratio", "max_grad_norm")
_UNLEARN_KEYS = ("unlearn_steps", "unlearn_lr") + _TRAIN_KEYS
_KEYS_USED = {
    "L": ("learn_steps", "lr", "replay_per_task") + _TRAIN_KEYS,
    "ga": _UNLEARN_KEYS,
    "graddiff": _UNLEARN_KEYS,
    "npo": _UNLEARN_KEYS + ("npo_beta",),
    "npo_rt": _UNLEARN_KEYS + ("npo_beta",),
    "ssd": ("batch_size", "importance_batches", "ssd_alpha", "ssd_lambda", "ssd_reference", "ssd_adaptive"),
    "salun": _UNLEARN_KEYS + ("salun_keep", "importance_batches"),
    "vlaforget": ("unlearn_steps", "batch_size") + tuple(f"vlaforget.{f.name}" for f in fields(VLAForgetConfig)),
    "purge": (),
}


def _flat(recipe: dict) -> dict:
    return {**{k: v for k, v in recipe.items() if k != "vlaforget"},
            **{f"vlaforget.{k}": v for k, v in recipe.get("vlaforget", {}).items()}}


def recipe_changes(node: Node, recipe: Recipe) -> dict[str, tuple]:
    """{field: (value recorded in the node's op.json, value the current code would use)} for the
    fields its op depends on, if the node was trained with other values, e.g. because a default
    changed since. A field missing from op.json (added later) counts as changed. Empty for a node
    without op.json or with an unknown method."""
    path = node.dir / "op.json"
    if node.is_base or not path.exists():
        return {}
    recorded = json.loads(path.read_text()).get("recipe")
    keys = _KEYS_USED.get("L" if node.last_op.kind == "L" else node.last_op.method, ())
    if recorded is None or not keys:
        return {}
    recorded, current = _flat(recorded), _flat(json.loads(json.dumps(asdict(recipe_for(node.last_op, recipe)))))
    return {k: (recorded.get(k, "<not recorded>"), current[k]) for k in keys if recorded.get(k, "<not recorded>") != current[k]}
