"""Ops, nodes and the on-disk run tree.

An op is a token:
    L<task>            learn a task, e.g. "LB"
    U<task>-<method>   unlearn a task with a method, e.g. "UB-ga"

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

from dataclasses import dataclass
import json
from pathlib import Path

from vlu.config import RUNS_DIR
from vlu.tasks import TASKS


@dataclass(frozen=True)
class Op:
    kind: str  # "L" or "U"
    task: str  # task letter
    method: str | None = None  # unlearning method, only for U ops

    @classmethod
    def parse(cls, token: str) -> "Op":
        kind, rest = token[0], token[1:]
        task, _, method = rest.partition("-")
        if kind not in ("L", "U") or task not in TASKS:
            raise ValueError(f"Bad op {token!r}: expected L<task> or U<task>-<method>, tasks {list(TASKS)}")
        if kind == "U" and not method:
            raise ValueError(f"U op {token!r} needs a method, e.g. U{task}-ga")
        if kind == "L" and method:
            raise ValueError(f"L op {token!r} takes no method")
        return cls(kind, task, method or None)

    def __str__(self) -> str:
        return f"{self.kind}{self.task}" + (f"-{self.method}" if self.method else "")


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
