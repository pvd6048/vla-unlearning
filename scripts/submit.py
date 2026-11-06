"""Add training and evaluation items to a run's plan, and start worker jobs that work through it.

    # Day 2: zero-shot floor, then a single-task run with snapshots, evaluated at each snapshot, 2 seeds
    python scripts/submit.py --run goal_s0 --seed 0 --zero-shot A --nodes LA --save-at 2000,5000 \\
        --eval-weights step_002000,step_005000,final --eval-seeds 0,1 --videos 5
    # Week 1: stream S1 for several unlearning methods, plus the retrain oracles
    python scripts/submit.py --run goal_s0 --stream S1 --methods purge,ga,graddiff
    # Only show the plan's progress (and start a worker if work is left but none is queued)
    python scripts/submit.py --run goal_s0

Why workers instead of one Slurm job per item: the cluster allows 1 running and 2 submitted gpu_large
jobs per user, and at most one GPU's worth of slices at a time (4 small, 1 medium or 1 large), so
work is serial anyway. A worker (vlu/worker.py) runs items one after another in one allocation and
queues its own successor before its time limit. Rerunning this script is always safe: items merge
into the plan, done items are skipped, and a worker is only started if none is queued or running.

Every trained node is evaluated on the tasks it has seen. A retrain-oracle node is evaluated on the
tasks of the stream nodes it stands in for (so including the unlearned ones). Purge nodes are not
evaluated: their weights are their parent's.
"""

import argparse
from collections import Counter
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu.config import PROJECT_DIR, RECIPE, RUNS_DIR  # noqa: E402
from vlu.nodes import Node, check_op, expand_stream, oracle_path, recipe_changes, run_seed  # noqa: E402
from vlu.tasks import STREAMS  # noqa: E402
from vlu.worker import failed_file, item_key, load_plan, plan_path, status, write_plan  # noqa: E402

# Worker pools, from the IIT Guwahati GPU facility job templates. "kinds" = what a worker may run:
# training needs a big slice; a small slice only evaluates.
POOLS = {
    "large": {"kinds": "train,eval,features", "sbatch": ["--partition=gpu_large", "--qos=gpu_large", "--gres=gpu:4g.96gb:1",
                                                "--cpus-per-gpu=24", "--mem-per-gpu=96G", "--time=2-00:00:00"]},
    "medium": {"kinds": "train,eval,features", "sbatch": ["--partition=gpu_medium", "--qos=gpu_medium", "--gres=gpu:2g.48gb:1",
                                                 "--cpus-per-gpu=12", "--mem-per-gpu=48G", "--time=2-00:00:00"]},
    "small": {"kinds": "eval", "sbatch": ["--partition=gpu_small", "--qos=gpu_small", "--gres=gpu:1g.24gb:1",
                                          "--cpus-per-gpu=6", "--mem-per-gpu=24G", "--time=1-00:00:00"]},
}


def build_items(args) -> list[dict]:
    """The plan items asked for on the command line, in the order they should run."""
    targets: dict[str, list[str]] = {}  # node path -> tasks to evaluate it on
    for path in filter(None, args.nodes.split(",")):
        targets[path] = Node.from_path(args.run, path).seen_tasks()
    if args.stream:
        methods = [m for m in args.methods.split(",") if m]
        if not methods:
            sys.exit("--stream needs --methods")
        for method in methods:
            for path in expand_stream(STREAMS[args.stream], method):
                seen = Node.from_path(args.run, path).seen_tasks()
                targets[path] = seen
                if not args.no_oracle and path.split("/")[-1].startswith("U"):
                    oracle = oracle_path(args.run, path)
                    targets[oracle] = sorted(set(targets.get(oracle, [])) | set(seen))

    save_at = sorted(int(s) for s in args.save_at.split(",") if s)
    items = []
    if args.zero_shot:
        items.append({"kind": "eval", "node": "", "weights": "final", "seed": 0,
                      "tasks": args.zero_shot.split(","), "videos": args.videos})
    if args.features:
        items.append({"kind": "features", "node": "", "weights": "final", "fisher": []})
    for path in sorted(targets, key=lambda p: (p.count("/"), p)):
        node = Node.from_path(args.run, path)
        # --save-at / snapshot evals are for L ops: U ops have their own, shorter step budget (and SSD
        # and VLA-Forget cannot save snapshots at all).
        snapshots = not node.is_base and node.last_op.kind == "L"
        for depth in range(1, node.depth + 1):  # missing ancestors first (done ones are skipped later)
            ancestor = "/".join(str(op) for op in node.ops[:depth])
            items.append({"kind": "train", "node": ancestor, "save_at": save_at if depth == node.depth and snapshots else []})
            if args.features and not node.ops[depth - 1].shares_parent_weights:  # e.g. purge: the parent's weights
                items.append({"kind": "features", "node": ancestor, "weights": "final", "fisher": []})
        if args.no_eval or (not node.is_base and node.last_op.shares_parent_weights):
            continue
        # The base model (the retrain oracle when nothing is retained) and U ops have no snapshots.
        for weights in args.eval_weights.split(",") if snapshots else ["final"]:
            for seed in map(int, args.eval_seeds.split(",")):
                items.append({"kind": "eval", "node": path, "weights": weights, "seed": seed,
                              "tasks": targets[path], "videos": args.videos})
    for spec in args.fisher:
        path, _, tasks = spec.rpartition(":")
        path = "" if path == "base" else path
        node = Node.from_path(args.run, path)
        for depth in range(1, node.depth + 1):  # the node must be trained (done ones are skipped later)
            items.append({"kind": "train", "node": "/".join(str(op) for op in node.ops[:depth]), "save_at": []})
        items.append({"kind": "features", "node": path, "weights": "final", "fisher": tasks.split(",")})
    for item in items:  # an op that cannot be applied fails now, not in a worker job
        if item["kind"] == "train":
            check_op(Node.from_path(args.run, item["node"]))
    return items


def warn_recipe_changes(run: str, plan: list[dict]) -> list[str]:
    """Print (and return) a warning for every trained node of the plan whose op.json records other
    values for the fields its op depends on than the current code would use (e.g. a default changed
    after it was trained): its results, and everything trained on top of it, are then not what the
    code now describes. Such a node is never retrained by itself: delete it to retrain it."""
    warnings = []
    for item in plan:
        if item["kind"] == "train":
            node = Node.from_path(run, item["node"])
            changes = node.done and recipe_changes(node, RECIPE)
            if changes:
                text = ", ".join(f"{k}: {old!r} -> now {new!r}" for k, (old, new) in changes.items())
                warnings.append(f"WARNING: {node} was trained with another recipe ({text})")
                print(warnings[-1])
    return warnings


def merge(plan: list[dict], new: list[dict]) -> list[dict]:
    """Add new items to the plan, keeping its order. Repeated items are merged (union of snapshot
    steps / tasks, max of videos; required if either is)."""
    by_key = {item_key(i): i for i in plan}
    for item in new:
        old = by_key.get(item_key(item))
        if old is None:
            by_key[item_key(item)] = item
            plan.append(item)
            continue
        if not item.get("optional"):
            old.pop("optional", None)
        if item["kind"] == "train":
            old["save_at"] = sorted(set(old.get("save_at", [])) | set(item["save_at"]))
        elif item["kind"] == "features":
            old["fisher"] = sorted(set(old.get("fisher", [])) | set(item["fisher"]))
        else:
            old["tasks"] = sorted(set(old["tasks"]) | set(item["tasks"]))
            old["videos"] = max(old.get("videos", 0), item["videos"])
    return plan


def queued_workers(run: str) -> Counter:
    """Number of this run's worker jobs that are pending or running, per pool."""
    out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%j"], capture_output=True, text=True, check=True)
    prefix = f"vlu-worker:{run}:"
    return Counter(name[len(prefix):] for name in out.stdout.split() if name.startswith(prefix))


def start_worker(run: str, pool: str, dry_run: bool) -> None:
    log_dir = RUNS_DIR / run / "_slurm"
    options = [f"--job-name=vlu-worker:{run}:{pool}", f"--output={log_dir}/worker-{pool}-%j.out",
               "--signal=B:USR1@900", *POOLS[pool]["sbatch"]]
    cmd = ["sbatch", "--parsable", *options, str(PROJECT_DIR / "slurm" / "worker.sbatch"), run, POOLS[pool]["kinds"]]
    print(" ".join(cmd))
    if dry_run:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "VLU_PROJECT_DIR": str(PROJECT_DIR), "VLU_WORKER_SBATCH": " ".join(map(shlex.quote, options))}
    job = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout.strip().split(";")[0]
    print(f"started worker job {job} ({pool}); log: {log_dir}/worker-{pool}-{job}.out")


def make_parser() -> argparse.ArgumentParser:
    """This script's options (also used by scripts/main_run.py to plan the main run)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, help="run seed; needed when the run does not exist yet")
    parser.add_argument("--nodes", default="", help="comma-separated node paths, e.g. LA,LA/LB")
    parser.add_argument("--stream", choices=sorted(STREAMS))
    parser.add_argument("--methods", default="", help="unlearning methods for --stream, e.g. purge,ga")
    parser.add_argument("--no-oracle", action="store_true", help="with --stream: skip the retrain oracles")
    parser.add_argument("--zero-shot", default="", help="evaluate the base model on these tasks, e.g. A")
    parser.add_argument("--save-at", default="", help="extra weight snapshots of the trained nodes, e.g. 2000,5000")
    parser.add_argument("--eval-weights", default="final")
    parser.add_argument("--eval-seeds", default="0")
    parser.add_argument("--videos", type=int, default=0, help="videos per task per eval")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--features", action="store_true",
                        help="also extract drift probe features of the base model and every trained node")
    parser.add_argument("--fisher", action="append", default=[],
                        help="NODE:TASKS, e.g. LA/LB/LC:B or base:A,B: diagonal Fisher of a node on tasks (repeatable)")
    parser.add_argument("--workers", default="large",
                        help="pools to keep a worker in, e.g. large, or large,small:2; 'none' = only update the plan")
    parser.add_argument("--retry-failed", action="store_true", help="forget earlier failures, so they are run again")
    parser.add_argument("--dry-run", action="store_true", help="print the new plan items and sbatch commands only")
    return parser


def main():
    args = make_parser().parse_args()

    if args.dry_run:
        run_json = RUNS_DIR / args.run / "run.json"
        if not run_json.exists() and args.seed is None:
            sys.exit(f"run {args.run} does not exist yet: pass --seed")
    else:
        run_seed(args.run, args.seed)  # creates the run, or checks that --seed matches it

    try:
        plan = merge(load_plan(args.run), build_items(args))
    except ValueError as error:
        sys.exit(f"cannot plan: {error}")
    warn_recipe_changes(args.run, plan)
    if args.dry_run:
        for item in plan:
            print(f"{status(args.run, item):8s} {item_key(item)}" + (f"  tasks {','.join(item['tasks'])}" if item["kind"] == "eval" else ""))
    else:
        if args.retry_failed:
            for item in plan:
                failed_file(args.run, item).unlink(missing_ok=True)
        write_plan(args.run, plan)

    counts = Counter(status(args.run, i) for i in plan)
    print(f"plan {plan_path(args.run)}: {len(plan)} items, " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    for item in plan:
        if status(args.run, item) == "failed":
            print(f"  failed: {item_key(item)}  ({json.loads(failed_file(args.run, item).read_text())['log']})")

    if args.workers == "none" or counts["ready"] + counts["waiting"] + counts["claimed"] == 0:
        return
    running = Counter() if args.dry_run else queued_workers(args.run)
    for spec in args.workers.split(","):
        pool, _, n = spec.partition(":")
        if pool not in POOLS:
            sys.exit(f"unknown pool {pool!r}; known: {sorted(POOLS)}")
        for _ in range(max(0, int(n or 1) - running[pool])):
            start_worker(args.run, pool, args.dry_run)


if __name__ == "__main__":
    main()
