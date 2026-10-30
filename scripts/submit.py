"""Submit the training and evaluation jobs for a set of nodes, each chained after its parent.

    # Day 2: single-task sanity run with snapshots, evaluated at each snapshot and with 2 seeds
    python scripts/submit.py --run goal_s0 --seed 0 --nodes LA --save-at 2000,5000 \\
        --eval-weights step_002000,step_005000,final --eval-seeds 0,1 --videos 5
    # Zero-shot floor of the base model on task A
    python scripts/submit.py --run goal_s0 --seed 0 --zero-shot A
    # Week 1: stream S1 for several unlearning methods, plus the retrain oracles
    python scripts/submit.py --run goal_s0 --seed 0 --stream S1 --methods purge,ga,graddiff

Nodes that are done are skipped, missing ancestors are added, and a node already queued or running
(same job name in squeue) is reused as a dependency instead of being submitted twice. Every node
is evaluated on the tasks it has seen; an oracle node is evaluated on the tasks of the stream node
it stands in for (so it includes the unlearned ones). Use --dry-run to print the commands only.
"""

import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu.config import PROJECT_DIR, RUNS_DIR  # noqa: E402
from vlu.nodes import Node, expand_stream, oracle_path  # noqa: E402
from vlu.tasks import STREAMS  # noqa: E402

# Cluster resources. EDIT to match the cluster's partitions/GRES names.
RESOURCES = {
    "train": ["--partition=gpu_large", "--gres=gpu:1", "--cpus-per-task=24", "--mem=90G", "--time=2-00:00:00"],
    "eval": ["--partition=gpu_small", "--gres=gpu:1", "--cpus-per-task=6", "--mem=22G", "--time=1-00:00:00"],
}


def queued_jobs() -> dict[str, str]:
    """{job name: job id} of this user's pending/running jobs."""
    out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-O", "Name:300,JobID:20"], capture_output=True, text=True)
    return dict(line.split() for line in out.stdout.splitlines() if line.strip())


class Submitter:
    def __init__(self, run: str, dry_run: bool):
        self.run, self.dry_run = run, dry_run
        self.queued = {} if dry_run else queued_jobs()
        self.train_jobs: dict[str, str | None] = {}  # node path -> job id (None = already done)
        self.log_dir = RUNS_DIR / run / "_slurm"

    def sbatch(self, name: str, kind: str, script: str, args: list[str], after: list[str]) -> str:
        if name in self.queued:
            print(f"[queued already] {name} = job {self.queued[name]}")
            return self.queued[name]
        log_name = name.replace("/", "_").replace(":", "_")
        cmd = ["sbatch", "--parsable", f"--job-name={name}", f"--output={self.log_dir}/{log_name}-%j.out", *RESOURCES[kind]]
        deps = [j for j in after if j]
        if deps:
            cmd.append("--dependency=afterok:" + ":".join(deps))
        cmd += [str(PROJECT_DIR / "slurm" / script), *args]
        print(" ".join(cmd))
        if self.dry_run:
            return f"<{name}>"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "VLU_PROJECT_DIR": str(PROJECT_DIR)}
        job = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout.strip().split(";")[0]
        self.queued[name] = job
        return job

    def train(self, path: str, extra: list[str]) -> str | None:
        """Submit `path` and any unfinished ancestors; returns the job id (None if done)."""
        if path in self.train_jobs:
            return self.train_jobs[path]
        node = Node.from_path(self.run, path)
        if node.done:
            self.train_jobs[path] = None
            return None
        parent_job = self.train(node.parent.path, extra) if node.depth > 1 else None
        kind = "eval" if node.last_op.method == "purge" else "train"  # purge only copies weights
        job = self.sbatch(f"vlu-train:{self.run}:{path}", kind, "train_op.sbatch", [self.run, path, *extra], [parent_job])
        self.train_jobs[path] = job
        return job

    def eval(self, path: str, tasks: list[str], weights: str, seed: int, videos: int, after: str | None):
        node = Node.from_path(self.run, path)
        todo = [t for t in tasks if not (node.eval_dir(weights, seed) / f"{t}.json").exists()]
        if not todo:
            return
        name = f"vlu-eval:{self.run}:{path or '<base>'}:{weights}:s{seed}:{''.join(todo)}"
        self.sbatch(name, "eval", "eval.sbatch", [self.run, path, ",".join(todo), weights, str(seed), str(videos)], [after])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, help="run seed; needed when the run does not exist yet")
    parser.add_argument("--nodes", default="", help="comma-separated node paths, e.g. LA,LA/LB")
    parser.add_argument("--stream", choices=sorted(STREAMS))
    parser.add_argument("--methods", default="", help="unlearning methods for --stream, e.g. purge,ga")
    parser.add_argument("--no-oracle", action="store_true", help="with --stream: skip the retrain oracles")
    parser.add_argument("--zero-shot", default="", help="evaluate the base model on these tasks, e.g. A")
    parser.add_argument("--save-at", default="", help="passed to vlu.run_op for every trained node")
    parser.add_argument("--eval-weights", default="final")
    parser.add_argument("--eval-seeds", default="0")
    parser.add_argument("--videos", type=int, default=0, help="videos per task per eval")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_json = RUNS_DIR / args.run / "run.json"
    if not run_json.exists() and args.seed is None:
        sys.exit(f"run {args.run} does not exist yet: pass --seed")
    extra = (["--seed", str(args.seed)] if args.seed is not None else []) + (["--save-at", args.save_at] if args.save_at else [])

    # Which nodes to train and evaluate, and on which tasks.
    targets: dict[str, list[str]] = {}  # node path -> eval tasks
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

    sub = Submitter(args.run, args.dry_run)
    for path in sorted(targets, key=lambda p: (p.count("/"), p)):
        job = sub.train(path, extra)
        if args.no_eval or path.endswith("-purge"):  # purge leaves the weights unchanged: same SR as parent
            continue
        for weights in args.eval_weights.split(","):
            for seed in map(int, args.eval_seeds.split(",")):
                sub.eval(path, targets[path], weights, seed, args.videos, job)
    if args.zero_shot:
        sub.eval("", args.zero_shot.split(","), "final", 0, args.videos, None)


if __name__ == "__main__":
    main()
