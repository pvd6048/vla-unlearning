"""A worker: runs the planned items of a run one after another, inside one Slurm allocation.

    python -m vlu.worker --run goal_s0 --kinds train,eval      (normally via slurm/worker.sbatch)

The plan is RUNS_DIR/<run>/plan.json, written by scripts/submit.py: a list of items such as
    {"kind": "train", "node": "LA/LB", "save_at": [2000, 5000]}
    {"kind": "eval", "node": "LA/LB", "weights": "final", "seed": 0, "tasks": ["A", "B"], "videos": 0}
The worker repeatedly takes the first item, in plan order, that is not done, not failed, ready (a
train item needs its parent trained, an eval item its node) and not taken by another worker, and
runs it in a subprocess: vlu.run_op for train items, slurm/eval.sbatch for eval items. It exits when
nothing is left that it can run.

Several workers may run at once: an item is taken by creating _claims/<item> (mkdir is atomic). A
claim whose Slurm job has ended (time-out, crash) is stale and is taken over; an interrupted
training resumes from its resume.pt. A failed item is recorded in _failed/<item>.json and skipped
until `scripts/submit.py --retry-failed`. Each item's output goes to _logs/<item>.<job>.log.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from vlu.config import PROJECT_DIR, RUNS_DIR
from vlu.nodes import Node

JOB = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")


def item_key(item: dict) -> str:
    node = item["node"] or "<base>"
    if item["kind"] == "train":
        return f"train:{node}"
    return f"eval:{node}:{item['weights']}:s{item['seed']}"


def _file_name(item: dict) -> str:
    """item_key made shell-friendly, e.g. eval_LA~LB_final_s0."""
    return item_key(item).replace("<base>", "base").replace("/", "~").replace(":", "_")


def plan_path(run: str) -> Path:
    return RUNS_DIR / run / "plan.json"


def load_plan(run: str) -> list[dict]:
    path = plan_path(run)
    return json.loads(path.read_text()) if path.exists() else []


def claim_dir(run: str, item: dict) -> Path:
    return RUNS_DIR / run / "_claims" / _file_name(item)


def failed_file(run: str, item: dict) -> Path:
    return RUNS_DIR / run / "_failed" / f"{_file_name(item)}.json"


# --- Item status -------------------------------------------------------------------------------


def eval_todo(run: str, item: dict) -> list[str]:
    """Tasks of an eval item without a result file yet."""
    out = Node.from_path(run, item["node"]).eval_dir(item["weights"], item["seed"])
    return [t for t in item["tasks"] if not (out / f"{t}.json").exists()]


def is_done(run: str, item: dict) -> bool:
    if item["kind"] == "train":
        return Node.from_path(run, item["node"]).done
    return not eval_todo(run, item)


def is_ready(run: str, item: dict) -> bool:
    node = Node.from_path(run, item["node"])
    return node.parent.done if item["kind"] == "train" else node.done


def status(run: str, item: dict) -> str:
    """done / failed / claimed / ready / waiting (for logs and scripts/submit.py)."""
    if is_done(run, item):
        return "done"
    if failed_file(run, item).exists():
        return "failed"
    if claim_dir(run, item).exists():
        return "claimed"
    return "ready" if is_ready(run, item) else "waiting"


# --- Claims ------------------------------------------------------------------------------------


def _job_alive(job: str) -> bool:
    out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
    if out.returncode == 0:
        return bool(out.stdout.strip())
    return "Invalid job id" not in out.stderr  # squeue itself failed: assume alive, to be safe


def try_claim(claim: Path) -> bool:
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError:
        owner = claim / "owner.json"
        if not owner.exists():
            if time.time() - claim.stat().st_mtime < 120:  # another worker is just creating it
                return False
        elif _job_alive(json.loads(owner.read_text())["job"]):
            return False
        print(f"taking over the stale claim {claim.name}", flush=True)
        shutil.rmtree(claim, ignore_errors=True)
        try:
            claim.mkdir()
        except FileExistsError:
            return False
    (claim / "owner.json").write_text(json.dumps({"job": JOB, "host": socket.gethostname(), "time": time.time()}))
    return True


# --- Running -----------------------------------------------------------------------------------


def command(run: str, item: dict) -> list[str]:
    if item["kind"] == "train":
        cmd = [sys.executable, "-m", "vlu.run_op", "--run", run, "--node", item["node"]]
        if item.get("save_at"):
            cmd += ["--save-at", ",".join(map(str, item["save_at"]))]
        return cmd
    tasks = ",".join(eval_todo(run, item))
    return ["bash", str(PROJECT_DIR / "slurm" / "eval.sbatch"), run, item["node"], tasks,
            item["weights"], str(item["seed"]), str(item.get("videos", 0))]


def run_item(run: str, item: dict) -> None:
    key = item_key(item)
    log = RUNS_DIR / run / "_logs" / f"{_file_name(item)}.{JOB}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(run, item)
    print(f"[{time.strftime('%H:%M:%S')}] start {key}\n    {' '.join(cmd)}\n    log: {log}", flush=True)
    start = time.time()
    with open(log, "a") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - start) / 60
    if code == 0 and is_done(run, item):
        print(f"[{time.strftime('%H:%M:%S')}] done  {key} ({minutes:.0f} min)", flush=True)
    else:
        path = failed_file(run, item)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"item": item, "exit_code": code, "log": str(log), "job": JOB}, indent=2) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] FAILED {key} (exit code {code}, {minutes:.0f} min): see {log}", flush=True)
    shutil.rmtree(claim_dir(run, item), ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--kinds", default="train,eval", help="item kinds this worker may run")
    args = parser.parse_args()
    kinds = set(args.kinds.split(","))

    while True:
        pending = [i for i in load_plan(args.run) if i["kind"] in kinds and status(args.run, i) in ("ready", "waiting", "claimed")]
        if not pending:
            print("nothing left to do", flush=True)
            return
        item = next((i for i in pending if is_ready(args.run, i) and try_claim(claim_dir(args.run, i))), None)
        if item is None:
            print(f"{len(pending)} item(s) left, but none is ready and free (they wait for other workers, "
                  f"or for a failed item); exiting", flush=True)
            return
        run_item(args.run, item)


if __name__ == "__main__":
    main()
