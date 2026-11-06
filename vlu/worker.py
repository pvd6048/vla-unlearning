"""A worker: runs the planned items of a run one after another, inside one Slurm allocation.

    python -m vlu.worker --run goal_s0 --kinds train,eval      (normally via slurm/worker.sbatch)

The plan is RUNS_DIR/<run>/plan.json, written by scripts/submit.py: a list of items such as
    {"kind": "train", "node": "LA/LB", "save_at": [2000, 5000]}
    {"kind": "eval", "node": "LA/LB", "weights": "final", "seed": 0, "tasks": ["A", "B"], "videos": 0}
    {"kind": "features", "node": "LA/LB", "weights": "final", "fisher": ["B"]}     (vlu/drift/extract.py)
An item marked "optional": true (the calibration's drift features) is run like any other, but the run
counts as complete without it: if it fails, that is reported, not an error.
The worker repeatedly takes the first item, in plan order, that is not done, not failed, ready (a
train item needs its parent trained, the other kinds their node) and not taken by another worker,
and runs it in a subprocess: vlu.run_op, slurm/eval.sbatch or vlu.drift.extract. It exits when
nothing is left that it can run.

Several workers may run at once: an item is taken by creating _claims/<item> (mkdir is atomic). A
claim whose Slurm job has ended (time-out, crash) is stale and is taken over; an interrupted
training resumes from its resume.pt. A failed item is recorded in _failed/<item>.json and skipped
until `scripts/submit.py --retry-failed`. Each item's output goes to _logs/<item>.<job>.log.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from vlu.config import PROJECT_DIR, RUNS_DIR
from vlu.drift.paths import features_path, fisher_path
from vlu.nodes import Node

# Who holds a claim: the Slurm job, or (without Slurm) this process on this host.
JOB = os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}@{socket.gethostname()}"


def item_key(item: dict) -> str:
    node = item["node"] or "<base>"
    if item["kind"] == "train":
        return f"train:{node}"
    if item["kind"] == "features":
        return f"features:{node}:{item['weights']}"
    return f"eval:{node}:{item['weights']}:s{item['seed']}"


def file_name(item: dict) -> str:
    """item_key made shell-friendly, e.g. eval_LA~LB_final_s0."""
    return item_key(item).replace("<base>", "base").replace("/", "~").replace(":", "_")


def plan_path(run: str) -> Path:
    return RUNS_DIR / run / "plan.json"


def load_plan(run: str) -> list[dict]:
    path = plan_path(run)
    return json.loads(path.read_text()) if path.exists() else []


def write_plan(run: str, plan: list[dict]) -> None:
    """Atomically: workers read the plan while it is being extended."""
    tmp = plan_path(run).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n")
    tmp.replace(plan_path(run))


def claim_dir(run: str, item: dict) -> Path:
    return RUNS_DIR / run / "_claims" / file_name(item)


def failed_file(run: str, item: dict) -> Path:
    return RUNS_DIR / run / "_failed" / f"{file_name(item)}.json"


# --- Item status -------------------------------------------------------------------------------


def eval_todo(run: str, item: dict) -> list[str]:
    """Tasks of an eval item without a result file yet."""
    out = Node.from_path(run, item["node"]).eval_dir(item["weights"], item["seed"])
    return [t for t in item["tasks"] if not (out / f"{t}.json").exists()]


def is_done(run: str, item: dict) -> bool:
    node = Node.from_path(run, item["node"])
    if item["kind"] == "train":
        return node.done
    if item["kind"] == "features":
        return features_path(node, item["weights"]).exists() and all(
            fisher_path(node, t, item["weights"]).exists() for t in item.get("fisher", []))
    return not eval_todo(run, item)


def is_ready(run: str, item: dict) -> bool:
    node = Node.from_path(run, item["node"])
    return node.parent.done if item["kind"] == "train" else node.done


def require_complete(run: str, kinds: set[str] | None = None) -> None:
    """A successful exit must mean all required items, including failed ones, are done. Optional items
    that are not done are only reported."""
    plan = load_plan(run)
    items = [i for i in plan if kinds is None or i["kind"] in kinds]
    missing = [i for i in items if not is_done(run, i)]
    optional = [item_key(i) for i in missing if i.get("optional")]
    if optional:
        print(f"WARNING: {len(optional)} optional item(s) not done: {', '.join(optional[:5])}"
              f"{', ...' if len(optional) > 5 else ''}", flush=True)
    incomplete = [item_key(i) for i in missing if not i.get("optional")]
    if not plan or incomplete:
        detail = ", ".join(incomplete[:5]) if incomplete else "the plan is empty"
        raise SystemExit(f"run {run} is incomplete: {detail}")


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


# squeue states of a job whose processes are gone (COMPLETING: only the epilog is still running).
ENDED_STATES = ("COMPLETING", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
                "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY")


def _job_alive(job: str) -> bool | None:
    """Whether the job holding a claim still runs; None if that cannot be checked (a process on
    another machine, no squeue here, or squeue failing)."""
    if job.startswith("local"):  # a worker without Slurm: a process on some host
        pid, _, host = job[len("local"):].partition("@")
        if host != socket.gethostname():
            return None
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if out.returncode == 0:  # listed while pending, running, ... and briefly after it ended
        return out.stdout.strip() not in ("", *ENDED_STATES)
    return False if "Invalid job id" in out.stderr else None


def _owner_job(claim: Path) -> str | None:
    """The job holding a claim, or None if its owner file is missing or not readable yet."""
    try:
        return json.loads((claim / "owner.json").read_text())["job"]
    except (OSError, ValueError, KeyError):
        return None


def _claim_is_live(claim: Path) -> bool | None:
    """As _job_alive: None if the owner cannot be checked."""
    job = _owner_job(claim)
    if job is None:  # being created by another worker right now, unless it has been like this for long
        try:
            return time.time() - claim.stat().st_mtime < 120
        except FileNotFoundError:
            return False
    return _job_alive(job)


def someone_working(run: str, verified: bool = False) -> bool:
    """Whether any claim of the run belongs to a live worker (or is being created right now). A claim
    whose owner cannot be checked counts as live, unless verified is set."""
    claims = RUNS_DIR / run / "_claims"
    states = [_claim_is_live(c) for c in claims.iterdir() if c.is_dir()] if claims.exists() else []
    return any(state or (state is None and not verified) for state in states)


@contextmanager
def _claims_lock(claims: Path):
    """Serializes claiming between the workers of a run (taking over a stale claim is a read-then-write)."""
    claims.mkdir(parents=True, exist_ok=True)
    with open(claims / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def try_claim(claim: Path) -> bool:
    with _claims_lock(claim.parent):
        if claim.exists():
            if _claim_is_live(claim) is not False:  # live, or cannot be checked: assume live, to be safe
                return False
            print(f"taking over the stale claim {claim.name}", flush=True)
            shutil.rmtree(claim, ignore_errors=True)
        claim.mkdir()
        tmp = claim / "owner.json.tmp"  # written atomically: readers never see half a file
        tmp.write_text(json.dumps({"job": JOB, "host": socket.gethostname(), "time": time.time()}))
        tmp.replace(claim / "owner.json")
    return True


def take(run: str, item: dict) -> bool:
    """Claim an item that still needs running. Checked again once claimed: another worker may have
    finished or failed it (and then released its claim) after this worker listed its pending items."""
    claim = claim_dir(run, item)
    if not try_claim(claim):
        return False
    if is_done(run, item) or failed_file(run, item).exists():
        shutil.rmtree(claim, ignore_errors=True)
        return False
    return True


# --- Running -----------------------------------------------------------------------------------


def command(run: str, item: dict) -> list[str]:
    if item["kind"] == "train":
        cmd = [sys.executable, "-m", "vlu.run_op", "--run", run, "--node", item["node"]]
        if item.get("save_at"):
            cmd += ["--save-at", ",".join(map(str, item["save_at"]))]
        return cmd
    if item["kind"] == "features":
        cmd = [sys.executable, "-m", "vlu.drift.extract", "--run", run, "--node", item["node"], "--weights", item["weights"]]
        if item.get("fisher"):
            cmd += ["--fisher", ",".join(item["fisher"])]
        return cmd
    tasks = ",".join(eval_todo(run, item))
    return ["bash", str(PROJECT_DIR / "slurm" / "eval.sbatch"), run, item["node"], tasks,
            item["weights"], str(item["seed"]), str(item.get("videos", 0))]


def run_item(run: str, item: dict) -> None:
    key = item_key(item)
    log = RUNS_DIR / run / "_logs" / f"{file_name(item)}.{JOB}.log"
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
    parser.add_argument("--kinds", default="train,eval,features", help="item kinds this worker may run")
    parser.add_argument("--wait", action="store_true",
                        help="when nothing is ready, wait while other workers are busy (several workers on one machine)")
    parser.add_argument("--check-complete", action="store_true", help="check the whole plan without starting work")
    args = parser.parse_args()
    if args.check_complete:
        require_complete(args.run)
        return
    kinds = set(args.kinds.split(","))

    idle_checks = 0
    while True:
        pending = [i for i in load_plan(args.run) if i["kind"] in kinds and status(args.run, i) in ("ready", "waiting", "claimed")]
        if not pending:
            require_complete(args.run, kinds)
            print("nothing left to do", flush=True)
            return
        item = next((i for i in pending if is_ready(args.run, i) and take(args.run, i)), None)
        if item is None:
            # Three idle checks a minute apart, so a worker between two items (or still starting) does
            # not look idle.
            idle_checks = 0 if args.wait and someone_working(args.run) else idle_checks + 1
            if not args.wait or idle_checks >= 3:
                print(f"{len(pending)} item(s) left, but none is ready and free (they wait for other workers, "
                      f"or for a failed item); exiting", flush=True)
                if someone_working(args.run, verified=True):
                    # Other workers are still on the plan (e.g. an eval-only Slurm worker while the trainer
                    # trains): leaving the rest to them is not a failure. Only if their jobs are known to
                    # run: a claim that cannot be checked (squeue failing, another machine's process) may
                    # be stale, and then nobody would finish the plan. (With --wait this branch is only
                    # reached when nobody is working.)
                    return
                if someone_working(args.run):
                    print("a claim's worker could not be checked (squeue failing, or another machine's process): "
                          "not leaving the rest to it", flush=True)
                require_complete(args.run, kinds)
                return
            time.sleep(60)
            continue
        idle_checks = 0
        run_item(args.run, item)


if __name__ == "__main__":
    main()
