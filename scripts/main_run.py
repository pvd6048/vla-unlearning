"""Write the plan of the main run: stream S1 (PLAN.md week 1) with the branches of PLAN.md section 5, on a
machine without Slurm. Normally called by scripts/run_local.sh --main (see CALIBRATION.md).

    python scripts/main_run.py [--run goal_s0]

Run `goal_s0` (seed 0), LIBERO-Goal tasks A-E (vlu/tasks.py), every op with the recipe of vlu/config.py
(10k steps per learned task with ER, 1000 steps per unlearning op):
    LA/LB/LC              the shared prefix; then, per method m of METHODS, the tail UB-m/LD/LE/UE-m
    LA/LC, LA/LC/LD       the retrain oracles of UB and UE (never saw B; never saw B or E)
    LA/LB/LC/LD(/LE)      branch C of PLAN.md section 5.1 (B still replayed): next to purge's tail (branch
                          F, B dropped from replay), B's success rate and the incidental-forgetting drift of B
    evals                 every node on every task it has seen (an oracle: on the tasks of the nodes it
                          stands in for), eval seed 0, VIDEOS videos per task
    drift features        of every node, and the Fisher of A and of B at LA/LB/LC (vlu.drift.extract);
                          optional, as in the calibration; scripts/main_drift.py turns them into metrics
This is the README's "Week 1" command plus branch C; scripts/metrics.py turns the evals into PLAN.md's metrics.

METHODS holds only the methods whose hyperparameters are set (the planner refuses an unlearning op whose
method has no learning rate in vlu/config.py). The others are added after calibration round 3; rerunning
run_local.sh --main then adds their tails and keeps everything already done.
Disk: ~6.5 GB per trained node (purge's are hard links to its parent's): ~110 GB for the methods below,
~300 GB with all eight.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import build_items, make_parser, merge, warn_recipe_changes  # noqa: E402
from vlu.nodes import run_seed  # noqa: E402
from vlu.worker import load_plan, plan_path, write_plan  # noqa: E402

SEED = 0
# Calibration round 2 set these (vlu/config.py); ga, graddiff, npo, npo_rt and vlaforget follow round 3.
METHODS = ["purge", "ssd", "salun"]
CONTROL = ["LA/LB/LC/LD", "LA/LB/LC/LD/LE"]  # PLAN.md section 5.1, branch C (both evaluated, as the tails)
FISHER = "LA/LB/LC:A,B"  # at theta0 of every U_B: the curvature of B and of a retained task (main_drift.py)
VIDEOS = 2  # per task per eval, to see how the models fail (PLAN.md section 5.3, question 4)


def items(run: str) -> list[dict]:
    """The plan items, in run order (scripts/submit.py's, from its own options)."""
    args = make_parser().parse_args(["--run", run, "--stream", "S1", "--methods", ",".join(METHODS), "--nodes", ",".join(CONTROL),
                                     "--videos", str(VIDEOS), "--features", "--fisher", FISHER])
    out = build_items(args)
    for item in out:
        if item["kind"] == "features":  # drift metrics are extra data: a failure is reported, not fatal
            item["optional"] = True
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="goal_s0")
    args = parser.parse_args()
    run_seed(args.run, SEED)  # creates the run, or checks its seed
    try:
        plan = merge(load_plan(args.run), items(args.run))
    except ValueError as error:
        sys.exit(f"cannot plan: {error}")
    write_plan(args.run, plan)
    warn_recipe_changes(args.run, plan)
    print(f"main run plan {plan_path(args.run)}: {len(plan)} items; methods {', '.join(METHODS)}")


if __name__ == "__main__":
    main()
