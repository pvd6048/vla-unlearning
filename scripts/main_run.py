"""Write the plan of the main run: stream S1 (PLAN.md week 1) with the branches of PLAN.md section 5, on a
machine without Slurm. Normally called by scripts/run_local.sh --main (see CALIBRATION.md).

    python scripts/main_run.py [--run goal_s0]
    python scripts/main_run.py --run goal_s0 --methods 'rmu@unlearn_lr=5e-5+rmu_layer=7'

--methods replaces METHODS for this call (purge is always added). The plan only grows (submit.merge): the
methods already planned stay. Add methods to goal_s0, not to a new run name: every run trains its own nodes
(runs/<run>/), so a new run would retrain the whole stream (LA..LE, 10k steps each, and the oracles).

Run `goal_s0` (seed 0), LIBERO-Goal tasks A-E (vlu/tasks.py), every op with the recipe of vlu/config.py
(10k steps per learned task with ER, 1000 steps per unlearning op):
    LA/LB/LC              the shared prefix; then, per method m of METHODS, the tail UB-m/LD/LE/UE-m
    LA/LC, LA/LC/LD       the retrain oracles of UB and UE (never saw B; never saw B or E)
    LA/LB/LC/LD(/LE)      branch C of PLAN.md section 5.1 (B still replayed): next to purge's tail (branch
                          F, B dropped from replay), B's success rate and the incidental-forgetting drift of B
    evals                 every node on every task it has seen (an oracle: on the tasks of the nodes it
                          stands in for), eval seed 0, VIDEOS videos per task
    drift features        of the nodes scripts/main_drift.py measures drifts between, and the Fisher of A
                          and of B at LA/LB/LC (vlu.drift.extract); optional, as in the calibration
This is the README's "Week 1" command plus branch C; scripts/metrics.py turns the evals into PLAN.md's metrics.

METHODS holds only the methods whose hyperparameters are set (the planner refuses an unlearning op whose
method has no learning rate in vlu/config.py). The others are added after calibration round 3; rerunning
run_local.sh --main then adds their tails and keeps everything already done.
Disk: ~6.5 GB per trained node (purge's are hard links to its parent's): ~110 GB for the methods below,
~380 GB with all of round 3's methods (purge, SSD, SalUn, NPO, NPO-RT, GU on NPO-RT, Sharp MinMax,
VLA-Forget, task arithmetic, RMU; GA and GradDiff are dropped; ~26 GB per method's tail).
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
# Calibration round 2 set these (vlu/config.py). npo, npo_rt, npo_rt_gu, sharp_minmax, vlaforget, task_arithmetic
# and rmu join after round 3 sets their rates (unlearn_lrs, vlaforget.lr; sharp_select, task_arithmetic_scale,
# rmu_layer too); ga and graddiff are dropped.
METHODS = ["purge", "ssd", "salun"]
CONTROL = ["LA/LB/LC/LD", "LA/LB/LC/LD/LE"]  # PLAN.md section 5.1, branch C (both evaluated, as the tails)
FISHER = "LA/LB/LC:A,B"  # at theta0 of every U_B: the curvature of B and of a retained task (main_drift.py)
VIDEOS = 2  # per task per eval, to see how the models fail (PLAN.md section 5.3, question 4)


def with_controls(methods: list[str]) -> list[str]:
    """The drift comparison always needs the vanilla purge branch, even for a custom method list."""
    return list(dict.fromkeys(["purge", *methods]))


def items(run: str, methods: list[str] | None = None) -> list[dict]:
    """The plan items, in run order (scripts/submit.py's, from its own options), with probe features only of
    the nodes the drift metrics use (each extraction loads the model on a training GPU)."""
    from main_drift import endpoints  # here: main_drift imports this module

    methods = with_controls(METHODS if methods is None else methods)
    args = make_parser().parse_args(["--run", run, "--stream", "S1", "--methods", ",".join(methods), "--nodes", ",".join(CONTROL),
                                     "--videos", str(VIDEOS), "--features", "--fisher", FISHER])
    used = set(endpoints(methods))
    out = [i for i in build_items(args) if i["kind"] != "features" or i["node"] in used]
    for item in out:
        if item["kind"] == "features":  # drift metrics are extra data: a failure is reported, not fatal
            item["optional"] = True
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="goal_s0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--methods", help="comma-separated methods, optionally with @recipe overrides; defaults to calibrated methods")
    args = parser.parse_args()
    methods = METHODS if args.methods is None else [m for m in args.methods.split(",") if m]
    if not methods:
        parser.error("--methods must not be empty")
    methods = with_controls(methods)
    run_seed(args.run, args.seed)  # creates the run, or checks its seed
    try:
        plan = merge(load_plan(args.run), items(args.run, methods))
    except ValueError as error:
        sys.exit(f"cannot plan: {error}")
    write_plan(args.run, plan)
    warn_recipe_changes(args.run, plan)
    print(f"main run plan {plan_path(args.run)}: {len(plan)} items; methods {', '.join(methods)}")


if __name__ == "__main__":
    main()
