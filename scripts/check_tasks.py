"""Every task of vlu/tasks.py exists in LIBERO-Goal with its training instruction and enough initial states
(make_env checks the instruction): found before a run, not at the first eval of a task, hours into it.
Run with the simulator's Python, rendering set up (slurm/check.sbatch, scripts/run_local.sh):

    $SIM_PY scripts/check_tasks.py [--required A,B]

Exit code 1 if a required task (default: every task) is not usable; the other tasks are checked too, and a
problem there is only a warning (the calibration evaluates only A and B).
"""

import argparse
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlu.config import RECIPE  # noqa: E402
from vlu.eval.libero_client import make_env  # noqa: E402
from vlu.tasks import TASKS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--required", default="", help="tasks that must be usable, e.g. A,B (default: every task)")
    args = parser.parse_args()
    required = set(args.required.split(",")) if args.required else set(TASKS)
    if required - set(TASKS):
        raise SystemExit(f"unknown task(s) {sorted(required - set(TASKS))}; tasks: {sorted(TASKS)}")
    failed = []
    for letter in TASKS:
        try:
            env, init_states = make_env(letter)  # raises if LIBERO's instruction differs from the training one
            env._env.close()  # its MuJoCo / EGL render context: one open at a time
            if len(init_states) < RECIPE.eval_episodes:
                raise ValueError(f"{len(init_states)} initial states, {RECIPE.eval_episodes} needed")
            print(f"task {letter} OK: {TASKS[letter]}, {len(init_states)} initial states", flush=True)
        except Exception:
            traceback.print_exc()
            if letter in required:
                failed.append(letter)
                print(f"task {letter} NOT USABLE: {TASKS[letter]}", flush=True)
            else:
                print(f"WARNING: task {letter} not usable ({TASKS[letter]}); this run does not evaluate it", flush=True)
    if failed:
        raise SystemExit(f"task(s) {', '.join(failed)} not usable in LIBERO (tracebacks above)")


if __name__ == "__main__":
    main()
