"""Print the success-rate table of a run: one row per evaluated node, one column per task.

    python scripts/results.py --run goal_s0 [--weights final] [--seed 0] [--ci] [--csv out.csv]

Rows are node paths; "-" = not evaluated. --ci adds the 95% Wilson interval over the rollouts of each
eval (PLAN.md reporting rules). Runs anywhere (no GPU, no GR00T imports).
"""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu.config import RUNS_DIR  # noqa: E402
from vlu.stream_metrics import wilson_interval  # noqa: E402
from vlu.tasks import TASKS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--weights", default="final")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci", action="store_true", help="show the 95%% Wilson interval of each success rate")
    parser.add_argument("--csv", help="also write the table here")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run
    eval_name = f"{args.weights}_seed{args.seed}"
    rows = {}
    for eval_dir in sorted(run_dir.rglob(f"eval/{eval_name}")):
        path = str(eval_dir.parent.parent.relative_to(run_dir))
        path = "<base>" if path == "." else path
        rows[path] = {f.stem: json.loads(f.read_text()) for f in eval_dir.glob("*.json")}

    def cell(result: dict) -> str:
        text = f"{result['success_rate']:.0%}"
        if args.ci:
            episodes = result["episodes"]
            lo, hi = wilson_interval(sum(e["success"] for e in episodes), len(episodes))
            text += f" [{lo:.0%},{hi:.0%}]"
        return text

    header = ["node"] + list(TASKS)
    lines = [header] + [[p] + [cell(r[t]) if t in r else "-" for t in TASKS] for p, r in rows.items()]
    width = max(len(line[0]) for line in lines)
    cell_width = max(len(c) for line in lines for c in line[1:])
    for line in lines:
        print(line[0].ljust(width), *(c.rjust(cell_width) for c in line[1:]))
    if args.csv:
        with open(args.csv, "w", newline="") as f:  # csv quotes cells such as "80% [67%,89%]"
            csv.writer(f).writerows(lines)


if __name__ == "__main__":
    main()
