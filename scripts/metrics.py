"""Print the stream metrics (NBT, collateral, relapse, oracle gap, ...) of each unlearning method,
averaged over runs (seeds). Definitions: PLAN.md, "Evaluation metrics"; code: vlu/stream_metrics.py.

    python scripts/metrics.py --stream S1 --methods purge,ga,graddiff --runs goal_s0,goal_s1

Runs anywhere (no GPU, no GR00T imports). NaN = an eval it needs is missing.
"""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu.stream_metrics import stream_matrix, summarize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="S1")
    parser.add_argument("--methods", required=True)
    parser.add_argument("--runs", required=True, help="comma-separated, e.g. goal_s0,goal_s1")
    parser.add_argument("--weights", default="final")
    parser.add_argument("--eval-seed", type=int, default=0)
    args = parser.parse_args()

    table = {}
    for method in args.methods.split(","):
        per_run = [summarize(*stream_matrix(run, args.stream, method, args.weights, args.eval_seed))
                   for run in args.runs.split(",")]
        table[method] = {k: [r[k] for r in per_run] for k in per_run[0]}

    names = list(dict.fromkeys(k for row in table.values() for k in row))
    print("method".ljust(10), *(n.rjust(16) for n in names))
    for method, row in table.items():
        cells = []
        for n in names:
            xs = row.get(n, [math.nan])
            mean = sum(xs) / len(xs)
            std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
            cells.append(f"{mean:.3f}" + (f" ±{std:.3f}" if std is not None else ""))
        print(method.ljust(10), *(c.rjust(16) for c in cells))


if __name__ == "__main__":
    main()
