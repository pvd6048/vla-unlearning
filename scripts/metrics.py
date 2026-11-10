"""Print the stream metrics (NBT, collateral, relapse, oracle gap, ...) of each unlearning method,
averaged over runs (seeds). Definitions: PLAN.md, "Evaluation metrics"; code: vlu/stream_metrics.py.

    python scripts/metrics.py --stream S1 --methods purge,ga,graddiff --runs goal_s0,goal_s1 [--eval-seeds 0,1]

Success rates are averaged over the eval seeds (more rollouts of the same model). "± x" is the
standard deviation over runs (training seeds), shown when there are several runs; it is not the
per-eval rollout interval of PLAN.md's reporting rules. Runs anywhere (no GPU, no GR00T imports).
NaN = an eval it needs is missing, or the metric is undefined for the stream (see vlu/stream_metrics.py).
"""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu.stream_metrics import stream_matrix, summarize  # noqa: E402


def _list(text: str) -> list[str]:
    return [x for x in text.split(",") if x]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="S1")
    parser.add_argument("--methods", required=True)
    parser.add_argument("--runs", required=True, help="comma-separated, e.g. goal_s0,goal_s1")
    parser.add_argument("--weights", default="final")
    parser.add_argument("--eval-seeds", default="0", help="eval seeds to average, e.g. 0,1")
    args = parser.parse_args()
    methods, runs, seeds = _list(args.methods), _list(args.runs), tuple(int(s) for s in _list(args.eval_seeds))
    if not (methods and runs and seeds):
        sys.exit("--methods, --runs and --eval-seeds must each name at least one entry")

    table = {}
    for method in methods:
        per_run = [summarize(*stream_matrix(run, args.stream, method, args.weights, seeds)) for run in runs]
        table[method] = {k: [r[k] for r in per_run] for k in per_run[0]}

    names = list(dict.fromkeys(k for row in table.values() for k in row))
    width = max(len("method"), *(len(m) for m in table))
    columns = {n: max(16, len(n) + 1) for n in names}  # every header fits: values sit under their own name
    print("method".ljust(width), *(n.rjust(columns[n]) for n in names))
    for method, row in table.items():
        cells = []
        for n in names:
            xs = row.get(n, [math.nan])
            mean = sum(xs) / len(xs)
            std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
            cells.append(f"{mean:.3f}" + (f" ±{std:.3f}" if std is not None else ""))
        print(method.ljust(width), *(c.rjust(columns[n]) for n, c in zip(names, cells)))


if __name__ == "__main__":
    main()
