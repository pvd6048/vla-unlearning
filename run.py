"""Run one experiment: a stream of learn/unlearn ops with one unlearning method and seed.

    python run.py --stream s1 --method ga --seed 0
    python run.py --stream s1_oracle --method none     # retrain oracle (no U ops)
"""

import argparse

from vla_unlearning.runner import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, help="configs/streams/<name>.yaml")
    ap.add_argument("--method", default="none", help="configs/methods/<name>.yaml, or 'none'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.stream, args.method, args.seed)
