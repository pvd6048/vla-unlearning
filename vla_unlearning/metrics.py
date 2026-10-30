"""Metrics from success matrices (definitions: PLAN.md, "Evaluation metrics"). No GPU needed.

    python -m vla_unlearning.metrics s1     # methods x metrics for stream s1, mean over seeds

A matrix has one row per op ("L A", "U B", ...) and one success-rate column per task.
"""

import argparse
from statistics import mean

import pandas as pd
import yaml

from vla_unlearning.paths import CONFIGS, RESULTS


def avg(xs):
    xs = list(xs)
    return mean(xs) if xs else float("nan")


def summarize(m, oracle=None):
    ops = list(m["op"])
    last = len(ops) - 1
    learned = {op.split()[1]: i for i, op in enumerate(ops) if op.startswith("L")}
    unlearned = {op.split()[1]: i for i, op in enumerate(ops) if op.startswith("U")}
    c = lambda t, k: m.at[k, t]  # noqa: E731  success rate of task t after op k
    kept_at = lambda k: [t for t, i in learned.items() if i <= k and unlearned.get(t, last + 1) > k]  # noqa: E731

    # Continual learning: is the kept knowledge safe?
    drops = {t: [c(t, k) - c(t, j) for j in range(k + 1, unlearned.get(t, last + 1))] for t, k in learned.items()}
    retain_curve = [avg(c(t, k) for t in kept_at(k)) for k in range(len(ops))]
    out = {
        "retain_sr": retain_curve[-1],
        "retain_auc": avg(retain_curve),
        "nbt": avg(avg(d) for d in drops.values() if d),
        "max_forgetting": max((max(d) for d in drops.values() if d), default=float("nan")),
        "retain_stability": avg(abs(b - a) for a, b in zip(retain_curve, retain_curve[1:])),
    }

    # Unlearning: is the removed task really gone, and what did removing it cost?
    if unlearned:
        out["forget_sr"] = avg(c(t, j) for t, j in unlearned.items())
        out["collateral"] = avg(c(t, j - 1) - c(t, j) for j in unlearned.values() for t in kept_at(j))
        out["relapse"] = avg(max(c(t, k) for k in range(j, last + 1)) - c(t, j) for t, j in unlearned.items())
        if oracle is not None:
            out["oracle_gap"] = avg(abs(c(t, last) - oracle[t].iloc[-1]) for t in unlearned)
            a, b = 1 - out["oracle_gap"], out["retain_sr"]
            out["score"] = 2 * a * b / (a + b) if a + b > 0 else 0.0
        out["unlearn_minutes"] = m["unlearn_minutes"].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    args = ap.parse_args()
    stream = yaml.safe_load((CONFIGS / "streams" / f"{args.stream}.yaml").read_text())

    oracles = {}
    if stream.get("oracle"):
        for f in (RESULTS / stream["oracle"]).glob("*/seed*/matrix.csv"):
            oracles[f.parent.name] = pd.read_csv(f)

    rows = []
    for f in sorted((RESULTS / args.stream).glob("*/seed*/matrix.csv")):
        method, seed = f.parent.parent.name, f.parent.name
        rows.append({"method": method, "seed": seed, **summarize(pd.read_csv(f), oracles.get(seed))})
    table = pd.DataFrame(rows).drop(columns="seed").groupby("method")
    print(table.mean().round(3).to_string())
    if table.size().max() > 1:
        print("\nstd over seeds:\n" + table.std().round(3).to_string())


if __name__ == "__main__":
    main()
