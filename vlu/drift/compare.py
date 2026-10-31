"""Representation and policy drift between checkpoints, from their probe features (vlu.drift.extract).
CPU is enough (the inputs are a few hundred probe samples).

    python -m vlu.drift.compare --run goal_s0 \\
        --drift "U=LA/LB/LC..LA/LB/LC/UB-ga" --drift "F=LA/LB/LC/LD/LE..LA/LB/LC/UB-purge/LD/LE" --out rep.csv

Drifts and checkpoint references are written as in vlu.drift.params. Every metric is computed per
probe task (A..E) and over all probe samples ("all"), for every feature (layer) of the probe file.
Output CSV columns:  feature, task, metric, drift_i, drift_j, value

Per drift X..Y:
    cka_distance     1 - linear CKA(h_X, h_Y)
    pca_overlap      overlap of the top-k PCA subspaces of h_X and h_Y (vlu.drift.metrics.pca_shift)
    pca_distance     movement of the samples in h_X's top-k PCA space, relative to h_X's spread
  feature "policy":
    flow_divergence  mean squared difference of the predicted velocities at the same noisy action and
                     timestep (normalized action units; a Monte-Carlo estimate over GR00T's training
                     distribution of timesteps)
    fm_loss_change   L_FM(Y) - L_FM(X) on the probe samples
Per pair of drifts i < j:
    shift_cka        linear CKA between the representation shifts h_Y - h_X of the two drifts
  feature "policy":
    velocity_shift_cos  cosine between the two drifts' changes of predicted velocity
"""

import argparse
import csv
import itertools
from pathlib import Path

import torch

from vlu.drift.metrics import cosine, linear_cka, pca_shift
from vlu.drift.params import parse_drift
from vlu.drift.paths import features_path
from vlu.drift.weights import Ref


def load(ref: Ref, cache: dict) -> dict:
    if ref not in cache:
        path = features_path(ref.node, ref.weights)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: extract features of {ref} first (scripts/submit.py --features)")
        cache[ref] = torch.load(path, map_location="cpu", weights_only=False)
    return cache[ref]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="default run of checkpoint references")
    parser.add_argument("--drift", action="append", required=True, help="NAME=X..Y (repeatable)")
    parser.add_argument("--pca-k", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = {}
    drifts = [(name, load(x, cache), load(y, cache)) for name, x, y in (parse_drift(d, args.run) for d in args.drift)]
    tasks = drifts[0][1]["tasks"]
    for _, a, b in drifts:
        if a["tasks"] != tasks or b["tasks"] != tasks or a["probe"] != b["probe"]:
            raise ValueError("probe sets differ between checkpoints; re-extract with the same vlu.drift.extract")
    subsets = {t: torch.tensor([i for i, s in enumerate(tasks) if s == t]) for t in dict.fromkeys(tasks)}
    subsets["all"] = torch.arange(len(tasks))

    rows = []
    for feature in drifts[0][1]["features"]:
        for task, idx in subsets.items():
            shifts = {}
            for name, a, b in drifts:
                ha, hb = a["features"][feature][idx].float(), b["features"][feature][idx].float()
                shifts[name] = hb - ha
                rows.append([feature, task, "cka_distance", name, "", 1 - linear_cka(ha, hb)])
                for metric, value in pca_shift(ha, hb, args.pca_k).items():
                    rows.append([feature, task, metric, name, "", value])
            for i, j in itertools.combinations(shifts, 2):
                rows.append([feature, task, "shift_cka", i, j, linear_cka(shifts[i], shifts[j])])

    for task, idx in subsets.items():
        vshift = {}
        for name, a, b in drifts:
            va, vb = a["velocity"][idx], b["velocity"][idx]
            vshift[name] = vb - va
            rows.append(["policy", task, "flow_divergence", name, "", float((vb - va).square().mean())])
            loss_a = (a["sse"][idx] / a["n_valid"][idx]).mean()
            loss_b = (b["sse"][idx] / b["n_valid"][idx]).mean()
            rows.append(["policy", task, "fm_loss_change", name, "", float(loss_b - loss_a)])
        for i, j in itertools.combinations(vshift, 2):
            rows.append(["policy", task, "velocity_shift_cos", i, j, cosine(vshift[i], vshift[j])])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", "task", "metric", "drift_i", "drift_j", "value"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
