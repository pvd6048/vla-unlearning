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
    pca_similarity, pca_centroid_shift   Xu et al.'s PCA similarity and PCA shift ("Unlearning Isn't
                     Deletion"), see vlu.drift.metrics.pca_shift; the mean of pca_centroid_shift over
                     the layers is their mean PCA distance
  feature "policy":
    flow_divergence  mean squared difference of the predicted velocities at the same noisy action and
                     timestep (normalized action units; a Monte-Carlo estimate over GR00T's training
                     distribution of timesteps)
    fm_loss_change   L_FM(Y) - L_FM(X) on the probe samples
Per pair of drifts i < j (with --pairs-with NAME, repeatable: only the pairs including a named drift):
    shift_cka        linear CKA between the representation shifts h_Y - h_X of the two drifts
  feature "policy":
    velocity_shift_cos  cosine between the two drifts' changes of predicted velocity

Cross-modal alignment, per drift, each as <metric>_start (at X), <metric>_end (at Y), <metric>_change:
    vl_alignment     (features vlm.layerXX / vl) mean cosine between a sample's pooled image tokens
                     and its pooled text tokens
    attn_ratio       (each DiT block) size of the attention update relative to the block input; in
                     x-text / x-image blocks: how much the action head draws on language / vision
    language_sensitivity  (feature "policy") mean ||v(swapped instruction) - v||^2 / mean ||v||^2:
                     0 = the policy ignores the instruction ("linguistic blindness")

Non-finite values, per drift and feature (task "all"):
    nonfinite_start, nonfinite_end   number of inf / NaN entries at X and at Y (a DiT block: its features
                     and attention ratios; feature "policy": the velocities and flow-matching errors).
                     Non-zero only for a diverged checkpoint whose activations overflow; the metrics on
                     that feature are then NaN or inf (vlu.drift.metrics). Each such checkpoint is named
                     once in the log.
A PCA whose SVD does not converge (on finite input) gives NaN for its PCA metrics and a warning; every
other number is still computed.
"""

import argparse
import csv
import itertools
from pathlib import Path

import torch

from vlu.drift.metrics import PCA_METRICS, cosine, linear_cka, pca_shift
from vlu.drift.params import parse_drift
from vlu.drift.paths import FEATURES_VERSION, features_path
from vlu.drift.weights import Ref


def load(ref: Ref, cache: dict) -> dict:
    if ref not in cache:
        path = features_path(ref.node, ref.weights)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: extract features of {ref} first (scripts/submit.py --features)")
        cache[ref] = torch.load(path, map_location="cpu", weights_only=False)
        if cache[ref]["probe"].get("version") != FEATURES_VERSION:
            raise ValueError(f"{path} has an old layout: delete it and extract again (scripts/submit.py --features)")
    return cache[ref]


def nonfinite(data: dict) -> dict[str, int]:
    """{feature: number of inf / NaN entries} of a checkpoint's probe file: a DiT block's count includes its
    attention ratios; "policy" counts the velocities and flow-matching errors."""
    bad = lambda t: int((~torch.isfinite(t)).sum())  # noqa: E731
    counts = {name: bad(h) for name, h in data["features"].items()}
    for block, ratio in data["attn_ratio"].items():
        counts[block] = counts.get(block, 0) + bad(ratio)
    counts["policy"] = sum(bad(data[k]) for k in ("velocity", "velocity_swapped", "sse"))
    return counts


def pca_metrics(ha: torch.Tensor, hb: torch.Tensor, k: int, where: str) -> dict[str, float]:
    """pca_shift, or NaN for each of its metrics (with a warning) if an SVD does not converge: one such
    cell does not cost every other number of the run."""
    try:
        return pca_shift(ha, hb, k)
    except torch.linalg.LinAlgError as e:
        print(f"WARNING: PCA of {where} failed ({e}); its PCA metrics are NaN", flush=True)
        return dict.fromkeys(PCA_METRICS, float("nan"))


def vl_alignment(data: dict, layer: str, idx: torch.Tensor) -> float:
    image = data["features"][f"{layer}.image"][idx].double()
    text = data["features"][f"{layer}.text"][idx].double()
    return float(torch.nn.functional.cosine_similarity(image, text, dim=1).mean())


def language_sensitivity(data: dict, idx: torch.Tensor) -> float:
    v, swapped = data["velocity"][idx].double(), data["velocity_swapped"][idx].double()
    return float((swapped - v).square().sum() / v.square().sum().clamp_min(1e-300))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="default run of checkpoint references")
    parser.add_argument("--drift", action="append", required=True, help="NAME=X..Y (repeatable)")
    parser.add_argument("--pca-k", type=int, default=10)
    parser.add_argument("--pairs-with", action="append", default=[],
                        help="pairwise metrics only for pairs including this drift (repeatable; default: all pairs)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = {}
    specs = [parse_drift(d, args.run) for d in args.drift]  # (name, X, Y)
    drifts = [(name, load(x, cache), load(y, cache)) for name, x, y in specs]
    tasks = drifts[0][1]["tasks"]
    for _, a, b in drifts:
        if a["tasks"] != tasks or b["tasks"] != tasks or a["probe"] != b["probe"]:
            raise ValueError("probe sets differ between checkpoints (or were extracted by different versions "
                             "of vlu.drift.extract); re-extract them")
    names = [name for name, _, _ in drifts]
    if set(args.pairs_with) - set(names):
        raise SystemExit(f"--pairs-with {sorted(set(args.pairs_with) - set(names))}: not among the drifts {names}")
    pairs = [(i, j) for i, j in itertools.combinations(names, 2) if not args.pairs_with or {i, j} & set(args.pairs_with)]
    subsets = {t: torch.tensor([i for i, s in enumerate(tasks) if s == t]) for t in dict.fromkeys(tasks)}
    subsets["all"] = torch.arange(len(tasks))

    rows = []
    counts = {}  # once per checkpoint: a checkpoint ends or starts many drifts (LA/LB starts every U drift)
    for ref in dict.fromkeys(r for _, x, y in specs for r in (x, y)):
        counts[ref] = nonfinite(cache[ref])
        if bad := {feature: n for feature, n in counts[ref].items() if n}:
            print(f"WARNING: {ref} has inf / NaN values, a diverged checkpoint? {bad}; "
                  "its metrics on these features are NaN or inf", flush=True)
    for name, x, y in specs:
        for feature in counts[x]:
            rows += [[feature, "all", "nonfinite_start", name, "", counts[x][feature]],
                     [feature, "all", "nonfinite_end", name, "", counts[y][feature]]]
    for feature in drifts[0][1]["features"]:
        for task, idx in subsets.items():
            shifts = {}
            for name, a, b in drifts:
                ha, hb = a["features"][feature][idx].float(), b["features"][feature][idx].float()
                shifts[name] = hb - ha
                rows.append([feature, task, "cka_distance", name, "", 1 - linear_cka(ha, hb)])
                for metric, value in pca_metrics(ha, hb, args.pca_k, f"{feature}, task {task}, drift {name}").items():
                    rows.append([feature, task, metric, name, "", value])
            for i, j in pairs:
                rows.append([feature, task, "shift_cka", i, j, linear_cka(shifts[i], shifts[j])])

    for task, idx in subsets.items():
        vshift = {}
        for name, a, b in drifts:
            va, vb = a["velocity"][idx].double(), b["velocity"][idx].double()  # float64, as vlu.drift.metrics
            vshift[name] = vb - va
            rows.append(["policy", task, "flow_divergence", name, "", float((vb - va).square().mean())])
            loss_a = (a["sse"][idx].double() / a["n_valid"][idx]).mean()
            loss_b = (b["sse"][idx].double() / b["n_valid"][idx]).mean()
            rows.append(["policy", task, "fm_loss_change", name, "", float(loss_b - loss_a)])
        for i, j in pairs:
            rows.append(["policy", task, "velocity_shift_cos", i, j, cosine(vshift[i], vshift[j])])

    def endpoints(feature, task, metric, name, start, end):
        rows.extend([[feature, task, f"{metric}_start", name, "", start], [feature, task, f"{metric}_end", name, "", end],
                     [feature, task, f"{metric}_change", name, "", end - start]])

    for task, idx in subsets.items():
        for name, a, b in drifts:
            for layer in sorted({k.rsplit(".", 1)[0] for k in a["features"] if k.endswith(".image")}):
                endpoints(layer, task, "vl_alignment", name, vl_alignment(a, layer, idx), vl_alignment(b, layer, idx))
            for block in a["attn_ratio"]:
                endpoints(block, task, "attn_ratio", name, float(a["attn_ratio"][block][idx].mean()),
                          float(b["attn_ratio"][block][idx].mean()))
            endpoints("policy", task, "language_sensitivity", name, language_sensitivity(a, idx), language_sensitivity(b, idx))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", "task", "metric", "drift_i", "drift_j", "value"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
