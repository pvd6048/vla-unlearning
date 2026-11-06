"""Parameter drift between checkpoints (PLAN.md §4, rows "Parameter" and "Curvature-weighted").

    python -m vlu.drift.params --run goal_s0 \\
        --drift "U=LA/LB/LC..LA/LB/LC/UB-ga" --drift "tauB=LA..LA/LB" --random \\
        --fisher "FB=LA/LB/LC:B" --subspace-k 8 --out params_UB.csv

A drift "name=X..Y" is theta_Y - theta_X over the trainable parameters, with X and Y checkpoint
references ([run:]path[#weights], see vlu/drift/weights.py). Noise floors are just more drifts, e.g.
the same op in a seed-twin run ("goal_s1:LA/LB/LC..goal_s1:LA/LB/LC/UB-ga"). --random adds a Gaussian
direction as the "unrelated" baseline. --fisher NAME=REF:TASK uses the diagonal Fisher saved by
vlu.drift.extract (ref's <node>/drift/fisher_<task>.safetensors) as a metric.

Everything is computed tensor by tensor, accumulating inner products, so memory stays small and one
pass gives every level: "global" (all trainable weights), "module" (vlm / projector / dit) and
"layer" (LLM layer, DiT block with its kind, projector part). Output CSV columns:
    level, group, metric, drift_i, drift_j, value
with the metrics
    norm        ||D_i||                              (drift_j empty)
    rel_norm    ||D_i|| / ||theta_X_i||               (drift_j empty)
    cos         cos(D_i, D_j)
    proj        <D_i, D_j> / ||D_j||^2: how much of D_j is in D_i (-1: D_i undoes all of D_j)
    <F>_norm, <F>_cos, <F>_proj   the same in the Fisher metric <a, b>_F = sum F a b, over the weights
                the Fisher was computed on (the trainable ones; frozen overrides get weight 0)
    subspace    layer level: mean over the layer's 2-D weights of the overlap of the top-k left
                singular subspaces of D_i and D_j (vlu.drift.metrics.subspace_overlap); random ~ k/rows
The pairwise metrics are for every pair of drifts, or with --pairs-with NAME (repeatable) only for the
pairs that include one of the named drifts (n drifts: O(n) instead of O(n^2) products and rows).
"""

import argparse
from collections import defaultdict
import csv
import itertools
from pathlib import Path
import sys
import zlib

from safetensors import safe_open
import torch

from vlu.drift.paths import fisher_path
from vlu.drift.metrics import subspace_overlap, top_singular_subspace
from vlu.drift.weights import Ref, WeightReader, groups_of


def parse_drift(text: str, run: str) -> tuple[str, Ref, Ref]:
    name, _, span = text.partition("=")
    start, sep, end = span.partition("..")
    if not (name and sep):
        raise ValueError(f"bad --drift {text!r}; expected NAME=X..Y")
    return name, Ref.parse(start, run), Ref.parse(end, run)


def parse_fisher(text: str, run: str) -> tuple[str, Path]:
    name, _, spec = text.partition("=")
    ref_text, _, task = spec.rpartition(":")
    ref = Ref.parse(ref_text, run)
    path = fisher_path(ref.node, task, ref.weights)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run vlu.drift.extract with --fisher {task} first")
    return name, path


def random_direction(key: str, shape, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(zlib.crc32(key.encode()))
    return torch.randn(shape, generator=generator).to(device)


def inner_products(flat: torch.Tensor, cols: list[int], metric: torch.Tensor | None = None) -> torch.Tensor:
    """n x n float64 matrix (on the CPU) of <D_i, D_j> = sum metric * D_i * D_j over one tensor, for the
    rows of flat: on the diagonal and where i or j is in cols (the pairs needed); 0 elsewhere."""
    weighted = flat if metric is None else flat * metric
    part = (weighted @ flat[cols].T).double().cpu()  # n x len(cols)
    g = torch.zeros(len(flat), len(flat), dtype=torch.float64)
    g[:, cols] = part
    g[cols, :] = part.T
    g.diagonal().copy_(torch.bmm(weighted.unsqueeze(1), flat.unsqueeze(2)).flatten().double().cpu())  # row-wise dot products
    return g


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="default run of checkpoint references")
    parser.add_argument("--drift", action="append", required=True, help="NAME=X..Y (repeatable)")
    parser.add_argument("--random", action="store_true", help="add a random Gaussian direction 'random'")
    parser.add_argument("--fisher", action="append", default=[], help="NAME=REF:TASK (repeatable)")
    parser.add_argument("--subspace-k", type=int, default=0, help="also compute top-k singular subspace overlaps")
    parser.add_argument("--pairs-with", action="append", default=[],
                        help="pairwise metrics only for pairs including this drift (repeatable; default: all pairs)")
    parser.add_argument("--out", required=True, help="CSV file to write")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    drifts = [parse_drift(d, args.run) for d in args.drift]
    fishers = [parse_fisher(f, args.run) for f in args.fisher]
    names = [d[0] for d in drifts] + (["random"] if args.random else [])
    reader = WeightReader(device)
    node_refs = [r for _, x, y in drifts for r in (x, y) if not r.is_base]
    if not node_refs:
        sys.exit("at least one checkpoint must be a node (the base model has no list of trainable weights)")
    keys = reader.keys(node_refs)
    fisher_files = {name: safe_open(str(path), framework="pt", device="cpu") for name, path in fishers}

    n = len(names)
    unknown = set(args.pairs_with) - set(names)
    if unknown:
        sys.exit(f"--pairs-with {sorted(unknown)}: not among the drifts {names}")
    cols = sorted(names.index(name) for name in set(args.pairs_with)) if args.pairs_with else list(range(n))
    pairs = [(i, j) for i, j in itertools.permutations(range(n), 2) if i in cols or j in cols]
    gram = defaultdict(lambda: torch.zeros(n, n, dtype=torch.float64))  # (level, group) -> <D_i, D_j>
    start_sq = defaultdict(lambda: torch.zeros(n, dtype=torch.float64))  # ||theta_X_i||^2
    fgram = {f: defaultdict(lambda: torch.zeros(n, n, dtype=torch.float64)) for f in fisher_files}
    subspace = defaultdict(lambda: [torch.zeros(n, n, dtype=torch.float64), torch.zeros(n, n, dtype=torch.float64)])  # sum, count

    for i_key, key in enumerate(keys):
        cache = {}

        def theta(ref: Ref) -> torch.Tensor:
            if ref not in cache:
                cache[ref] = reader.get(ref, key)
            return cache[ref]

        # The drifts' deltas of this tensor, one row each (written in place: no second copy of them)
        shape = theta(drifts[0][2]).shape
        flat = torch.empty(n, shape.numel(), device=device)
        starts = []
        for i, (_, x, y) in enumerate(drifts):
            flat[i] = (theta(y) - theta(x)).flatten()
            starts.append(theta(x).double().square().sum())
        if args.random:
            flat[-1] = random_direction(key, shape, device).flatten()
            starts.append(torch.tensor(float("nan"), dtype=torch.float64, device=device))
        cache.clear()
        g = inner_products(flat, cols)
        s = torch.stack(starts).cpu()
        fg = {}
        for f, handle in fisher_files.items():
            if key not in handle.keys():  # a Fisher covers the weights it was computed on (the trainable ones)
                fg[f] = torch.zeros(n, n, dtype=torch.float64)
                continue
            fg[f] = inner_products(flat, cols, reader.reduce(key, handle.get_tensor(key)).flatten())
        sub = None
        if args.subspace_k and len(shape) == 2:
            bases = [top_singular_subspace(d.view(shape), args.subspace_k) for d in flat]
            # pairs where a drift leaves this weight unchanged are left out of the layer's average
            sub = torch.full((n, n), float("nan"), dtype=torch.float64)
            for i, j in pairs:
                if i < j and bases[i] is not None and bases[j] is not None:
                    sub[i, j] = subspace_overlap(bases[i], bases[j])
        for group in groups_of(key):
            gram[group] += g
            start_sq[group] += s
            for f in fisher_files:
                fgram[f][group] += fg[f]
            if sub is not None:
                subspace[group][0] += sub.nan_to_num(0.0)
                subspace[group][1] += (~sub.isnan()).double()
        if i_key % 50 == 0:
            print(f"{i_key}/{len(keys)} tensors", flush=True)

    rows = []
    for (level, group), g in gram.items():
        for i in range(n):
            rows.append([level, group, "norm", names[i], "", g[i, i].sqrt().item()])
            rows.append([level, group, "rel_norm", names[i], "", (g[i, i] / start_sq[(level, group)][i]).sqrt().item()])
        for i, j in pairs:
            rows.append([level, group, "proj", names[i], names[j], (g[i, j] / g[j, j]).item()])
            if i < j:
                rows.append([level, group, "cos", names[i], names[j], (g[i, j] / (g[i, i] * g[j, j]).sqrt()).item()])
        for f in fisher_files:
            fg = fgram[f][(level, group)]
            for i in range(n):
                rows.append([level, group, f"{f}_norm", names[i], "", fg[i, i].sqrt().item()])
            for i, j in pairs:
                rows.append([level, group, f"{f}_proj", names[i], names[j], (fg[i, j] / fg[j, j]).item()])
                if i < j:
                    rows.append([level, group, f"{f}_cos", names[i], names[j], (fg[i, j] / (fg[i, i] * fg[j, j]).sqrt()).item()])
        if (level, group) in subspace:
            total, count = subspace[(level, group)]
            for i, j in pairs:
                if i < j and count[i, j] > 0:
                    rows.append([level, group, "subspace", names[i], names[j], (total[i, j] / count[i, j]).item()])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["level", "group", "metric", "drift_i", "drift_j", "value"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
