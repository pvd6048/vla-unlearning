"""Known-answer checks of the drift math (vlu/drift/metrics.py) and of vlu/drift/params.py on fake
checkpoints: a node that undoes a task vector exactly, LIBERO-row slicing of per-embodiment weights,
Fisher weighting, singular subspaces, and frozen overrides.   python tests/test_drift_params.py
"""
import csv
import json
import os
import subprocess
import sys

import stubs  # first: temp dirs + GR00T stand-ins
from safetensors.torch import save_file
import torch

from vlu.drift.metrics import linear_cka, pca_shift, subspace_overlap, top_singular_subspace

torch.manual_seed(0)

# --- metrics -------------------------------------------------------------------------------------
x = torch.randn(200, 50)
q, _ = torch.linalg.qr(torch.randn(50, 50))
assert abs(linear_cka(x, 3 * x @ q + 5) - 1) < 1e-9, "CKA is invariant to rotation, scale and shift"
s = pca_shift(x, x)
assert abs(s["pca_overlap"] - 1) < 1e-9 and s["pca_distance"] < 1e-12
for shape in [(300, 200), (200, 300), (2048, 1536)]:
    w = torch.randn(*shape)
    exact = torch.linalg.svd(w, full_matrices=False)[0][:, :8].T
    assert abs(subspace_overlap(exact, top_singular_subspace(w, 8)) - 1) < 1e-3, shape
assert top_singular_subspace(torch.zeros(5, 4), 2) is None
print("metrics: CKA invariances, PCA shift, exact singular subspaces OK")

# Known answers from Xu et al.'s code (Representational_Analysis_Tools): cka_analysis.linear_cka (feature- and
# Gram-centered), pca_sim_analysis (cosine of the two PC1s), pca_shift_analysis (PCA(2) fit on the reference;
# shift = mean PC1 projection of new minus ref's, PC2 = each model's mean PC2 projection).
import numpy as np  # noqa: E402
try:
    from sklearn.decomposition import PCA

    def components(a, n):
        return PCA(n_components=n).fit(a).components_
except ImportError:  # the same as sklearn's PCA: right singular vectors of the centered data
    def components(a, n):
        return np.linalg.svd(a - a.mean(0), full_matrices=False)[2][:n]
ref = torch.randn(64, 20) @ torch.diag(torch.linspace(3, 0.1, 20))
new = ref + 0.4 * torch.randn(64, 20) + torch.linspace(-1, 1, 20)
xr, xn = ref.double().numpy(), new.double().numpy()


def their_cka(a, b):
    def center_gram(k):
        u = np.ones_like(k) / len(k)
        return k - u @ k - k @ u + u @ k @ u
    kx, ky = center_gram((a - a.mean(0)) @ (a - a.mean(0)).T), center_gram((b - b.mean(0)) @ (b - b.mean(0)).T)
    return np.trace(kx @ ky) / np.sqrt(np.trace(kx @ kx) * np.trace(ky @ ky))


c_ref, c_new = components(xr, 1)[0], components(xn, 1)[0]
their_sim = c_ref @ c_new / (np.linalg.norm(c_ref) * np.linalg.norm(c_new))
comp = components(xr, 2)
shift = (xn @ comp[0]).mean() - (xr @ comp[0]).mean()
their_point_distance = np.hypot(shift, (xn @ comp[1]).mean() - (xr @ comp[1]).mean())
ours = pca_shift(ref, new)
assert abs(linear_cka(ref, new) - their_cka(xr, xn)) < 1e-9, (linear_cka(ref, new), their_cka(xr, xn))
assert abs(ours["pca_similarity"] - abs(their_sim)) < 1e-9, (ours, their_sim)
assert abs(ours["pca_centroid_shift"] - their_point_distance) < 1e-9, (ours, their_point_distance)
assert 0.1 < ours["pca_similarity"] < 1 and ours["pca_centroid_shift"] > 0.1  # the toy is not degenerate
one = pca_shift(ref, new, k=1)  # their 1 and 2 directions do not depend on our k (compare.py --pca-k)
assert abs(one["pca_centroid_shift"] - ours["pca_centroid_shift"]) < 1e-12 and abs(one["pca_similarity"] - ours["pca_similarity"]) < 1e-12
print(f"metrics: linear CKA, PCA similarity ({ours['pca_similarity']:.3f}) and PCA shift distance "
      f"({ours['pca_centroid_shift']:.3f}) equal Xu et al.'s code")

# --- checkpoint references: runs, snapshots and hyperparameter-variant nodes ----------------------------
from vlu.drift.weights import Ref
assert Ref.parse("LA/LB", "r0") == Ref("r0", "LA/LB", "final")
assert Ref.parse("goal_s1:LA/LB#step_002000", "r0") == Ref("goal_s1", "LA/LB", "step_002000")
assert Ref.parse("base", "r0") == Ref("r0", "", "final")
assert Ref.parse("LA/LB/UB-ga@unlearn_lr=3e-5", "r0") == Ref("r0", "LA/LB/UB-ga@unlearn_lr=3e-5", "final")
assert Ref.parse("s1:LA/UA-vlaforget@vlaforget.stage_ratio=60:60:90#final", "r0") == Ref("s1", "LA/UA-vlaforget@vlaforget.stage_ratio=60:60:90", "final")
print("refs: run prefix, #snapshot and @overrides parse unambiguously")

# --- params.py on fake checkpoints ---------------------------------------------------------------
base, runs = stubs.TMP / "base", stubs.TMP / "runs" / "toy"
shapes = {"backbone.model.language_model.model.layers.12.mlp.down_proj.weight": (40, 30),
          "action_head.model.transformer_blocks.2.attn1.to_q.weight": (30, 30),
          "action_head.state_encoder.layer1.W": (32, 10, 20),  # per-embodiment stack; LIBERO = row 2
          "action_head.vlln.weight": (30,)}
FROZEN = "backbone.model.vision_model.vision_model.encoder.layers.26.mlp.fc1.weight"
theta0 = {k: torch.randn(*s) for k, s in {**shapes, FROZEN: (8, 8)}.items()}
tau = {k: 0.1 * torch.randn(*s) for k, s in shapes.items()}
tau["action_head.state_encoder.layer1.W"] = torch.zeros(32, 10, 20)
tau["action_head.state_encoder.layer1.W"][2] = 0.1 * torch.randn(10, 20)
d_frozen = torch.randn(8, 8)
save_file(theta0, str(base / "shard.safetensors"))
(base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "shard.safetensors" for k in theta0}}))
nodes = {"LA": ({k: theta0[k] + tau[k] for k in shapes}, []),
         "LA/LB": ({k: theta0[k] for k in shapes}, []),                                      # undoes tau exactly
         "LA/LB/UB-x": ({**{k: theta0[k] for k in shapes}, FROZEN: theta0[FROZEN] + d_frozen}, [FROZEN])}  # frozen override
for path, (w, overrides) in nodes.items():
    (runs / path).mkdir(parents=True)
    save_file(w, str(runs / path / "weights.safetensors"), metadata={"frozen_overrides": json.dumps(overrides)})
(runs / "LA" / "drift").mkdir()
save_file({k: torch.ones(*s) for k, s in shapes.items()}, str(runs / "LA" / "drift" / "fisher_B.safetensors"))

out = stubs.TMP / "params.csv"
root = str(stubs.Path(__file__).resolve().parents[1])
subprocess.run([sys.executable, "-m", "vlu.drift.params", "--run", "toy", "--drift", "tau=base..LA", "--drift", "U=LA..LA/LB",
                "--drift", "V=LA/LB..LA/LB/UB-x", "--random", "--fisher", "F=LA:B", "--subspace-k", "4", "--out", str(out)],
               check=True, env={**os.environ, "PYTHONPATH": root})
rows = {(r["level"], r["group"], r["metric"], r["drift_i"], r["drift_j"]): float(r["value"]) for r in csv.DictReader(open(out))}
assert abs(rows[("global", "all", "cos", "tau", "U")] + 1) < 1e-6 and abs(rows[("global", "all", "proj", "U", "tau")] + 1) < 1e-6
assert abs(rows[("global", "all", "F_cos", "tau", "U")] + 1) < 1e-6
assert abs(rows[("global", "all", "F_norm", "tau", "")] - rows[("global", "all", "norm", "tau", "")]) < 1e-6
for group in ["vlm.layer12", "dit.block02.x-image"]:
    assert abs(rows[("layer", group, "subspace", "tau", "U")] - 1) < 1e-4, group
assert abs(rows[("global", "all", "rel_norm", "tau", "")] - 0.1) < 0.02, "LIBERO-row slicing keeps the relative norm at ~0.1"
assert abs(rows[("module", "vision", "norm", "V", "")] - d_frozen.norm().item()) < 1e-4, "frozen override measured against base"
assert rows[("module", "vision", "norm", "tau", "")] == 0.0
print("params: task-vector negation, Fisher, subspaces, embodiment slicing, frozen overrides OK;",
      "cos(tau, random) =", round(rows[("global", "all", "cos", "tau", "random")], 4))

# --pairs-with U: the pairs with U, with the same values; no other pair.
subprocess.run([sys.executable, "-m", "vlu.drift.params", "--run", "toy", "--drift", "tau=base..LA", "--drift", "U=LA..LA/LB",
                "--drift", "V=LA/LB..LA/LB/UB-x", "--random", "--fisher", "F=LA:B", "--subspace-k", "4", "--pairs-with", "U",
                "--out", str(out)], check=True, env={**os.environ, "PYTHONPATH": root})
some = {(r["level"], r["group"], r["metric"], r["drift_i"], r["drift_j"]): float(r["value"]) for r in csv.DictReader(open(out))}
pairs = {k[3:] for k in some if k[4]}
assert pairs == {(a, b) for a in ("tau", "U", "V", "random") for b in ("tau", "U", "V", "random") if a != b and "U" in (a, b)}, pairs
assert set(some) < set(rows) and all(abs(v - rows[k]) <= 1e-6 * max(1.0, abs(v)) or (v != v and rows[k] != rows[k])
                                     for k, v in some.items()), "same values as with all pairs (up to fp32 summation order)"
assert ("layer", "vlm.layer12", "subspace", "tau", "U") in some and ("global", "all", "F_cos", "tau", "U") in some
print("params: --pairs-with keeps the pairs with the named drift (same values, Fisher and subspaces too), drops the others")

# The LIBERO row of a per-embodiment stack is a tensor of its own (a view would keep all 32 rows allocated).
from vlu.drift.weights import WeightReader  # noqa: E402
row = WeightReader("cpu").get(Ref.parse("LA", "toy"), "action_head.state_encoder.layer1.W")
assert row.shape == (10, 20) and row.untyped_storage().nbytes() == row.numel() * 4
print("reader: the LIBERO row is read without keeping the 32-robot stack")
print("ALL DRIFT PARAMS CHECKS PASSED")
