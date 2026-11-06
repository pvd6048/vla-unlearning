"""Known-answer checks of scripts/calibration_drift.py on fake checkpoints and probe features: the
distances to the retrain oracle (weights, policy, DiT representations) against the toy tensors, the
pairwise numbers of the restricted pairs against an all-pairs run, the Fisher selectivity, the DiT
share, nodes left out, and a failed tool (no numbers reported, also none of an earlier run).
    python tests/test_calibration_drift.py
"""
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import stubs  # first: temp dirs + GR00T stand-ins
from safetensors.torch import save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import calibration_drift as cd  # noqa: E402
from vlu.drift.metrics import linear_cka  # noqa: E402
from vlu.drift.paths import FEATURES_VERSION, features_path, fisher_path  # noqa: E402
from vlu.nodes import Node  # noqa: E402

torch.manual_seed(0)
RUN = "cal"
GA, NPO_RT = "ga@unlearn_lr=1e-7", "npo_rt@unlearn_lr=1e-6"  # two of the calibration's variants
UB, UB2 = f"LA/LB/UB-{GA}", f"LA/LB/UB-{NPO_RT}"
DIT = "action_head.model.transformer_blocks.0.attn1.to_q.weight"
shapes = {DIT: (8, 8), "backbone.model.language_model.model.layers.12.mlp.down_proj.weight": (6, 8),
          "action_head.vlln.weight": (8,)}

# Weights: tauA, tauB random; the GA variant removes 60% of tauB plus noise, the NPO-RT one only noise.
base = {k: torch.randn(*s) for k, s in shapes.items()}
save_file(base, str(stubs.TMP / "base" / "shard.safetensors"))
(stubs.TMP / "base" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "shard.safetensors" for k in base}}))
la = {k: v + 0.1 * torch.randn_like(v) for k, v in base.items()}
lalb = {k: v + 0.1 * torch.randn_like(v) for k, v in la.items()}
weights = {"LA": la, "LA/LB": lalb,
           UB: {k: lalb[k] - 0.6 * (lalb[k] - la[k]) + 0.02 * torch.randn_like(v) for k, v in lalb.items()},
           UB2: {k: v + 0.05 * torch.randn_like(v) for k, v in lalb.items()}}
for path, w in weights.items():
    node = Node.from_path(RUN, path)
    node.dir.mkdir(parents=True, exist_ok=True)
    save_file(w, str(node.weights_path), metadata={"frozen_overrides": "[]"})
fisher_a = {k: torch.rand(*s) for k, s in shapes.items()}
for task, f in (("A", fisher_a), ("B", {k: 4 * v for k, v in fisher_a.items()})):  # F_B = 4 F_A: selectivity exactly 2
    path = fisher_path(Node.from_path(RUN, "LA/LB"), task)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(f, str(path))

# Probe features (vlu.drift.extract's layout) of base, LA, LA/LB and the GA variant only.
TASKS, PER = "ABCDE", 12
N = len(TASKS) * PER


def features(velocity):
    return {"tasks": [t for t in TASKS for _ in range(PER)],
            "features": {name: torch.randn(N, 16).half() for name in ("vlm.layer12.image", "vlm.layer12.text", "vl.image",
                                                                     "vl.text", "dit.block00.x-text", "dit.block01.self")},
            "attn_ratio": {"dit.block00.x-text": torch.rand(N), "dit.block01.self": torch.rand(N)},
            "velocity": velocity, "velocity_swapped": velocity + 0.1 * torch.randn_like(velocity),
            "sse": 10 * torch.rand(N), "n_valid": torch.full((N,), 112.0),
            "probe": {"per_task": PER, "seed": 1, "version": FEATURES_VERSION, "swap": "next task"}}


v_base = torch.randn(N, 16, 7)
v_la = v_base + 0.3 * torch.randn_like(v_base)
v_lalb = v_la + 0.3 * torch.randn_like(v_la)
velocities = {"": v_base, "LA": v_la, "LA/LB": v_lalb, UB: v_lalb - 0.6 * (v_lalb - v_la) + 0.05 * torch.randn_like(v_la)}
for path, v in velocities.items():
    p = features_path(Node.from_path(RUN, path))
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features(v), p)

# --- the script, as run_local.sh runs it -------------------------------------------------------------
env = {**os.environ, "PYTHONPATH": str(ROOT)}
script = [sys.executable, str(ROOT / "scripts" / "calibration_drift.py"), "--run", RUN]
subprocess.run(script, check=True, env=env)
out = cd.report_dir(RUN)
p, r = cd.read(out / "params.csv"), cd.read(out / "representation.csv")
u, u2, o = cd.drift_name(GA), cd.drift_name(NPO_RT), cd.drift_name(GA, "O")
assert "=" not in u and cd.label(u) == GA
assert cd.input_changes(RUN) == ([], []), "inputs.json records what the report was computed from"

# Weights: the distance to the oracle, ||theta_U - theta_LA|| / ||theta_LA/LB - theta_LA||, from the tensors.
sq = lambda a, b: sum((a[k].double() - b[k].double()).square().sum() for k in shapes)  # noqa: E731
truth = math.sqrt(sq(weights[UB], la) / sq(lalb, la))
w = cd.weight_metrics(p, u)
assert math.isclose(w["to_oracle"], truth, rel_tol=1e-5), (w["to_oracle"], truth)
assert 0.3 < w["to_oracle"] < 0.6 and w["proj"] < -0.5, w  # removes ~60% of tauB
# The tools' pairwise numbers agree with it: theta_U - theta_LA = U + tauB.
assert math.isclose(w["to_oracle"] ** 2, w["size"] ** 2 + 2 * w["proj"] + 1, rel_tol=1e-5), w
assert math.isclose(w["selectivity"], 2.0, rel_tol=1e-6) and math.isclose(cd.weight_metrics(p, "random")["selectivity"], 2.0, rel_tol=1e-6)
delta = {k: weights[UB][k] - lalb[k] for k in shapes}
dit_share = (delta[DIT].square().sum() / sum(d.square().sum() for d in delta.values())).item()
assert math.isclose(w["dit_share"], dit_share, rel_tol=1e-5), (w["dit_share"], dit_share)
assert cd.weight_metrics(p, u2) is not None and cd.weight_metrics(p, "tauA") is not None
print(f"weights: distance to the oracle = from the tensors ({w['to_oracle']:.6f} vs {truth:.6f}) and = sqrt(size^2 + 2 proj + 1); "
      "Fisher selectivity 2 for F_B = 4 F_A; DiT share as computed from the tensors")

# Policy and representations, from the probe files.
data = {path: torch.load(features_path(Node.from_path(RUN, path)), weights_only=False) for path in ("LA", "LA/LB", UB)}
b = slice(PER, 2 * PER)  # B's probe frames
fd = lambda x, y: (data[y]["velocity"][b].double() - data[x]["velocity"][b].double()).square().mean().item()  # noqa: E731
truth = fd("LA", UB) / fd("LA", "LA/LB")
cka_truth = statistics.mean(1 - linear_cka(data["LA"]["features"][f][b].float(), data[UB]["features"][f][b].float())
                            for f in ("dit.block00.x-text", "dit.block01.self"))
m = cd.policy_metrics(r, cd.dit_cka(r), u)
assert math.isclose(m["B_to_oracle"], truth, rel_tol=1e-5), (m["B_to_oracle"], truth)
assert 0.1 < m["B_to_oracle"] < 0.3, m  # (1 - 0.6)^2 plus noise
assert math.isclose(m["cka_oracle_B"], cka_truth, rel_tol=1e-6), (m["cka_oracle_B"], cka_truth)
fd_u, fd_tau, c = m["fd_B"], cd.policy_metrics(r, cd.dit_cka(r), "tauB")["fd_B"], cd.pair(r, u, "tauB", "velocity_shift_cos", ("policy", "B"))
assert math.isclose(m["B_to_oracle"], (fd_u + fd_tau + 2 * c * math.sqrt(fd_u * fd_tau)) / fd_tau, rel_tol=1e-4), "dv_O = dv_U + dv_tauB"
assert m["lang_start"] is not None and m["lang_end"] is not None and m["cka_dit_A"] is not None
assert cd.policy_metrics(r, cd.dit_cka(r), u2) is None, "the NPO-RT variant has no features"
print(f"policy: B's distance to the oracle = from the velocities ({m['B_to_oracle']:.6f} vs {truth:.6f}); "
      "CKA distance to the oracle = from the features")

# Pairwise numbers only with tauA / tauB, and the same as in an all-pairs run.
g = ("global", "all")
assert cd.pair(p, u, "tauA", "cos", g) is not None and cd.pair(p, u, u2, "cos", g) is None and cd.pair(p, u, o, "cos", g) is None
assert cd.pair(r, u, "tauA", "velocity_shift_cos", ("policy", "B")) is not None and cd.pair(r, u, o, "velocity_shift_cos", ("policy", "B")) is None
subprocess.run([sys.executable, "-m", "vlu.drift.params", "--run", RUN, "--drift", "tauB=LA..LA/LB", "--drift", f"U=LA/LB..{UB}",
                "--drift", f"V=LA/LB..{UB2}", "--random", "--fisher", "FB=LA/LB:B", "--out", str(stubs.TMP / "all_pairs.csv")], check=True, env=env)
full = cd.read(stubs.TMP / "all_pairs.csv")
for metric in ("cos", "FB_cos"):
    assert math.isclose(full[(*g, metric, "tauB", "U")], cd.pair(p, u, "tauB", metric, g), rel_tol=1e-6), metric
assert math.isclose(full[(*g, "proj", "U", "tauB")], p[(*g, "proj", u, "tauB")], rel_tol=1e-6)
assert math.isclose(full[(*g, "norm", "V", "")], p[(*g, "norm", u2, "")], rel_tol=1e-6) and (*g, "cos", "U", "V") in full
print("pairs: only those with tauA / tauB, with the values of an all-pairs run")

summary = (out / "summary.md").read_text()
assert "## Drift metrics" in summary and f"| {GA} |" in summary and f"| {NPO_RT} |" in summary and "| *random* |" in summary
assert "| O:" not in summary and "Incomplete" not in summary
left_out = summary.split("Left out", 1)[1].split("\n", 1)[0]
weights_left, features_left = left_out.split("weights: ", 1)[1].split("; features: ")
assert "ssd@ssd_reference=retain" in weights_left and NPO_RT not in weights_left and GA not in weights_left, weights_left
assert NPO_RT in features_left and GA not in features_left, features_left
print("summary: every computed drift has a row; variants without weights / features are listed")

# A failed tool reports no number, also none left by an earlier run; the script exits non-zero.
(out / "representation.csv").write_text("feature,task,metric,drift_i,drift_j,value\npolicy,all,flow_divergence,tauB,,123\n")
(out / "compare.log").write_text("log of an earlier version\n")
broken = features_path(Node.from_path(RUN, UB))
broken.write_bytes(b"not a torch file")
assert subprocess.run(script, env=env, capture_output=True).returncode == 1
summary = (out / "summary.md").read_text()
assert not (out / "representation.csv").exists() and (out / "representation.log").exists() and not (out / "compare.log").exists()
assert "**Incomplete**: weights: ok; features: FAILED (exit code 1; log `drift/representation.log`)." in summary, summary
assert "| *tauB* |" not in summary.split("### Policy", 1)[1], "no policy row without representation.csv"
assert cd.weight_metrics(cd.read(out / "params.csv"), u) is not None
print("failure: a failed tool's numbers (and an earlier run's) are not reported; the summary says it failed")
print("ALL CALIBRATION DRIFT CHECKS PASSED")
