"""Known-answer checks of scripts/main_drift.py on fake checkpoints and probe features: the drifts of PLAN.md
section 5 (unlearning U, incidental forgetting F_B = branch F minus branch C, tauB, distances to the retrain
oracle LA/LC), each reported number against a direct computation on the toy tensors, and nodes left out.
    python tests/test_main_drift.py
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import stubs  # first: temp dirs + GR00T stand-ins
from safetensors.torch import save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import calibration_drift as cd  # noqa: E402
import main_drift as md  # noqa: E402
from vlu.drift.metrics import linear_cka  # noqa: E402
from vlu.drift.paths import FEATURES_VERSION, features_path, fisher_path  # noqa: E402
from vlu.nodes import Node  # noqa: E402

torch.manual_seed(0)
RUN = "goal_s0"
T0, C, F, ORACLE = "LA/LB/LC", "LA/LB/LC/LD/LE", "LA/LB/LC/UB-purge/LD/LE", "LA/LC"
SSD, SALUN = "LA/LB/LC/UB-ssd", "LA/LB/LC/UB-salun"
assert (md.THETA0, md.C, md.F, md.ORACLE) == (T0, C, F, ORACLE) and md.unlearned(md.main_run.METHODS) == ["ssd", "salun"]
assert md.unlearned(["purge", "ga", "vlaforget"]) == ["ga", "vlaforget"]
assert {n: (x, y) for n, x, y in md.wanted()} == {
    "tauB": ("LA", "LA/LB"), "tauD": (T0, "LA/LB/LC/LD"), "F_B": (C, F), "O:theta0": (ORACLE, T0),
    "U:ssd": (T0, SSD), "UF:ssd": (C, SSD + "/LD/LE"), "O:ssd": (ORACLE, SSD),
    "U:salun": (T0, SALUN), "UF:salun": (C, SALUN + "/LD/LE"), "O:salun": (ORACLE, SALUN)}
print("drifts: tauB, tauD, F_B = C..F, O:theta0, and U / UF / O per unlearning method (not purge)")

shapes = {"action_head.model.transformer_blocks.0.attn1.to_q.weight": (8, 8),
          "backbone.model.language_model.model.layers.12.mlp.down_proj.weight": (6, 8), "action_head.vlln.weight": (8,)}
base = {k: torch.randn(*s) for k, s in shapes.items()}
save_file(base, str(stubs.TMP / "base" / "shard.safetensors"))
(stubs.TMP / "base" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "shard.safetensors" for k in base}}))
step = lambda w, s=0.1: {k: v + s * torch.randn_like(v) for k, v in w.items()}  # noqa: E731
w = {"LA": step(base)}
w["LA/LB"] = step(w["LA"])
w[T0] = step(w["LA/LB"])
w[ORACLE] = step(w["LA"])
w["LA/LB/LC/LD"] = step(w[T0])
w[C] = step(w["LA/LB/LC/LD"])
forget = {k: 0.1 * torch.randn_like(v) for k, v in base.items()}  # F_B
w[F] = {k: w[C][k] + forget[k] for k in base}
w[SSD] = {k: w[T0][k] + 0.5 * forget[k] - 0.3 * (w["LA/LB"][k] - w["LA"][k]) + 0.02 * torch.randn_like(v) for k, v in base.items()}
w[SALUN] = step(w[T0], 0.05)
w[SSD + "/LD/LE"] = {k: w[C][k] + 0.7 * forget[k] + 0.03 * torch.randn_like(v) for k, v in base.items()}
w[SALUN + "/LD/LE"] = step(w[C], 0.05)
for path, ws in w.items():
    node = Node.from_path(RUN, path)
    node.dir.mkdir(parents=True, exist_ok=True)
    save_file(ws, str(node.weights_path), metadata={"frozen_overrides": "[]"})
fisher_a = {k: torch.rand(*s) for k, s in shapes.items()}
fisher_b = {k: 4 * v for k, v in fisher_a.items()}  # F_B = 4 F_A: selectivity exactly 2
for task, f in (("A", fisher_a), ("B", fisher_b)):
    path = fisher_path(Node.from_path(RUN, T0), task)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(f, str(path))

TASKS, PER = "ABCDE", 12
N = len(TASKS) * PER
v = {p: torch.randn(N, 16, 7) for p in w}  # predicted velocities; the UB-ssd policy moves partly along F_B's change
v[SSD] = v[T0] + 0.5 * (v[F] - v[C]) + 0.1 * torch.randn(N, 16, 7)
b = slice(TASKS.index("B") * PER, (TASKS.index("B") + 1) * PER)
v[SSD][b] = 0.8 * v[ORACLE][b] + 0.2 * v[SSD][b]  # on B's frames only, the policy moves most of the way to the oracle's
dit = {p: torch.randn(N, 16) for p in w}
for path in w:
    if path == SALUN + "/LD/LE":
        continue  # features missing: its drift is left out
    feats = {"tasks": [t for t in TASKS for _ in range(PER)],
             "features": {"vlm.layer12.image": torch.randn(N, 16).half(), "vlm.layer12.text": torch.randn(N, 16).half(),
                          "vl.image": torch.randn(N, 16).half(), "vl.text": torch.randn(N, 16).half(),
                          "dit.block00.x-text": dit[path].half(), "dit.block01.self": dit[path].half()},
             "attn_ratio": {"dit.block00.x-text": torch.rand(N), "dit.block01.self": torch.rand(N)},
             "velocity": v[path], "velocity_swapped": v[path] + 0.1 * torch.randn(N, 16, 7),
             "sse": 10 * torch.rand(N), "n_valid": torch.full((N,), 112.0),
             "probe": {"per_task": PER, "seed": 1, "version": FEATURES_VERSION, "swap": "next task"}}
    p = features_path(Node.from_path(RUN, path))
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, p)

env = {**os.environ, "PYTHONPATH": str(ROOT)}
result = subprocess.run([sys.executable, str(ROOT / "scripts" / "main_drift.py"), "--run", RUN], env=env, text=True, capture_output=True)
assert result.returncode == cd.LEFT_OUT, result.stdout + result.stderr  # a drift is left out: reported as such
out = cd.report_dir(RUN)
summary = (out / "summary.md").read_text()
assert "Left out (checkpoint or features missing): weights: none; features: UF:salun." in summary, summary
assert "Incomplete" not in summary and (out / "params.csv").exists() and (out / "representation.csv").exists()
assert json.loads((out / "inputs.json").read_text()) == md.inputs(RUN) and md.input_changes(RUN) == ([], [])
print("main_drift: every drift computed; the one without features listed as left out; inputs recorded")

# --- Known answers ---------------------------------------------------------------------------------
flat = lambda d: torch.cat([d[k].flatten().double() for k in shapes])  # noqa: E731
delta = lambda x, y: {k: w[y][k] - w[x][k] for k in shapes}  # noqa: E731
FB = flat(fisher_b)
dot = lambda a, b, m=None: float((a * b * (1 if m is None else m)).sum())  # noqa: E731
cos = lambda a, b, m=None: dot(a, b, m) / math.sqrt(dot(a, a, m) * dot(b, b, m))  # noqa: E731
u, f, tau_b = flat(delta(T0, SSD)), flat(delta(C, F)), flat(delta("LA", "LA/LB"))
row = next(line for line in summary.splitlines() if line.startswith("| U:ssd |")).split(" | ")
expect = [math.sqrt(dot(u, u)), cos(u, tau_b), dot(u, tau_b) / dot(tau_b, tau_b), cos(u, f), cos(u, f, FB),
          math.sqrt(dot(*[flat(delta(ORACLE, SSD))] * 2) / dot(*[flat(delta(ORACLE, T0))] * 2)), 2.0]
for got, want in zip(row[1:8], expect):
    assert math.isclose(float(got), float(f"{want:.3g}"), rel_tol=1e-6), (row, expect)
fr = next(line for line in summary.splitlines() if line.startswith("| *F_B* |")).split(" | ")
assert fr[4] == "1" and fr[5] == "1" and fr[6] == "-", fr  # cos with itself; no oracle distance
uf = next(line for line in summary.splitlines() if line.startswith("| UF:ssd |")).split(" | ")
assert math.isclose(float(uf[4]), float(f"{cos(flat(delta(C, SSD + '/LD/LE')), f):.3g}"), rel_tol=1e-6), uf
print(f"weights: U:ssd norm, cos/proj on tauB, cos with F_B ({expect[3]:.3f}; Fisher-weighted {expect[4]:.3f}), "
      "distance to the oracle LA/LC and F_B/F_A as computed directly; UF against C")

fd = lambda x, y: float((v[y][b] - v[x][b]).double().square().mean())  # noqa: E731
shift = lambda x, y: (v[y][b] - v[x][b]).double().flatten()  # noqa: E731
cka_b = lambda x, y: 1 - linear_cka(dit[x][b].half().float(), dit[y][b].half().float())  # noqa: E731
pr = next(line for line in summary.splitlines() if line.startswith("| U:ssd |") and "->" in line).split(" | ")
for got, want in [(pr[1], fd(T0, SSD)), (pr[4], fd(ORACLE, SSD) / fd(ORACLE, T0)), (pr[7], cka_b(T0, SSD)),
                  (pr[8], cka_b(ORACLE, SSD)), (pr[9].rstrip(" |"), cos(shift(T0, SSD), shift(C, F)))]:
    assert math.isclose(float(got), float(f"{want:.3g}"), rel_tol=1e-6, abs_tol=1e-3), (pr, got, want)
a = slice(0, PER)
fd_a = lambda x, y: float((v[y][a] - v[x][a]).double().square().mean())  # noqa: E731
assert fd(ORACLE, SSD) / fd(ORACLE, T0) < 0.5 < fd_a(ORACLE, SSD) / fd_a(ORACLE, T0)  # the toy tells B from A
print(f"policy: fd on B, B's distance to the oracle, DiT CKA (to theta0 and to the oracle) and the velocity-shift "
      f"cosine with F_B ({cos(shift(T0, SSD), shift(C, F)):.3f}) as computed directly")
print("ALL MAIN DRIFT CHECKS PASSED")
