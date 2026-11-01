"""Base-level checks of VLA-Forget's pieces (PCGrad, LoRA, frozen overrides) and of the whole op on a
fake model with GR00T N1.6's module layout.   python tests/test_vlaforget.py
"""
import copy
import dataclasses
import json
from types import SimpleNamespace

import stubs  # noqa: F401  (first: temp dirs + GR00T stand-ins)
import torch
from torch import nn

import vlu.data, vlu.vlaforget as vf
from vlu import lora
from vlu.config import RECIPE
from vlu.model import frozen_overrides, load_weights, save_weights, trainable_named_parameters
from vlu.nodes import Node
from vlu.ops import OpContext, plan_op

torch.manual_seed(0)

# --- 1. PCGrad known answers -----------------------------------------------------------------
g = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0])]
seen = set()
for seed in range(20):
    out, conflicts = vf.pcgrad(g, torch.Generator().manual_seed(seed))
    seen.add((tuple(round(x, 6) for x in out.tolist()), conflicts))
assert seen == {((-0.25, 0.75), 1), ((0.5, 0.5), 1)}, seen
out, conflicts = vf.pcgrad([torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])], torch.Generator().manual_seed(0))
assert torch.allclose(out, torch.tensor([1.0, 0.5])) and conflicts == 0
print("1. PCGrad matches the hand-computed answers")

# --- 2. LoRA: identity at init, merge == adapter --------------------------------------------
lin = nn.Linear(16, 8).cuda()
x = torch.randn(5, 16, device="cuda")
holder = nn.Module(); holder.l = lin
ref_out = lin(x).detach().clone()
ad = lora.attach(holder, ["l"], r=4, alpha=4.0, dropout=0.0)
assert torch.equal(holder.l(x), ref_out), "fresh adapter must not change the output"
with torch.no_grad():
    ad["l"].lora_B.normal_()
adapted = holder.l(x).detach().clone()
retained = lora.merge(holder, ad)
assert isinstance(holder.l, nn.Linear) and list(retained) == ["l.weight"] and torch.allclose(holder.l(x), adapted, atol=1e-5)
assert abs(retained["l.weight"] - 1) < 1e-4, retained
# A bf16 weight cannot hold a delta far below its precision: merge must report the loss.
holder.l = nn.Linear(16, 8).cuda().to(torch.bfloat16)
ad = lora.attach(holder, ["l"], r=4, alpha=4.0, dropout=0.0)
with torch.no_grad():
    ad["l"].lora_B.fill_(1e-6)
assert lora.merge(holder, ad)["l.weight"] < 0.5
print("2. LoRA: identity at init, merged == adapted, retained delta 1 (fp32) / reported lost (bf16)")

# --- 3. Fake GR00T-shaped model --------------------------------------------------------------
from fakes import Demos, Fake, P, Proc, collate, install, make_sample  # noqa: E402

install(vf)
demos = {t: Demos(t) for t in "ABC"}
proc = Proc()
probes = {t: collate([make_sample(None, None, demos[t], 0, s) for s in range(4)])["inputs"] for t in "ABC"}

recipe = dataclasses.replace(RECIPE, unlearn_steps=21, num_workers=2, log_every=5, batch_size=4,
                             vlaforget=dataclasses.replace(RECIPE.vlaforget, selection_batches=2, boundary_forget_pool=16,
                                                           boundary_retain_pool=32, top_k_vision=2, top_k_reasoning=3))
node = Node.from_path("vf", "LA/LB/LC/UB-vlaforget")
plan = plan_op(node, recipe)
assert plan.custom_train is not None and plan.steps == 21

model = Fake().cuda()
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
flags_before = {n: p.requires_grad for n, p in model.named_parameters()}
ctx = OpContext(node, model, demos, proc, {"action": SimpleNamespace(delta_indices=list(range(16)))}, seed=0)

# Step-0 property: theta == theta0 -> both KL stand-ins exactly 0 (eval mode: no dropout mismatch).
ref = copy.deepcopy(model).eval(); taps, ref_taps = vf.Taps(model), vf.Taps(ref)
model.eval()
out, mine, theirs = vf.paired_forward(model, ref, taps, ref_taps, probes["A"])
assert vf.velocity_kl(out, mine, theirs).item() == 0.0 and vf.feature_loss(mine, theirs, 1.0).item() == 0.0
print("3. step 0: velocity KL and feature loss are exactly 0")
cands = vf.candidates(model, {n for n, f in flags_before.items() if f}, vision_blocks=4)
assert all(".encoder.layers." in n and int(n.split(".layers.")[1].split(".")[0]) >= 2 for n in cands["vision"])
assert cands["projector"] == ["backbone.model.mlp1.1", "backbone.model.mlp1.3"]
assert "backbone.model.language_model.model.layers.2.self_attn.q_proj" in cands["reasoning"]
assert not any("layers.0." in n or "layers.1." in n for n in cands["reasoning"] if n.startswith("backbone"))
print("4. candidates: late vision layers / mlp1 linears / trainable LLM layer + DiT:", {k: len(v) for k, v in cands.items()})

# --- 4. The whole op ---------------------------------------------------------------------------
model = Fake().cuda(); model.load_state_dict(base_state)
ctx.model = model
node.dir.mkdir(parents=True, exist_ok=True)
plan.custom_train(ctx, probes)
report = json.loads((node.dir / "vlaforget.json").read_text())
assert not any(isinstance(mod, lora.LoRALinear) for mod in model.modules()), "adapters left in the model"
assert {n: p.requires_grad for n, p in model.named_parameters()} == flags_before, "requires_grad flags not restored"
sel = {s: report["stages"][s]["selected"] for s in vf.STAGES}
assert all(n in cands[s] for s in vf.STAGES for n in sel[s]) and [report["stages"][s]["steps"] for s in vf.STAGES] == [6, 6, 9]
ov = frozen_overrides(model)
assert ov and all(n.startswith(("backbone.model.vision_model", "backbone.model.mlp1")) for n in ov), ov
assert report["merge"]["max_relative_change"] < 0.01, report["merge"]   # bf16 rounding of probe forward passes
assert all(abs(r - 1) < 1e-3 for r in report["merge"]["retained_delta_fraction"].values()), "fp32 fake weights keep the whole delta"
# Exact check in fp32 (no autocast): a model with adapters on every stage's layers == the merged model.
m2 = Fake().cuda(); m2.load_state_dict(base_state); m2.eval()
ads = lora.attach(m2, cands["vision"][:2] + cands["projector"] + cands["reasoning"][:3], r=4, alpha=4.0, dropout=0.0)
with torch.no_grad():
    for a in ads.values():
        a.lora_B.normal_(std=0.05)
torch.manual_seed(1); before_fp32 = m2(probes["B"])["loss"].item()
lora.merge(m2, ads)
torch.manual_seed(1); after_fp32 = m2(probes["B"])["loss"].item()
assert abs(before_fp32 - after_fp32) < 1e-5 * abs(before_fp32), (before_fp32, after_fp32)
after_state = model.state_dict()
changed = {k for k in base_state if not torch.equal(base_state[k], after_state[k])}
assert changed == set(report["merge"]["retained_delta_fraction"]), (changed, report["merge"]["retained_delta_fraction"])
log = [json.loads(l) for l in open(node.dir / "train_log.jsonl")]
assert log[0]["stage"] == "start" and log[-1]["stage"] == "merged" and any(r.get("conflicts", 0) > 0 for r in log)
print("5. op: selections", sel)
print("   adapters merged, flags restored, overrides", sorted(ov), "merge rel. change", report["merge"]["max_relative_change"])

# --- 5. Save/load round trip with frozen overrides --------------------------------------------
path = node.dir / "weights.safetensors"
save_weights(model, path)
from safetensors import safe_open
keys = set(safe_open(str(path), "pt").keys())
assert keys == set(trainable_named_parameters(model)) | ov
fresh = Fake().cuda(); fresh.load_state_dict(base_state)
load_weights(fresh, path)
assert all(torch.equal(fresh.state_dict()[k], after_state[k]) for k in after_state) and frozen_overrides(fresh) == ov
child = node.dir / "child.safetensors"; save_weights(fresh, child)   # a descendant carries the overrides forward
assert set(safe_open(str(child), "pt").keys()) == keys
bad = Fake().cuda(); bad.action_head.vlln.requires_grad_(False)
try:
    load_weights(bad, path); raise AssertionError("loading into a different trainable scope must fail")
except ValueError:
    pass
print("6. weights: trainable + frozen overrides saved, reloaded exactly, carried to descendants; scope mismatch rejected")

# --- 6. The global RNG is the same whether the boundary set is computed or read from its cache ------
node3 = Node.from_path("vf_rng", "LA/LB/LC/UB-vlaforget"); node3.dir.mkdir(parents=True)
m3 = Fake().cuda(); m3.load_state_dict(base_state)
ref3 = copy.deepcopy(m3).eval(); ref3_taps = vf.Taps(ref3)
ctx3 = OpContext(node3, m3, demos, proc, ctx.modality_configs, seed=0)
states = []
for _ in range(2):  # first call computes and caches, second reads the cache
    torch.manual_seed(123)
    frames, _ = vf.boundary_set(ctx3, ref3, ref3_taps, "B", ("A", "C"), recipe.vlaforget)
    states.append((torch.get_rng_state(), torch.cuda.get_rng_state(), frames))
assert torch.equal(states[0][0], states[1][0]) and torch.equal(states[0][1], states[1][1]) and states[0][2] == states[1][2]
print("7. boundary set: same frames and same global RNG state whether computed or cached")

# --- 7. A merge that changes the model beyond the tolerance fails the op -----------------------------
strict = dataclasses.replace(recipe, vlaforget=dataclasses.replace(recipe.vlaforget, merge_tolerance=1e-12))
node4 = Node.from_path("vf_tol", "LA/LB/LC/UB-vlaforget"); node4.dir.mkdir(parents=True)
m4 = Fake().cuda(); m4.load_state_dict(base_state)
try:
    plan_op(node4, strict).custom_train(OpContext(node4, m4, demos, proc, ctx.modality_configs, seed=0), probes)
    raise AssertionError("a merge beyond the tolerance must fail the op")
except RuntimeError as e:
    assert "merging the adapters changed a probe loss" in str(e)
print("8. merge beyond tolerance -> the op fails (node not saved)")
print("ALL VLA-FORGET CHECKS PASSED")
