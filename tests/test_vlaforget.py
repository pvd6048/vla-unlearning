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
changed = lora.merge(holder, ad)
assert isinstance(holder.l, nn.Linear) and changed == ["l.weight"] and torch.allclose(holder.l(x), adapted, atol=1e-5)
print("2. LoRA: identity at init, merged == adapted")

# --- 3. Fake GR00T-shaped model --------------------------------------------------------------
V, D, DD, IMG, P = 16, 24, 20, 99, 4


class VLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module(); self.self_attn.q_proj = nn.Linear(V, V); self.self_attn.out_proj = nn.Linear(V, V)
        self.mlp = nn.Module(); self.mlp.fc1 = nn.Linear(V, V); self.mlp.fc2 = nn.Linear(V, V)
    def forward(self, h):
        h = h + self.self_attn.out_proj(torch.tanh(self.self_attn.q_proj(h)))
        return h + self.mlp.fc2(torch.tanh(self.mlp.fc1(h)))


class LLMLayer(nn.Module):
    def __init__(self):
        super().__init__(); self.self_attn = nn.Module(); self.self_attn.q_proj = nn.Linear(D, D)
    def forward(self, h):
        return (h + torch.tanh(self.self_attn.q_proj(h)),)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = nn.Module(); self.attn1.to_q = nn.Linear(DD, DD); self.attn1.to_k = nn.Linear(D, DD)
        self.ff = nn.Linear(DD, DD); self.drop = nn.Dropout(0.2)
    def forward(self, h, enc):
        h = h + self.attn1.to_q(h) + self.attn1.to_k(enc).mean(dim=1, keepdim=True)
        return h + self.drop(self.ff(h))


class Fake(nn.Module):
    """Module paths and calling conventions of GR00T N1.6 that VLA-Forget relies on."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module(); bm = self.backbone.model = nn.Module()
        vm = bm.vision_model = nn.Module(); vt = vm.vision_model = nn.Module()
        vt.patch = nn.Linear(3, V); vt.encoder = nn.Module(); vt.encoder.layers = nn.ModuleList([VLayer() for _ in range(6)])
        vt.post_layernorm = nn.LayerNorm(V)
        bm.mlp1 = nn.Sequential(nn.LayerNorm(V), nn.Linear(V, D), nn.GELU(), nn.Linear(D, D))
        lm = bm.language_model = nn.Module(); lm.model = nn.Module(); lm.model.layers = nn.ModuleList([LLMLayer() for _ in range(3)])
        lm.embed = nn.Embedding(100, D)
        ah = self.action_head = nn.Module()
        ah.vlln = nn.LayerNorm(D); ah.model = nn.Module(); ah.model.transformer_blocks = nn.ModuleList([Block() for _ in range(4)])
        ah.action_encoder = nn.Linear(128, DD); ah.state_encoder = nn.Linear(8, DD)
        ah.action_decoder = nn.Sequential(nn.Linear(DD, 128))
        # GR00T's trainable scope: frozen vision + projector + LLM except the top layers.
        for mod in [vm, bm.mlp1, lm]:
            mod.requires_grad_(False)
        lm.model.layers[2].requires_grad_(True)

    def forward(self, inputs):
        pix = inputs["pixel_values"].cuda()                      # (B * 2 views, P, 3)
        h = self.backbone.model.vision_model.vision_model.patch(pix)
        for layer in self.backbone.model.vision_model.vision_model.encoder.layers:
            h = layer(h)
        h = self.backbone.model.vision_model.vision_model.post_layernorm(h)
        vis = self.backbone.model.mlp1(h).reshape(inputs["input_ids"].shape[0], -1, D)  # image tokens per sample
        txt = self.backbone.model.language_model.embed(inputs["input_ids"].cuda())
        x = torch.cat([vis, txt], dim=1)
        for layer in self.backbone.model.language_model.model.layers:
            x = layer(x)[0]
        vl = self.action_head.vlln(x)
        action, mask = inputs["action"].cuda(), inputs["action_mask"].cuda()
        noise, t = torch.randn_like(action), torch.rand(action.shape[0], 1, 1, device="cuda")
        sa = torch.cat([self.action_head.state_encoder(inputs["state"].cuda())[:, None],
                        self.action_head.action_encoder((1 - t) * noise + t * action)], dim=1)
        for block in self.action_head.model.transformer_blocks:
            sa = block(sa, vl)
        pred = self.action_head.action_decoder(sa)[:, -16:]
        err = (pred - (action - noise)).square() * mask
        return {"loss": err.sum() / mask.sum(), "action_loss": err, "action_mask": mask}


class Demos:
    def __init__(self, letter):
        self.letter, self.index = letter, [(0, s) for s in range(40)]


def make_sample(processor, mcfg, demos, episode, step, text=None):
    return {"letter": demos.letter, "step": step}


def collate(samples):
    gens = [torch.Generator().manual_seed(s["step"] * 7 + ord(s["letter"])) for s in samples]
    pix = torch.stack([torch.randn(2, P, 3, generator=g) + (ord(s["letter"]) - 65) for g, s in zip(gens, samples)]).reshape(-1, P, 3)
    act = torch.stack([torch.randn(16, 128, generator=g) for g in gens])
    mask = torch.zeros(len(samples), 16, 128); mask[..., :7] = 1
    ids = torch.tensor([[ord(s["letter"]) % 90, 5, 6] for s in samples])
    return {"inputs": {"pixel_values": pix, "input_ids": ids, "state": torch.randn(len(samples), 8), "action": act * mask, "action_mask": mask}}


class Proc:
    training = True
    collator = staticmethod(collate)
    def eval(self): self.training = False
    def train(self): self.training = True


vlu.data.make_sample = make_sample
vf.make_sample = make_sample
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
assert changed == set(report["merge"]["changed_weights"]), (changed, report["merge"]["changed_weights"])
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
print("ALL VLA-FORGET CHECKS PASSED")
