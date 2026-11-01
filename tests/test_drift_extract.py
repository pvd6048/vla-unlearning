"""Base-level checks of the drift feature extraction (vlu/drift/extract.py) and of compare.py, on a
fake model with GR00T N1.6's module layout.   python tests/test_drift_extract.py
"""
import zlib
from types import SimpleNamespace

import stubs  # first: temp dirs + GR00T stand-ins
import torch
from torch import nn

import vlu.data
import vlu.drift.extract as ex
from vlu.tasks import TASKS, instruction

D, DD, IMG = 32, 24, 99


class Layer(nn.Module):  # HF decoder layers return tuples
    def __init__(self):
        super().__init__(); self.lin = nn.Linear(D, D)
    def forward(self, h):
        return (h + torch.tanh(self.lin(h)),)


class Attn(nn.Module):
    def __init__(self):
        super().__init__(); self.q = nn.Linear(DD, DD); self.kv = nn.Linear(D, DD)
    def forward(self, x, encoder_hidden_states=None):
        out = self.q(x)
        if encoder_hidden_states is not None:
            out = out + self.kv(encoder_hidden_states).mean(dim=1, keepdim=True)
        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.norm1 = nn.LayerNorm(DD); self.attn1 = Attn(); self.ff = nn.Linear(DD, DD)
    def forward(self, h, encoder_hidden_states=None, temb=None):
        h = self.attn1(self.norm1(h), encoder_hidden_states=encoder_hidden_states) + h
        return self.ff(h) + h


class Fake(nn.Module):
    def __init__(self, ignore_text=False):
        super().__init__()
        self.ignore_text = ignore_text
        self.embed = nn.Embedding(100, D)
        lm = nn.Module(); lm.model = nn.Module(); lm.model.layers = nn.ModuleList([Layer() for _ in range(3)])
        self.backbone = nn.Module(); self.backbone.model = nn.Module(); self.backbone.model.language_model = lm
        self.backbone.model.config = SimpleNamespace(image_token_index=IMG)
        lm.model.layers[0].requires_grad_(False)  # like GR00T: only the top layers train
        self.action_head = nn.Module(); ah = self.action_head
        ah.vlln = nn.LayerNorm(D); ah.model = nn.Module(); ah.model.transformer_blocks = nn.ModuleList([Block() for _ in range(4)])
        ah.action_encoder = nn.Linear(128, DD); ah.state_encoder = nn.Linear(8, DD); ah.action_decoder = nn.Linear(DD, 128)

    def forward(self, inputs):
        ids = inputs["input_ids"].cuda()
        h = self.embed(ids)
        if self.ignore_text:
            h = h * (ids == IMG)[..., None]
        for layer in self.backbone.model.language_model.model.layers:
            h = layer(h)[0]
        vl = self.action_head.vlln(h)
        action, mask = inputs["action"].cuda(), inputs["action_mask"].cuda()
        noise, t = torch.randn_like(action), torch.rand(action.shape[0], 1, 1, device="cuda")
        sa = torch.cat([self.action_head.state_encoder(inputs["state"].cuda())[:, None],
                        self.action_head.action_encoder((1 - t) * noise + t * action)], dim=1)
        for i, block in enumerate(self.action_head.model.transformer_blocks):
            sa = block(sa, encoder_hidden_states=vl if i % 2 == 0 else None)
        pred = self.action_head.action_decoder(sa)
        err = (pred[:, -16:] - (action - noise)).square() * mask
        return {"loss": err.sum() / mask.sum(), "action_loss": err, "action_mask": mask}


class Demos:
    def __init__(self, letter):
        self.letter, self.index = letter, [(0, s) for s in range(100)]


def make_sample(processor, mcfg, demos, episode, step, text=None):
    return {"letter": demos.letter, "step": step, "text": text or instruction(demos.letter)}


def collate(samples):
    ids, act = [], []
    for s in samples:
        words = [zlib.crc32(w.encode()) % 90 for w in s["text"].split()][:8]
        ids.append([0] * (8 - len(words)) + [IMG] * 4 + words)  # left padding, 4 image tokens, text
        g = torch.Generator().manual_seed(s["step"] + 1000 * ord(s["letter"]))
        act.append(torch.randn(16, 128, generator=g))
    ids = torch.tensor(ids)
    mask = torch.zeros(len(samples), 16, 128); mask[..., :7] = 1
    return {"inputs": {"input_ids": ids, "attention_mask": (ids != 0).long(), "state": torch.randn(len(samples), 8),
                       "action": torch.stack(act) * mask, "action_mask": mask}}


processor = SimpleNamespace(collator=collate, training=False, eval=lambda: None, train=lambda: None)
mcfg = {"action": SimpleNamespace(delta_indices=list(range(16)))}
ex.load_demos = lambda mcfg, letters: {letters[0]: Demos(letters[0])}
ex.make_sample = make_sample
vlu.data.make_sample = make_sample  # used by OpBatches inside accumulate_grads (Fisher)

torch.manual_seed(0); model = Fake().cuda().eval()
a = ex.extract_features(model, processor, mcfg)
b = ex.extract_features(model, processor, mcfg)
N = len(TASKS) * ex.PROBE_PER_TASK
assert len(a["tasks"]) == N and a["velocity"].shape == (N, 16, 7) and a["velocity_swapped"].shape == (N, 16, 7)
assert sorted(a["features"]) == ["dit.block00.x-text", "dit.block01.self", "dit.block02.x-image", "dit.block03.self",
                                 "vl.image", "vl.text", "vlm.layer01.image", "vlm.layer01.text", "vlm.layer02.image", "vlm.layer02.text"], sorted(a["features"])
assert all(v.shape[0] == N for v in a["features"].values()) and sorted(a["attn_ratio"]) == sorted(k for k in a["features"] if k.startswith("dit."))
assert all(torch.equal(a["features"][k], b["features"][k]) for k in a["features"]) and torch.equal(a["velocity"], b["velocity"]), "not deterministic"
assert not torch.equal(a["velocity"], a["velocity_swapped"])
blind = ex.extract_features(Fake(ignore_text=True).cuda().eval(), processor, mcfg)
assert torch.equal(blind["velocity"], blind["velocity_swapped"]), "a model that ignores text must have zero language sensitivity"
assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()), "extraction hooks left on the model"
print("features: shapes, names, determinism, swap OK; no hooks left behind")

# Fisher of task B: non-negative, same names/shapes as the trainable weights, summary shares sum to 1.
fisher, summary = ex.extract_fisher(SimpleNamespace(), model, processor, mcfg, "B")
trainable = {n: p.shape for n, p in model.named_parameters() if p.requires_grad}
assert {k: v.shape for k, v in fisher.items()} == trainable and all((v >= 0).all() for v in fisher.values())
assert abs(sum(summary["module_mass_share"].values()) - 1) < 1e-6 and 0 < summary["top1pct_mass_share"] <= 1
print("fisher OK:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})

# Write two checkpoints' files and run compare.py on them.
from vlu.drift.paths import features_path
from vlu.nodes import Node
for path, data in [("LA", a), ("LA/LB", ex.extract_features(Fake().cuda().eval(), processor, mcfg))]:
    p = features_path(Node.from_path("toy", path)); p.parent.mkdir(parents=True, exist_ok=True); torch.save(data, p)

# compare.py on the two checkpoints: a no-op drift changes nothing.
import csv
import os
import subprocess
import sys
out = stubs.TMP / "rep.csv"
root = str(stubs.Path(__file__).resolve().parents[1])
subprocess.run([sys.executable, "-m", "vlu.drift.compare", "--run", "toy", "--drift", "none=LA..LA", "--drift", "B=LA..LA/LB",
                "--out", str(out)], check=True, env={**os.environ, "PYTHONPATH": root})
rows = list(csv.DictReader(open(out)))
get = lambda f, t, m, i: [float(r["value"]) for r in rows if (r["feature"], r["task"], r["metric"], r["drift_i"]) == (f, t, m, i)][0]
import math
assert abs(get("vlm.layer02.text", "all", "cka_distance", "none")) < 1e-9 and get("policy", "all", "flow_divergence", "none") == 0
assert math.isnan(get("vlm.layer02.image", "all", "cka_distance", "none")), "constant features: CKA undefined -> NaN"
assert get("policy", "all", "language_sensitivity_change", "none") == 0 and get("dit.block00.x-text", "all", "attn_ratio_change", "none") == 0
print("compare: no-op drift is exactly zero on every metric family")
print("ALL DRIFT EXTRACTION CHECKS PASSED")
