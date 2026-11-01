"""Fake model and data with GR00T N1.6's module layout and calling conventions, shared by the tests.

Fake mirrors what vlu relies on: module paths (SigLIP2 encoder layers, post_layernorm, mlp1, LLM layers
with tuple outputs, DiT transformer_blocks with attn1 and dropout, action_decoder, per-embodiment state
encoder), the frozen/trainable scope, and forward(inputs) -> {"loss", "action_loss", "action_mask"} with
random flow-matching noise. install() makes vlu build samples from the fake data.
"""
import torch
from torch import nn

LIBERO = 2  # embodiment row (as in tests/stubs.py's embodiment_id.json)
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
        ah.action_encoder = nn.Linear(128, DD)
        ah.state_encoder = nn.Module()  # like GR00T's CategorySpecificMLP: one weight row per embodiment
        ah.state_encoder.W = nn.Parameter(0.1 * torch.randn(32, 8, DD)); ah.state_encoder.b = nn.Parameter(torch.zeros(32, DD))
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
        se = self.action_head.state_encoder
        sa = torch.cat([(inputs["state"].cuda() @ se.W[LIBERO] + se.b[LIBERO])[:, None],
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




def install(*modules):
    """Make vlu.data (and any module that imported make_sample by name) use the fake samples."""
    import vlu.data

    vlu.data.make_sample = make_sample
    for module in modules:
        module.make_sample = make_sample
