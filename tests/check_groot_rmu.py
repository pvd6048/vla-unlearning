"""Optional CPU check against GR00T's actual pinned DiT block, without its model weights/data.

Requires diffusers==0.35.1 and GR00T's dit.py at the pinned commit. Without an argument it uses the
checkout of the GR00T environment ($GR00T_DIR, set by slurm/env.sh), so scripts/run_local.sh runs it as a
self-test; without $GR00T_DIR (no GR00T environment, e.g. a laptop) it reports a skip. This compares
RMU's block-only reference shortcut to an independently evaluated frozen full miniature model.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import runpy
import sys

import stubs  # noqa: F401
import torch
from torch import nn

from vlu.rmu import RMUUpdate


def check(block_type):
    class Model(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.upstream = nn.Linear(8, 8)
            self.decoder = nn.Linear(8, 3)
            self.action_head = nn.Module()
            self.action_head.model = nn.Module()
            self.action_head.model.transformer_blocks = nn.ModuleList([
                block_type(dim=8, num_attention_heads=2, attention_head_dim=4, dropout=.2,
                           final_dropout=True, cross_attention_dim=8 if i == 0 else None,
                           norm_type="ada_norm", ff_inner_dim=16,
                           positional_embeddings="sinusoidal", num_positional_embeddings=4)
                for i in range(2)])

        def forward(self, inputs):
            h = self.upstream(inputs["x"] + .1 * torch.randn_like(inputs["x"]))
            for i, block in enumerate(self.action_head.model.transformer_blocks):
                h = block(hidden_states=h, attention_mask=None,
                          encoder_hidden_states=inputs["context"] if i == 0 else None,
                          encoder_attention_mask=inputs["context_mask"] if i == 0 else None,
                          temb=inputs["temb"])
                if i == self.layer:
                    selected = h
            error = (self.decoder(h[:, -3:]) - inputs["action"]).square() * inputs["action_mask"]
            return {"loss": error.sum() / inputs["action_mask"].sum(), "hidden": selected}

    maximum = 0.
    for layer in (0, 1):
        for alpha in (0., 3.):
            torch.manual_seed(17)
            model = Model(layer)
            expected, frozen = copy.deepcopy(model).eval(), copy.deepcopy(model).eval().requires_grad_(False)
            rule = RMUUpdate(model, layer, alpha, 2., 19)
            selected = expected.action_head.model.transformer_blocks[layer]
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
            other = torch.optim.AdamW(selected.parameters(), lr=.001, weight_decay=0.)
            inputs = {"x": torch.randn(2, 4, 8), "context": torch.randn(2, 3, 8),
                      "temb": torch.randn(2, 8), "action": torch.randn(2, 3, 3),
                      "context_mask": torch.tensor([[True, True, False], [True, False, True]]),
                      "action_mask": torch.tensor([[[1., 1., 0.], [1., 0., 0.], [0., 0., 0.]],
                                                    [[1., 1., 1.], [0., 0., 0.], [1., 0., 1.]]])}
            data = {"forget": inputs, "retain": {**inputs, "x": inputs["x"] * .5}}
            vector = torch.rand(8, generator=torch.Generator().manual_seed(19))
            control = vector / vector.norm() * 2.
            valid = inputs["action_mask"].bool().any(dim=-1)
            for step in range(4):
                torch.manual_seed(1000 + step)
                rng = torch.get_rng_state()
                _, metrics = rule.step(model, data, optimizer, float("inf"))
                actual_rng = torch.get_rng_state()

                torch.set_rng_state(rng)
                f = expected(data["forget"])["hidden"][:, -3:]
                retain_rng = torch.get_rng_state()
                r = expected(data["retain"])["hidden"][:, -3:]
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.set_rng_state(retain_rng)
                    ref = frozen(data["retain"])["hidden"][:, -3:]
                forget = (f[valid] - control).square().mean()
                retain = (r[valid] - ref[valid]).square().mean()
                loss = forget + alpha * retain
                expected.zero_grad(set_to_none=True)
                loss.backward()
                other.step()
                torch.testing.assert_close(torch.get_rng_state(), actual_rng, atol=0, rtol=0)
                torch.testing.assert_close(torch.tensor(metrics["retain_rmu"]), retain.detach(), atol=1e-7, rtol=1e-5)
                for name, p in model.named_parameters():
                    q = dict(expected.named_parameters())[name]
                    maximum = max(maximum, (p-q).abs().max().item())
                    torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
                optimizer.zero_grad(set_to_none=True)
    print(f"Pinned GR00T DiT: 4 configurations x 4 AdamW steps; max parameter difference {maximum:.3g}")
    print("Checked real cross/self attention, adaptive normalization, masks, keyword inputs and full frozen references.")
    print("This is a tiny CPU architecture check, not a GR00T checkpoint or rollout benchmark.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dit_source", type=Path, nargs="?",
                        help="GR00T's gr00t/model/modules/dit.py (default: the one in $GR00T_DIR)")
    args = parser.parse_args()
    if args.dit_source is None:
        if not os.environ.get("GR00T_DIR"):
            print("SKIP: no $GR00T_DIR (GR00T environment) and no dit.py given")
            sys.exit(0)
        args.dit_source = Path(os.environ["GR00T_DIR"]) / "gr00t" / "model" / "modules" / "dit.py"
    import diffusers  # GR00T's pinned dependency; only needed for the check itself
    source = args.dit_source.read_text(encoding="utf-8").rstrip() + "\n"
    if hashlib.sha256(source.encode()).hexdigest() != "26809bbe5872ef3e27e59cc0935334d57f868e98c5805a1a54f0f03e39f8f3e4":
        parser.error("source does not match the inspected GR00T revision")
    if diffusers.__version__ != "0.35.1":
        parser.error("GR00T's pinned diffusers==0.35.1 is required")
    torch.set_num_threads(1)
    check(runpy.run_path(str(args.dit_source))["BasicTransformerBlock"])
