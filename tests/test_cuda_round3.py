"""Round 3's new methods end to end on CUDA, as a worker runs them.

Every calibration variant of GU on NPO-RT, Sharp MinMax, task arithmetic and RMU (scripts/calibrate.py) is
prepared, trained for a few steps by the shared loop and saved through vlu/run_op.py's _run, with the real bf16
autocast, DataLoader worker processes and an active cost meter, on the GR00T-shaped fake (tests/fakes.py).
The CPU tests replace autocast, CUDA and the DataLoader; bugs that only appear with them show here (Sharp
MinMax's stale autocast weight cache did). RMU's block 7 becomes the fake's last block (it has 4). Sharp
MinMax's SGD rates are divided by 1e4: relative to its weights the fake's gradient is ~6000x GR00T's (per
weight ~0.09 at |w| ~0.09, GR00T ~5e-6 at ~0.03), so GR00T's rates make SGD diverge on the fake (AdamW,
which normalizes the step, is unaffected). The other overrides are the calibration's. Skipped without CUDA.
"""
import dataclasses
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import stubs  # noqa: F401
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate import VARIANTS  # noqa: E402
from fakes import Demos, Fake, Proc, collate, make_sample  # noqa: E402
from vlu import cost  # noqa: E402
from vlu.config import RECIPE  # noqa: E402
from vlu.nodes import Node, recipe_for  # noqa: E402
from vlu.ops import plan_op  # noqa: E402
from vlu.run_op import _run  # noqa: E402

NEW_METHODS = ("npo_rt_gu", "sharp_minmax", "task_arithmetic", "rmu")
FAKE_BLOCKS = 4


def for_fake(variant):
    variant = variant.replace("rmu_layer=7", f"rmu_layer={FAKE_BLOCKS - 1}")
    if variant.startswith("sharp_minmax@"):
        variant = re.sub(r"unlearn_lr=([0-9.]+(?:e-?[0-9]+)?)", lambda m: f"unlearn_lr={float(m[1]) / 1e4:g}", variant)
    return variant


def new_variants():
    return [for_fake(v) for v in VARIANTS if v.split("@")[0] in NEW_METHODS]


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (bf16 autocast, as the workers)")
class Round3OnCuda(unittest.TestCase):
    def test_every_new_variant_trains_and_saves(self):
        variants = new_variants()
        self.assertEqual({v.split("@")[0] for v in variants}, set(NEW_METHODS))
        recipe = dataclasses.replace(RECIPE, unlearn_steps=4, batch_size=2, log_every=2, save_every=2, num_workers=2)
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                with patch.object(Node, "dir", new_callable=lambda: property(lambda self: directory)), \
                     patch("vlu.data.make_sample", make_sample):
                    node = Node.from_path("smoke", f"LA/LB/UB-{variant}")
                    op = node.last_op
                    plan = plan_op(node, recipe)
                    torch.manual_seed(0)
                    model = Fake().cuda()
                    before = {n: p.detach().clone() for n, p in model.named_parameters()}
                    demos = {t: Demos(t) for t in "AB"}
                    probes = {t: collate([make_sample(None, None, d, 0, 0)])["inputs"] for t, d in demos.items()}
                    with cost.Meter().active():
                        _run(node, op, plan, recipe_for(op, recipe), model, demos, Proc(), {}, probes, 0, set())
                self.assertTrue((directory / "weights.safetensors").exists())
                self.assertFalse((directory / "resume.pt").exists())
                log = [json.loads(line) for line in (directory / "train_log.jsonl").read_text().splitlines()]
                changed = set()
                for name, p in model.named_parameters():
                    self.assertTrue(torch.isfinite(p).all(), name)
                    if not p.requires_grad:
                        self.assertTrue(torch.equal(p, before[name]), f"frozen {name} changed")
                    elif not torch.equal(p, before[name]):
                        changed.add(name)
                self.assertTrue(changed, "no trainable weight changed")
                if op.method == "rmu":
                    prefix = f"action_head.model.transformer_blocks.{recipe_for(op, recipe).rmu_layer}."
                    self.assertTrue(all(n.startswith(prefix) for n in changed), sorted(changed))
                    self.assertTrue(all(r["hidden_norm"] > 0 for r in log if "hidden_norm" in r))
                if op.method == "task_arithmetic":
                    self.assertEqual(log[-1]["stage"], "finalize")
                if op.method == "sharp_minmax":
                    notes = json.loads((directory / "op.json").read_text())["notes"]
                    self.assertAlmostEqual(notes["actual_forget_fraction"], .3, delta=.01)
                    self.assertEqual(notes["saliency_frames"], len(demos["B"].index))
                if op.method == "npo_rt_gu":
                    self.assertTrue(all("basis_size" in r for r in log[1:]))


if __name__ == "__main__":
    unittest.main()
