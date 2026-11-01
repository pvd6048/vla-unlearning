"""Base-level checks of the unlearning plans in vlu/ops.py (NPO, SSD, SalUn), of vlu/train.py with a
gradient mask, and of vlu/run_op.py's argument checks, on the fake GR00T-shaped model.
    python tests/test_ops.py
"""
import dataclasses
import sys
from types import SimpleNamespace

import stubs  # noqa: F401  (first: temp dirs + GR00T stand-ins)
import torch

from fakes import LIBERO, Demos, Fake, Proc, collate, install, make_sample
from vlu.config import RECIPE
from vlu.data import OpBatches
from vlu.model import save_weights, trainable_named_parameters
from vlu.nodes import Node, run_seed
from vlu.ops import OpContext, plan_op
import vlu.run_op
from vlu.train import train

install()
torch.manual_seed(0)
demos = {t: Demos(t) for t in "ABC"}
proc = Proc()
mcfg = {"action": SimpleNamespace(delta_indices=list(range(16)))}
probes = {t: collate([make_sample(None, None, demos[t], 0, s) for s in range(4)])["inputs"] for t in "ABC"}
recipe = dataclasses.replace(RECIPE, unlearn_steps=6, batch_size=4, num_workers=2, log_every=2, importance_batches=4,
                             probe_every=1000, save_every=1000)
STATE = "action_head.state_encoder.W"  # per-embodiment stack: only row LIBERO can change


def context(path: str, model) -> OpContext:
    node = Node.from_path("ops", path)
    node.dir.mkdir(parents=True, exist_ok=True)
    return OpContext(node, model, demos, proc, mcfg, seed=0)


# --- NPO: theta == reference at step 0, in train mode (DiT dropout on) ------------------------------
model = Fake().cuda().train()
ctx = context("LA/LB/UB-npo", model)
plan = plan_op(ctx.node, recipe)
plan.prepare(ctx)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss, metrics = plan.loss_fn(model, {"forget": probes["B"]})
assert metrics["log_ratio"] == 0.0 and metrics["npo_weight"] == 1.0, metrics
print("NPO: log-ratio exactly 0 at step 0 with dropout on (the reference replays the same random numbers)")

# --- SSD: zero steps, only shrinks weights, never touches rows that cannot change --------------------
model = Fake().cuda()
before = {n: p.detach().clone() for n, p in trainable_named_parameters(model).items()}
ctx = context("LA/LB/UB-ssd", model)
plan = plan_op(ctx.node, recipe)
assert plan.steps == 0
plan.prepare(ctx)
for n, p in trainable_named_parameters(model).items():
    assert (p.abs() <= before[n].abs() + 1e-7).all(), n
dead = [r for r in range(32) if r != LIBERO]
assert torch.equal(dict(model.named_parameters())[STATE][dead], before[STATE][dead])
print(f"SSD: shrink-only, dead embodiment rows untouched, dampened fraction {plan.notes['dampened_fraction']:.3f}")

# --- SalUn: mask over the weights that can change; bool; reused on resume; masked weights frozen -----
model = Fake().cuda()
ctx = context("LA/LB/UB-salun", model)
plan = plan_op(ctx.node, recipe)
plan.prepare(ctx)
mask = plan.grad_mask
assert all(m.dtype == torch.bool for m in mask.values())
assert not mask[STATE][dead].any(), "rows that never get a gradient must not be selected"
assert 0.4 < plan.notes["kept_fraction"] < 0.6, plan.notes
plan2 = plan_op(ctx.node, recipe)
plan2.prepare(ctx)
assert all(torch.equal(mask[n], plan2.grad_mask[n]) for n in mask), "a resumed op must reuse the saved mask"

before = {n: p.detach().clone() for n, p in trainable_named_parameters(model).items()}
batches = OpBatches(demos, plan.parts, proc, mcfg, seed=0, num_steps=plan.steps)
train(ctx.node, model, plan, batches, probes, recipe, save_at=set())
after = dict(trainable_named_parameters(model))
frozen_ok = all(torch.equal(after[n][~mask[n]], before[n][~mask[n]]) for n in mask)
moved = sum(int((after[n][mask[n]] != before[n][mask[n]]).sum()) for n in mask)
assert frozen_ok, "masked-out weights changed (weight decay?)"
assert moved > 0 and ctx.node.weights_path.exists()
print(f"SalUn: kept {plan.notes['kept_fraction']:.3f} of the changeable weights, bool mask, same mask on resume;",
      f"after training, masked weights bit-identical and {moved} selected weights moved")

# --- run_op rejects impossible snapshot requests before loading anything ------------------------------
run_seed("args", 0)
for path in ["LA", "LA/LB"]:
    node = Node.from_path("args", path)
    node.dir.mkdir(parents=True, exist_ok=True)
    save_weights(Fake(), node.weights_path)
cases = [(["--node", "LA/LB/UB-vlaforget", "--save-at", "5"], "own training loop"),
         (["--node", "LA/LB/UB-ssd", "--steps", "10", "--save-at", "5"], "(0 steps"),
         (["--node", "LA/LB/UB-ga", "--save-at", "5000"], "(1000 steps")]
for argv, expected in cases:
    sys.argv = ["run_op", "--run", "args", *argv]
    try:
        vlu.run_op.main()
        raise AssertionError(f"{argv} should have been rejected")
    except ValueError as e:
        assert expected in str(e), (argv, str(e))
print("run_op: --save-at rejected for VLA-Forget, for SSD (still 0 steps under --steps) and beyond the step budget")
print("ALL OPS CHECKS PASSED")
