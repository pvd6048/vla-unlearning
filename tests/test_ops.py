"""Base-level checks of the unlearning plans in vlu/ops.py (NPO, SSD, SalUn), of vlu/train.py with a
gradient mask, and of vlu/run_op.py's argument checks, on the fake GR00T-shaped model.
    python tests/test_ops.py
"""
import copy
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
import vlu.ops
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


# --- Recipe overrides in op tokens (hyperparameter variants as separate nodes) ---------------------------
from vlu.nodes import Op
from vlu.ops import recipe_for

op = Op.parse("UB-npo@unlearn_lr=3e-5+npo_beta=1+vlaforget.lr=5e-5+replay_per_task=None")
assert (op.kind, op.task, op.method) == ("U", "B", "npo") and str(op) == "UB-npo@unlearn_lr=3e-5+npo_beta=1+vlaforget.lr=5e-5+replay_per_task=None"
r = recipe_for(op, RECIPE)
assert (r.unlearn_lr, r.npo_beta, r.vlaforget.lr, r.replay_per_task) == (3e-5, 1.0, 5e-5, None)
assert (r.unlearn_steps, r.vlaforget.lambda_forget) == (RECIPE.unlearn_steps, RECIPE.vlaforget.lambda_forget)  # rest unchanged
assert recipe_for(Op.parse("UB-ga"), RECIPE) is RECIPE and Op.parse("UB-ga") == Op("U", "B", "ga")
assert recipe_for(Op.parse("LA@learn_steps=5000"), RECIPE).learn_steps == 5000
for bad in ["UB-ga@no_such_field=1", "UB-ga@unlearn_lr", "UB-ga@vlaforget.nope=1"]:
    try:
        Op.parse(bad)
        raise AssertionError(bad)
    except ValueError:
        pass
try:
    recipe_for(Op.parse("UB-ga@unlearn_steps=2.5"), RECIPE)
    raise AssertionError("a non-integer step count must be rejected")
except ValueError:
    pass
plan = plan_op(Node.from_path("ops", "LA/LB/UB-ga@unlearn_lr=3e-5+unlearn_steps=200"), recipe)
assert (plan.lr, plan.steps, plan.probe_every) == (3e-5, 200, recipe.unlearn_probe_every)
print("overrides: parsed, printed back, typed, validated, applied to the op only")

# --- ER (Liu et al.'s continual-GR00T): full batch of the new task + equally many replay-buffer frames --
import vlu.data
from vlu.data import ReplayBuffer

er = dataclasses.replace(recipe, replay_per_task=10)
plan = plan_op(Node.from_path("ops", "LA/LB/LC"), er)
slots = plan.parts["train"]
assert slots[:4] == [("C",)] * 4 and slots[4:] == [ReplayBuffer(("A", "B"), 10)] * 4, slots
assert plan_op(Node.from_path("ops", "LA"), er).parts["train"] == [("A",)] * 4  # nothing to replay yet
buffer = ReplayBuffer(("A", "B"), 10)
frames = buffer.frames(demos, run_seed=0)
assert len(frames) == 20 and {f[0] for f in frames} == {"A", "B"} and len(set(frames)) == 20
assert buffer.frames(demos, run_seed=0) == frames and buffer.frames(demos, run_seed=1) != frames
assert ReplayBuffer(("A", "B", "C"), 10).frames(demos, 0)[:20] == frames  # a task's buffer does not depend on the others
drawn = []
real_make_sample = vlu.data.make_sample
vlu.data.make_sample = lambda processor, mcfg, demos_, episode, step, text=None: drawn.append((demos_.letter, episode, step)) or real_make_sample(processor, mcfg, demos_, episode, step)
for op_seed in (5, 6):
    data = OpBatches(demos, plan.parts, proc, mcfg, seed=op_seed, num_steps=20, run_seed=0)
    assert data.buffers[buffer] == frames  # the same buffer in every op of the run
    for t in range(20):
        data[t]
vlu.data.make_sample = real_make_sample
replayed = [f for f in drawn if f[0] != "C"]
assert len(replayed) == 2 * 20 * 4 and set(replayed) <= set(frames), "replay must come from the fixed buffer only"
print("ER: 4 new + 4 replay per batch (16 + 16 in the recipe), fixed per-task buffers, replay drawn only from them")

# --- NPO: theta == reference at step 0, in train mode (DiT dropout on) ------------------------------
model = Fake().cuda().train()
ctx = context("LA/LB/UB-npo", model)
plan = plan_op(ctx.node, recipe)
plan.prepare(ctx)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss, metrics = plan.loss_fn(model, {"forget": probes["B"]})
assert metrics["log_ratio"] == 0.0 and metrics["npo_weight"] == 1.0, metrics
print("NPO: log-ratio exactly 0 at step 0 with dropout on (the reference replays the same random numbers)")

# Known answer: the official loss (licong-lin/negative-preference-optimization@1207bbe, TOFU/dataloader.py:255-270)
#   neg_log_ratios = NLL_current - NLL_ref;  loss = -logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
# with NLL := the chunk's SSE and both models on the same noise and timesteps.
ref = copy.deepcopy(model).requires_grad_(False)  # = the reference prepare() copied
with torch.no_grad():
    for p in trainable_named_parameters(model).values():
        p.add_(0.01 * torch.randn_like(p))
torch.manual_seed(7)
nll_cur = model(probes["B"])["action_loss"].float().sum(dim=(1, 2))
torch.manual_seed(7)
with torch.no_grad():
    nll_ref = ref(probes["B"])["action_loss"].float().sum(dim=(1, 2))
official = -torch.nn.functional.logsigmoid(recipe.npo_beta * (nll_cur - nll_ref)).mean() * 2 / recipe.npo_beta
torch.manual_seed(7)
ours, _ = plan.loss_fn(model, {"forget": probes["B"]})
assert 1 < (recipe.npo_beta * (nll_cur - nll_ref)).abs().max() < 20, (nll_cur - nll_ref)  # the regime where the weighting matters
assert torch.allclose(ours, official, rtol=1e-5, atol=1e-6), (ours.item(), official.item())
print(f"NPO: loss equals the official formula on a perturbed model ({ours.item():.6f} vs {official.item():.6f})")

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


# Known answer: given the same importances, SSD must change the weights exactly as the official code.
def official_modify_weight(params, original_importance, forget_importance, selection_weighting, dampening_constant):
    """if-loops/selective-synaptic-dampening@cdfdc0e, src/ssd.py:199-218 (exponent 1, lower_bound 1)."""
    with torch.no_grad():
        for n, p in params.items():
            oimp, fimp = original_importance[n], forget_importance[n]
            oimp_norm = oimp.mul(selection_weighting)
            locations = torch.where(fimp > oimp_norm)
            weight = ((oimp.mul(dampening_constant)).div(fimp)).pow(1)
            update = weight[locations]
            min_locs = torch.where(update > 1)
            update[min_locs] = 1
            p[locations] = p[locations].mul(update)


model = Fake().cuda()
params = trainable_named_parameters(model)
imp = {s: {n: torch.rand_like(p, dtype=torch.float32) ** 4 for n, p in params.items()} for s in (0, 1)}  # 0: forget, 1: all
expected = copy.deepcopy(params)
official_modify_weight(expected, imp[1], imp[0], recipe.ssd_alpha, recipe.ssd_lambda)
real_accumulate = vlu.ops.accumulate_grads
vlu.ops.accumulate_grads = lambda ctx, slots, n, stream, square: imp[stream]
try:
    ctx = context("LA/LB/UB-ssd", model)
    plan = plan_op(ctx.node, recipe)
    plan.prepare(ctx)
finally:
    vlu.ops.accumulate_grads = real_accumulate
assert 0 < plan.notes["dampened_fraction"] < 1, plan.notes
assert all(torch.equal(params[n], expected[n]) for n in params), "SSD differs from the official modify_weight"
print(f"SSD: bit-identical to the official modify_weight on the same importances (dampened {plan.notes['dampened_fraction']:.3f})")

# --- SalUn: mask over the weights that can change; bool; reused on resume; masked weights frozen -----
model = Fake().cuda()
ctx = context("LA/LB/UB-salun", model)
plan = plan_op(ctx.node, recipe)
plan.prepare(ctx)
mask = plan.grad_mask
assert plan.notes["batch_forget_retain"] == [2, 2], plan.notes  # forget B: 40 frames, retain A: 40 -> 4 * 40/80 = 2
assert len(plan.parts["forget"]) == 2 and len(plan.parts["retain"]) == 2
p3 = plan_op(Node.from_path("ops", "LA/LB/LC/UB-salun"), recipe)  # retain A + C: 80 frames -> 4 * 40/120 = 1.3 -> 1
p3.prepare(context("LA/LB/LC/UB-salun", Fake().cuda()))
assert p3.notes["batch_forget_retain"] == [1, 3], p3.notes
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
         (["--node", "LA/LB/UB-ga", "--save-at", "5000"], "(1000 steps"),
         (["--node", "LA/LB/UC-purge"], "cannot unlearn C")]
for argv, expected in cases:
    sys.argv = ["run_op", "--run", "args", *argv]
    try:
        vlu.run_op.main()
        raise AssertionError(f"{argv} should have been rejected")
    except ValueError as e:
        assert expected in str(e), (argv, str(e))
print("run_op: --save-at rejected for VLA-Forget, for SSD (still 0 steps under --steps) and beyond the step budget; purge of a task never learned rejected")
print("ALL OPS CHECKS PASSED")
