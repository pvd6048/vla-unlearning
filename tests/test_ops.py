"""Base-level checks of the unlearning plans in vlu/ops.py (NPO, NPO-RT, SSD, SalUn), of vlu/train.py with a
gradient mask, and of vlu/run_op.py's argument checks, on the fake GR00T-shaped model.
    python tests/test_ops.py
"""
import copy
import dataclasses
import json
import sys
from types import SimpleNamespace

import stubs  # noqa: F401  (first: temp dirs + GR00T stand-ins)
import numpy as np
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
# unlearn_lr set for the whole recipe: every method plans, including those not calibrated yet (vlu/config.py)
recipe = dataclasses.replace(RECIPE, unlearn_steps=6, batch_size=4, num_workers=2, log_every=2, importance_batches=4,
                             probe_every=1000, save_every=1000, unlearn_lr=1e-5)
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

# --- Learning rate per method (RECIPE.unlearn_lrs) ------------------------------------------------------
# A U op gets its method's calibrated rate; an op's override wins (a float even where the method has none);
# a method that trains with unlearn_lr but has no rate yet is refused when planning (check_op), unless the
# op or the recipe sets one; the others (SSD, VLA-Forget, purge) do not need one.
from vlu.nodes import _KEYS_USED, check_op  # noqa: E402

lr_methods = [m for m in vlu.ops.UNLEARN_METHODS if "unlearn_lr" in _KEYS_USED[m]]
assert set(lr_methods) == {"ga", "graddiff", "npo", "npo_rt", "salun"}, lr_methods
assert set(RECIPE.unlearn_lrs) <= set(lr_methods) and RECIPE.unlearn_lr is None
for method in vlu.ops.UNLEARN_METHODS:
    node = Node.from_path("ops", f"LA/LB/UB-{method}")
    lr = recipe_for(node.last_op, RECIPE).unlearn_lr
    assert lr == RECIPE.unlearn_lrs.get(method), (method, lr)
    try:
        check_op(node)
        refused = False
    except ValueError as error:
        refused = True
        assert "has no learning rate yet" in str(error) and "unlearn_lrs in vlu/config.py" in str(error), error
    assert refused == (method in lr_methods and method not in RECIPE.unlearn_lrs), method
    check_op(Node.from_path("ops", f"LA/LB/UB-{method}@unlearn_lr=3e-7"))  # the calibration's variants
    check_op(node, dataclasses.replace(RECIPE, unlearn_lr=1e-5))  # a recipe-wide rate (the tests')
    if method in lr_methods:
        assert recipe_for(Op.parse(f"UB-{method}@unlearn_lr=3e-7"), RECIPE).unlearn_lr == 3e-7
        assert plan_op(Node.from_path("ops", f"LA/LB/UB-{method}@unlearn_lr=3e-7"), RECIPE).lr == 3e-7
assert recipe_for(Op.parse("LB"), RECIPE).unlearn_lr is None
salun = dataclasses.replace(RECIPE, unlearn_lrs={"salun": 2e-5})
assert plan_op(Node.from_path("ops", "LA/LB/UB-salun"), salun).lr == 2e-5
assert recipe_for(Op.parse("UB-salun@unlearn_lr=1e-4"), salun).unlearn_lr == 1e-4
try:
    Op.parse("UB-salun@unlearn_lrs=1")
    raise AssertionError("the table of rates is not a per-op override")
except ValueError:
    pass
print(f"learning rates: per method ({RECIPE.unlearn_lrs}), op overrides win, methods without one "
      f"({sorted(set(lr_methods) - set(RECIPE.unlearn_lrs))}) refused when planning")

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

# --- Which methods need a retained task: one list (vlu.nodes.NEEDS_RETAINED_TASK), checked by plan_op ----
from vlu.nodes import NEEDS_RETAINED_TASK  # noqa: E402

for method in vlu.ops.UNLEARN_METHODS:
    try:
        plan_op(Node.from_path("ops", f"LA/UA-{method}"), recipe)
        rejected = False
    except ValueError:
        rejected = True
    assert rejected == (method in NEEDS_RETAINED_TASK), method
    plan_op(Node.from_path("ops", f"LA/LB/UA-{method}"), recipe)  # with B retained every method plans
assert NEEDS_RETAINED_TASK <= set(vlu.ops.UNLEARN_METHODS)
print(f"plan_op: unlearning a parent's only task is rejected exactly for {sorted(NEEDS_RETAINED_TASK)}")

# --- NPO: theta == reference at step 0, in train mode (DiT dropout on) ------------------------------
model = Fake().cuda().train()
ctx = context("LA/LB/UB-npo", model)
plan = plan_op(ctx.node, recipe)
plan.prepare(ctx)
state = next(c.cell_contents for c in plan.loss_fn.__closure__ if isinstance(c.cell_contents, dict) and "ref" in c.cell_contents)
ref_params = dict(state["ref"].named_parameters())
for name, p in model.named_parameters():
    assert (ref_params[name] is p) == (not p.requires_grad), f"{name}: the reference must share exactly the frozen weights"
assert not any(p.requires_grad for p in ref_params.values())
print("NPO: the reference shares the frozen weights and copies the trainable ones")
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

# --- NPO-RT: the authors' default npo_grad_diff (TOFU/dataloader.py:272-291, config/forget.yaml) -------
#   loss = npo_coeff * forget_loss + grad_diff_coeff * retain_loss,   npo_coeff = grad_diff_coeff = 1
# with forget_loss the NPO loss above and retain_loss the ordinary training loss (L_FM) on retained data.
rt_ctx = context("LA/LB/UB-npo_rt", model)
rt_plan = plan_op(rt_ctx.node, recipe)
assert rt_plan.parts["forget"] == [("B",)] * 4 and rt_plan.parts["retain"] == [("A",)] * 4, rt_plan.parts
assert plan.parts.keys() == {"forget"} and rt_plan.notes["retain_coeff"] == 1.0
assert rt_plan.notes["npo_beta"] == recipe.npo_beta
rt_state = next(c.cell_contents for c in rt_plan.loss_fn.__closure__ if isinstance(c.cell_contents, dict))
rt_state["ref"] = ref  # = what its prepare() copies: the model before the perturbation
torch.manual_seed(7)
rt, rt_metrics = rt_plan.loss_fn(model, {"forget": probes["B"], "retain": probes["A"]})
torch.manual_seed(7)
model(probes["B"])  # the forget pass draws its noise first, as in the loss
retain_fm = model(probes["A"])["loss"]
official_rt = 1.0 * official + 1.0 * retain_fm
assert torch.allclose(rt, official_rt, rtol=1e-5, atol=1e-6), (rt.item(), official_rt.item())
assert rt_metrics["retain_fm"] == retain_fm.item() and rt_metrics["log_ratio"] != 0, rt_metrics
# Its gradient is NPO's plus the retain gradient: the retain term never fades with NPO's weight.
model.zero_grad(set_to_none=True)
torch.manual_seed(7)
rt_plan.loss_fn(model, {"forget": probes["B"], "retain": probes["A"]})[0].backward()
g_rt = {n: p.grad.clone() for n, p in trainable_named_parameters(model).items() if p.grad is not None}
model.zero_grad(set_to_none=True)
torch.manual_seed(7)
plan.loss_fn(model, {"forget": probes["B"]})[0].backward()
g_npo = {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p) for n, p in trainable_named_parameters(model).items()}
model.zero_grad(set_to_none=True)
torch.manual_seed(7)
model(probes["B"])
model(probes["A"])["loss"].backward()
g_ret = {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p) for n, p in trainable_named_parameters(model).items()}
model.zero_grad(set_to_none=True)
for n, g in g_rt.items():
    assert torch.allclose(g, g_npo[n] + g_ret[n], rtol=1e-4, atol=1e-6), n
print(f"NPO-RT: loss equals the official npo_grad_diff (NPO + 1.0 x retain L_FM: {rt.item():.6f} vs {official_rt.item():.6f}), "
      "gradient = NPO's + the retain gradient")

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
asked = {}  # stream -> the slots SSD computed importances on


def fixed_importances(ctx, slots, n, stream, square):
    asked[stream] = set(slots)
    return {k: v.clone() for k, v in imp[stream].items()}  # SSD may change the importances it gets in place


def run_ssd(path, model):
    vlu.ops.accumulate_grads = fixed_importances
    try:
        ctx = context(path, model)
        plan = plan_op(ctx.node, recipe_for(ctx.node.last_op, recipe))
        plan.prepare(ctx)
    finally:
        vlu.ops.accumulate_grads = real_accumulate
    return plan


plan = run_ssd("LA/LB/UB-ssd", model)
assert 0 < plan.notes["dampened_fraction"] < 1, plan.notes
assert all(torch.equal(params[n], expected[n]) for n in params), "SSD differs from the official modify_weight"
print(f"SSD: bit-identical to the official modify_weight on the same importances (dampened {plan.notes['dampened_fraction']:.3f})")

# Reference data: the retained tasks by default. ssd_reference=all is the paper's D as the official code
# computes it: full_train_dl = an unshuffled ConcatDataset(retain, forget) with the forget loader's batch
# size, so its batches are all-retain or all-forget and I_full = (1 - s) * I_retain + s * I_forget, s the
# forget share of the data (if-loops/selective-synaptic-dampening@cdfdc0e, forget_full_class_main.py:152-156).
asked.clear(); run_ssd("LA/LB/LC/UB-ssd", Fake().cuda())
assert asked == {0: {("B",)}, 1: {("A", "C")}}, asked
model = Fake().cuda()
params = trainable_named_parameters(model)
expected = copy.deepcopy(params)
asked.clear(); plan = run_ssd("LA/LB/LC/UB-ssd@ssd_reference=all", model)
assert asked == {0: {("B",)}, 1: {("A", "C")}}, asked  # the forget part of I_full is the forget importance
share = 40 / 120  # fakes.Demos: 40 frames per task; forget B, retain A and C
assert plan.notes["forget_share"] == share, plan.notes
full = {n: imp[1][n].clone().mul_(1 - share).add_(imp[0][n], alpha=share) for n in params}
official_modify_weight(expected, full, imp[0], recipe.ssd_alpha, recipe.ssd_lambda)
assert all(torch.equal(params[n], expected[n]) for n in params), "SSD (ssd_reference=all) differs from the official update"
try:
    run_ssd("LA/UA-ssd", Fake().cuda())
    raise AssertionError("ssd needs a retained task")
except ValueError:
    pass
print("SSD: I_ref on the retained tasks (default); ssd_reference=all = (1-s) I_retain + s I_forget as the official "
      "unshuffled full loader, then the official update; no retained task -> error")

# Known answer for ASSD: alpha = the official percentile of I_forget / I_ref (if-loops/adaptive-selective-
# synaptic-dampening@f04c49d, error_unlearning.ipynb: p = 100 - ln(1 + share% ), np.nanpercentile of the
# ratios without NaN/inf), then the official SSD update with that alpha. As in GR00T, the per-embodiment
# rows that never get a gradient have zero importance (0/0 = NaN, left out by both).
dead = [r for r in range(32) if r != LIBERO]
for s in (0, 1):
    for n, v in imp[s].items():
        if vlu.ops.live_part(n, v, LIBERO) is not v:
            v[dead] = 0
model = Fake().cuda()
params = trainable_named_parameters(model)
expected = copy.deepcopy(params)
plan = run_ssd("LA/LB/UB-ssd@ssd_adaptive=1", model)
share = 40 / 80  # fakes.Demos: 40 frames per task; forget B, retain A
ratios = np.concatenate([(imp[0][n] / imp[1][n]).flatten().cpu().numpy() for n in params])
ratios = ratios[np.isfinite(ratios)]
official_alpha = float(np.nanpercentile(ratios, 100 - np.log(1 + share * 100)))
assert abs(plan.notes["assd_percentile"] - (100 - np.log(1 + share * 100))) < 1e-12, plan.notes
assert abs(plan.notes["assd_alpha"] - official_alpha) <= 1e-5 * official_alpha, (plan.notes["assd_alpha"], official_alpha)
official_modify_weight(expected, imp[1], imp[0], plan.notes["assd_alpha"], recipe.ssd_lambda)
assert all(torch.equal(params[n], expected[n]) for n in params), "ASSD differs from the official update"
# dampened_fraction counts the weights that can change (as SalUn's kept_fraction), so it is ASSD's target
assert abs(plan.notes["dampened_fraction"] - np.log(1 + share * 100) / 100) < 2e-3, plan.notes
# Degenerate importances (no finite ratio) must fail, not silently dampen nothing.
imp_saved = imp
imp = {s: {n: torch.zeros_like(v) for n, v in imp_saved[s].items()} for s in (0, 1)}
try:
    run_ssd("LA/LB/UB-ssd@ssd_adaptive=1", Fake().cuda())
    raise AssertionError("ASSD with all-NaN ratios must fail")
except RuntimeError:
    pass
imp = imp_saved
print(f"ASSD: alpha {plan.notes['assd_alpha']:.4g} = the official percentile ({official_alpha:.4g}) at p = "
      f"{plan.notes['assd_percentile']:.2f}; dampens the top {np.log(1 + share * 100):.2f}% of the weights; update as the official code")

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
         (["--node", "LA/LB/UB-ga@unlearn_lr=1e-6", "--save-at", "5000"], "(1000 steps"),
         (["--node", "LA/LB/UC-purge"], "cannot unlearn C")]
for argv, expected in cases:
    sys.argv = ["run_op", "--run", "args", *argv]
    try:
        vlu.run_op.main()
        raise AssertionError(f"{argv} should have been rejected")
    except ValueError as e:
        assert expected in str(e), (argv, str(e))
print("run_op: --save-at rejected for VLA-Forget, for SSD (still 0 steps under --steps) and beyond the step budget; purge of a task never learned rejected")

# Purge where hard links fail: the weights are copied through a temporary file (a partial file would mark the node done).
real_link = vlu.run_op.os.link
vlu.run_op.os.link = lambda *a: (_ for _ in ()).throw(OSError("no hard links here"))
try:
    sys.argv = ["run_op", "--run", "args", "--node", "LA/LB/UB-purge"]
    vlu.run_op.main()
finally:
    vlu.run_op.os.link = real_link
purged = Node.from_path("args", "LA/LB/UB-purge")
assert purged.weights_path.read_bytes() == purged.parent.weights_path.read_bytes()
assert not purged.weights_path.is_symlink() and not list(purged.dir.glob("*.tmp"))
print("run_op: purge without hard links copies the parent's weights via a temporary file")

# run_op end to end on the fake model (loading stubbed): SSD and SalUn as the main run trains them, with the
# recipe's own learning rates (no recipe-wide one); op.json records the method's rate, cost.json the op's cost.
probe_of = {id(demos[t]): probes[t] for t in demos}
real = {name: getattr(vlu.run_op, name) for name in ("RECIPE", "gr00t_config", "build_processor", "load_model",
                                                      "modality_configs", "load_demos", "fixed_batch")}
patched = {"RECIPE": dataclasses.replace(recipe, unlearn_lr=None), "gr00t_config": lambda: None,
           "build_processor": lambda config: proc, "load_model": lambda config, weights: Fake().cuda(),
           "modality_configs": lambda config: mcfg, "load_demos": lambda m: demos,
           "fixed_batch": lambda d, n, processor, m: probe_of[id(d)]}
try:
    for name, value in patched.items():
        setattr(vlu.run_op, name, value)
    for method in ("ssd", "salun"):
        sys.argv = ["run_op", "--run", "args", "--node", f"LA/LB/UB-{method}"]
        vlu.run_op.main()
        node = Node.from_path("args", f"LA/LB/UB-{method}")
        op_json, cost = json.loads((node.dir / "op.json").read_text()), json.loads((node.dir / "cost.json").read_text())
        assert node.done and cost["seconds"] > 0 and cost["resumed"] is False and cost["gpu"], cost
        assert op_json["lr"] == (0.0 if method == "ssd" else RECIPE.unlearn_lrs["salun"]), op_json["lr"]
        assert op_json["recipe"]["unlearn_lr"] == (None if method == "ssd" else RECIPE.unlearn_lrs["salun"])
finally:
    for name, value in real.items():
        setattr(vlu.run_op, name, value)
print("run_op: SSD and SalUn end to end with the configured rates; op.json and cost.json written")

# --- The training log after an interrupted attempt -----------------------------------------------------
from vlu.train import jsonl_logger  # noqa: E402

log_path = Node.from_path("args", "LA").dir / "log_test.jsonl"
log_path.write_text("".join(json.dumps({"step": s}) + "\n" for s in range(6)) + '{"step": 6, "lo')
log = jsonl_logger(log_path, keep_through=3)  # resumed from the checkpoint at step 3
log({"step": 4})
assert [json.loads(line)["step"] for line in log_path.read_text().splitlines()] == [0, 1, 2, 3, 4]
jsonl_logger(log_path)  # a fresh start
assert log_path.read_text() == ""
print("train log: a resumed op drops the records after its checkpoint (and a cut-off line); a fresh start empties it")
print("ALL OPS CHECKS PASSED")
