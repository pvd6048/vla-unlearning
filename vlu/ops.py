"""What each op trains on and which loss it minimizes.

`plan_op(node)` returns the batch composition, loss, step budget and learning rate for the node's
last op. Adding an unlearning method = writing one `_plan_<method>` function and registering it in
UNLEARN_METHODS. All losses are built from GR00T's flow-matching loss L_FM (GR00T has no token
cross-entropy).

A plan may also have a `prepare(ctx)` step, run once by vlu/run_op.py after the parent's weights are
loaded and before training (NPO: frozen reference copy; SSD: the whole method; SalUn: weight mask).
"""

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch

from vlu.config import Recipe
from vlu.data import OpBatches, Slots
from vlu.model import trainable_named_parameters
from vlu.nodes import Node

# loss_fn(model, batch) -> (loss to minimize, {metric name: float}); batch = {part name: inputs}
LossFn = Callable[[torch.nn.Module, dict], tuple[torch.Tensor, dict[str, float]]]


@dataclass
class OpContext:
    """What a plan's prepare step can use: the model (parent weights loaded) and the demos."""

    node: Node
    model: torch.nn.Module
    demos: dict
    processor: object
    modality_configs: dict
    seed: int

    def batches(self, slots: Slots, n: int, stream: int = 0):
        """n batches with the given slots, without augmentation, reproducible from (seed, stream)."""
        was_training = self.processor.training
        self.processor.eval()
        try:
            data = OpBatches(self.demos, {"x": slots}, self.processor, self.modality_configs,
                             seed=self.seed * 1000 + 100 + stream, num_steps=n)
            for i in range(n):
                yield data[i]["x"]
        finally:
            if was_training:
                self.processor.train()


@dataclass
class OpPlan:
    parts: dict[str, Slots]
    loss_fn: LossFn
    steps: int
    lr: float
    notes: dict = field(default_factory=dict)  # method hyperparameters, recorded in op.json
    prepare: Callable[[OpContext], None] | None = None
    # {parameter name: 0/1 tensor}: gradients are multiplied by it before every step (SalUn)
    grad_mask: dict[str, torch.Tensor] | None = None


def fm_loss(model, inputs: dict) -> torch.Tensor:
    """GR00T's flow-matching loss on one batch (mean over the valid action entries of the batch)."""
    return model(inputs)["loss"]


def fm_sse_per_sample(out: dict) -> torch.Tensor:
    """Per-sample sum of squared flow-matching errors over the valid action entries, shape (B,).
    -SSE is our stand-in for an action chunk's log-likelihood (as for a Gaussian)."""
    return out["action_loss"].float().sum(dim=(1, 2))


def accumulate_grads(ctx: OpContext, slots: Slots, n: int, stream: int, square: bool) -> dict[str, torch.Tensor]:
    """Sum over n batches of each trainable parameter's L_FM gradient (square=False) or squared
    gradient (square=True, i.e. n x the diagonal empirical Fisher of batch gradients)."""
    model = ctx.model
    params = trainable_named_parameters(model)
    total = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in params.items()}
    was_training = model.training
    model.eval()
    for inputs in ctx.batches(slots, n, stream):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fm_loss(model, inputs)
        loss.backward()
        for name, p in params.items():
            if p.grad is not None:
                total[name] += p.grad.float().square() if square else p.grad.float()
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return total


# --- Learning --------------------------------------------------------------------------------------


def _plan_learn(node: Node, recipe: Recipe) -> OpPlan:
    """Fine-tune on the new task with ER: part of every batch replays the retained earlier tasks.
    Unlearned tasks are not retained, so they are never replayed."""
    task = node.last_op.task
    replay = tuple(t for t in node.parent.retained_tasks() if t != task)
    n_replay = round(recipe.batch_size * recipe.replay_fraction) if replay else 0
    slots = [(task,)] * (recipe.batch_size - n_replay) + [replay] * n_replay

    def loss_fn(model, batch):
        loss = fm_loss(model, batch["train"])
        return loss, {"fm": loss.item()}

    return OpPlan({"train": slots}, loss_fn, recipe.learn_steps, recipe.lr, {"replay_tasks": replay})


# --- Unlearning ------------------------------------------------------------------------------------
# Every method gets the same step budget (recipe.unlearn_steps) and learning rate.


def _forget_retain(node: Node, recipe: Recipe) -> tuple[Slots, Slots]:
    task = node.last_op.task
    retain = tuple(t for t in node.parent.retained_tasks() if t != task)
    if task not in node.parent.retained_tasks():
        raise ValueError(f"{node}: cannot unlearn {task}, it is not retained by the parent")
    return [(task,)] * recipe.batch_size, ([retain] * recipe.batch_size if retain else [])


def _plan_ga(node: Node, recipe: Recipe) -> OpPlan:
    """Gradient ascent: maximize L_FM on the forget task."""
    forget, _ = _forget_retain(node, recipe)

    def loss_fn(model, batch):
        forget_loss = fm_loss(model, batch["forget"])
        return -forget_loss, {"forget_fm": forget_loss.item()}

    return OpPlan({"forget": forget}, loss_fn, recipe.unlearn_steps, recipe.unlearn_lr)


def _plan_graddiff(node: Node, recipe: Recipe) -> OpPlan:
    """Gradient difference: ascent on the forget task plus ordinary descent on the retained tasks."""
    forget, retain = _forget_retain(node, recipe)
    if not retain:
        raise ValueError(f"{node}: graddiff needs at least one retained task")

    def loss_fn(model, batch):
        forget_loss = fm_loss(model, batch["forget"])
        retain_loss = fm_loss(model, batch["retain"])
        return retain_loss - forget_loss, {"forget_fm": forget_loss.item(), "retain_fm": retain_loss.item()}

    return OpPlan({"forget": forget, "retain": retain}, loss_fn, recipe.unlearn_steps, recipe.unlearn_lr)


def _plan_npo(node: Node, recipe: Recipe) -> OpPlan:
    """NPO (Zhang et al. 2024) on the forget task:
        L = 2/beta * mean softplus(beta * (log p_theta - log p_ref)),   log p := -SSE (fm_sse_per_sample)
    with p_ref the model before the op (frozen copy). Its gradient is GA's, weighted per sample by
    2*sigmoid(beta * (log p_theta - log p_ref)), which starts at 1 and fades as the forget data
    becomes unlikely relative to the reference. theta and the reference see the same flow-matching
    noise and timesteps (the RNG state is replayed), so their SSEs are directly comparable."""
    forget, _ = _forget_retain(node, recipe)
    beta = recipe.npo_beta
    state = {}

    def prepare(ctx: OpContext):
        ref = copy.deepcopy(ctx.model)
        ref.requires_grad_(False)
        state["ref"] = ref

    def loss_fn(model, batch):
        inputs, ref = batch["forget"], state["ref"]
        rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        out = model(inputs)
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
        ref.train(model.training)
        with torch.no_grad():
            ref_out = ref(inputs)
        log_ratio = fm_sse_per_sample(ref_out) - fm_sse_per_sample(out)  # log p_theta - log p_ref
        loss = (2 / beta) * torch.nn.functional.softplus(beta * log_ratio).mean()
        weight = 2 * torch.sigmoid(beta * log_ratio.detach())
        return loss, {"forget_fm": out["loss"].item(), "log_ratio": log_ratio.mean().item(), "npo_weight": weight.mean().item()}

    return OpPlan({"forget": forget}, loss_fn, recipe.unlearn_steps, recipe.unlearn_lr,
                  {"npo_beta": beta, "log_likelihood_proxy": "-sum of squared flow-matching errors of the chunk"},
                  prepare=prepare)


def _plan_ssd(node: Node, recipe: Recipe) -> OpPlan:
    """Selective Synaptic Dampening (Foster et al. 2024), no training: with diagonal Fisher
    importances I_f on the forget task and I_D on all the parent's training tasks (forget included),
    every trainable weight with I_f > alpha * I_D is multiplied by min(lambda * I_D / I_f, 1)."""
    task = node.last_op.task
    _forget_retain(node, recipe)  # checks that the task is retained
    everything = tuple(node.parent.retained_tasks())
    plan = OpPlan({}, lambda model, batch: None, 0, 0.0,
                  {"ssd_alpha": recipe.ssd_alpha, "ssd_lambda": recipe.ssd_lambda, "importance_batches": recipe.importance_batches,
                   "data_tasks": everything})

    def prepare(ctx: OpContext):
        n = recipe.importance_batches
        imp_f = accumulate_grads(ctx, [(task,)] * recipe.batch_size, n, stream=0, square=True)
        imp_d = accumulate_grads(ctx, [everything] * recipe.batch_size, n, stream=1, square=True)
        dampened = total = 0
        with torch.no_grad():
            for name, p in trainable_named_parameters(ctx.model).items():
                f, d = imp_f[name], imp_d[name]
                selected = f > recipe.ssd_alpha * d
                factor = torch.clamp(recipe.ssd_lambda * d / f.clamp_min(1e-30), max=1.0)
                p.mul_(torch.where(selected, factor, torch.ones_like(factor)).to(p.dtype))
                dampened += int(selected.sum())
                total += selected.numel()
        plan.notes["dampened_fraction"] = dampened / total
        print(f"SSD dampened {dampened:,} of {total:,} trainable weights ({dampened / total:.4%})", flush=True)

    plan.prepare = prepare
    return plan


def _plan_salun(node: Node, recipe: Recipe) -> OpPlan:
    """SalUn (Fan et al. 2024) with random labels: only the weights most salient to the forget task
    (largest |gradient of L_FM on it|, the top `salun_keep` fraction) are updated, to minimize
        L_FM(forget observations with random action labels) + L_FM(retained tasks).
    A "random label" for an action chunk is the (normalized) action chunk of a random retained-task
    frame: the analogue of a wrong class label. The mask is saved in the node directory, so a resumed
    op uses the same one."""
    forget, retain = _forget_retain(node, recipe)
    if not retain:
        raise ValueError(f"{node}: salun needs at least one retained task")
    plan = OpPlan({"forget": forget, "retain": retain}, None, recipe.unlearn_steps, recipe.unlearn_lr,
                  {"salun_keep": recipe.salun_keep, "importance_batches": recipe.importance_batches,
                   "random_label": "action chunk of a random retained-task frame"})

    def prepare(ctx: OpContext):
        mask_path = ctx.node.dir / "salun_mask.pt"
        if mask_path.exists():
            masks = torch.load(mask_path, map_location="cuda", weights_only=True)
        else:
            grads = accumulate_grads(ctx, forget, recipe.importance_batches, stream=0, square=False)
            saliency = {name: g.abs() for name, g in grads.items()}
            threshold = _top_fraction_threshold(saliency, recipe.salun_keep)
            masks = {name: s >= threshold for name, s in saliency.items()}
            ctx.node.dir.mkdir(parents=True, exist_ok=True)
            torch.save(masks, mask_path)
        kept = sum(int(m.sum()) for m in masks.values())
        total = sum(m.numel() for m in masks.values())
        plan.notes["kept_fraction"] = kept / total
        plan.grad_mask = {name: m.to(torch.float32) for name, m in masks.items()}
        print(f"SalUn updates {kept:,} of {total:,} trainable weights ({kept / total:.2%})", flush=True)

    def loss_fn(model, batch):
        relabeled = dict(batch["forget"])
        relabeled["action"] = batch["retain"]["action"]
        relabeled["action_mask"] = batch["retain"]["action_mask"]
        forget_loss = fm_loss(model, relabeled)
        retain_loss = fm_loss(model, batch["retain"])
        return forget_loss + retain_loss, {"relabeled_fm": forget_loss.item(), "retain_fm": retain_loss.item()}

    plan.loss_fn, plan.prepare = loss_fn, prepare
    return plan


def _top_fraction_threshold(values: dict[str, torch.Tensor], keep: float, sample_size: int = 10_000_000) -> float:
    """Value above which `keep` of all entries lie, estimated from a uniform random sample (an exact
    quantile over ~1e9 entries does not fit torch.quantile)."""
    total = sum(v.numel() for v in values.values())
    samples = [v.flatten()[torch.randint(v.numel(), (max(1, round(sample_size * v.numel() / total)),), device=v.device)]
               for v in values.values()]
    return torch.quantile(torch.cat(samples).float(), 1 - keep).item()


# "purge" is handled in run_op.py: it trains nothing and only drops the task from the replay
# buffer, which happens automatically because the task is no longer retained after the U op.
UNLEARN_METHODS: dict[str, Callable[[Node, Recipe], OpPlan]] = {
    "ga": _plan_ga,
    "graddiff": _plan_graddiff,
    "npo": _plan_npo,
    "ssd": _plan_ssd,
    "salun": _plan_salun,
}
ZERO_STEP_METHODS = {"purge"}


def plan_op(node: Node, recipe: Recipe) -> OpPlan:
    op = node.last_op
    if op.kind == "L":
        return _plan_learn(node, recipe)
    if op.method not in UNLEARN_METHODS:
        raise ValueError(f"unknown unlearning method {op.method!r}; known: {sorted(UNLEARN_METHODS)}")
    return UNLEARN_METHODS[op.method](node, recipe)
