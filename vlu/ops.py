"""What each op trains on and which loss it minimizes.

`plan_op(node)` returns the batch composition, loss, step budget and learning rate for the node's
last op. Adding an unlearning method = writing one `_plan_<method>` function and registering it in
UNLEARN_METHODS. All losses are built from GR00T's flow-matching loss L_FM (GR00T has no token
cross-entropy).

A plan may also have a `prepare(ctx)` step, run once by vlu/run_op.py after the parent's weights are
loaded and before training (NPO: frozen reference copy; SSD: the whole method; SalUn: weight mask).
"""

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

from vlu.config import Recipe
from vlu.data import OpBatches, ReplayBuffer, Slots
from vlu.drift.weights import embodiment_index, live_part
from vlu.model import trainable_named_parameters
from vlu.nodes import Node, check_op, recipe_for  # noqa: F401  (recipe_for: used by vlu.run_op and the tests)

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
    # {parameter name: bool tensor}: gradients are multiplied by it before every step (SalUn)
    grad_mask: dict[str, torch.Tensor] | None = None
    # A method with its own optimization loop (VLA-Forget) replaces vlu.train.train with
    # custom_train(ctx, probes); it must leave ctx.model ready to be saved.
    custom_train: Callable[[OpContext, dict], None] | None = None
    probe_every: int | None = None  # steps between probe losses; None = recipe.probe_every


def fm_loss(model, inputs: dict) -> torch.Tensor:
    """GR00T's flow-matching loss on one batch (mean over the valid action entries of the batch)."""
    return model(inputs)["loss"]


def fm_sse_per_sample(out: dict) -> torch.Tensor:
    """Per-sample sum of squared flow-matching errors over the valid action entries, shape (B,).
    -SSE is our stand-in for an action chunk's log-likelihood (as for a Gaussian)."""
    return out["action_loss"].float().sum(dim=(1, 2))


def frozen_reference(model: torch.nn.Module) -> torch.nn.Module:
    """A frozen copy of the model as it is now (theta0), for methods that compare against it (NPO,
    VLA-Forget). The weights that are frozen in `model` are shared, not copied (~3.3 GB in GR00T):
    an op never changes them while it uses the reference. (VLA-Forget's final merge writes into
    frozen weights, so it drops its reference first.)"""
    shared = {id(p): p for p in model.parameters() if not p.requires_grad}
    ref = copy.deepcopy(model, memo=shared)
    ref.requires_grad_(False)
    return ref


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
    """Fine-tune on the new task with ER as in Liu et al. 2026 (continual-GR00T): every batch is a full
    batch of the new task plus an equally large batch from the fixed replay buffer of the retained
    earlier tasks. Unlearned tasks are not retained, so they are never replayed."""
    task = node.last_op.task
    replay = tuple(t for t in node.parent.retained_tasks() if t != task)
    slots = [(task,)] * recipe.batch_size
    if replay:
        slots += [ReplayBuffer(replay, recipe.replay_per_task)] * recipe.batch_size

    def loss_fn(model, batch):
        loss = fm_loss(model, batch["train"])
        return loss, {"fm": loss.item()}

    return OpPlan({"train": slots}, loss_fn, recipe.learn_steps, recipe.lr,
                  {"replay_tasks": replay, "replay_per_task": recipe.replay_per_task})


# --- Unlearning ------------------------------------------------------------------------------------
# Every method gets the same step budget (recipe.unlearn_steps) and learning rate.


def _forget_retain(node: Node, recipe: Recipe) -> tuple[Slots, Slots]:
    """Batch slots of the forget task and of the retained tasks (drawn uniformly); plan_op has
    already checked that the op can be applied (check_op)."""
    task = node.last_op.task
    retain = tuple(t for t in node.parent.retained_tasks() if t != task)
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
        state["ref"] = frozen_reference(ctx.model)

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
    importances I_f on the forget task and I_ref as the reference, every trainable weight with
    I_f > alpha * I_ref is multiplied by min(lambda * I_ref / I_f, 1).

    recipe.ssd_reference = "retain" (default): I_ref = I_r, the importance on the retained tasks.
    "all": the paper's full training set D, computed as the official code does (an unshuffled
    ConcatDataset of the retain and forget sets, so every batch is all-retain or all-forget):
    I_ref = (1 - s) * I_r + s * I_f, with s the forget task's share of the frames. The paper's s is
    1-5%, where I_D ~ I_r; ours is 0.2-0.45, where alpha >= 1/s can select nothing and the dampening
    factor cannot go below lambda * s (METHODS.md section 4).
    recipe.ssd_adaptive: alpha is ASSD's (Schoepf et al. 2024), the p-th percentile of I_f / I_ref over
    the weights, p = 100 - ln(1 + 100 s)."""
    task = node.last_op.task
    _, retain = _forget_retain(node, recipe)  # plan_op checked that there is a retained task (check_op)
    if recipe.ssd_reference not in ("retain", "all"):
        raise ValueError(f"ssd_reference must be 'retain' or 'all', not {recipe.ssd_reference!r}")
    retain_tasks = retain[0]
    plan = OpPlan({}, lambda model, batch: None, 0, 0.0,
                  {"ssd_alpha": "adaptive" if recipe.ssd_adaptive else recipe.ssd_alpha, "ssd_lambda": recipe.ssd_lambda,
                   "ssd_reference": recipe.ssd_reference, "importance_batches": recipe.importance_batches,
                   "retain_tasks": retain_tasks})

    def prepare(ctx: OpContext):
        n = recipe.importance_batches
        imp_f = accumulate_grads(ctx, [(task,)] * recipe.batch_size, n, stream=0, square=True)
        imp_ref = accumulate_grads(ctx, [retain_tasks] * recipe.batch_size, n, stream=1, square=True)
        # |D_f| / |D| in frames, as the official codes count samples (SSD: the batches of the full set;
        # ASSD: len(forget set) / (len(retain set) + len(forget set)))
        n_forget = len(ctx.demos[task].index)
        share = n_forget / (n_forget + sum(len(ctx.demos[t].index) for t in retain_tasks))
        plan.notes["forget_share"] = share
        if recipe.ssd_reference == "all":
            for name, r in imp_ref.items():  # the same number of batches in both sums: the mean's 1/n cancels
                r.mul_(1 - share).add_(imp_f[name], alpha=share)
        embodiment = embodiment_index()
        alpha = recipe.ssd_alpha
        if recipe.ssd_adaptive:
            # ASSD (if-loops/adaptive-selective-synaptic-dampening@f04c49d, error_unlearning.ipynb: ssd() and
            # modify_weight): share in percent, natural log; np.nanpercentile of I_f / I_ref over the weights,
            # NaN (0/0) and inf (x/0) left out.
            percentile = 100 - math.log(1 + share * 100)
            alpha = top_fraction_threshold({k: live_part(k, v, embodiment) for k, v in imp_f.items()}, 1 - percentile / 100,
                                           divide_by={k: live_part(k, v, embodiment) for k, v in imp_ref.items()})
            if not math.isfinite(alpha):
                raise RuntimeError(f"ASSD: alpha is {alpha} (no finite importance ratio); the importances are degenerate")
            plan.notes.update({"assd_percentile": percentile, "assd_alpha": alpha})
        dampened = total = 0
        with torch.no_grad():
            for name, p in trainable_named_parameters(ctx.model).items():
                f, d = imp_f[name], imp_ref[name]
                selected = f > alpha * d
                factor = torch.clamp(recipe.ssd_lambda * d / f.clamp_min(1e-30), max=1.0)
                p.mul_(torch.where(selected, factor, torch.ones_like(factor)).to(p.dtype))
                # counted over the weights that can change (as SalUn's kept_fraction): the other
                # embodiments' rows have zero importance and are never selected
                live = live_part(name, selected, embodiment)
                dampened += int(live.sum())
                total += live.numel()
        plan.notes["dampened_fraction"] = dampened / total
        print(f"SSD (alpha {alpha:.4g}, reference {recipe.ssd_reference}, forget share {share:.3f}) dampened "
              f"{dampened:,} of the {total:,} trainable weights that can change ({dampened / total:.4%})", flush=True)

    plan.prepare = prepare
    return plan


def _plan_salun(node: Node, recipe: Recipe) -> OpPlan:
    """SalUn (Fan et al. 2024) with random labels: only the weights most salient to the forget task
    (largest |gradient of L_FM on it|, the top `salun_keep` fraction) are updated, to minimize the
    flow-matching loss over a batch that mixes forget observations with random action labels and
    retained-task samples. As the official code (ConcatDataset of the relabeled forget set and the
    retain set), the mix follows the data sizes and the loss is the mean over the mixed batch.
    A "random label" for an action chunk is the (normalized) action chunk of a random retained-task
    frame: the analogue of a wrong class label. Masked-out weights stay exactly as they were (the
    official code restores them after every step). The mask is saved in the node directory, so a
    resumed op uses the same one."""
    forget, retain = _forget_retain(node, recipe)
    forget_task, retain_tasks = forget[0][0], retain[0]
    plan = OpPlan({}, None, recipe.unlearn_steps, recipe.unlearn_lr,
                  {"salun_keep": recipe.salun_keep, "importance_batches": recipe.importance_batches,
                   "random_label": "action chunk of a random retained-task frame"})

    def prepare(ctx: OpContext):
        # Batch composition: forget and retain samples in proportion to their frame counts.
        n_forget_frames = len(ctx.demos[forget_task].index)
        n_retain_frames = sum(len(ctx.demos[t].index) for t in retain_tasks)
        n_f = min(recipe.batch_size - 1, max(1, round(recipe.batch_size * n_forget_frames / (n_forget_frames + n_retain_frames))))
        plan.parts = {"forget": [(forget_task,)] * n_f, "retain": [retain_tasks] * (recipe.batch_size - n_f)}
        plan.notes["batch_forget_retain"] = [n_f, recipe.batch_size - n_f]
        embodiment = embodiment_index()
        mask_path = ctx.node.dir / "salun_mask.pt"
        if mask_path.exists():
            masks = torch.load(mask_path, map_location="cuda", weights_only=True)
        else:
            grads = accumulate_grads(ctx, forget, recipe.importance_batches, stream=0, square=False)
            saliency = {name: g.abs() for name, g in grads.items()}
            # The fraction refers to the weights that can change: LIBERO's row of the per-embodiment
            # projectors (the other 31 rows never get a gradient); zero saliency is never selected.
            threshold = top_fraction_threshold({n: live_part(n, s, embodiment) for n, s in saliency.items()}, recipe.salun_keep)
            masks = {name: (s >= threshold) & (s > 0) for name, s in saliency.items()}
            ctx.node.dir.mkdir(parents=True, exist_ok=True)
            tmp = mask_path.with_name(mask_path.name + ".tmp")  # atomic: a resumed op trusts an existing mask
            torch.save(masks, tmp)
            tmp.replace(mask_path)
        kept = sum(int(live_part(n, m, embodiment).sum()) for n, m in masks.items())
        total = sum(live_part(n, m, embodiment).numel() for n, m in masks.items())
        plan.notes["kept_fraction"] = kept / total
        plan.grad_mask = masks  # bool: 1 byte per weight
        print(f"SalUn updates {kept:,} of the {total:,} trainable weights that can change ({kept / total:.2%})", flush=True)

    def loss_fn(model, batch):
        n_f, n_r = len(batch["forget"]["action"]), len(batch["retain"]["action"])
        pick = torch.arange(n_f) % n_r  # random labels: the retain samples' (randomly drawn) action chunks
        relabeled = dict(batch["forget"])
        relabeled["action"] = batch["retain"]["action"][pick]
        relabeled["action_mask"] = batch["retain"]["action_mask"][pick]
        forget_loss = fm_loss(model, relabeled)
        retain_loss = fm_loss(model, batch["retain"])
        # = the mean over one mixed batch (every sample has the same number of valid action entries)
        loss = (n_f * forget_loss + n_r * retain_loss) / (n_f + n_r)
        return loss, {"relabeled_fm": forget_loss.item(), "retain_fm": retain_loss.item()}

    plan.loss_fn, plan.prepare = loss_fn, prepare
    return plan


def top_fraction_threshold(values: dict[str, torch.Tensor], keep: float, sample_size: int = 10_000_000,
                           divide_by: dict[str, torch.Tensor] | None = None) -> float:
    """Value above which `keep` of all entries lie (NaN entries ignored), estimated from a uniform random
    sample (an exact quantile over ~1e9 entries does not fit torch.quantile); exact when there are at
    most `sample_size` entries. With divide_by, the entries are values / divide_by (non-finite ratios
    ignored), computed only where sampled."""
    total = sum(v.numel() for v in values.values())
    generator = {}  # one fixed-seed generator per device: the global RNG is left untouched

    def entries(name):
        v = values[name].flatten()
        if total > sample_size:
            g = generator.setdefault(v.device, torch.Generator(device=v.device).manual_seed(0))
            index = torch.randint(v.numel(), (max(1, round(sample_size * v.numel() / total)),), device=v.device, generator=g)
        else:
            index = slice(None)
        x = v[index].float()
        if divide_by is not None:
            x = x / divide_by[name].flatten()[index].float()
            x = torch.where(torch.isfinite(x), x, torch.nan)
        return x

    return torch.nanquantile(torch.cat([entries(name) for name in values]), 1 - keep).item()


def _plan_vlaforget(node: Node, recipe: Recipe) -> OpPlan:
    """VLA-Forget (Ranjan & Polyzou 2026), staged LoRA unlearning of vision, projector and
    reasoning/action layers; see vlu/vlaforget.py for the port and its choices."""
    forget, retain = _forget_retain(node, recipe)
    task, retain_tasks = forget[0][0], retain[0]
    plan = OpPlan({}, None, recipe.unlearn_steps, recipe.vlaforget.lr, {"vlaforget": asdict(recipe.vlaforget)})

    def custom_train(ctx: OpContext, probes: dict):
        from vlu import vlaforget  # imported here: vlaforget imports this module

        vlaforget.run(ctx, probes, recipe, task, retain_tasks, plan.steps)

    plan.custom_train = custom_train
    return plan


# "purge" is handled in run_op.py: it trains nothing and only drops the task from the replay
# buffer, which happens automatically because the task is no longer retained after the U op.
UNLEARN_METHODS: dict[str, Callable[[Node, Recipe], OpPlan]] = {
    "ga": _plan_ga,
    "graddiff": _plan_graddiff,
    "npo": _plan_npo,
    "ssd": _plan_ssd,
    "salun": _plan_salun,
    "vlaforget": _plan_vlaforget,
}
ZERO_STEP_METHODS = {"purge"}


def plan_op(node: Node, recipe: Recipe) -> OpPlan:
    op = node.last_op
    check_op(node)  # e.g. the methods in NEEDS_RETAINED_TASK need a retained task besides the forget task
    recipe = recipe_for(op, recipe)
    if op.kind == "L":
        return _plan_learn(node, recipe)
    if op.method not in UNLEARN_METHODS:
        raise ValueError(f"unknown unlearning method {op.method!r}; known: {sorted(UNLEARN_METHODS)}")
    plan = UNLEARN_METHODS[op.method](node, recipe)
    plan.probe_every = recipe.unlearn_probe_every  # U ops are short: watch them closely
    return plan
