"""What each op trains on and which loss it minimizes.

`plan_op(node)` returns the batch composition, loss, step budget and learning rate for the node's
last op. Adding an unlearning method = writing one `_plan_<method>` function and registering it in
UNLEARN_METHODS. All losses are built from GR00T's flow-matching loss L_FM (GR00T has no token
cross-entropy).
"""

from dataclasses import dataclass, field
from typing import Callable

import torch

from vlu.config import Recipe
from vlu.data import Slots
from vlu.nodes import Node

# loss_fn(model, batch) -> (loss to minimize, {metric name: float}); batch = {part name: inputs}
LossFn = Callable[[torch.nn.Module, dict], tuple[torch.Tensor, dict[str, float]]]


@dataclass
class OpPlan:
    parts: dict[str, Slots]
    loss_fn: LossFn
    steps: int
    lr: float
    notes: dict = field(default_factory=dict)  # method hyperparameters, recorded in op.json


def fm_loss(model, inputs: dict) -> torch.Tensor:
    """GR00T's flow-matching loss on one batch (mean over the batch, horizon and action dims)."""
    return model(inputs)["loss"]


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


def _pending(name: str):
    def plan(node: Node, recipe: Recipe) -> OpPlan:
        raise NotImplementedError(f"unlearning method {name!r} is not implemented yet (see README, open decisions)")

    return plan


# "purge" is handled in run_op.py: it trains nothing and only drops the task from the replay
# buffer, which happens automatically because the task is no longer retained after the U op.
UNLEARN_METHODS: dict[str, Callable[[Node, Recipe], OpPlan]] = {
    "ga": _plan_ga,
    "graddiff": _plan_graddiff,
    "npo": _pending("npo"),
    "ssd": _pending("ssd"),
    "salun": _pending("salun"),
}
ZERO_STEP_METHODS = {"purge"}


def plan_op(node: Node, recipe: Recipe) -> OpPlan:
    op = node.last_op
    if op.kind == "L":
        return _plan_learn(node, recipe)
    if op.method not in UNLEARN_METHODS:
        raise ValueError(f"unknown unlearning method {op.method!r}; known: {sorted(UNLEARN_METHODS)}")
    return UNLEARN_METHODS[op.method](node, recipe)
