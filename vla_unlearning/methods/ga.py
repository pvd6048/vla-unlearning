"""Gradient Ascent: maximize the flow-matching loss on the forget task.

The loss has no upper bound, so GA is controlled only by its step budget and gradient clipping.
"""

from vla_unlearning import groot
from vla_unlearning.methods.loop import optimize


def unlearn(model, ref, data, cfg):
    def loss_fn(step):
        l, _ = groot.loss(model, data["forget"]())
        return -l.mean(), {"forget": l.mean().item()}

    optimize(model, loss_fn, cfg)
