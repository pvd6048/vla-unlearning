"""Gradient Difference: gradient ascent on the forget task + normal training on kept tasks."""

from vla_unlearning import groot
from vla_unlearning.methods.loop import optimize, retain_term


def unlearn(model, ref, data, cfg):
    def loss_fn(step):
        l, _ = groot.loss(model, data["forget"]())
        r, logs = retain_term(model, data, cfg)
        return -l.mean() + r, {"forget": l.mean().item(), **logs}

    optimize(model, loss_fn, {"retain_weight": 1.0, **cfg})
