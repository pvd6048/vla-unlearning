"""SalUn (Fan et al., 2024) adapted to a robot policy.

1. Saliency mask: the weights with the largest forget-loss gradient. Per tensor, the top
   `saliency` fraction are kept (the original uses one global threshold; per-tensor avoids
   sorting billions of values).
2. "Random labeling": forget-task observations are trained toward *another kept task's*
   actions, plus normal training on kept tasks. Only masked weights are updated.
"""

import torch

from vla_unlearning import groot
from vla_unlearning.methods.loop import optimize, retain_term
from vla_unlearning.methods.ssd import fisher


def saliency_mask(model, data, frac, n):
    grads = fisher(model, data["forget"], n)  # mean squared gradient; same ranking as |grad|
    masks = {}
    for p, g in zip(groot.trainable(model), grads):
        k = max(1, int(frac * g.numel()))
        threshold = g.flatten().kthvalue(g.numel() - k + 1).values
        masks[p] = (g >= threshold).to(p.dtype)
    return masks


def unlearn(model, ref, data, cfg):
    mask = saliency_mask(model, data, cfg.get("saliency", 0.5), cfg.get("saliency_batches", 20))

    def loss_fn(step):
        forget, other = data["forget"](), data["retain"]()
        n = min(len(forget["inputs"]["action"]), len(other["inputs"]["action"]))
        forget["inputs"]["action"][:n] = other["inputs"]["action"][:n]  # relabel with another task's actions
        l, _ = groot.loss(model, forget)
        r, logs = retain_term(model, data, cfg)
        return l.mean() + r, {"relabeled": l.mean().item(), **logs}

    optimize(model, loss_fn, {"retain_weight": 1.0, **cfg}, grad_mask=mask)
