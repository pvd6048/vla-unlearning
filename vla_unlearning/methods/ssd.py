"""Selective Synaptic Dampening (Foster et al., 2024). Retraining-free.

1. Estimate the diagonal Fisher (mean squared gradient) on the forget data and on the retain data.
2. Weights much more important to the forget task (F_forget > alpha * F_retain) are shrunk
   by a factor min(lambda * F_retain / F_forget, 1).
"""

import torch

from vla_unlearning import groot


def fisher(model, batches, n):
    params = groot.trainable(model)
    fim = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    model.train()
    for _ in range(n):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, _ = groot.loss(model, batches())
        l.mean().backward()
        for f, p in zip(fim, params):
            if p.grad is not None:
                f += p.grad.float() ** 2 / n
    model.zero_grad(set_to_none=True)
    return fim


def unlearn(model, ref, data, cfg):
    alpha, lam, n = cfg.get("alpha", 10.0), cfg.get("lambda", 1.0), cfg.get("fisher_batches", 50)
    f_forget = fisher(model, data["forget"], n)
    f_retain = fisher(model, data["retain"], n)
    dampened = 0
    with torch.no_grad():
        for p, ff, fr in zip(groot.trainable(model), f_forget, f_retain):
            select = ff > alpha * fr
            factor = (lam * fr / (ff + 1e-12)).clamp(max=1.0)
            p.mul_(torch.where(select, factor, torch.ones_like(factor)).to(p.dtype))
            dampened += select.sum().item()
    print(f"SSD dampened {dampened} weights")
