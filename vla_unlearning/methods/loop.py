"""The one optimization loop shared by all gradient-based unlearning methods."""

import json

import torch

from vla_unlearning import groot


def optimize(model, loss_fn, cfg, grad_mask=None):
    """Run cfg["steps"] AdamW steps on `loss_fn(step) -> (loss, logs)`.

    Uses bf16 autocast like NVIDIA's trainer. `grad_mask` ({param: 0/1 tensor}) restricts
    updates to selected weights (SalUn).
    """
    params = groot.trainable(model)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    model.train()
    for step in range(cfg["steps"]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, logs = loss_fn(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_mask is not None:
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(grad_mask[p])
        torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 1.0))
        opt.step()
        if step % 10 == 0 or step == cfg["steps"] - 1:
            print(json.dumps({"step": step, "loss": loss.item(), **logs}), flush=True)


def retain_term(model, data, cfg):
    """Ordinary training loss on the replay buffer of kept tasks, weighted by cfg["retain_weight"]."""
    w = cfg.get("retain_weight", 0.0)
    if w == 0 or data["retain"] is None:
        return 0.0, {}
    l, _ = groot.loss(model, data["retain"]())
    return w * l.mean(), {"retain": l.mean().item()}
