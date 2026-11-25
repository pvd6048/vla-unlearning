"""Small interface for methods whose update is more than loss.backward(); optimizer.step().

A rule may additionally define finalize(model) -> metrics. train.py calls it before publishing
the final checkpoint, and logs new probes of the transformed weights (e.g. task-vector negation).
"""
from typing import Protocol

import torch

from vlu.model import check_finite


class UpdateRule(Protocol):
    disable_weight_decay: bool

    def initialize(self, optimizer): ...
    def step(self, model, batch, optimizer, max_grad_norm: float) -> tuple[torch.Tensor, dict]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict): ...


def gradients(loss, params):
    """Detached, finite gradients; disconnected parameters remain None."""
    check_finite({"loss": loss}, "method loss")
    values = torch.autograd.grad(loss, params, allow_unused=True)
    check_finite({str(i): g for i, g in enumerate(values) if g is not None}, "method gradients")
    return [g.detach() if g is not None else None for g in values]


def set_gradients(params, values, max_grad_norm):
    for p, g in zip(params, values):
        p.grad = g
    return torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True).item()
