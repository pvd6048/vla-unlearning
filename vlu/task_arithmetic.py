"""Forget-task fine-tuning followed by task-vector negation (Ilharco et al., ICLR 2023).

Starting at theta0, fit the forget task to obtain theta_f, then return
theta0 - scale * (theta_f - theta0). Auxiliary fitting is part of the method's compute budget.
"""
import torch

from vlu.model import check_finite, trainable_named_parameters
from vlu.updates import gradients, set_gradients


class TaskArithmeticUpdate:
    disable_weight_decay = False

    def __init__(self, scale):
        self.scale = scale
        self.anchor = None  # CPU copies: do not keep a second full set of weights on the GPU

    def initialize(self, optimizer):
        pass

    def step(self, model, batch, optimizer, max_grad_norm):
        named = trainable_named_parameters(model)
        if self.anchor is None:
            self.anchor = {n: p.detach().cpu().clone() for n, p in named.items()}
        if set(self.anchor) != set(named) or any(self.anchor[n].shape != p.shape for n, p in named.items()):
            raise ValueError("task arithmetic anchor does not match the trainable parameters")
        params = list(named.values())
        loss = model(batch["forget"])["loss"]
        norm = set_gradients(params, gradients(loss, params), max_grad_norm)
        optimizer.step()
        return loss.detach(), {"fit_forget_fm": loss.item(), "grad_norm": norm}

    @torch.no_grad()
    def finalize(self, model):
        """Negate once, before the shared loop publishes the completed checkpoint."""
        named = trainable_named_parameters(model)
        if self.anchor is None or set(self.anchor) != set(named):
            raise ValueError("task arithmetic needs a fitted model and matching anchor")
        squared_norm = 0.
        for name, p in named.items():
            anchor = self.anchor[name].to(device=p.device, dtype=p.dtype)
            delta = p - anchor
            squared_norm += float(delta.norm(dtype=torch.float64).square())
            p.copy_(anchor - self.scale * delta)
        check_finite(named, "task-vector negation")
        return {"task_vector_norm": squared_norm ** .5,
                "task_edit_norm": self.scale * squared_norm ** .5}

    def state_dict(self):
        return {"kind": "task_arithmetic", "scale": self.scale, "anchor": self.anchor}

    def load_state_dict(self, state):
        if state.get("kind") != "task_arithmetic" or state["scale"] != self.scale:
            raise ValueError("task arithmetic checkpoint has different method settings")
        if not state.get("anchor"):
            raise ValueError("task arithmetic checkpoint is missing its anchor")
        check_finite(state["anchor"], "task arithmetic anchor")
        self.anchor = {n: p.cpu() for n, p in state["anchor"].items()}
