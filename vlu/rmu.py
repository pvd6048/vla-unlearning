"""Representation Misdirection for Unlearning, adapted to one GR00T DiT block.

Forget action-token activations approach a fixed random target; retain activations match the
pre-unlearning block. Only that block is edited, so its frozen copy can evaluate exactly the same
incoming activations without duplicating the rest of GR00T. FM noise/timesteps therefore match.
"""
import copy

import torch
from torch.utils._pytree import tree_map

from vlu.model import check_finite, trainable_named_parameters
from vlu.updates import gradients, set_gradients


def hidden_tensor(output):
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("RMU expects block activations shaped [batch, tokens, hidden]")
    return hidden


def action_mse(hidden, target, action_mask):
    """Mean squared hidden error over valid action tokens; exclude state/padding tokens."""
    valid = action_mask.bool().any(dim=-1)
    if valid.ndim != 2 or hidden.shape[0] != valid.shape[0] or hidden.shape[1] < valid.shape[1]:
        raise ValueError("RMU activation and action-mask shapes do not match")
    if not valid.any():
        raise ValueError("RMU needs at least one valid action token")
    hidden = hidden[:, -valid.shape[1]:].float()
    if target.ndim == 3:
        target = target[:, -valid.shape[1]:]
    error = hidden - target.float()
    return error.square()[valid].mean()


class RMUUpdate:
    disable_weight_decay = True

    def __init__(self, model, layer, alpha, steering, seed):
        blocks = model.action_head.model.transformer_blocks
        if layer >= len(blocks):
            raise ValueError(f"rmu_layer {layer} is outside the {len(blocks)} DiT blocks")
        self.module = blocks[layer]
        prefix = f"action_head.model.transformer_blocks.{layer}."
        self.names = tuple(n for n in trainable_named_parameters(model) if n.startswith(prefix))
        if not self.names:
            raise ValueError("RMU target block has no trainable parameters")
        self.reference = copy.deepcopy(self.module).requires_grad_(False).eval()
        self.settings = (layer, alpha, steering, seed)
        self.alpha, self.steering, self.seed = alpha, steering, seed
        self.control = None  # width is learned from the first actual block output

    def initialize(self, optimizer):
        pass

    def forward(self, model, inputs, with_reference=False):
        captured = {}

        def before(module, args, kwargs):
            # Copy inputs: a reference implementation with in-place operations must not change
            # what the learner sees. All upstream weights are unchanged throughout this method.
            detach = lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x
            with torch.no_grad():
                captured["reference"] = hidden_tensor(self.reference(
                    *tree_map(detach, args), **tree_map(detach, kwargs)))

        def after(module, args, output):
            if "hidden" in captured:
                raise ValueError("RMU target block was called more than once in one forward")
            captured["hidden"] = hidden_tensor(output)

        handle = self.module.register_forward_hook(after)
        ref_handle = self.module.register_forward_pre_hook(before, with_kwargs=True) if with_reference else None
        try:
            output = model(inputs)
        finally:
            handle.remove()
            if ref_handle is not None:
                ref_handle.remove()
        return output, captured

    def step(self, model, batch, optimizer, max_grad_norm):
        named = trainable_named_parameters(model)
        params = [named[n] for n in self.names]
        modes = {module: module.training for module in model.modules()}
        model.eval()  # isolate activation steering from dropout, as for the frozen reference
        model.zero_grad(set_to_none=True)
        try:
            forget, f = self.forward(model, batch["forget"])
            if self.control is None:
                generator = torch.Generator().manual_seed(self.seed)
                vector = torch.rand(f["hidden"].shape[-1], generator=generator)
                self.control = (vector / vector.norm() * self.steering).to(f["hidden"].device)
            if self.control.shape != (f["hidden"].shape[-1],):
                raise ValueError("RMU control-vector width does not match the target block")
            forget_loss = action_mse(f["hidden"], self.control, batch["forget"]["action_mask"])
            retain, r = self.forward(model, batch["retain"], with_reference=True)
            retain_loss = action_mse(r["hidden"], r["reference"], batch["retain"]["action_mask"])
            loss = forget_loss + self.alpha * retain_loss
            norm = set_gradients(params, gradients(loss, params), max_grad_norm)
            optimizer.step()
        finally:
            for module, training in modes.items():
                module.training = training
        return loss.detach(), {"forget_rmu": forget_loss.item(), "retain_rmu": retain_loss.item(),
                               "forget_fm": forget["loss"].item(), "retain_fm": retain["loss"].item(),
                               "grad_norm": norm}

    def state_dict(self):
        return {"kind": "rmu", "settings": self.settings, "control": self.control,
                "reference": self.reference.state_dict()}

    def load_state_dict(self, state):
        if state.get("kind") != "rmu" or tuple(state["settings"]) != self.settings:
            raise ValueError("RMU checkpoint has different method settings")
        if state.get("control") is None:
            raise ValueError("RMU checkpoint is missing its control vector")
        check_finite({"control": state["control"], **state["reference"]}, "RMU checkpoint")
        self.reference.load_state_dict(state["reference"])
        self.control = state["control"].to(next(self.module.parameters()).device)
