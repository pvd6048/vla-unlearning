"""Representation Misdirection for Unlearning, adapted to one GR00T DiT block.

Forget action-token activations approach a fixed random target; retain activations match the
pre-unlearning block. Only that block is edited, so its frozen copy can evaluate exactly the same
incoming activations without duplicating the rest of GR00T. FM noise/timesteps therefore match.
Each forward stops right after the target block: nothing downstream enters the objective, so running
the rest of GR00T would only add compute to the method's measured cost.
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


class _Captured(Exception):
    """Raised by the target block's hook to end a forward once its output is captured."""


def _valid_actions(hidden, action_mask):
    valid = action_mask.to(hidden.device).bool().any(dim=-1)
    if valid.ndim != 2 or hidden.shape[0] != valid.shape[0] or hidden.shape[1] < valid.shape[1]:
        raise ValueError("RMU activation and action-mask shapes do not match")
    if not valid.any():
        raise ValueError("RMU needs at least one valid action token")
    return valid


def action_tokens(hidden, action_mask):
    """The valid action tokens' hidden states, [n, width] in float32 (state/padding tokens excluded)."""
    valid = _valid_actions(hidden, action_mask)
    return hidden[:, -valid.shape[1]:].float()[valid]


def action_mse(hidden, target, action_mask):
    """Mean squared hidden error over valid action tokens; exclude state/padding tokens."""
    valid = _valid_actions(hidden, action_mask)
    if target.ndim == 3:
        target = target[:, -valid.shape[1]:][valid]
    error = action_tokens(hidden, action_mask) - target.float()
    return error.square().mean()


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
        """The target block's output (and, with_reference, the frozen block's on the same inputs); the
        model's forward ends right after that block."""
        captured = {}

        def before(module, args, kwargs):
            # Copy inputs: a reference implementation with in-place operations must not change
            # what the learner sees. All upstream weights are unchanged throughout this method.
            detach = lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x
            with torch.no_grad():
                captured["reference"] = hidden_tensor(self.reference(
                    *tree_map(detach, args), **tree_map(detach, kwargs)))

        def after(module, args, output):
            captured["hidden"] = hidden_tensor(output)
            raise _Captured  # the block's first call ends the forward, so it is never called twice

        handle = self.module.register_forward_hook(after)
        ref_handle = self.module.register_forward_pre_hook(before, with_kwargs=True) if with_reference else None
        try:
            model(inputs)
        except _Captured:
            pass
        finally:
            handle.remove()
            if ref_handle is not None:
                ref_handle.remove()
        if "hidden" not in captured:
            raise ValueError("RMU target block was not called by the model's forward")
        return captured

    def step(self, model, batch, optimizer, max_grad_norm):
        named = trainable_named_parameters(model)
        params = [named[n] for n in self.names]
        modes = {module: module.training for module in model.modules()}
        model.eval()  # isolate activation steering from dropout, as for the frozen reference
        model.zero_grad(set_to_none=True)
        try:
            f = self.forward(model, batch["forget"])
            check_finite({"hidden": f["hidden"]}, "RMU forget activations")
            # The steering target's norm is per token: compare it with the activations' own scale.
            hidden_norm = action_tokens(f["hidden"].detach(), batch["forget"]["action_mask"]).norm(dim=-1).mean().item()
            if self.control is None:
                generator = torch.Generator().manual_seed(self.seed)
                vector = torch.rand(f["hidden"].shape[-1], generator=generator)
                self.control = (vector / vector.norm() * self.steering).to(f["hidden"].device)
            if self.control.shape != (f["hidden"].shape[-1],):
                raise ValueError("RMU control-vector width does not match the target block")
            forget_loss = action_mse(f["hidden"], self.control, batch["forget"]["action_mask"])
            r = self.forward(model, batch["retain"], with_reference=True)
            check_finite({"hidden": r["hidden"], "reference": r["reference"]}, "RMU retain activations")
            retain_loss = action_mse(r["hidden"], r["reference"], batch["retain"]["action_mask"])
            loss = forget_loss + self.alpha * retain_loss
            norm = set_gradients(params, gradients(loss, params), max_grad_norm)
            optimizer.step()
        finally:
            for module, training in modes.items():
                module.training = training
        return loss.detach(), {"forget_rmu": forget_loss.item(), "retain_rmu": retain_loss.item(),
                               "hidden_norm": hidden_norm,
                               "steering_ratio": self.steering / hidden_norm if hidden_norm > 0 else float("inf"),
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
