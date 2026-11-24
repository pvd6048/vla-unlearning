"""Saliency-split Sharp MinMax: retain SAM descent, then forget sharpness maximization.

Tang & Khanna, arXiv:2506.13715, Algorithm 2 and sharp-unlearn/sam/sam.py. Each phase uses
g = g0 + sign * lambda * (g_perturbed - g0), with sign +1 for retain and -1 for forget.
The forget objective is negative FM. Both gradients and perturbations respect the partition.
"""
from contextlib import contextmanager

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from vlu.model import check_finite, trainable_named_parameters
from vlu.updates import gradients, set_gradients


@contextmanager
def perturbed(params, grads, rho, adaptive):
    """Restore exact original weights, also if the second forward/backward fails."""
    originals = [p.detach().clone() for p in params]
    norms = [(p.detach().abs() * g if adaptive else g).norm() for p, g in zip(params, grads) if g is not None]
    norm = torch.stack(norms).norm() if norms else params[0].new_zeros(())
    try:
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None:
                    p.add_((p.square() if adaptive else 1.) * g * (rho / (norm + 1e-12)))
        yield
    finally:
        with torch.no_grad():
            for p, old in zip(params, originals):
                p.copy_(old)


@contextmanager
def replay_randomness(model, cpu_rng, cuda_rng):
    """Use the same FM noise/dropout for both evaluations; count BN statistics only once."""
    norms = [(m, m.momentum) for m in model.modules() if isinstance(m, _BatchNorm)]
    try:
        for m, _ in norms:
            m.momentum = 0.
        with torch.random.fork_rng(devices=list(cuda_rng)):
            torch.set_rng_state(cpu_rng)
            for device, state in cuda_rng.items():
                torch.cuda.set_rng_state(state, device)
            yield
    finally:
        for m, momentum in norms:
            m.momentum = momentum


class SharpMinMaxUpdate:
    disable_weight_decay = True  # AdamW decay would move the inactive partition

    def __init__(self, masks, alpha, rho, strength, adaptive=False):
        self.masks = masks  # True = forget partition, False = retain partition
        self.alpha, self.rho, self.strength, self.adaptive = alpha, rho, strength, adaptive
        self.forget_optimizer = None

    def initialize(self, optimizer):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("Sharp MinMax uses two independent AdamW optimizers")
        groups = [{**group, "params": list(group["params"])} for group in optimizer.param_groups]
        # Groups already carry lr/betas/eps/etc.; defaults may include internal keys that are not
        # constructor arguments in another torch release (e.g. decoupled_weight_decay).
        self.forget_optimizer = torch.optim.AdamW(groups)

    def _phase(self, model, inputs, named, optimizer, forget, max_grad_norm):
        params = list(named.values())
        masks = [self.masks[n] if forget else ~self.masks[n] for n in named]
        coefficient = -(1 - self.alpha) if forget else self.alpha
        cpu_rng = torch.get_rng_state()
        devices = {p.device.index for p in params if p.is_cuda}
        cuda_rng = {d: torch.cuda.get_rng_state(d) for d in devices}
        loss = model(inputs)["loss"]
        original = gradients(coefficient * loss, params)
        original = [g * mask if g is not None else None for g, mask in zip(original, masks)]
        with perturbed(params, original, self.rho, self.adaptive):
            with replay_randomness(model, cpu_rng, cuda_rng):
                second_loss = model(inputs)["loss"]
                second = gradients(coefficient * second_loss, params)
        sign = -1 if forget else 1
        combined = []
        for g0, g1, mask in zip(original, second, masks):
            if g0 is None and g1 is None:
                combined.append(None)
                continue
            g0 = torch.zeros_like(g1) if g0 is None else g0
            g1 = torch.zeros_like(g0) if g1 is None else g1 * mask
            combined.append((g0 + sign * self.strength * (g1 - g0)) * mask)
        check_finite({str(i): g for i, g in enumerate(combined) if g is not None}, "sharpness gradients")
        norm = set_gradients(params, combined, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.detach(), norm

    def step(self, model, batch, optimizer, max_grad_norm):
        named = trainable_named_parameters(model)
        if set(named) != set(self.masks):
            raise ValueError("Sharp MinMax mask does not match the trainable parameters")
        for name, p in named.items():
            mask = self.masks[name]
            if mask.dtype != torch.bool or mask.shape != p.shape or mask.device != p.device:
                raise ValueError(f"invalid Sharp MinMax mask for {name}")
        for retain_group, forget_group in zip(optimizer.param_groups, self.forget_optimizer.param_groups):
            forget_group["lr"] = retain_group["lr"]
        retain, r_norm = self._phase(model, batch["retain"], named, optimizer, False, max_grad_norm)
        # Check before the forget phase as well: invalid weights must not enter the next forward.
        check_finite(named, "Sharp MinMax retain update")
        forget, f_norm = self._phase(model, batch["forget"], named, self.forget_optimizer, True, max_grad_norm)
        return self.alpha * retain - (1 - self.alpha) * forget, {
            "retain_fm": retain.item(), "forget_fm": forget.item(), "grad_norm": max(r_norm, f_norm)}

    def state_dict(self):
        return {"kind": "sharp_minmax", "forget_optimizer": self.forget_optimizer.state_dict(),
                "masks": self.masks, "settings": (self.alpha, self.rho, self.strength, self.adaptive)}

    def load_state_dict(self, state):
        if state.get("kind") != "sharp_minmax" or tuple(state["settings"]) != (
                self.alpha, self.rho, self.strength, self.adaptive):
            raise ValueError("Sharp MinMax checkpoint has different method settings")
        self.forget_optimizer.load_state_dict(state["forget_optimizer"])
        check_finite({f"{i}.{k}": v for i, s in enumerate(self.forget_optimizer.state.values())
                      for k, v in s.items() if isinstance(v, torch.Tensor)}, "forget optimizer state")
        self.masks = state["masks"]

    def to(self, named):
        self.masks = {n: m.to(named[n].device) for n, m in self.masks.items()}
