"""Saliency-split Sharp MinMax: retain SAM descent, then forget sharpness maximization.

Tang & Khanna, arXiv:2506.13715, Algorithm 2 and sharp-unlearn/sam/sam.py. Each phase uses
g = g0 + sign * lambda * (g_perturbed - g0), with sign +1 for retain and -1 for forget.
The forget objective is negative FM. As in the paper and the authors' code, each phase perturbs all trainable
weights along the full gradient g0 (SAM's first_step, norm over all weights) and then keeps the blended
gradient on its own partition only. The authors mask only g_perturbed before blending, so their forget step
2 g0 - g1 * mask also moves the retain partition by 2 g0; here the blend is masked as a whole. The retain
phase at the default lambda = 1 is exactly their update (their g0 + (g1 * mask - g0) is g1 * mask); otherwise,
and in the forget phase, theirs also moves the other partition and ours does not.

Memory (GR00T: one fp32 copy of the trainable weights is 6.5 GB). The authors use two AdamW optimizers
over all weights (unlearn/impl.py, "Adam Override"). With the partition masks and no weight decay, each
of them only ever has nonzero moments on its own partition, so one pair of moment buffers holds both
exactly: each phase updates the moments and the weights inside its partition only, with AdamW's formulas
and its own step count. That is the same update as two torch AdamW optimizers (tests/test_sharp_minmax.py)
with half their state (13 GB instead of 26). The blend is formed in place; the perturbation saves the
weights it moves (all of them, the authors' old_p), so one step holds the weights, the moments, the masks,
two gradient copies and the saved weights (toy measurement: about NPO-RT's peak plus 0.3 copies).

alpha: the losses are scaled by alpha and -(1 - alpha) as in the authors' code, but under AdamW (theirs and
ours) a constant scale of a phase's gradient cancels in m / sqrt(v), so alpha changes almost nothing (only
through eps and clipping). It is kept for fidelity, not calibrated.
"""
from contextlib import contextmanager
import math

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from vlu.model import check_finite, trainable_named_parameters
from vlu.updates import clip_, gradients


@contextmanager
def perturbed(params, grads, rho, adaptive):
    """Move to w + e(w) (SAM; ASAM's parameter-scaled e with adaptive) and restore the exact original
    weights afterwards, also if the second forward/backward fails.
    Both in-place changes clear autocast's weight cache: within one autocast region (train.py runs the whole
    step in one) a forward reuses its cached bf16 copy of a trainable fp32 weight even after the weight
    changed in place, so the perturbed forward would see the old weights and g1 would equal g0. The clear
    after restoring also lets the forget phase's first forward see the retain phase's update (no forward
    runs in between, so nothing is cached again)."""
    norms = [(p.detach().abs() * g if adaptive else g).norm() for p, g in zip(params, grads) if g is not None]
    norm = torch.stack(norms).norm() if norms else params[0].new_zeros(())
    saved = []
    try:
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is None:
                    saved.append(None)
                    continue
                saved.append(p.detach().clone())
                p.add_((p.square() if adaptive else 1.) * g * (rho / (norm + 1e-12)))
        torch.clear_autocast_cache()
        yield
    finally:
        with torch.no_grad():
            for p, old in zip(params, saved):
                if old is not None:
                    p.copy_(old)
        torch.clear_autocast_cache()


@contextmanager
def replay_randomness(model, cpu_rng, cuda_rng):
    """Use the same FM noise/dropout for both evaluations; count BN statistics only once."""
    norms = [(m, m.momentum, {n: b.clone() for n, b in m.named_buffers()})
             for m in model.modules() if isinstance(m, _BatchNorm)]
    try:
        for m, _, _ in norms:
            m.momentum = 0.
        with torch.random.fork_rng(devices=list(cuda_rng)):
            torch.set_rng_state(cpu_rng)
            for device, state in cuda_rng.items():
                torch.cuda.set_rng_state(state, device)
            yield
    finally:
        for m, momentum, buffers in norms:
            m.momentum = momentum
            with torch.no_grad():
                for name, buffer in m.named_buffers():
                    buffer.copy_(buffers[name])


class SharpMinMaxUpdate:
    disable_weight_decay = True  # AdamW decay would move the inactive partition

    def __init__(self, masks, alpha, rho, strength, adaptive=False):
        self.masks = masks  # True = forget partition, False = retain partition
        self.alpha, self.rho, self.strength, self.adaptive = alpha, rho, strength, adaptive
        self.moments = {}  # name -> [exp_avg, exp_avg_sq], each partition's AdamW moments on its own entries
        self.steps = {"retain": 0, "forget": 0}  # each partition's AdamW step count
        self.optimizer = None

    def initialize(self, optimizer):
        """The loop's AdamW supplies the (scheduled) learning rate, betas and eps; its own step is not used."""
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("Sharp MinMax applies AdamW per partition; it needs an AdamW optimizer")
        for group in optimizer.param_groups:
            if group["weight_decay"] or group.get("amsgrad") or group.get("maximize"):
                raise ValueError("Sharp MinMax's AdamW has no weight decay, amsgrad or maximize")
        self.optimizer = optimizer

    def _active(self, name, forget):
        return self.masks[name] if forget else ~self.masks[name]

    def _phase(self, model, inputs, named, forget, max_grad_norm):
        """One partition's sharpness-aware gradient (the perturbation over all weights, the blend masked to
        the partition, clipped) and loss."""
        names, params = list(named), list(named.values())
        coefficient = -(1 - self.alpha) if forget else self.alpha
        cpu_rng = torch.get_rng_state()
        devices = {p.device.index for p in params if p.is_cuda}
        cuda_rng = {d: torch.cuda.get_rng_state(d) for d in devices}
        loss = model(inputs)["loss"]
        g0 = gradients(coefficient * loss, params)
        with perturbed(params, g0, self.rho, self.adaptive):  # all weights, as the authors' first_step
            with replay_randomness(model, cpu_rng, cuda_rng):
                second_loss = model(inputs)["loss"]
                g1 = gradients(coefficient * second_loss, params)
        sign = -1 if forget else 1
        for i, name in enumerate(names):  # g0 <- (g0 + sign * strength * (g1 - g0)) * mask, in place
            if g0[i] is None and g1[i] is None:
                continue
            mask = self._active(name, forget)
            if g0[i] is None:
                g0[i] = torch.zeros_like(g1[i])
            if g1[i] is not None:
                g0[i].add_(g1[i].sub_(g0[i]), alpha=sign * self.strength)
            else:
                g0[i].mul_(1 - sign * self.strength)
            g0[i].mul_(mask)
            g1[i] = None
        check_finite({str(i): g for i, g in enumerate(g0) if g is not None}, "sharpness gradients")
        norm = clip_(g0, max_grad_norm)
        return loss.detach(), g0, norm

    @torch.no_grad()
    def _adamw(self, named, grads, forget):
        """torch AdamW (weight_decay 0) restricted to the partition: moments and weights change only inside
        it, so the other partition's moments, kept in the same buffers, are untouched."""
        phase = "forget" if forget else "retain"
        self.steps[phase] += 1
        step = self.steps[phase]
        # Looked up every step: the scheduler sets the lr there, and load_state_dict (resume) replaces the dicts.
        groups = {id(p): group for group in self.optimizer.param_groups for p in group["params"]}
        for (name, p), g in zip(named.items(), grads):
            if g is None:
                continue
            group = groups[id(p)]
            beta1, beta2 = group["betas"]
            mask = self._active(name, forget)
            if name not in self.moments:
                self.moments[name] = [torch.zeros_like(p, memory_format=torch.preserve_format),
                                      torch.zeros_like(p, memory_format=torch.preserve_format)]
            exp_avg, exp_avg_sq = self.moments[name]
            exp_avg.copy_(torch.where(mask, exp_avg.lerp(g, 1 - beta1), exp_avg))
            exp_avg_sq.copy_(torch.where(mask, torch.addcmul(exp_avg_sq * beta2, g, g, value=1 - beta2), exp_avg_sq))
            # As torch's AdamW (single-tensor path): Python floats for the bias corrections.
            bias_correction1 = 1 - beta1 ** float(step)
            bias_correction2_sqrt = math.sqrt(1 - beta2 ** float(step))
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
            p.addcdiv_(exp_avg * mask, denom, value=-group["lr"] / bias_correction1)

    def step(self, model, batch, optimizer, max_grad_norm):
        named = trainable_named_parameters(model)
        if set(named) != set(self.masks):
            raise ValueError("Sharp MinMax mask does not match the trainable parameters")
        retain, retain_grads, r_norm = self._phase(model, batch["retain"], named, False, max_grad_norm)
        self._adamw(named, retain_grads, False)
        del retain_grads
        # Check before the forget phase as well: invalid weights must not enter the next forward.
        check_finite(named, "Sharp MinMax retain update")
        forget, forget_grads, f_norm = self._phase(model, batch["forget"], named, True, max_grad_norm)
        self._adamw(named, forget_grads, True)
        del forget_grads
        optimizer.step()  # no parameter has a gradient: a no-op, so the LR scheduler sees a step
        return self.alpha * retain - (1 - self.alpha) * forget, {
            "retain_fm": retain.item(), "forget_fm": forget.item(), "grad_norm": max(r_norm, f_norm)}

    def check_masks(self, named):
        for name, p in named.items():
            mask = self.masks[name]
            if mask.dtype != torch.bool or mask.shape != p.shape or mask.device != p.device:
                raise ValueError(f"invalid Sharp MinMax mask for {name}")

    def state_dict(self):
        return {"kind": "sharp_minmax", "moments": self.moments, "steps": dict(self.steps),
                "masks": self.masks, "settings": (self.alpha, self.rho, self.strength, self.adaptive)}

    def load_state_dict(self, state):
        if state.get("kind") != "sharp_minmax" or tuple(state["settings"]) != (
                self.alpha, self.rho, self.strength, self.adaptive):
            raise ValueError("Sharp MinMax checkpoint has different method settings")
        if "moments" not in state:
            raise ValueError("Sharp MinMax checkpoint from the two-optimizer version: start the op again")
        check_finite({f"{n}.{i}": v for n, pair in state["moments"].items() for i, v in enumerate(pair)},
                     "Sharp MinMax optimizer state")
        self.moments, self.steps, self.masks = state["moments"], dict(state["steps"]), state["masks"]

    def to(self, named):
        self.masks = {n: m.to(named[n].device) for n, m in self.masks.items()}
        self.moments = {n: [v.to(named[n].device) for v in pair] for n, pair in self.moments.items()}
        self.check_masks({n: named[n] for n in self.masks})
