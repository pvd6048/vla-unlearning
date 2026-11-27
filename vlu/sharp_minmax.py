"""Sharp MinMax, Tang & Khanna (arXiv:2506.13715), Algorithms 1/2.

The second gradient is masked BEFORE blending: g0 + sign*lambda*(mask*g1 - g0).
In particular, forget maximization at lambda=1 uses 2*g0 - mask*g1, including
updates outside the forget partition. This is the literal published/released update.

Class defaults: the paper's constant-rate SGD, momentum .9 and decay 5e-4 (the GR00T recipe, vlu/config.py,
uses decay 0 and explains why). The released
training script's AdamW override is available explicitly. Each phase owns an
independent native optimizer over all trainable weights. See PAPER_ALIGNMENT.md.
"""
from contextlib import contextmanager

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from vlu.model import check_finite, trainable_named_parameters
from vlu.updates import gradients, total_norm


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
    disable_weight_decay = False
    constant_lr = True

    def __init__(self, masks, alpha, rho, strength, adaptive=False,
                 optimizer_name="sgd", momentum=.9, weight_decay=5e-4):
        self.masks = masks  # True = forget selection
        self.alpha, self.rho, self.strength, self.adaptive = alpha, rho, strength, adaptive
        self.optimizer_name, self.momentum, self.weight_decay = optimizer_name, momentum, weight_decay
        self.optimizer = self.forget_optimizer = None

    def make_optimizer(self, model, lr):
        params = list(trainable_named_parameters(model).values())
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        raise ValueError("sharp_optimizer must be 'sgd' or 'adamw'")

    def initialize(self, optimizer):
        expected = torch.optim.SGD if self.optimizer_name == "sgd" else torch.optim.AdamW
        if not isinstance(optimizer, expected):
            raise TypeError(f"Sharp MinMax needs {expected.__name__} for this recipe")
        self.optimizer = optimizer
        # Copy group settings, not defaults: newer torch versions add constructor-incompatible defaults.
        groups = [{**group, "params": list(group["params"])} for group in optimizer.param_groups]
        self.forget_optimizer = expected(groups, lr=optimizer.param_groups[0]["lr"])

    def _active(self, name, forget):
        return self.masks[name] if forget else ~self.masks[name]

    def _phase(self, model, inputs, named, forget):
        """Full-gradient SAM perturbation, then the authors' second-gradient-only mask. Returns the loss, the
        blended gradients and their norm. No gradient clipping: Algorithm 2 and the released training loop
        do not clip."""
        names, params = list(named), list(named.values())
        coefficient = -(1 - self.alpha) if forget else self.alpha
        cpu_rng = torch.get_rng_state()
        devices = {p.device.index for p in params if p.is_cuda}
        cuda_rng = {d: torch.cuda.get_rng_state(d) for d in devices}
        loss = model(inputs)["loss"]
        g0 = gradients(coefficient * loss, params)
        with perturbed(params, g0, self.rho, self.adaptive):
            with replay_randomness(model, cpu_rng, cuda_rng):
                second_loss = model(inputs)["loss"]
                g1 = gradients(coefficient * second_loss, params)
        sign = -1 if forget else 1
        blended = []
        for i, name in enumerate(names):
            base, second = g0[i], g1[i]
            g0[i] = g1[i] = None  # memory: drop each tensor's inputs once its blend exists
            if base is None and second is None:
                blended.append(None)
                continue
            base = torch.zeros_like(second) if base is None else base
            masked = torch.zeros_like(base) if second is None else second * self._active(name, forget)
            blended.append(base + sign * self.strength * (masked - base))
            del base, second, masked
        check_finite({str(i): g for i, g in enumerate(blended) if g is not None}, "sharpness gradients")
        return loss.detach(), blended, total_norm(blended)

    def step(self, model, batch, optimizer, max_grad_norm):
        named = trainable_named_parameters(model)
        self.check_masks(named)
        for source, target in zip(optimizer.param_groups, self.forget_optimizer.param_groups):
            target["lr"] = source["lr"]
        norms, losses = [], []
        for forget, opt in ((False, optimizer), (True, self.forget_optimizer)):
            phase = "forget" if forget else "retain"
            loss, grads, norm = self._phase(model, batch[phase], named, forget)
            for p, g in zip(named.values(), grads):
                p.grad = g
            p = g = None  # the loop variables would keep the last gradient alive
            opt.step()
            opt.zero_grad(set_to_none=True)
            # Memory: the retain phase's gradients (one weight copy) must not stay alive through the forget
            # phase, whose second backward is the step's peak (about NPO-RT's peak on GR00T, ~40 GiB).
            del grads
            # AdamW/SGD changed weights inside train.py's enclosing autocast region.
            torch.clear_autocast_cache()
            check_finite(named, f"Sharp MinMax {phase} update")
            check_finite({f"{i}.{key}": value for i, state in enumerate(opt.state.values())
                          for key, value in state.items() if isinstance(value, torch.Tensor)},
                         f"Sharp MinMax {phase} optimizer")
            losses.append(loss)
            norms.append(norm)
        retain, forget = losses
        return self.alpha * retain - (1 - self.alpha) * forget, {
            "retain_fm": retain.item(), "forget_fm": forget.item(), "grad_norm": max(norms)}

    def check_masks(self, named):
        if set(named) != set(self.masks):
            raise ValueError("Sharp MinMax mask does not match the trainable parameters")
        for name, p in named.items():
            mask = self.masks[name]
            if mask.dtype != torch.bool or mask.shape != p.shape or mask.device != p.device:
                raise ValueError(f"invalid Sharp MinMax mask for {name}")

    def _settings(self):
        return (self.alpha, self.rho, self.strength, self.adaptive,
                self.optimizer_name, self.momentum, self.weight_decay)

    def state_dict(self):
        return {"kind": "sharp_minmax", "version": 3, "masks": self.masks,
                "settings": self._settings(), "forget_optimizer": self.forget_optimizer.state_dict()}

    def load_state_dict(self, state):
        if state.get("version") != 3:
            raise ValueError("Sharp checkpoint predates the literal Algorithm 2 update; start a fresh operation")
        if state.get("kind") != "sharp_minmax" or tuple(state["settings"]) != self._settings():
            raise ValueError("Sharp MinMax checkpoint has different method settings")
        self.forget_optimizer.load_state_dict(state["forget_optimizer"])
        self.masks = state["masks"]

    def to(self, named):
        self.masks = {n: m.to(named[n].device) for n, m in self.masks.items()}
        self.check_masks(named)
