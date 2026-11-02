"""Minimal LoRA (Hu et al. 2021) for nn.Linear layers, with merging back into the base weights.

Used by VLA-Forget (vlu/vlaforget.py). Same parametrization as peft, with the initialization of the
authors' OpenVLA script (peft's init_lora_weights="gaussian"): the output is
    base(x) + (alpha / r) * B(A(dropout(x))),   A ~ N(0, (1/r)^2), B = 0,
so a freshly attached adapter leaves the model unchanged. A and B are fp32 (like our trainable weights).
"""

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
        nn.init.normal_(self.lora_A, std=1 / r)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = nn.functional.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A)  # under autocast: bf16 matmuls
        return out + (nn.functional.linear(h, self.lora_B) * self.scale).to(out.dtype)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scale


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    path, _, child = name.rpartition(".")
    return (model.get_submodule(path) if path else model), child


def attach(model: nn.Module, names: list[str], r: int, alpha: float, dropout: float) -> dict[str, LoRALinear]:
    """Wrap the nn.Linear modules `names` (as in model.named_modules()) with LoRA adapters."""
    adapters = {}
    for name in names:
        parent, child = _parent(model, name)
        module = getattr(parent, child)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} is a {type(module).__name__}, not nn.Linear")
        adapters[name] = LoRALinear(module, r, alpha, dropout)
        setattr(parent, child, adapters[name])
    return adapters


@torch.no_grad()
def merge(model: nn.Module, adapters: dict[str, LoRALinear]) -> dict[str, float]:
    """Fold every adapter into its base weight (computed in fp32, stored in the weight's dtype) and put
    the plain nn.Linear back. Returns {changed weight name: retained fraction}, where the retained
    fraction ||W_merged - W|| / ||delta|| shows how much of the delta survived rounding to the weight's
    dtype (1 for fp32 weights; a bf16 weight loses the part of the delta below its precision)."""
    retained = {}
    for name, adapter in adapters.items():
        weight = adapter.base.weight
        old, delta = weight.float().clone(), adapter.delta_weight()
        weight.copy_((old + delta).to(weight.dtype))
        retained[f"{name}.weight"] = float((weight.float() - old).norm() / delta.norm().clamp_min(1e-30))
        parent, child = _parent(model, name)
        setattr(parent, child, adapter.base)
    return retained
