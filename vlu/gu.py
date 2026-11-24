"""GU: retain-null projection of the actual AdamW update in its induced metric.

For each parameter tensor, U holds recent normalized retain gradients and P is Adam's diagonal
preconditioner. For candidate displacement d, use d - P U (U^T P U)^+ U^T d. Thus U^T d_new = 0
up to roundoff, including the candidate's momentum and weight decay. This strict FM adaptation
solves a constrained displacement problem; the authors instead whiten/project raw gradients,
with reference-KL anchors and optional tangential updates, before applying Adam.
"""
import torch

from vlu.model import check_finite, trainable_named_parameters
from vlu.updates import gradients, set_gradients


def retain_null(displacement, basis, preconditioner):
    """Metric projection via a whitened SVD, without squaring the condition number.

    Column equilibration prevents large Adam scaling in one direction from causing a smaller,
    independent retain constraint to be discarded by a relative cutoff on U^T P U.
    """
    if not basis:
        return displacement
    u = torch.stack([v.reshape(-1).double() for v in basis], dim=1)
    p = preconditioner.reshape(-1).double()
    d = displacement.reshape(-1).double()
    if not torch.isfinite(p).all() or not (p > 0).all():
        raise ValueError("GU preconditioner must be finite and strictly positive")
    root = p.sqrt()
    whitened = root[:, None] * u
    lengths = torch.linalg.vector_norm(whitened, dim=0)
    whitened = whitened[:, lengths > 0]
    if whitened.shape[1] == 0:
        return displacement
    whitened = whitened / lengths[lengths > 0]
    q, singular, _ = torch.linalg.svd(whitened, full_matrices=False)
    tolerance = torch.finfo(whitened.dtype).eps * max(whitened.shape) * singular[0]
    q = q[:, singular > tolerance]
    scaled = d / root
    projected = root * (scaled - q @ (q.T @ scaled))
    return projected.reshape_as(displacement).to(displacement.dtype)


class GeometricUpdate:
    disable_weight_decay = False  # decay is included in the projected proposal

    def __init__(self, rank):
        self.rank = rank
        self.basis = {}

    def initialize(self, optimizer):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("GU requires AdamW to define its optimizer metric")

    def step(self, model, batch, optimizer, max_grad_norm):
        named = trainable_named_parameters(model)
        params = list(named.values())
        retain_loss = model(batch["retain"])["loss"]
        retain_grads = gradients(retain_loss, params)
        for (name, _), g in zip(named.items(), retain_grads):
            if g is not None:
                direction = g.float()
                scale = direction.abs().max()
                if scale > 0:
                    # Scaling first avoids overflow/underflow when finite gradients are squared.
                    direction = direction / scale
                    direction = direction / direction.norm()
                    history = self.basis.setdefault(name, [])
                    history.append(direction.clone())
                    del history[:-self.rank]
        del retain_grads
        forget_loss = model(batch["forget"])["loss"]
        grad_norm = set_gradients(params, gradients(-forget_loss, params), max_grad_norm)
        before = [p.detach().clone() for p in params]
        optimizer.step()
        check_finite(named, "GU AdamW proposal")
        group_of = {id(p): group for group in optimizer.param_groups for p in group["params"]}
        with torch.no_grad():
            for (name, p), old in zip(named.items(), before):
                state = optimizer.state.get(p, {})
                if "exp_avg_sq" not in state:
                    continue
                group = group_of[id(p)]
                variance = state.get("max_exp_avg_sq", state["exp_avg_sq"])
                correction = 1 - group["betas"][1] ** float(state["step"])
                metric = (variance.sqrt() / correction ** .5 + group["eps"]).reciprocal()
                displacement = retain_null(p - old, self.basis.get(name, []), metric)
                p.copy_(old + displacement)
        return -forget_loss.detach(), {"forget_fm": forget_loss.item(), "retain_fm": retain_loss.item(),
                                       "grad_norm": grad_norm}

    def state_dict(self):
        return {"kind": "gu", "rank": self.rank, "basis": self.basis}

    def load_state_dict(self, state):
        if state.get("kind") != "gu" or state["rank"] != self.rank:
            raise ValueError("GU checkpoint has a different method or basis rank")
        self.basis = state["basis"]

    def to(self, named):
        self.basis = {n: [v.to(named[n].device) for v in history] for n, history in self.basis.items()}
