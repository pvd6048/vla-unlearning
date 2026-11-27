"""GU (Geometric-disentanglement Unlearning, arXiv:2511.17100) on NPO-RT, adapted to flow matching.

GU is a plug-in for a base unlearning objective (the paper: "any forget loss"; the authors' config uses
SimNPO, their code also NPO). Here the base is NPO-RT (vlu/ops.py: NPO on the forget task plus the FM loss
on the retained tasks, weights gamma = alpha = 1, NPO-RT's own; the authors' trainer defaults to gamma = 1.5,
but keeping the base method's weights makes npo_rt vs npo_rt_gu differ only by the projection). We follow
the paper's Algorithm 1 / Appendix B and the authors' code (Lemutisme/geo-unlearning@bbe4d50,
src/trainer/unlearn/geometric.py, RetainNullProjector and GeometricUnlearn.optimizer_step):

- Projected tensors: the last K blocks (K = gu_last_blocks, the authors' auto_last_k_layers = 3) plus the
  output layers. In GR00T the output is produced by the action head, so the blocks are the DiT's
  transformer_blocks and the output layers its proj_out_* and the action decoder (the authors' regex finds
  ".layers.i." / ".blocks.i." and the LM head). All other trainable tensors take NPO-RT's gradient.
- Metric: Adam's diagonal; whitened coordinates g~ = g / (sqrt(v) + 1e-8), v = the optimizer's exp_avg_sq.
- Retain basis: per projected tensor, up to k = gu_basis_k (null_k = 8) unit vectors, stored in float16, from
  the gradient of a retain anchor, every gu_refresh_every-th step (basis_update_every = 2): Gram-Schmidt in
  float32 against the current basis, kept if the residual's norm is >= 1e-3 x the raw gradient's
  (residual_keep_thresh). Once k vectors are kept the basis no longer changes, as in the authors' code.
  The anchor is the paper's KL(pi_theta || pi_ref) on retain data; for a flow policy we use VLA-Forget's
  stand-in, the masked mean squared difference of the predicted velocities of theta and the frozen
  reference (vlu/vlaforget.py velocity_kl), with the same FM noise and timesteps (replayed RNG). At theta =
  theta_ref it and its gradient are zero, so, as for the authors, the basis starts filling once the model
  has moved (from the 3rd step).
- Update (per projected tensor, whitened): forget part = P_perp g~_f plus the components a_j u_j of g~_f
  whose sign is opposite to the retain gradient's along u_j (a_j b_j < -tau, tau = 0), their norm capped at
  gu_cap (sign_cap_ratio = 0.5) x ||P_perp g~_f||; retain part = P_U g~_r; both mapped back by
  x (sqrt(v) + 1e-8) and added. Then clipping and the usual AdamW step (NPO-RT's, weight decay included).

Choices where the sources differ, or where GR00T needs one:
- g_r is the gradient of the base method's retain loss (the paper's Appendix B.1: g~_r = W grad L_r). The
  authors' code instead uses the anchor's gradient for the tangent and sign terms and recovers g_f as
  (g_tot - alpha g_anchor) / gamma, which mixes the retain loss's gradient into "g_f"; we compute g_f and
  g_r separately (two backward passes on the forward graphs NPO-RT builds anyway), so g_tot = g_f + g_r.
- The authors fall back to an EMA of squared gradients before the optimizer has a second moment; that is
  only the first step here, where the anchor's gradient is zero and the basis empty, so there is nothing
  to whiten: the first step is NPO-RT's.
- The projection uses the authors' sequential subtraction along the basis (exact when the stored vectors
  are orthogonal; float16 rounding makes them nearly so).
With an empty basis (always the first two steps) the update is exactly NPO-RT's
(tests/test_gu_npo_rt.py).

Memory: the reference copy (as NPO), the basis (k float16 copies of the projected tensors) and their
per-step gradients; the projected tensors are the last 3 of GR00T's DiT blocks plus the output layers.
"""
import re

import torch

from vlu.model import check_finite, trainable_named_parameters
from vlu.ops import fm_sse_per_sample
from vlu.vlaforget import velocity_kl

EPS = 1e-8  # the authors' whitening: g / (sqrt(v) + 1e-8)
KEEP_THRESHOLD = 1e-3  # residual_keep_thresh
SIGN_TAU = 0.0  # sign_tau
_BLOCK = re.compile(r"^action_head\.model\.transformer_blocks\.(\d+)\.")
_OUTPUT = ("action_head.model.proj_out", "action_head.model.norm_out", "action_head.action_decoder.")


def projected_names(names: list[str], last_blocks: int) -> list[str]:
    """The trainable tensors GU projects: the last `last_blocks` DiT blocks and the output layers."""
    blocks = sorted({int(m.group(1)) for n in names if (m := _BLOCK.match(n))})
    if not blocks:
        raise ValueError("GU found no trainable DiT transformer blocks (action_head.model.transformer_blocks)")
    chosen = set(blocks[-last_blocks:])
    return [n for n in names if ((m := _BLOCK.match(n)) and int(m.group(1)) in chosen) or n.startswith(_OUTPUT)]


class GuNpoRt:
    disable_weight_decay = False  # NPO-RT's AdamW, decay included

    def __init__(self, beta: float, names: list[str], k: int, refresh_every: int, cap: float):
        self.beta, self.names, self.k, self.refresh_every, self.cap = beta, list(names), k, refresh_every, cap
        self.basis = {n: [] for n in self.names}
        self.calls = 0
        self.last_anchor = 0.0
        self.ref = None
        self.optimizer = None
        self._capture, self._taps = False, {}

    def attach(self, model, ref):
        """The frozen reference (theta0) and velocity taps on both action decoders (they keep an output
        only while a retain anchor is being computed)."""
        self.ref = ref
        for key, m in (("mine", model), ("ref", ref)):
            m.action_head.action_decoder.register_forward_hook(
                lambda _m, _i, output, key=key: self._taps.__setitem__(key, output) if self._capture else None)

    def initialize(self, optimizer):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("GU needs AdamW: its second moment is the metric")
        if any(group.get("amsgrad") for group in optimizer.param_groups):
            raise ValueError("GU uses AdamW's exp_avg_sq, not amsgrad's maximum")
        self.optimizer = optimizer

    def _scale(self, p):
        """sqrt(v) + eps for one parameter, or None before AdamW's first step."""
        v = self.optimizer.state.get(p, {}).get("exp_avg_sq")
        return None if v is None else v.float().sqrt().add_(EPS)

    def _paired(self, model, inputs):
        """theta's forward, and a function running the reference on the same inputs with the same random
        draws, without changing theta's RNG stream (as vlu/ops.py's NPO)."""
        devices = sorted({p.device.index for p in (*model.parameters(), *model.buffers()) if p.is_cuda})
        cpu_rng = torch.get_rng_state()
        cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
        out = model(inputs)

        def reference():
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.set_rng_state(cpu_rng)
                for device, rng in cuda_rng.items():
                    torch.cuda.set_rng_state(rng, device)
                self.ref.train(model.training)
                return self.ref(inputs)
        return out, reference

    @torch.no_grad()
    def _grow_basis(self, named, grads):
        for name, g in zip(self.names, grads):
            basis = self.basis[name]
            if g is None or len(basis) >= self.k:
                continue
            scale = self._scale(named[name])
            if scale is None:
                continue
            g = g.float()
            residual = g / scale
            for u in basis:
                uu = u.float()
                residual -= (residual * uu).sum() / (uu * uu).sum().clamp_min(1e-12) * uu
            norm = residual.norm()
            if norm > 1e-12 and norm / (g.norm() + 1e-12) >= KEEP_THRESHOLD:
                basis.append((residual / norm).half())

    @torch.no_grad()
    def _project(self, name, p, g_f, g_r):
        """The final gradient of one projected tensor from its forget and retain gradients."""
        basis = self.basis[name]
        scale = self._scale(p) if basis else None
        if scale is None:  # no basis yet: NPO-RT's gradient
            if g_f is None or g_r is None:
                return g_f if g_r is None else g_r
            return g_f + g_r
        us = [u.float() for u in basis]
        out = None
        f = r = None
        if g_r is not None:
            r = g_r.float() / scale
        if g_f is not None:
            f = g_f.float() / scale
            perp = f.clone()
            for uu in us:
                perp -= (perp * uu).sum() / (uu * uu).sum().clamp_min(1e-12) * uu
            selected = perp
            if r is not None:
                tangent = torch.zeros_like(f)
                for uu in us:
                    a, b = (f * uu).sum(), (r * uu).sum()
                    if a * b < -SIGN_TAU:  # locally lowers the retain loss: keep it
                        tangent += a * uu
                perp_norm, tan_norm = perp.norm().clamp_min(1e-12), tangent.norm()
                if tan_norm > self.cap * perp_norm:
                    tangent *= self.cap * perp_norm / tan_norm
                selected = perp + tangent
            out = selected * scale
        if r is not None:
            tangent_r = torch.zeros_like(r)
            for uu in us:
                tangent_r += (r * uu).sum() / (uu * uu).sum().clamp_min(1e-12) * uu
            out = tangent_r * scale if out is None else out + tangent_r * scale
        return out.to(p.dtype)

    def step(self, model, batch, optimizer, max_grad_norm):
        named = trainable_named_parameters(model)
        params = list(named.values())
        # NPO on the forget task, exactly as vlu/ops.py's NPO-RT (same forwards, same random draws).
        out, reference = self._paired(model, batch["forget"])
        ref_out = reference()
        log_ratio = fm_sse_per_sample(ref_out) - fm_sse_per_sample(out)
        npo = (2 / self.beta) * torch.nn.functional.softplus(self.beta * log_ratio).mean()
        check_finite({"loss": npo}, "NPO loss")
        metrics = {"forget_fm": out["loss"].item(), "log_ratio": log_ratio.mean().item(),
                   "npo_weight": (2 * torch.sigmoid(self.beta * log_ratio.detach())).mean().item()}
        del out, ref_out, log_ratio
        npo.backward()
        g_f = {}
        for name in self.names:  # keep the forget gradient of the projected tensors apart
            g_f[name], named[name].grad = named[name].grad, None
        # The retain FM loss (NPO-RT's retain term) and, on refresh steps, the anchor for the basis.
        refresh = self.calls % self.refresh_every == 0
        self.calls += 1
        self._capture = refresh
        try:
            out_r, reference = self._paired(model, batch["retain"])
            if refresh:
                mine = self._taps.pop("mine")
                reference()
                theirs = self._taps.pop("ref")
        finally:
            self._capture = False
            self._taps.clear()
        retain_loss = out_r["loss"]
        check_finite({"loss": retain_loss}, "retain loss")
        if refresh:
            anchor = velocity_kl(out_r, {"velocity": mine}, {"velocity": theirs})
            del mine, theirs
            check_finite({"anchor": anchor}, "GU retain anchor")
            grads = torch.autograd.grad(anchor, [named[n] for n in self.names], retain_graph=True, allow_unused=True)
            self._grow_basis(named, grads)
            self.last_anchor = anchor.item()
            del grads, anchor
        metrics["retain_fm"] = retain_loss.item()
        del out_r
        retain_loss.backward()
        for name in self.names:
            p = named[name]
            p.grad = self._project(name, p, g_f.pop(name), p.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True).item()
        optimizer.step()
        metrics.update(anchor=self.last_anchor, basis_size=sum(map(len, self.basis.values())) / len(self.basis),
                       grad_norm=grad_norm)
        return npo.detach() + retain_loss.detach(), metrics

    def state_dict(self):
        return {"kind": "npo_rt_gu", "names": self.names, "basis": self.basis, "calls": self.calls,
                "last_anchor": self.last_anchor, "settings": (self.beta, self.k, self.refresh_every, self.cap)}

    def load_state_dict(self, state):
        if state.get("kind") != "npo_rt_gu" or state["names"] != self.names or tuple(state["settings"]) != (
                self.beta, self.k, self.refresh_every, self.cap):
            raise ValueError("GU checkpoint has different method settings or projected tensors")
        self.basis, self.calls, self.last_anchor = state["basis"], state["calls"], state["last_anchor"]

    def to(self, named):
        self.basis = {n: [u.to(named[n].device) for u in basis] for n, basis in self.basis.items()}
