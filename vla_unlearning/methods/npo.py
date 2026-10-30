"""NPO (Zhang et al., 2024) adapted to a flow-matching policy.

GR00T has no token likelihood, so the flow-matching loss is used as a negative log-likelihood:
log(pi / pi_ref) ~= loss_ref - loss. NPO's loss  -(2/b) log sigmoid(-b log(pi/pi_ref))
then becomes  (2/b) softplus(b (loss_ref - loss)). It pushes the forget loss up, but the push
fades once the model is already much worse than the reference, unlike plain GA.
Model and reference see identical noise (same seed), otherwise the difference is noise.
"""

import torch
import torch.nn.functional as F

from vla_unlearning import groot
from vla_unlearning.methods.loop import optimize, retain_term


def unlearn(model, ref, data, cfg):
    beta = cfg.get("beta", 0.1)
    ref.train()  # same mode as the model, so both see the same dropout / augmentation behaviour

    def loss_fn(step):
        batch = data["forget"]()
        l, _ = groot.loss(model, batch, seed=step)
        with torch.no_grad():
            l_ref, _ = groot.loss(ref, batch, seed=step)
        npo = (2 / beta) * F.softplus(beta * (l_ref - l)).mean()
        r, logs = retain_term(model, data, cfg)
        return npo + r, {"forget": l.mean().item(), "forget_ref": l_ref.mean().item(), **logs}

    optimize(model, loss_fn, cfg)
