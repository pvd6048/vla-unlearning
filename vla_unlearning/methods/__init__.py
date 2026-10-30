"""Unlearning methods. Each is `unlearn(model, ref, data, cfg)` and edits `model` in place.

    model : GR00T in train mode; only its trainable params (NVIDIA's tuning flags) are changed
    ref   : frozen copy of the model right before the U op
    data  : {"forget": callable -> batch, "retain": callable -> batch or None}
    cfg   : method config dict (configs/methods/<name>.yaml)
"""

from vla_unlearning.methods import ga, graddiff, npo, salun, ssd

METHODS = {
    "ga": ga.unlearn,
    "graddiff": graddiff.unlearn,
    "npo": npo.unlearn,
    "ssd": ssd.unlearn,
    "salun": salun.unlearn,
}
