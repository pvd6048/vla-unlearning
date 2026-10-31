"""Loading GR00T N1.6, building its processor, and saving/loading the weights of a node.

A node stores only the trainable parameters, in fp32. Everything else is frozen and identical to the
base checkpoint, so a node's model is always "base checkpoint + node weights on top".

The construction below mirrors NVIDIA's single-node fine-tune (gr00t/experiment/launch_finetune.py
and gr00t/model/gr00t_n1d6/setup.py at the pinned Isaac-GR00T commit). If you change the GR00T
commit, re-check it against those two files.
"""

import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.configs.base_config import Config, get_default_config
import gr00t.model  # noqa: F401  (registers the GR00T model classes with transformers)

from vlu.config import BASE_MODEL, EMBODIMENT, GOAL_DATASET
from vlu.data import episode_loader

# Image augmentation of NVIDIA's examples/finetune.sh
COLOR_JITTER = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
HF_KWARGS = {"trust_remote_code": True}


def gr00t_config() -> Config:
    """The GR00T Config that launch_finetune.py builds for a LIBERO-Goal fine-tune.
    Trainable scope (NVIDIA default): DiT + projectors + VL layer norm + top-4 LLM layers."""
    config = get_default_config().load_dict(
        {
            "data": {
                "datasets": [
                    {"dataset_paths": [str(GOAL_DATASET)], "mix_ratio": 1.0, "embodiment_tag": EMBODIMENT}
                ]
            }
        }
    )
    m = config.model
    m.tune_llm, m.tune_visual, m.tune_projector, m.tune_diffusion_model = False, False, True, True
    m.state_dropout_prob = 0.0
    m.color_jitter_params = COLOR_JITTER
    m.extra_augmentation_config = None
    m.load_bf16, m.reproject_vision, m.eagle_collator = False, False, True
    m.model_name = "nvidia/Eagle-Block2A-2B-v2"
    m.backbone_trainable_params_fp32 = True
    m.use_relative_action = True
    config.validate()  # keeps only the libero_panda modality config and fills action defaults
    return config


def modality_configs(config: Config) -> dict:
    return config.data.modality_configs[EMBODIMENT]


def goal_statistics(config: Config) -> dict:
    """Normalization statistics of the full LIBERO-Goal dataset (all 10 tasks).

    These are pinned for every op. GR00T would otherwise recompute them from whatever data an op
    trains on, so the action normalization would change from op to op (and differ between a branch
    and its oracle), which would show up as drift that has nothing to do with learning."""
    return episode_loader(modality_configs(config)).get_dataset_statistics()  # reads meta/stats.json only


def build_processor(config: Config):
    """GR00T's processor (image transforms, state/action normalization, tokenization), in train mode.
    Call .eval() on it for evaluation (disables augmentation)."""
    m = config.model
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        modality_configs=config.data.modality_configs,
        image_crop_size=m.image_crop_size,
        image_target_size=m.image_target_size,
        random_rotation_angle=m.random_rotation_angle,
        color_jitter_params=m.color_jitter_params,
        model_name=m.model_name,
        model_type=m.backbone_model_type,
        formalize_language=m.formalize_language,
        apply_sincos_state_encoding=m.apply_sincos_state_encoding,
        max_action_horizon=m.action_horizon,
        use_albumentations=m.use_albumentations_transforms,
        extra_augmentation_config=m.extra_augmentation_config,
        shortest_image_edge=m.shortest_image_edge,
        crop_fraction=m.crop_fraction,
        transformers_loading_kwargs=HF_KWARGS,
        use_alternate_vl_dit=m.use_alternate_vl_dit,
        use_relative_action=m.use_relative_action,
        **HF_KWARGS,
    )
    processor.set_statistics({EMBODIMENT: goal_statistics(config)}, override=True)
    processor.train()
    return processor


def load_model(config: Config, weights: Path | None, device: str = "cuda"):
    """Base GR00T N1.6 with the trainable scope of `config`, plus `weights` (a node's
    weights.safetensors) on top. weights=None gives the base model."""
    m = config.model
    model, info = AutoModel.from_pretrained(
        BASE_MODEL,
        tune_llm=m.tune_llm,
        tune_visual=m.tune_visual,
        tune_projector=m.tune_projector,
        tune_diffusion_model=m.tune_diffusion_model,
        tune_vlln=m.tune_vlln,
        state_dropout_prob=m.state_dropout_prob,
        backbone_trainable_params_fp32=m.backbone_trainable_params_fp32,
        transformers_loading_kwargs=HF_KWARGS,
        output_loading_info=True,
        **HF_KWARGS,
    )
    # As in NVIDIA's setup.py: the base checkpoint may not contain the state-dropout mask token.
    # It is trainable, so for nodes it is overwritten by the saved weights below.
    if any("mask_token" in k for k in info.get("missing_keys", [])) and model.action_head.mask_token is not None:
        with torch.no_grad():
            model.action_head.mask_token.data.copy_(0.02 * torch.randn_like(model.action_head.mask_token))
    if weights is not None:
        load_weights(model, weights)
    return model.to(device)


def trainable_named_parameters(model) -> dict[str, torch.nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


# Frozen overrides: an op may change parameters that are frozen during learning (VLA-Forget edits the
# vision tower and the vision-language projector). Their new values are stored in the node's weights
# file next to the trainable parameters, and every descendant node carries them forward. The names of
# the overridden frozen parameters of a loaded model are kept in this attribute.
OVERRIDES_ATTR = "vlu_frozen_overrides"


def frozen_overrides(model) -> set[str]:
    return getattr(model, OVERRIDES_ATTR, set())


def add_frozen_overrides(model, names) -> None:
    setattr(model, OVERRIDES_ATTR, frozen_overrides(model) | set(names))


def save_weights(model, path: Path) -> None:
    """Save the trainable parameters (fp32) and any frozen overrides atomically: a crash never leaves
    a half-written file under the final name, and the file's existence is what marks a node as done."""
    params = dict(model.named_parameters())
    overrides = sorted(frozen_overrides(model))
    names = set(trainable_named_parameters(model)) | set(overrides)
    state = {name: params[name].detach().float().cpu().contiguous() for name in sorted(names)}
    tmp = path.with_name(path.name + ".tmp")
    save_file(state, str(tmp), metadata={"frozen_overrides": json.dumps(overrides)})
    tmp.replace(path)


def load_weights(model, path: Path) -> None:
    """Load a node's weights. The file's trainable parameters must be exactly the model's trainable
    parameters (else it was saved with another trainable scope); its frozen overrides (listed in the
    file's metadata) must be frozen parameters of the model."""
    state = load_file(str(path))
    with safe_open(str(path), framework="pt") as f:
        overrides = set(json.loads((f.metadata() or {}).get("frozen_overrides", "[]")))
    all_params = dict(model.named_parameters())
    trainable = set(trainable_named_parameters(model))
    saved_trainable = set(state) - overrides
    bad_overrides = sorted(n for n in overrides if n not in all_params or n in trainable)
    if saved_trainable != trainable or bad_overrides:
        missing, extra = sorted(trainable - saved_trainable), sorted(saved_trainable - trainable)
        raise ValueError(
            f"{path} does not match the model's trainable parameters (missing {missing[:5]}..., unexpected "
            f"{extra[:5]}..., bad frozen overrides {bad_overrides[:5]}...). Was it saved with another trainable scope?"
        )
    with torch.no_grad():
        for name, value in state.items():
            all_params[name].copy_(value.to(all_params[name].dtype))
    add_frozen_overrides(model, overrides)


def describe(model) -> dict:
    """Parameter counts, for logs."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
    return {"total_params": total, "trainable_params": trainable, "trainable_dtypes": dtypes}
