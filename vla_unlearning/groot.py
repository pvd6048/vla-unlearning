"""The only module that touches GR00T's model and data code.

Everything here calls NVIDIA's own classes (config, pipeline, dataset factory, model forward)
exactly as their training code does. Nothing inside third_party/Isaac-GR00T is modified.
"""

import copy
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gr00t.configs.base_config import get_default_config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.model.gr00t_n1d6.setup import Gr00tN1d6Pipeline

EMBODIMENT = "libero_panda"
BASE_MODEL = "nvidia/GR00T-N1.6-3B"

# Parameter groups by module-name prefix (used by staged methods and drift analysis).
GROUPS = {
    "vision": ("backbone.model.vision_model.", "backbone.model.mlp1."),
    "llm": ("backbone.model.language_model.",),
    "action_io": (
        "action_head.state_encoder.",
        "action_head.action_encoder.",
        "action_head.action_decoder.",
        "action_head.vlln.",
        "action_head.position_embedding.",
        "action_head.mask_token",
    ),
    "dit": ("action_head.model.",),
}


def dataset_mix(mix):
    """`mix` = [(dataset_paths, mix_ratio), ...] -> NVIDIA's data.datasets entries.

    E.g. [([task_dir], 1.0), (replay_dirs, 1.0)] is a 1:1 mixture of new task and replay buffer.
    """
    specs = [
        {"dataset_paths": [str(p) for p in paths], "mix_ratio": r, "embodiment_tag": EMBODIMENT}
        for paths, r in mix
    ]
    return get_default_config().load_dict({"data": {"datasets": specs}}).data.datasets


def make_config(init, mix, out_dir, steps=10000, lr=1e-4, batch=16, seed=0, workers=4):
    """Training config with the same settings as NVIDIA's launch_finetune.py + finetune.sh."""
    cfg = get_default_config().load_dict({"data": {"download_cache": False}})
    cfg.data.datasets = dataset_mix(mix)
    cfg.load_config_path = None

    # Model: NVIDIA defaults (tunes projector, DiT, vlln and the top-4 LLM layers; vision frozen).
    m = cfg.model
    m.tune_llm, m.tune_visual, m.tune_projector, m.tune_diffusion_model = False, False, True, True
    m.state_dropout_prob = 0.0
    m.random_rotation_angle = None
    m.color_jitter_params = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
    m.extra_augmentation_config = None
    m.load_bf16 = False
    m.reproject_vision = False
    m.eagle_collator = True
    m.model_name = "nvidia/Eagle-Block2A-2B-v2"
    m.backbone_trainable_params_fp32 = True
    m.use_relative_action = True

    t = cfg.training
    t.start_from_checkpoint = str(init)
    t.optim = "adamw_torch"
    t.global_batch_size = batch
    t.dataloader_num_workers = workers
    t.learning_rate = lr
    t.gradient_accumulation_steps = 1
    t.output_dir = str(out_dir)
    t.save_steps = steps  # one checkpoint, at the end
    t.save_total_limit = 1
    t.num_gpus = 1
    t.use_wandb = False
    t.max_steps = steps
    t.weight_decay = 1e-5
    t.warmup_ratio = 0.05

    d = cfg.data
    d.shard_size = 2**10
    d.episode_sampling_rate = 0.1
    d.num_shards_per_epoch = int(1e5)
    d.seed = seed
    return cfg


class Batches:
    """Endless stream of collated training batches; call it to get the next batch."""

    def __init__(self, dataset, collator, batch, workers=2):
        self.it = iter(DataLoader(dataset, batch_size=batch, collate_fn=collator, num_workers=workers))

    def __call__(self):
        return next(self.it)


def load(ckpt, forget, retain, work_dir, batch=16, seed=0):
    """Load a checkpoint the way NVIDIA's training does, plus forget/retain batch streams.

    `forget` and `retain` are lists of dataset folders. Returns (model, {"forget", "retain"}).
    """
    cfg_dir = Path(work_dir) / "experiment_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(ckpt, [(forget, 1.0)], work_dir, batch=batch, seed=seed)
    pipe = Gr00tN1d6Pipeline(cfg, cfg_dir)
    pipe.setup()  # NVIDIA's own model + processor + (forget) dataset creation
    model = pipe.return_model().cuda()
    processor, collator = pipe.return_processor(), pipe.return_collator()

    data = {"forget": Batches(pipe.return_dataset()[0], collator, batch), "retain": None}
    if retain:
        cfg.data.datasets = dataset_mix([(retain, 1.0)])
        retain_set, _ = DatasetFactory(config=cfg).build(processor=processor)
        data["retain"] = Batches(retain_set, collator, batch)
    return model, data


def loss(model, batch, seed=None):
    """Per-sample flow-matching loss from NVIDIA's own forward (its `action_loss` output).

    With `seed`, the random noise and timestep are fixed, so two models (e.g. current and
    reference) can be compared on identical noise. Returns (loss[B], full model output).
    """
    with torch.random.fork_rng(devices=[model.device.index]):
        if seed is not None:
            torch.manual_seed(seed)
        out = model(**batch)
    per_elem, mask = out["action_loss"], out["action_mask"]
    return per_elem.sum((1, 2)) / (mask.sum((1, 2)) + 1e-6), out


def trainable(model):
    """Parameters NVIDIA's config marks as trainable; all methods only change these."""
    return [p for p in model.parameters() if p.requires_grad]


def groups(model):
    """{group name: [trainable params]} using GROUPS prefixes."""
    out = {g: [] for g in GROUPS}
    for name, p in model.named_parameters():
        for g, prefixes in GROUPS.items():
            if p.requires_grad and name.startswith(prefixes):
                out[g].append(p)
    return out


def frozen_copy(model):
    ref = copy.deepcopy(model)
    ref.requires_grad_(False)
    return ref


def save(model, src_ckpt, dst):
    """Write an NVIDIA-format checkpoint: processor/config files of `src_ckpt` + new weights."""
    skip = shutil.ignore_patterns(
        "*.safetensors", "*.safetensors.index.json", "optimizer*", "scheduler*", "rng_state*", "trainer_state.json"
    )
    shutil.copytree(src_ckpt, dst, ignore=skip, dirs_exist_ok=True)
    model.save_pretrained(dst)
