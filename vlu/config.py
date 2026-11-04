"""Paths and the frozen experiment recipe.

Change a default here (and commit it) rather than overriding it on the command line, so that every
run in the paper uses the same recipe. Paths come from environment variables set in slurm/env.sh.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("VLU_RUNS_DIR", PROJECT_DIR / "runs"))
DATA_DIR = Path(os.environ.get("VLU_DATA_DIR", PROJECT_DIR / "data"))
GOAL_DATASET = DATA_DIR / "libero_goal_no_noops_1.0.0_lerobot"

BASE_MODEL = os.environ.get("VLU_BASE_MODEL", "nvidia/GR00T-N1.6-3B")
EMBODIMENT = "libero_panda"
# Video decoder for the demos (AV1 videos). "pyav" (default) uses the FFmpeg bundled in the pinned av
# wheel, so it works on any node. "torchcodec" is GR00T's default but needs system FFmpeg libraries
# with an AV1 decoder. Do not use GR00T's "opencv" reader: it returns BGR frames (LIBERO gives RGB).
VIDEO_BACKEND = os.environ.get("VLU_VIDEO_BACKEND", "pyav")


@dataclass(frozen=True)
class VLAForgetConfig:
    """VLA-Forget (vlu/vlaforget.py). Values from the paper (Table 5) and the authors' code; PROVISIONAL
    until checked on GR00T."""

    lr: float = 2e-4  # Table 5 and code (the paper's appendix also mentions 5e-5)
    lambda_forget: float = 0.7  # paper, OpenVLA setting (1.2 in the authors' small-model code)
    lambda_mismatch: float = 0.8
    lambda_feat: float = 0.7
    beta_kl: float = 0.5
    gamma: float = 1.0  # weight of the projector features in the feature loss (code default)
    alpha_ratio: float = 1.0
    top_k_vision: int = 2
    top_k_projector: int = 2
    top_k_reasoning: int = 3  # code default (paper: 3 or 4)
    vision_blocks: int = 4  # last SigLIP2 layers (of 27) that are candidates; the paper does not say
    lora_r: int = 16
    lora_alpha: float = 16.0
    stage_ratio: tuple = (60, 60, 90)  # vision / projector / reasoning share of the op's step budget
    max_grad_norm: float = 1.0
    selection_batches: int = 16  # batches for the layer-selection gradients (paper: whole set; code: 1)
    boundary_forget_pool: int = 256  # forget frames whose nearest retain frames form the boundary set
    boundary_retain_pool: int = 2048  # retain frames searched
    merge_tolerance: float = 0.01  # max relative change of a probe loss when merging the adapters; above it the op fails


@dataclass(frozen=True)
class Recipe:
    # --- Learning (L ops) -------------------------------------------------------------------------
    # CL-paper recipe (Liu et al. 2026, GR00T): lr 1e-4, batch 16, AdamW, wd 1e-5, cosine decay.
    # The optimizer and LR schedule restart at every op: each op is a fresh fine-tune of its parent.
    learn_steps: int = 10_000
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05  # NVIDIA finetune.sh default
    max_grad_norm: float = 1.0  # NVIDIA TrainingConfig default
    # ER as in Liu et al.'s continual-GR00T: each L-op batch = batch_size frames of the new task +
    # batch_size frames from a fixed replay buffer of replay_per_task random frames per retained task
    # (their n_memories = 1000; None = all frames).
    replay_per_task: int | None = 1_000

    # --- Unlearning (U ops) -----------------------------------------------------------------------
    # PROVISIONAL: same step budget for every method; values still to be agreed (see README).
    unlearn_steps: int = 1_000
    unlearn_lr: float = 1e-5
    # Method hyperparameters, PROVISIONAL (each paper's default, mapped to the flow-matching loss):
    npo_beta: float = 0.1  # NPO paper default; log-likelihood stand-in = -sum of squared FM errors of a chunk
    ssd_alpha: float = 5.0  # SSD official code: 10 for ResNet, 5 for ViT (transformers); dampen weights with I_forget > alpha * I_ref ...
    ssd_lambda: float = 1.0  # ... by the factor min(lambda * I_ref / I_forget, 1)
    # I_ref, the reference importance: "retain" = the retained tasks (default; METHODS.md section 4: the paper's
    # full-data importance is ~ the retain importance at its forget shares of 1-5%, not at ours of 0.2-0.45),
    # or "all" = all tasks the parent keeps, forget task included (the paper's literal D).
    ssd_reference: str = "retain"
    # ASSD (Schoepf, Foster & Brintrup 2024): alpha = the p-th percentile of I_forget / I_ref over the weights,
    # p = 100 - ln(1 + 100 * forget share of the frames); ssd_alpha is then ignored, lambda as set (ASSD: 1).
    ssd_adaptive: bool = False
    salun_keep: float = 0.5  # SalUn paper default: update the 50% most salient weights
    importance_batches: int = 64  # batches for SSD's Fisher importances and SalUn's saliency
    vlaforget: VLAForgetConfig = field(default_factory=VLAForgetConfig)

    # --- Bookkeeping ------------------------------------------------------------------------------
    num_workers: int = 8  # DataLoader processes building batches
    log_every: int = 50
    save_every: int = 2_500  # resumable checkpoint (weights + optimizer, ~20 GB, ~3 min to write)
    probe_every: int = 500  # flow-matching loss on a fixed batch of every task (L ops)
    unlearn_probe_every: int = 50  # the same for U ops, which are short
    probe_size: int = 32

    # --- Evaluation (LIBERO) ----------------------------------------------------------------------
    eval_episodes: int = 50  # one per fixed LIBERO initial state
    eval_max_steps: int = 300  # standard LIBERO-Goal limit (OpenVLA protocol)
    eval_wait_steps: int = 10  # no-op steps after reset so objects settle (standard)
    n_action_steps: int = 8  # actions executed from each predicted 16-step chunk (NVIDIA default)


RECIPE = Recipe()
