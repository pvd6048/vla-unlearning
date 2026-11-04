# Paths and environment for every job and setup step. Source it; do not run it.
#   source slurm/env.sh
# Per-user settings go in slurm/env.local.sh (git-ignored), e.g.
#   export VLU_STORE=/scratch/$USER/vla-unlearning     # big disk: code deps, data, runs
#   export VLU_HF_HOME=/path/to/existing/hf_home       # reuse an HF cache that already has GR00T-N1.6-3B
#   export VLU_GOAL_SOURCE=/path/to/libero_goal_no_noops_1.0.0_lerobot   # reuse a downloaded copy

VLU_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLU_PROJECT_DIR
[ -f "$VLU_PROJECT_DIR/slurm/env.local.sh" ] && source "$VLU_PROJECT_DIR/slurm/env.local.sh"

# Everything large lives under VLU_STORE: the two Python environments (~20 GB), uv's download
# cache, the GR00T checkpoint (7 GB), the dataset (0.4 GB) and the run tree (hundreds of GB).
export VLU_STORE="${VLU_STORE:-$VLU_PROJECT_DIR}"
export VLU_DATA_DIR="$VLU_STORE/data"
export VLU_RUNS_DIR="$VLU_STORE/runs"
export HF_HOME="${VLU_HF_HOME:-$VLU_STORE/hf_cache}"
export UV_CACHE_DIR="$VLU_STORE/uv_cache"   # same disk as the venvs, so uv can hard-link
# LIBERO keeps its asset paths in <LIBERO_CONFIG_PATH>/config.yaml (default ~/.libero). Ours is
# private, so other LIBERO installs of the same user are left alone.
export LIBERO_CONFIG_PATH="$VLU_STORE/libero_config"

# Isaac-GR00T, pinned. vlu/model.py mirrors this commit's fine-tune code.
export GR00T_DIR="$VLU_STORE/external/Isaac-GR00T"
export GR00T_COMMIT="ead52833afbbf4243f8cd5e7664f48a94de03b19"   # n1.6-release, 2026-04-14

# Two Python environments, both created by scripts/setup_cluster.sh:
MAIN_PY="$GR00T_DIR/.venv/bin/python"                                  # GR00T, torch, GPU
SIM_PY="$GR00T_DIR/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"   # LIBERO simulator, CPU
export MAIN_PY SIM_PY
export PYTHONPATH="$VLU_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Jobs never download: everything is fetched once by setup_cluster.sh.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# GPU numbering: nvidia-smi / EGL order everywhere (CUDA's default puts the fastest GPU first).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# MuJoCo rendering (EGL). VLU_RENDER=mesa (default): Mesa's software EGL, needed on the cluster's MIG
# slices, which have no graphics API. VLU_RENDER=gpu: the NVIDIA EGL driver of a full GPU
# (scripts/run_local.sh picks it when a test render works).
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export VLU_RENDER="${VLU_RENDER:-mesa}"
MESA_EGL=/usr/share/glvnd/egl_vendor.d/50_mesa.json
if [ "$VLU_RENDER" = mesa ] && [ -f "$MESA_EGL" ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_EGL"
else  # e.g. inherited from a shell that sourced this file in mesa mode: it would hide the GPU driver
    unset __EGL_VENDOR_LIBRARY_FILENAMES
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"   # simulator workers: one thread each
# The tokenizer runs in the DataLoader workers: without this, every fork prints a warning and turns
# the tokenizer's threads off anyway.
export TOKENIZERS_PARALLELISM=false
# CPUs this job may use (its cgroup / CPU affinity). Not plain `nproc`: GNU nproc also honours
# OMP_NUM_THREADS, so after the line above it would always say 1.
VLU_CPUS=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
