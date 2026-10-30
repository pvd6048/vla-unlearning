#!/bin/bash
# One-time setup on the cluster LOGIN node (needs internet; compute nodes run offline).
#   bash scripts/setup_cluster.sh
# Safe to rerun: finished steps are skipped. Creates, under $VLU_STORE (see slurm/env.sh):
#   external/Isaac-GR00T                     pinned GR00T code + its LIBERO submodule
#   external/Isaac-GR00T/.venv               main env: GR00T, torch, flash-attn       -> $MAIN_PY
#   .../gr00t/eval/sim/LIBERO/libero_uv/.venv  simulator env: LIBERO, robosuite, MuJoCo -> $SIM_PY
#   libero_config/config.yaml                LIBERO's asset paths
#   hf_cache/                                GR00T-N1.6-3B weights
#   data/libero_goal_no_noops_1.0.0_lerobot  LIBERO-Goal demos (LeRobot format) + goal_report.json
#   envs/{main,sim}_packages.txt             exact package versions installed (for debugging)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../slurm/env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$VLU_DATA_DIR" "$VLU_RUNS_DIR" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$VLU_STORE/envs"

command -v uv >/dev/null || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# 1. GR00T code at the pinned commit. Only the LIBERO submodule is needed. (git-lfs only covers
#    GR00T's demo_data/ and media/, which we do not use.)
if [ ! -d "$GR00T_DIR/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/Isaac-GR00T.git "$GR00T_DIR"
fi
git -C "$GR00T_DIR" checkout --quiet "$GR00T_COMMIT"
git -C "$GR00T_DIR" submodule update --init external_dependencies/LIBERO

# 2. Main environment: exactly GR00T's uv.lock (torch 2.7.1+cu128, flash-attn 2.7.4.post1,
#    transformers 4.51.3, numpy 1.26.4). Differences from NVIDIA's install_deps.sh:
#    - Python 3.10 is pinned (as in NVIDIA's Dockerfile): the lock only has prebuilt flash-attn
#      wheels for 3.10 and 3.12; any other Python would compile flash-attn from source.
#    - deepspeed (source-only package, needs nvcc to build) and TensorRT (deployment only, several GB)
#      are left out: nothing we import uses them.
#    - its sudo apt step (FFmpeg, CUDA toolkit) is skipped: we decode videos with PyAV (vlu/config.py).
if [ ! -f "$GR00T_DIR/.venv/.vlu_ok" ]; then
    (cd "$GR00T_DIR" && uv sync --frozen --python 3.10 \
        --no-install-package deepspeed --no-install-package tensorrt --no-install-package tensorrt-cu13 \
        --no-install-package tensorrt-cu13-bindings --no-install-package tensorrt-cu13-libs)
    "$MAIN_PY" -c "import torch, transformers, av, gr00t.model; print('main env ok: torch', torch.__version__)"
    "$MAIN_PY" -c "import flash_attn; print('flash_attn', flash_attn.__version__)" ||
        echo "WARNING: flash_attn does not import on this node (may only need a GPU driver; smoke.sbatch checks again)"
    touch "$GR00T_DIR/.venv/.vlu_ok"
fi

# 3. Simulator environment: the install steps of NVIDIA's gr00t/eval/sim/LIBERO/setup_libero.sh,
#    run here instead of the script itself because the script deletes ~/.libero (which other LIBERO
#    installs may use) and ends with a render test that cannot work on a login node.
#    Unpinned packages (mujoco, robosuite's deps, ...) are resolved as of the GR00T commit date,
#    so every rebuild gets the same versions.
LIBERO_REPO="$GR00T_DIR/external_dependencies/LIBERO"
SIM_VENV="$(dirname "$(dirname "$SIM_PY")")"
if [ ! -f "$SIM_VENV/.vlu_ok" ]; then
    rm -rf "$SIM_VENV"
    uv venv "$SIM_VENV" --python 3.10
    sim_install() { uv pip install --python "$SIM_PY" --exclude-newer 2026-04-15T00:00:00Z "$@"; }
    sim_install --requirements "$LIBERO_REPO/requirements.txt"
    sim_install -e "$LIBERO_REPO" --config-settings editable_mode=compat
    sim_install torch==2.5.1 torchvision==0.20.1 pydantic av tianshou==0.5.1 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
    sim_install transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
    sim_install numpy==1.26.4
    sim_install -e "$GR00T_DIR" --no-deps
    # Writes $LIBERO_CONFIG_PATH/config.yaml with LIBERO's default asset paths ("n" = no custom dataset path).
    [ -f "$LIBERO_CONFIG_PATH/config.yaml" ] || echo n | "$SIM_PY" -c "import libero.libero"
    "$SIM_PY" -c "from libero.libero import benchmark; print('sim env ok:', benchmark.get_benchmark_dict()['libero_goal']().get_num_tasks(), 'LIBERO-Goal tasks')"
    touch "$SIM_VENV/.vlu_ok"
fi
uv pip freeze --python "$MAIN_PY" > "$VLU_STORE/envs/main_packages.txt"
uv pip freeze --python "$SIM_PY" > "$VLU_STORE/envs/sim_packages.txt"

# 4. Model and data. HF downloads skip files that are already complete.
"$GR00T_DIR/.venv/bin/huggingface-cli" download nvidia/GR00T-N1.6-3B
DATASET="$VLU_DATA_DIR/libero_goal_no_noops_1.0.0_lerobot"
if [ ! -f "$DATASET/meta/info.json" ] && [ -n "${VLU_GOAL_SOURCE:-}" ]; then
    # Copy (not link): the next steps write into the dataset, and the source may belong to another project.
    echo "copying LIBERO-Goal from $VLU_GOAL_SOURCE"
    cp -rL "$VLU_GOAL_SOURCE" "$DATASET"
    rm -f "$DATASET/meta/stats.json" "$DATASET/meta/relative_stats.json"   # recomputed below by our GR00T
fi
# Downloads whatever is missing (everything if there was no copy).
"$GR00T_DIR/.venv/bin/huggingface-cli" download --repo-type dataset \
    IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot --local-dir "$DATASET"
# NVIDIA's LIBERO README: add the modality description and replace one corrupted wrist video.
cp "$GR00T_DIR/examples/LIBERO/modality.json" "$DATASET/meta/"
cp "$GR00T_DIR/examples/LIBERO/patches/episode_000082.mp4" "$DATASET/videos/chunk-000/observation.images.wrist_image/"

# 5. Dataset statistics and a per-task report (also checks the A–E mapping against the data).
"$MAIN_PY" "$VLU_PROJECT_DIR/scripts/prepare_data.py"
echo "setup done"
