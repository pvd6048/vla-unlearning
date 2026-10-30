#!/usr/bin/env bash
# One-time setup on the GPU server. Follows NVIDIA's own README steps for GR00T N1.6 + LIBERO.
#   bash scripts/setup_server.sh
# LIBERO rendering needs (once, with sudo): sudo apt install libegl1-mesa-dev libglu1-mesa
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GROOT="$ROOT/third_party/Isaac-GR00T"
DATA="$ROOT/data"

# 1. NVIDIA's code at the pinned commit (n1.6.1-release), and its Python environment.
git -C "$ROOT" submodule update --init
cd "$GROOT"
uv sync
uv pip install -e .
uv pip install pyyaml pandas

# 2. LIBERO-Goal dataset, with NVIDIA's modality file and their fix for one corrupted video.
uv run huggingface-cli download --repo-type dataset IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
    --local-dir "$DATA/libero_goal"
cp examples/LIBERO/modality.json "$DATA/libero_goal/meta/"
cp examples/LIBERO/patches/episode_000082.mp4 \
    "$DATA/libero_goal/videos/chunk-000/observation.images.wrist_image/"

# 3. Normalization stats over the FULL suite, then one folder per task sharing those stats.
uv run python gr00t/data/stats.py "$DATA/libero_goal" libero_panda
uv run python "$ROOT/scripts/split_tasks.py" "$DATA/libero_goal" "$DATA/goal" --replay-episodes 10

# 4. LIBERO simulator in its own environment (NVIDIA's script).
bash gr00t/eval/sim/LIBERO/setup_libero.sh

echo "Setup done. Activate with: source $GROOT/.venv/bin/activate"
