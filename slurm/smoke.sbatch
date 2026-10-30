#!/bin/bash
# Day-1 smoke test on ONE slice that can train (see README): 200 training steps of L_A in the
# throw-away run "smoke", then a 5-episode LIBERO eval of the result with 2 videos.
# Measures training steps/s, peak GPU memory and seconds per rollout.
#   sbatch <train resources> slurm/smoke.sbatch
set -euo pipefail
source "${VLU_PROJECT_DIR:?export VLU_PROJECT_DIR=<repo root> first}/slurm/env.sh"

nvidia-smi
"$MAIN_PY" -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0), 'flash_attn', flash_attn.__version__)"
"$SIM_PY" -c "import mujoco, robosuite, torch; print('sim env: mujoco', mujoco.__version__, 'robosuite', robosuite.__version__, 'torch', torch.__version__)"

"$MAIN_PY" -m vlu.run_op --run smoke --node LA --seed 0 --steps 200 --save-at 100

# Eval needs the run to exist; reuse the eval job script in this allocation.
EPISODES=5 bash "$VLU_PROJECT_DIR/slurm/eval.sbatch" smoke LA A final 0 2
echo "smoke test finished: see $VLU_RUNS_DIR/smoke/LA/{train_log.jsonl,eval/final_seed0/}"
