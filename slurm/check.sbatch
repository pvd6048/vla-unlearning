#!/bin/bash
# Pre-flight checks on ONE small slice, before a long run on the cluster (scripts/run_local.sh does the
# same on a machine without Slurm): the base tests (tests/README.md), and every task A-E exists in
# LIBERO-Goal with its training instruction and enough initial states (else a task's first eval fails,
# hours into the run). Exit code 0 only if everything passed.
#   export VLU_PROJECT_DIR=$PWD
#   sbatch --partition=gpu_small --qos=gpu_small --gres=gpu:1g.24gb:1 --cpus-per-gpu=6 --mem-per-gpu=24G \
#       --time=1:00:00 --output=check-%j.out slurm/check.sbatch
set -uo pipefail
source "${VLU_PROJECT_DIR:?export VLU_PROJECT_DIR=<repo root> first}/slurm/env.sh"
cd "$VLU_PROJECT_DIR"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  commit $(git rev-parse HEAD 2>/dev/null)  render $VLU_RENDER"
nvidia-smi -L || true

OK=1
LOGS="$VLU_RUNS_DIR/_check/${SLURM_JOB_ID:-local}"
mkdir -p "$LOGS"
for t in tests/test_*.py; do
    if "$MAIN_PY" "$t" > "$LOGS/$(basename "$t" .py).log" 2>&1; then echo "PASS $t"
    else echo "FAIL $t (log: $LOGS/$(basename "$t" .py).log)"; tail -20 "$LOGS/$(basename "$t" .py).log"; OK=0; fi
done

# The simulator needs no CUDA; on MIG slices it renders with Mesa (one EGL device: 0).
if CUDA_VISIBLE_DEVICES="" MUJOCO_EGL_DEVICE_ID=0 "$SIM_PY" - <<'PY'
import sys
sys.path.insert(0, ".")
from vlu.config import RECIPE
from vlu.eval.libero_client import make_env
from vlu.tasks import TASKS
for letter in TASKS:
    env, init_states = make_env(letter)  # raises if LIBERO's instruction differs from the training one
    env._env.close()  # its MuJoCo / EGL render context: one open at a time
    assert len(init_states) >= RECIPE.eval_episodes, f"task {letter}: {len(init_states)} initial states"
    print(f"task {letter} OK: {TASKS[letter]}, {len(init_states)} initial states")
PY
then echo "PASS LIBERO tasks"
else echo "FAIL LIBERO tasks (see the traceback above)"; OK=0; fi

if [ "$OK" = 1 ]; then echo "all checks passed"; else echo "SOME CHECKS FAILED"; exit 1; fi
