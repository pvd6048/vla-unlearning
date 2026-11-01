#!/bin/bash
# Calibration on a machine without Slurm (see CALIBRATION.md). After scripts/setup_cluster.sh:
#   bash scripts/run_local.sh
# Safe to rerun: finished work is skipped and interrupted work resumes. Everything to send back ends up
# in $VLU_STORE/calibration_<host>/ and $VLU_STORE/calibration_<host>.tar.gz (no model weights).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../slurm/env.sh"
cd "$VLU_PROJECT_DIR"
RUN="${RUN:-cal}"
OUT="$VLU_STORE/calibration_$(hostname)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run_local.log") 2>&1
echo "=== $(date)  run $RUN  output $OUT"

package() {
    "$MAIN_PY" scripts/calibration_report.py --run "$RUN" --out "$OUT" || echo "WARNING: report failed"
    tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
    echo "=== $(date)  done. Please send back: $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
}

# 1. Environment ---------------------------------------------------------------------------------
{
    echo "host $(hostname)  date $(date)  commit $(git rev-parse HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo -dirty)"
    nvidia-smi
    nproc; free -g; df -h "$VLU_STORE"
    "$MAIN_PY" -c "import torch, flash_attn, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda,
'flash_attn', flash_attn.__version__, 'transformers', transformers.__version__,
[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
    "$SIM_PY" -c "import mujoco, robosuite, torch; print('sim: mujoco', mujoco.__version__, 'robosuite', robosuite.__version__, 'torch', torch.__version__)"
} > "$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt"

# 2. Rendering: the GPU's EGL driver if a test render works, else Mesa (software) ---------------------
render_test() {
    VLU_RENDER="$1" bash -c 'source slurm/env.sh; CUDA_VISIBLE_DEVICES="" MUJOCO_EGL_DEVICE_ID=0 "$SIM_PY" - <<"PY"
import sys
sys.path.insert(0, ".")
from vlu.eval.libero_client import LIBERO_NOOP, make_env
env, init_states = make_env("A")
env.reset()
env._env.set_init_state(init_states[0])
raw_obs, _, _, _ = env._env.step(LIBERO_NOOP)  # as in vlu/eval/libero_client.py
image = raw_obs["agentview_image"]
assert image.shape == (256, 256, 3) and image.std() > 1, "blank render"
print("render OK", image.shape, float(image.mean()))
PY'
}
if render_test gpu >> "$OUT/environment.txt" 2>&1; then export VLU_RENDER=gpu
elif render_test mesa >> "$OUT/environment.txt" 2>&1; then export VLU_RENDER=mesa
else echo "ERROR: LIBERO cannot render with the GPU or with Mesa (see environment.txt)"; package; exit 1; fi
echo "rendering: $VLU_RENDER" | tee -a "$OUT/environment.txt"

# 3. Base tests (tests/README.md) ------------------------------------------------------------------
: > "$OUT/tests.txt"
TESTS_OK=1
for t in tests/test_stream_metrics.py tests/test_calibration.py tests/test_drift_params.py tests/test_drift_extract.py tests/test_ops.py tests/test_vlaforget.py; do
    if "$MAIN_PY" "$t" >> "$OUT/tests.txt" 2>&1; then echo "PASS $t" | tee -a "$OUT/tests.txt"
    else echo "FAIL $t" | tee -a "$OUT/tests.txt"; TESTS_OK=0; fi
done
if [ "$TESTS_OK" = 0 ]; then echo "ERROR: a base test failed; not starting the runs"; package; exit 1; fi

# 4. Plan ------------------------------------------------------------------------------------------
"$MAIN_PY" scripts/calibrate.py --run "$RUN"

# 5. One worker per GPU: GPUs with >= 40 GB train and evaluate, smaller ones only evaluate --------------
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits)
export VLU_EVAL_WORKERS=$(( $(nproc) / ${#GPUS[@]} - 1 > 1 ? $(nproc) / ${#GPUS[@]} - 1 : 1 ))
PIDS=()
for line in "${GPUS[@]}"; do
    index="${line%%,*}"; mem="${line##*, }"
    kinds=$([ "$mem" -ge 40000 ] && echo train,eval,features || echo eval)
    echo "GPU $index (${mem} MiB): $kinds, $VLU_EVAL_WORKERS simulator processes per eval"
    CUDA_VISIBLE_DEVICES="$index" "$MAIN_PY" -m vlu.worker --run "$RUN" --kinds "$kinds" --wait \
        > "$OUT/worker_gpu$index.log" 2>&1 &
    PIDS+=($!)
    sleep 30  # start one at a time (each loads a 3B model)
done
wait "${PIDS[@]}"

# 6. Report ----------------------------------------------------------------------------------------
"$MAIN_PY" scripts/submit.py --run "$RUN" --workers none
package
