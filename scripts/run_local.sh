#!/bin/bash
# Calibration on a machine without Slurm (see CALIBRATION.md). After scripts/setup_cluster.sh:
#   bash scripts/run_local.sh
#   bash scripts/run_local.sh --pack    only pack what exists now (e.g. while a stuck run is still going)
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
    cp -f "$VLU_PROJECT_DIR/setup.log" "$OUT/" 2>/dev/null || true
    "$MAIN_PY" scripts/calibration_report.py --run "$RUN" --out "$OUT" || echo "WARNING: report failed"
    tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
    echo "=== $(date)  done. Please send back: $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
}
if [ "${1:-}" = "--pack" ]; then
    [ -f "$OUT/environment.txt" ] || nvidia-smi > "$OUT/environment.txt" 2>&1
    package
    exit 0
fi

# 1. Environment ---------------------------------------------------------------------------------
{
    echo "host $(hostname)  date $(date)  commit $(git rev-parse HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo -dirty)"
    nvidia-smi
    echo "cpus $VLU_CPUS"; free -g; df -h "$VLU_STORE"
    "$MAIN_PY" -c "import torch, flash_attn, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda,
'flash_attn', flash_attn.__version__, 'transformers', transformers.__version__,
[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
    "$SIM_PY" -c "import mujoco, robosuite, torch; print('sim: mujoco', mujoco.__version__, 'robosuite', robosuite.__version__, 'torch', torch.__version__)"
} > "$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt"

# 2. Rendering: the GPUs' EGL driver if a test render works on every GPU (each GPU's worker renders its
#    evals there, see slurm/eval.sbatch), else Mesa (software, one device: 0) -----------------------
render_test() {  # render_test <gpu|mesa> <EGL device>
    VLU_RENDER="$1" RENDER_DEVICE="$2" bash -c 'source slurm/env.sh; CUDA_VISIBLE_DEVICES="" MUJOCO_EGL_DEVICE_ID=$RENDER_DEVICE "$SIM_PY" - <<"PY"
import os
import sys
sys.path.insert(0, ".")
print("render test: VLU_RENDER", os.environ["VLU_RENDER"], "EGL device", os.environ["MUJOCO_EGL_DEVICE_ID"],
      "EGL vendor file", os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "(system default)"))
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
gpu_render_ok() {
    local index
    for index in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
        render_test gpu "$index" >> "$OUT/environment.txt" 2>&1 || return 1
    done
}
if gpu_render_ok; then export VLU_RENDER=gpu
elif render_test mesa 0 >> "$OUT/environment.txt" 2>&1; then export VLU_RENDER=mesa
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
# A rerun (e.g. after a code fix) tries earlier failures again.
"$MAIN_PY" scripts/submit.py --run "$RUN" --retry-failed --workers none

# 5. One worker per GPU: GPUs with >= TRAIN_FREE_MIB free train and evaluate, the others only evaluate
#    (training peaks at ~34 GB, NPO's with its frozen reference at ~41 GB; CALIBRATION.md says 44 GB) ---
TRAIN_FREE_MIB=44000
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits)
export VLU_EVAL_WORKERS=$(( VLU_CPUS / ${#GPUS[@]} - 1 > 1 ? VLU_CPUS / ${#GPUS[@]} - 1 : 1 ))
if ! printf '%s\n' "${GPUS[@]}" | awk -F', ' -v need="$TRAIN_FREE_MIB" '$2 >= need {ok = 1} END {exit !ok}'; then
    echo "ERROR: no GPU has $TRAIN_FREE_MIB MiB free for training (nvidia-smi: ${GPUS[*]}); please stop other jobs on the GPUs"
    package; exit 1
fi
PIDS=()
for line in "${GPUS[@]}"; do
    IFS=', ' read -r index free used <<< "$line"
    kinds=$([ "$free" -ge "$TRAIN_FREE_MIB" ] && echo train,eval,features || echo eval)
    [ "$used" -gt 1000 ] && echo "WARNING: GPU $index already has ${used} MiB in use by another program; a job started there later can still run out of memory"
    echo "GPU $index (${free} MiB free): $kinds, $VLU_EVAL_WORKERS simulator processes per eval"
    CUDA_VISIBLE_DEVICES="$index" "$MAIN_PY" -m vlu.worker --run "$RUN" --kinds "$kinds" --wait \
        > "$OUT/worker_gpu$index.log" 2>&1 &
    PIDS+=($!)
    sleep 30  # start one at a time (each loads a 3B model)
done
wait "${PIDS[@]}"

# 6. Report ----------------------------------------------------------------------------------------
"$MAIN_PY" scripts/submit.py --run "$RUN" --workers none
package
