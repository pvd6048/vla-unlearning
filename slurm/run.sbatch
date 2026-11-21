#!/bin/bash
#SBATCH --partition=gpu_large
#SBATCH --qos=gpu_large
#SBATCH --gres=gpu:4g.96gb:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem-per-gpu=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=vlu-run-%j.out
# A whole experiment on one gpu_large slice, as scripts/run_local.sh does on a machine without Slurm:
#   cd <repo root> && sbatch slurm/run.sbatch            the calibration (run cal, scripts/calibrate.py)
#   cd <repo root> && sbatch slurm/run.sbatch --main     the main run (run goal_s0, scripts/main_run.py)
# 1. first launch only: base tests + every task A-E in LIBERO (slurm/check.sbatch) and a flash-attn kernel
#    test on this GPU; the plan; earlier failures are tried again
# 2. one worker runs the plan (train, eval, features). 15 minutes before the time limit this job queues its
#    own successor (same options, VLU_SUCCESSOR=1), which carries on; a training resumes from its checkpoint
# 3. when the worker is done: drift metrics, the completeness check, and the report, packed into
#    $VLU_STORE/<calibration|main>_cluster.tar.gz (no model weights)
# The resources must match scripts/submit.py's POOLS["large"] (tests/test_completion.py checks it). Do not run
# a submit.py worker (--workers large) on the same run at the same time: one GPU's worth is the cluster cap.
set -uo pipefail
ARGS=("$@")
export VLU_PROJECT_DIR="${VLU_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$VLU_PROJECT_DIR/slurm/env.sh" ] || { echo "submit from the repo root (no slurm/env.sh in $VLU_PROJECT_DIR)"; exit 2; }
source "$VLU_PROJECT_DIR/slurm/env.sh"
cd "$VLU_PROJECT_DIR"

MODE=calibration
for arg in "$@"; do
    case "$arg" in
        --main) MODE=main ;;
        *) echo "unknown option $arg (expected --main)"; exit 2 ;;
    esac
done
if [ "$MODE" = main ]; then
    RUN=goal_s0 PLAN=scripts/main_run.py REPORT=scripts/main_report.py DRIFT=scripts/main_drift.py
else
    RUN=cal PLAN=scripts/calibrate.py REPORT=scripts/calibration_report.py DRIFT=scripts/calibration_drift.py
fi
OUT="$VLU_STORE/${MODE}_cluster"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date)  $MODE: run $RUN  job ${SLURM_JOB_ID:-none} on $(hostname)$([ -n "${VLU_SUCCESSOR:-}" ] && echo ", successor")  output $OUT"

package() {
    "$MAIN_PY" "$REPORT" --run "$RUN" --out "$OUT" || echo "WARNING: report failed"
    tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
    echo "=== $(date)  done: $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
}

# 1. Environment and checks ---------------------------------------------------------------------------
{
    echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  date $(date)  commit $(git rev-parse HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo -dirty)  render $VLU_RENDER"
    nvidia-smi -L
    echo "cpus $VLU_CPUS"; free -g; df -h "$VLU_STORE"
} >> "$OUT/environment.txt" 2>&1
if [ -z "${VLU_SUCCESSOR:-}" ]; then
    # flash-attn's kernels on this GPU (Blackwell, sm_120): a missing kernel image fails here, not hours later
    if ! "$MAIN_PY" - <<'PY' >> "$OUT/environment.txt" 2>&1; then
import torch, flash_attn, transformers
from flash_attn import flash_attn_func
print("torch", torch.__version__, "cuda", torch.version.cuda, "flash_attn", flash_attn.__version__,
      "transformers", transformers.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
q, k, v = (torch.randn(2, 64, 8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
out = flash_attn_func(q, k, v)
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
err = float((out.float() - ref.float()).abs().max())
assert err < 2e-2, f"flash-attn differs from PyTorch's attention by {err}"
print(f"flash-attn runs on this GPU (max difference to PyTorch's attention {err:.1e})")
PY
        echo "ERROR: flash-attn does not run on this GPU (see $OUT/environment.txt)"; tail -5 "$OUT/environment.txt"; exit 1
    fi
    tail -2 "$OUT/environment.txt"
    if ! bash slurm/check.sbatch; then
        echo "ERROR: a check failed (above); not starting the run"; exit 1
    fi
fi

# 2. The plan and the worker ---------------------------------------------------------------------------
if ! "$MAIN_PY" "$PLAN" --run "$RUN"; then echo "ERROR: $MODE planning failed"; exit 1; fi
RETRY=$([ -z "${VLU_SUCCESSOR:-}" ] && echo --retry-failed)  # a successor does not retry: a failure would repeat each launch
# shellcheck disable=SC2086  # RETRY is empty or one option
if ! "$MAIN_PY" scripts/submit.py --run "$RUN" $RETRY --workers none; then echo "ERROR: writing the plan failed"; exit 1; fi

queue_successor() {
    echo "=== $(date)  time limit is near: queueing a successor"
    sbatch --dependency="afterany:$SLURM_JOB_ID" --export="ALL,VLU_SUCCESSOR=1" "$VLU_PROJECT_DIR/slurm/run.sbatch" "${ARGS[@]}" ||
        echo "could not queue a successor: submit again with  sbatch slurm/run.sbatch ${ARGS[*]}"
}
trap queue_successor USR1
"$MAIN_PY" -m vlu.worker --run "$RUN" --kinds train,eval,features &
WORKER=$!
while true; do   # a signal interrupts `wait`; keep waiting until the worker has really exited
    wait "$WORKER"; WORKER_STATUS=$?
    kill -0 "$WORKER" 2>/dev/null || break
done

# 3. Drift metrics, completeness, report (as scripts/run_local.sh step 6) ---------------------------------
"$MAIN_PY" scripts/submit.py --run "$RUN" --workers none
OMP_NUM_THREADS="$VLU_CPUS" "$MAIN_PY" "$DRIFT" --run "$RUN"
DRIFT_STATUS=$?
if [ "$DRIFT_STATUS" != 0 ]; then
    echo "WARNING: some drift metrics are missing (see summary.md; logs in $VLU_RUNS_DIR/$RUN/drift_report/, packed as drift/)"
fi
if ! "$MAIN_PY" -m vlu.worker --run "$RUN" --check-complete; then
    echo "ERROR: the experiment did not complete every planned item"; package; exit 1
fi
if [ "$WORKER_STATUS" != 0 ]; then echo "ERROR: the worker exited unsuccessfully"; package; exit 1; fi
package
