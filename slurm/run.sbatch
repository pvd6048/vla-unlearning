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
# 1. first launch only: a flash-attn kernel test on this GPU, the base tests and the LIBERO tasks
#    (slurm/check.sbatch; the calibration needs only A and B usable); the plan; earlier failures are tried again
# 2. one worker runs the plan (train, eval, features); then the drift metrics, the completeness check and the
#    report, packed into $VLU_STORE/<calibration|main>_cluster.tar.gz (no model weights)
# 15 minutes before the time limit (Slurm's USR1) this job queues its own successor (same options,
# VLU_SUCCESSOR=1), which carries on where it stops: a training resumes from its checkpoint, finished items
# are skipped, the drift metrics are computed again. If this job finishes before the limit after all, it
# cancels the successor. Step 2 runs in the background so that the signal is handled at once (bash runs a
# trap only between foreground commands).
# The resources must match scripts/submit.py's POOLS["large"] (tests/test_completion.py checks it). Do not run
# a submit.py worker (--workers large) on the same run at the same time: one GPU's worth is the cluster cap.
set -uo pipefail
ARGS=("$@")
export VLU_PROJECT_DIR="${VLU_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$VLU_PROJECT_DIR/slurm/env.sh" ] || { echo "submit from the repo root (no slurm/env.sh in $VLU_PROJECT_DIR)"; exit 2; }
source "$VLU_PROJECT_DIR/slurm/env.sh"
source "$VLU_PROJECT_DIR/slurm/lib.sh"
cd "$VLU_PROJECT_DIR"

MODE=calibration
for arg in "$@"; do
    case "$arg" in
        --main) MODE=main ;;
        *) echo "unknown option $arg (expected --main)"; exit 2 ;;
    esac
done
unset RUN  # a RUN from the submitting shell's environment would pick another run
select_mode "$MODE"
OUT="$VLU_STORE/${MODE}_cluster"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date)  $MODE: run $RUN  job ${SLURM_JOB_ID:-none} on $(hostname)$([ -n "${VLU_SUCCESSOR:-}" ] && echo ", successor")  output $OUT"
fail() { echo "ERROR: $1"; package; exit 1; }

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
        tail -5 "$OUT/environment.txt"; fail "flash-attn does not run on this GPU (see $OUT/environment.txt)"
    fi
    tail -2 "$OUT/environment.txt"
    bash slurm/check.sbatch ${REQUIRED_TASKS:+--required "$REQUIRED_TASKS"} || fail "a check failed (above); not starting the run"
fi

# 2. The plan, the worker, drift metrics, completeness, report (scripts/run_local.sh steps 4-6) -----------
"$MAIN_PY" "$PLAN" --run "$RUN" || fail "$MODE planning failed"
RETRY=$([ -z "${VLU_SUCCESSOR:-}" ] && echo --retry-failed)  # a successor does not retry: a failure would repeat each launch
# shellcheck disable=SC2086  # RETRY is empty or one option
"$MAIN_PY" scripts/submit.py --run "$RUN" $RETRY --workers none || fail "writing the plan failed"

SUCCESSOR=
queue_successor() {
    echo "=== $(date)  time limit is near: queueing a successor"
    local queued
    if queued=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" --export="ALL,VLU_SUCCESSOR=1" \
                "$VLU_PROJECT_DIR/slurm/run.sbatch" "${ARGS[@]}"); then
        SUCCESSOR=${queued%%;*}  # --parsable prints <job id>[;<cluster>]
        echo "successor: job $SUCCESSOR"
    else
        echo "could not queue a successor: submit again with  sbatch slurm/run.sbatch ${ARGS[*]}"
    fi
}
trap queue_successor USR1
work() {
    local worker_ok=0
    "$MAIN_PY" -m vlu.worker --run "$RUN" --kinds train,eval,features && worker_ok=1
    finish_run 0 "$worker_ok"
}
work &
wait_for $!
STATUS=$?
trap '' USR1  # done: a late signal must not queue a successor now
if [ -n "$SUCCESSOR" ]; then
    echo "finished before the time limit: cancelling the successor (job $SUCCESSOR)"
    scancel "$SUCCESSOR" || echo "WARNING: could not cancel job $SUCCESSOR (it would only repeat the drift metrics and the report)"
fi
exit "$STATUS"
