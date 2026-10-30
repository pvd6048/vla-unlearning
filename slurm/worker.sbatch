#!/bin/bash
# A worker job: runs the planned items of a run until none is left (see vlu/worker.py).
# Submitted by scripts/submit.py, which sets the resources and --signal=B:USR1@900: 15 minutes before
# the time limit this job queues its own successor (same resources, VLU_WORKER_SBATCH), which carries
# on with the plan. A training cut off by the time limit resumes from its last checkpoint.
#   sbatch <resources> slurm/worker.sbatch <run> <kinds, e.g. train,eval>
set -euo pipefail
source "${VLU_PROJECT_DIR:?export VLU_PROJECT_DIR (scripts/submit.py does)}/slurm/env.sh"

RUN="$1"; KINDS="$2"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  run $RUN  kinds $KINDS  cpus $(nproc)"
nvidia-smi -L || true

queue_successor() {
    echo "time limit is near: queueing a successor worker"
    # shellcheck disable=SC2086  # VLU_WORKER_SBATCH is a list of sbatch options
    sbatch ${VLU_WORKER_SBATCH:?} --dependency="afterany:$SLURM_JOB_ID" \
        "$VLU_PROJECT_DIR/slurm/worker.sbatch" "$RUN" "$KINDS" ||
        echo "could not queue a successor: rerun scripts/submit.py --run $RUN"
}
trap queue_successor USR1

"$MAIN_PY" -m vlu.worker --run "$RUN" --kinds "$KINDS" &
WORKER=$!
set +e
while true; do   # a signal interrupts `wait`; keep waiting until the worker has really exited
    wait "$WORKER"; STATUS=$?
    kill -0 "$WORKER" 2>/dev/null || break
done
exit "$STATUS"
