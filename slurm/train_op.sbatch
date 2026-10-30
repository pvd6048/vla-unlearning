#!/bin/bash
# Train one node. Usually submitted by scripts/submit.py, which sets the resources.
#   sbatch <resources> slurm/train_op.sbatch <run> <node path> [extra vlu.run_op args]
# Rerunning (e.g. after a time-out) resumes from the node's resume.pt.
set -euo pipefail
source "${VLU_PROJECT_DIR:?export VLU_PROJECT_DIR (scripts/submit.py does)}/slurm/env.sh"

RUN="$1"; NODE="$2"; shift 2
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  run $RUN  node $NODE"
nvidia-smi -L || true
exec "$MAIN_PY" -m vlu.run_op --run "$RUN" --node "$NODE" "$@"
