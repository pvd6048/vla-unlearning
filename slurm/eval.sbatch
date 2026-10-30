#!/bin/bash
# LIBERO evaluation of one node: a policy server on the GPU plus parallel simulator workers.
#   sbatch <resources> slurm/eval.sbatch <run> <node path|""> <tasks e.g. A,B> [weights=final] [seed=0] [videos=0]
# Results: <node dir>/eval/<weights>_seed<seed>/<task>.json. Finished tasks are skipped on rerun.
# Set EPISODES=<n> in the environment to run fewer than the recipe's 50 episodes per task.
set -euo pipefail
source "${VLU_PROJECT_DIR:?export VLU_PROJECT_DIR (scripts/submit.py does)}/slurm/env.sh"

RUN="$1"; NODE="$2"; TASKS="$3"; WEIGHTS="${4:-final}"; SEED="${5:-0}"; VIDEOS="${6:-0}"
JOB="${SLURM_JOB_ID:-$$}"
PORT=$(( 20000 + JOB % 20000 ))
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-6} - 1 ))   # one core stays with the policy server
echo "host $(hostname)  job $JOB  run $RUN  node ${NODE:-<base>}  tasks $TASKS  weights $WEIGHTS  seed $SEED  port $PORT  workers $WORKERS"

"$MAIN_PY" -m vlu.eval.serve --run "$RUN" --node "$NODE" --weights "$WEIGHTS" --port "$PORT" --seed "$SEED" &
SERVER=$!
"$SIM_PY" "$VLU_PROJECT_DIR/vlu/eval/libero_client.py" --port "$PORT" --run "$RUN" --node "$NODE" \
    --weights "$WEIGHTS" --seed "$SEED" --tasks "$TASKS" --workers "$WORKERS" --videos "$VIDEOS" ${EPISODES:+--episodes "$EPISODES"} &
CLIENT=$!

# Whichever process exits first decides: the client finishing is success; the server dying is a failure
# (the client would otherwise wait for it forever).
set +e
wait -n "$SERVER" "$CLIENT"
STATUS=$?
if kill -0 "$CLIENT" 2>/dev/null; then
    echo "policy server exited first (status $STATUS); stopping the client" >&2
    kill "$CLIENT"; exit 1
fi
kill "$SERVER" 2>/dev/null
exit "$STATUS"
