#!/bin/bash
# LIBERO evaluation of one node: a policy server on the GPU plus parallel simulator workers.
# Run by vlu/worker.py inside a worker job (or by hand inside any allocation with a GPU):
#   bash slurm/eval.sbatch <run> <node path|""> <tasks e.g. A,B> [weights=final] [seed=0] [videos=0]
# Results: <node dir>/eval/<weights>_seed<seed>/<task>.json. Finished tasks are skipped on rerun.
# Set EPISODES=<n> in the environment to run fewer than the recipe's 50 episodes per task.
set -euo pipefail
source "${VLU_PROJECT_DIR:?export VLU_PROJECT_DIR (scripts/submit.py does)}/slurm/env.sh"

RUN="$1"; NODE="$2"; TASKS="$3"; WEIGHTS="${4:-final}"; SEED="${5:-0}"; VIDEOS="${6:-0}"
JOB="${SLURM_JOB_ID:-none}"
PORT=$("$MAIN_PY" -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")  # a free port
WORKERS="${VLU_EVAL_WORKERS:-$(( $(nproc) - 1 ))}"   # CPUs of this allocation; one stays with the policy server
# The simulator client needs no CUDA. robosuite renders on EGL device $CUDA_VISIBLE_DEVICES unless
# MUJOCO_EGL_DEVICE_ID is set, so give it its render device explicitly: Mesa has one device (0); with
# VLU_RENDER=gpu, render on this job's GPU.
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
if [ "$VLU_RENDER" = gpu ] && [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then RENDER_DEVICE="$GPU_ID"; else RENDER_DEVICE=0; fi
echo "host $(hostname)  job $JOB  run $RUN  node ${NODE:-<base>}  tasks $TASKS  weights $WEIGHTS  seed $SEED  port $PORT  workers $WORKERS  render $VLU_RENDER:$RENDER_DEVICE"

"$MAIN_PY" -m vlu.eval.serve --run "$RUN" --node "$NODE" --weights "$WEIGHTS" --port "$PORT" --seed "$SEED" &
SERVER=$!
CUDA_VISIBLE_DEVICES="" MUJOCO_EGL_DEVICE_ID="$RENDER_DEVICE" "$SIM_PY" "$VLU_PROJECT_DIR/vlu/eval/libero_client.py" --port "$PORT" --run "$RUN" --node "$NODE" \
    --weights "$WEIGHTS" --seed "$SEED" --tasks "$TASKS" --workers "$WORKERS" --videos "$VIDEOS" ${EPISODES:+--episodes "$EPISODES"} &
CLIENT=$!

# Whichever process exits first decides: the client finishing is success; the server dying is a failure
# (the client would otherwise wait for it forever).
set +e
wait -n "$SERVER" "$CLIENT"
STATUS=$?
if kill -0 "$CLIENT" 2>/dev/null; then
    echo "policy server exited first (status $STATUS); stopping the client" >&2
    pkill -P "$CLIENT"; kill "$CLIENT"; wait "$CLIENT"   # also its simulator processes
    exit 1
fi
kill "$SERVER" 2>/dev/null; wait "$SERVER"   # free the GPU before the worker's next item
exit "$STATUS"
