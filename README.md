# vla-unlearning

Forgetting vs. unlearning in VLAs: GR00T N1.6 on LIBERO-Goal. The research plan is [PLAN.md](PLAN.md).
This file explains how to run things. Everything runs on the Slurm cluster.

## Layout

```
vlu/                 the experiment code (python package)
  config.py          paths + the frozen recipe (steps, lr, batch, eval protocol). Change defaults here.
  tasks.py           A–E -> LIBERO-Goal tasks, and the op streams (S1, ...)
  nodes.py           op syntax, the run tree on disk, oracle construction
  model.py           load GR00T / build its processor (mirrors NVIDIA's fine-tune), save/load weights
  data.py            demos per task in RAM, batch composition, deterministic per-step batches
  ops.py             what each op trains on + its loss: L ops (ER) and unlearning methods
  train.py           the training loop (AdamW + cosine, bf16, resumable)
  run_op.py          CLI: train one node
  worker.py          runs a run's plan item by item inside one Slurm job (see "Jobs" below)
  eval/serve.py      CLI: serve a node's policy (GPU, main env)
  eval/libero_client.py  CLI: LIBERO rollouts against the server (CPU, simulator env)
  drift/             drift metrics (PLAN §4): params.py (weights), extract.py (GPU: probe features,
                     Fisher), compare.py (representation + policy drift); see drift/__init__.py
slurm/               env.sh (paths, environments), worker.sbatch, eval.sbatch, smoke.sbatch, train_op.sbatch
scripts/             setup_cluster.sh, prepare_data.py, submit.py (plan + workers), results.py (tables)
```

## Concepts

- **Op**: `LB` = learn task B; `UB-ga` = unlearn task B with gradient ascent (`UB-purge`, `UB-graddiff`, ...).
- **Node**: the model after a sequence of ops, stored at `runs/<run>/<op>/<op>/...`. Shared prefixes are
  trained once and branches sit next to each other:
  ```
  runs/goal_s0/LA/LB/LC/                        prefix shared by all methods
  runs/goal_s0/LA/LB/LC/UB-ga/LD/LE/UE-ga/      the GA tail of S1
  runs/goal_s0/LA/LC/LD/                        retrain oracle of S1 (never saw B or E)
  ```
  A node is done when its `weights.safetensors` exists. Weights hold only the trainable
  parameters (fp32); the frozen rest is always the base checkpoint.
- **Run**: one seed of everything (`goal_s0`, `goal_s1`, ...). The seed is fixed in `runs/<run>/run.json`.
- **Replay / retain set** of a node = tasks learned and not unlearned since. L ops replay them (ER,
  half of every batch); U ops use them as the retain set. Unlearned tasks are never replayed.
- **Normalization statistics** are pinned to the full LIBERO-Goal dataset for every op, so they
  never change between ops or branches.

Every job is safe to rerun: done nodes and finished evals are skipped, interrupted training resumes
from `resume.pt` (saved every 1000 steps), interrupted evals redo only unfinished tasks.

## One-time setup (login node)

```bash
git clone https://github.com/pvd6048/vla-unlearning.git && cd vla-unlearning
echo 'export VLU_STORE=/path/to/big/disk/vla-unlearning' > slurm/env.local.sh   # ~30 GB + runs
bash scripts/setup_cluster.sh
```

Things already on the cluster can be reused by adding to `slurm/env.local.sh` (see `slurm/env.sh`):
`VLU_HF_HOME` for an HF cache that already holds GR00T-N1.6-3B (7 GB), and `VLU_GOAL_SOURCE` for an
existing copy of `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` (only 0.4 GB, so downloading is
also fine; the copy is completed and patched, the source is not modified). Other LIBERO formats
(raw HDF5, openpi's `physical-intelligence/libero`) are not usable.

The script pins Isaac-GR00T, builds two Python environments, downloads the model and data, and writes
`$VLU_STORE/data/goal_report.json` (demos per task, dims, action stats) plus PNGs of task A's first
frames. Check the report before going further.

### The two environments

GR00T and LIBERO need incompatible packages, so (as in NVIDIA's own LIBERO evaluation) training and
the policy server run in one environment and the simulator in another; they talk over ZMQ on localhost.

| | main env (`$MAIN_PY`) | simulator env (`$SIM_PY`) |
|---|---|---|
| used by | `vlu.run_op`, `vlu.eval.serve`, `scripts/prepare_data.py` | `vlu/eval/libero_client.py` only |
| source | GR00T's `uv.lock` at the pinned commit (exact) | NVIDIA's `setup_libero.sh` steps, resolved as of the commit date |
| Python | 3.10 | 3.10 |
| key packages | torch 2.7.1+cu128, flash-attn 2.7.4.post1, transformers 4.51.3, numpy 1.26.4, av 16.1.0, pyzmq 27.0.1 | LIBERO, robosuite 1.4.0, MuJoCo, torch 2.5.1 (CPU use only), numpy 1.26.4, msgpack 1.1.0 |
| GPU | yes | no (software rendering, Mesa EGL) |

Both get GR00T installed as an editable package; our code is found via `PYTHONPATH` (set in
`slurm/env.sh`). Exact installed versions are written to `$VLU_STORE/envs/`. Left out of the main env
on purpose: deepspeed and TensorRT (unused), and NVIDIA's `sudo apt` step. Videos are decoded with
PyAV, whose wheel bundles FFmpeg and the AV1 decoder the dataset needs, so no system FFmpeg is required.
Do not `pip install` into either env by hand; change `scripts/setup_cluster.sh` instead.

Known risks (the smoke test shows which ones bite):
- flash-attn on Blackwell: the pinned wheel contains sm_120 kernels, but this has not been run on the cluster.
- MuJoCo rendering through Mesa EGL on the MIG nodes (worked for an earlier project on gpu01).
- The simulator env builds a few packages from source (e.g. `evdev` via robosuite → pynput), which
  needs a C compiler and kernel headers on the login node.

Slurm resources for training and eval jobs are in `RESOURCES` at the top of `scripts/submit.py`.

## Day 1: smoke test

```bash
export VLU_PROJECT_DIR=$PWD
sbatch --partition=gpu_large --qos=gpu_large --gres=gpu:4g.96gb:1 --cpus-per-gpu=24 --mem-per-gpu=96G \
    --time=2:00:00 --output=smoke-%j.out slurm/smoke.sbatch
```

200 training steps of L_A in the throw-away run `smoke`, then 5 rollouts with 2 videos. From the
output take: `steps_per_s` and `peak_mem_gb` (train_log.jsonl), seconds per rollout (eval json),
and look at the videos in `runs/smoke/LA/eval/final_seed0/videos/`.

## Jobs: the plan and the workers

Cluster limits per user: gpu_large 1 running / 2 submitted jobs (48 h max), gpu_medium 1 / 4,
gpu_small 4 / 20 (24 h max), and at most one GPU's worth of slices at a time (4 small, or 1
medium, or 1 large). So work runs one item at a time, and one Slurm job per item is not possible.

`scripts/submit.py` therefore only (1) adds items to the run's plan, `runs/<run>/plan.json`, and
(2) starts a worker job if none is queued or running. The worker (`vlu/worker.py`, on gpu_large by
default) runs the plan in order: a node's training as soon as its parent is trained, then the node's
evals (on the same slice, 23 simulator processes). 15 minutes before its time limit it queues its own
successor, which carries on; a training cut off by the time limit resumes from its last checkpoint.
Rerunning `submit.py` is always safe, and `python scripts/submit.py --run <run>` alone prints progress.

Files a run directory gets besides the node tree: `plan.json`, `_logs/<item>.<job>.log` (output of
each training/eval), `_slurm/worker-*.out` (the workers' own logs), `_failed/<item>.json` (failed items
are skipped, and so are the items depending on them, until `--retry-failed`) and `_claims/` (items
being worked on; a claim of a dead job is taken over automatically).

`--workers small:4` adds eval-only workers on gpu_small (e.g. to catch up on evals while no large job
is running); `--workers none` only updates the plan.

## Day 2: pipeline sanity check on task A

```bash
python scripts/submit.py --run goal_s0 --seed 0 --zero-shot A --nodes LA --save-at 2000,5000 \
    --eval-weights step_002000,step_005000,final --eval-seeds 0,1 --videos 5
python scripts/submit.py --run goal_s0                      # progress
python scripts/results.py --run goal_s0 --weights final --seed 0
```

Pass bar (PLAN.md): ≥ 80% SR on A after 10k steps. The `--eval-seeds 0,1` pair gives the eval noise.
`train_log.jsonl` also has a fixed-noise flow-matching "probe" loss for every task A–E every 500 steps.

## Week 1: stream S1

```bash
python scripts/submit.py --run goal_s0 --stream S1 --methods purge,ga,graddiff,npo,ssd,salun --dry-run   # look first
python scripts/submit.py --run goal_s0 --stream S1 --methods purge,ga,graddiff,npo,ssd,salun
python scripts/results.py --run goal_s0
```

This plans the shared prefix once, every method's tail, the oracles (`LA/LC`, `LA/LC/LD`), and an
eval after every op on every task seen so far.

## Drift metrics

PLAN §5's branches are nodes of the tree (target task B, θ₀ = `LA/LB/LC`):

| branch | node | drift |
|---|---|---|
| C, control (B still replayed) | `LA/LB/LC/LD/LE` | |
| F, forgetting (B dropped from replay) | `LA/LB/LC/UB-purge/LD/LE` | Δ_F\|B = `LA/LB/LC/LD/LE..LA/LB/LC/UB-purge/LD/LE` |
| U, unlearning | `LA/LB/LC/UB-ga` | Δ_U = `LA/LB/LC..LA/LB/LC/UB-ga` |
| UF | `LA/LB/LC/UB-ga/LD/LE` | |
| task vector τ_B | | `LA..LA/LB` |

1. Train what is missing and extract probe features (and a Fisher where needed) inside the workers:
   ```bash
   python scripts/submit.py --run goal_s0 --nodes LA/LB/LC/LD/LE --stream S1 --methods purge,ga --features --fisher LA/LB/LC:B
   ```
2. Compare (main env; `params.py` streams the weight files tensor by tensor, `compare.py` reads small files):
   ```bash
   source slurm/env.sh
   $MAIN_PY -m vlu.drift.params --run goal_s0 --random --subspace-k 8 --fisher "FB=LA/LB/LC:B" \
       --drift "U=LA/LB/LC..LA/LB/LC/UB-ga" --drift "FB=LA/LB/LC/LD/LE..LA/LB/LC/UB-purge/LD/LE" \
       --drift "tauB=LA..LA/LB" --drift "U_s1=goal_s1:LA/LB/LC..goal_s1:LA/LB/LC/UB-ga" \
       --out $VLU_RUNS_DIR/goal_s0/_analysis/params_B.csv
   $MAIN_PY -m vlu.drift.compare --run goal_s0 \
       --drift "U=LA/LB/LC..LA/LB/LC/UB-ga" --drift "FB=LA/LB/LC/LD/LE..LA/LB/LC/UB-purge/LD/LE" \
       --out $VLU_RUNS_DIR/goal_s0/_analysis/rep_B.csv
   ```
   Outputs are tidy CSVs (one value per row: level/group or feature/task, metric, drift pair).
   A cosine only means something next to its controls: `random` (unrelated direction) and the
   seed twin (`U` vs `U_s1`: the best agreement two runs of the same op reach).

## Adding an unlearning method

Write `_plan_<name>(node, recipe) -> OpPlan` in `vlu/ops.py` (batch parts + loss) and add it to
`UNLEARN_METHODS`. The op token is then `U<task>-<name>`. A method that needs a one-off computation on
the parent model (a reference copy, importances, a weight mask) gives its plan a `prepare(ctx)`;
`OpPlan.grad_mask` restricts which weights may change. See NPO, SSD and SalUn there.

How the methods map onto GR00T's flow-matching loss (PROVISIONAL values in `vlu/config.py`):
- **NPO**: log-likelihood of a chunk := −(sum of squared flow-matching errors); θ and the frozen
  reference see the same noise and timesteps; β = 0.1.
- **SSD**: diagonal Fisher of L_FM gradients on the forget task vs. on all the parent's tasks; α = 10, λ = 1; no training steps.
- **SalUn**: saliency = |∇L_FM| on the forget task, top 50% of trainable weights updated; "random
  label" = the action chunk of a random retained-task frame; loss = relabeled forget + retain.

## Open decisions (provisional values in the code)

- A–E task mapping (`vlu/tasks.py`): currently LIBERO-Goal tasks 0–4.
- Unlearning budget and lr (`vlu/config.py`): 1000 steps at 1e-5 for every method.
- GA stopping rule / loss ceiling; NPO's β (depends on the scale of L_FM, see the probe losses); SSD's α, λ.
- ER: total batch 16 (8 new + 8 replay), replay from all demos of retained tasks.
