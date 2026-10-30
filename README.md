# VLA Unlearning

Continual learning + unlearning on NVIDIA GR00T N1.6 (LIBERO). Research plan: [PLAN.md](PLAN.md).

NVIDIA's Isaac-GR00T is a pinned, **unmodified** submodule (`third_party/Isaac-GR00T`, tag `n1.6.1-release`).
Their code does model loading, data loading, training (L ops) and LIBERO evaluation; ours only adds
the unlearning methods, the stream runner and metrics.

## Setup (once, on the GPU server)

```bash
git clone --recursive https://github.com/pvd6048/vla-unlearning.git && cd vla-unlearning
sudo apt install libegl1-mesa-dev libglu1-mesa      # LIBERO rendering
bash scripts/setup_server.sh                         # GR00T env, LIBERO-Goal data, per-task split, simulator
source third_party/Isaac-GR00T/.venv/bin/activate
```

Then fix which LIBERO-Goal tasks are A–E: look at `data/goal/tasks.json` and edit `tasks:` in `configs/streams/s1*.yaml`.

## Run

```bash
python run.py --stream smoke --method ga            # whole pipeline in minutes; numbers meaningless
PYTHONPATH=. python tests/test_groot.py <ckpt> data/goal/tasks/task_0   # wrapper == NVIDIA's loss

python run.py --stream s1_oracle --method none      # retrain oracle (never learns B, E)
for m in purge ga graddiff npo ssd salun; do python run.py --stream s1 --method $m --seed 0; done
python -m vla_unlearning.metrics s1                 # methods x metrics table
```

Each run writes `results/<stream>/<method>/seed<k>/matrix.csv`: success rate of every task after every op.
Checkpoints are shared by op prefix under `results/ckpt/`, so `L_A L_B L_C` is trained once for all methods.
Re-running skips finished ops and evaluations.

## Code

| File | What it does |
|---|---|
| `vla_unlearning/groot.py` | The only GR00T-facing module: config (same as NVIDIA's launcher), loading, per-sample loss, saving |
| `vla_unlearning/learn.py` | L op: NVIDIA's training with a 1:1 new-task / replay mixture |
| `vla_unlearning/unlearn.py` | U op: load, run a method, save an NVIDIA-format checkpoint |
| `vla_unlearning/methods/` | One file per method (`ga`, `graddiff`, `npo`, `ssd`, `salun`) + the shared loop |
| `vla_unlearning/evaluate.py` | NVIDIA's LIBERO server + rollout client → success rate |
| `vla_unlearning/runner.py` | Runs a stream of ops, writes the success matrix |
| `vla_unlearning/metrics.py` | NBT, collateral, relapse, oracle gap, score, … from success matrices |
| `configs/streams/`, `configs/methods/` | Experiments as YAML. To try other hyperparameters, copy a method file under a new name. |

**Adding a method:** write `vla_unlearning/methods/<name>.py` with `unlearn(model, ref, data, cfg)`,
register it in `methods/__init__.py`, add `configs/methods/<name>.yaml`.
