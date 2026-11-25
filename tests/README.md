# Base-level tests

Known-answer checks of the code that can be tested without GR00T, its data or a cluster GPU: each
test builds small fake models with GR00T N1.6's module layout (`stubs.py` stands in for the GR00T
packages and makes a temporary runs/base directory; `fakes.py` holds the fake model and data). They need torch with CUDA, safetensors,
transformers and numpy, and take a minute on a laptop GPU. Run them after changing the files they cover:

```bash
python tests/test_drift_params.py    # vlu/drift/{metrics,weights,params}.py
python tests/test_drift_extract.py   # vlu/drift/{extract,compare}.py
python tests/test_ops.py             # NPO/NPO-RT/SSD/SalUn plans (vlu/ops.py), learning rates per method, vlu/train.py with a mask, vlu/run_op.py checks
python tests/test_vlaforget.py       # vlu/{vlaforget,lora}.py, frozen overrides in vlu/model.py
python tests/test_stream_metrics.py  # vlu/stream_metrics.py (plain Python, no torch needed)
python tests/test_calibration.py     # local workers (no Slurm), calibration plan and report (plain Python)
python tests/test_calibration_drift.py # scripts/calibration_drift.py: distances to the oracle, restricted pairs, diverged checkpoints, SVD failures, failed tools
python tests/test_completion.py      # workers and run_local.sh fail on failed / unfinished required items (vlu/worker.py); run_local.sh options
python tests/test_main_run.py        # main run: plan of S1 + oracles + control branch, report (plain Python)
python tests/test_main_drift.py      # scripts/main_drift.py: forgetting vs. unlearning drifts against direct computation
python tests/test_resume.py          # per-step preprocessing randomness, interrupted training continues identically
python tests/test_training_safety.py # non-finite losses, gradients, weights and checkpoints are rejected
python tests/test_rank_subspace.py   # drift subspaces of rank-deficient matrices (vlu/drift/metrics.py)
python tests/test_cost.py            # op cost meter (vlu/cost.py): phases, peak memory, FLOPs, steps and waits, energy source per GPU kind, idle, flags, aggregate
```

What they cannot check: anything that needs real GR00T weights, the dataset or the simulator. That is
what the cluster smoke test is for.

`python tests/test_npo.py` runs on CPU: NPO/NPO-RT formula and gradient checks, invalid-temperature
rejection, fixed reference, isolated random-state replay, and interrupted/resumed updates. Its optional
non-current-device RNG check is skipped unless two CUDA devices are available. These tests use the
actual operation plans; only GR00T imports and the training loop's CUDA/worker plumbing are stubbed.

`python tests/test_gu.py` checks metric projections, actual AdamW displacements and exact resume.
`python tests/test_sharp_minmax.py` checks sharp min/max signs, partition isolation, ASAM perturbations,
temporary-weight restoration, paired randomness, saved saliency masks and both-optimizer resume.
It also trains both new plans with probes/snapshots on the GR00T-shaped fake under CPU plumbing.
These run on CPU and are included in the local runner (Slurm's check already discovers all tests).

`python tests/test_task_arithmetic_rmu.py` checks task-vector negation against an independent
forget-task fit, RMU's two-step reference objective, action masks, restricted update scope, reference
inputs, hook cleanup and deterministic stochastic-resume behavior, including interrupted finalization.
`python tests/test_method_benchmarks.py` checks benchmark variants, S1/oracle/drift planning, every
stream metric, report generation and efficiency exports on synthetic fixtures. On Windows its
fcntl import stand-in is only for pure planner/report helpers; worker locking is not tested.
Both run on CPU; set `USE_TF=0` if optional TensorFlow imports slow down transformers startup.

Optional original-source checks: `check_task_vectors.py` compares against the authors' arithmetic,
and `check_groot_rmu.py` compares RMU with a full frozen reference using GR00T's actual DiT blocks
on CPU (diffusers 0.35.1 required). Source downloads, commit pins, hashes and audit scope are
documented in [TASK_ARITHMETIC_RMU.md](../TASK_ARITHMETIC_RMU.md). These checks need no model weights
or LIBERO data and are not rollout or GPU-cost benchmarks.
