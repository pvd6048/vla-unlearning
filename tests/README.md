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
`python tests/test_sharp_minmax.py` checks literal masked-g1 blends, cross-partition forget updates,
SAM/ASAM perturbations, current weights under bf16 autocast (CPU or CUDA), restored temporary weights,
paired randomness, BatchNorm buffers, full-pass saved masks, independent native optimizer counters
and exact resume. It also trains the methods on the GR00T-shaped CPU fake.
`python tests/test_saliency.py` checks exact ranks, tie counts, unused embodiment rows and full data coverage.
`python tests/test_gu_npo_rt.py` checks whitened projections, sign-selective caps, basis growth,
Algorithm 1 recovery using the current anchor between refreshes, weighted empty-basis behavior,
projected tensors and exact resume. See [PAPER_ALIGNMENT.md](../PAPER_ALIGNMENT.md) for the optional
pinned-author source comparisons and their limits.
These run on CPU and are included in the local runner (Slurm's check already discovers all tests).

`python tests/test_task_arithmetic_rmu.py` checks task-vector negation against an independent
forget-task fit, RMU's two-step reference objective, action masks, restricted update scope, reference
inputs, hook cleanup, that RMU's forwards stop after the target block (and its logged hidden-state norm),
and deterministic stochastic-resume behavior, including interrupted finalization.
`python tests/test_method_benchmarks.py` checks benchmark variants, S1/oracle/drift planning, every
stream metric, report generation and efficiency exports on synthetic fixtures. On Windows its
fcntl import stand-in is only for pure planner/report helpers; worker locking is not tested.
Both run on CPU; set `USE_TF=0` if optional TensorFlow imports slow down transformers startup.

`python tests/test_cuda_round3.py` runs every round-3 calibration variant of GU on NPO-RT, Sharp MinMax, task
arithmetic and RMU through `vlu/run_op.py`'s preparation, training and saving on CUDA, with bf16 autocast,
DataLoader workers and the cost meter, on the GR00T-shaped fake (Sharp's SGD rates scaled to the fake's
gradients); skipped without CUDA.

Original-source checks: `check_task_vectors.py` (optional) compares against the authors' arithmetic,
and `check_groot_rmu.py` compares RMU with a full frozen reference using GR00T's actual DiT blocks
on CPU (diffusers 0.35.1 required). Without an argument it uses the pinned GR00T checkout's `dit.py`
(`$GR00T_DIR`), so `scripts/run_local.sh` runs it as a self-test; without a GR00T environment it reports
a skip. Source downloads, commit pins, hashes and audit scope are
documented in [TASK_ARITHMETIC_RMU.md](../TASK_ARITHMETIC_RMU.md). These checks need no model weights
or LIBERO data and are not rollout or GPU-cost benchmarks.
