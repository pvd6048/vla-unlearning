# Task arithmetic and RMU on GR00T

Implemented and checked on 2026-11-25. Both methods use the normal operation tree, replay exclusion,
checkpoint format, rollout evaluations, fixed FM probes, cost instrumentation and drift tools.
Neither has a calibrated GR00T learning rate: an explicit `unlearn_lr` is required.

## Task arithmetic: forget-task fitting, then vector negation

Reference: Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023;
[author code at 826a64c67082fab0f40628233287948f0f8d7fa3](https://github.com/mlfoundations/task_vectors/blob/826a64c67082fab0f40628233287948f0f8d7fa3/src/task_vectors.py).

`task_arithmetic` starts from the current pre-unlearning checkpoint theta0. It fine-tunes on the
forget task's ordinary FM loss for `unlearn_steps`, yielding theta_f, and publishes:

```
tau = theta_f - theta0
theta_unlearned = theta0 - task_arithmetic_scale * tau
```

This is a freshly fitted forget-task vector, not subtraction of an old continual-learning delta
that also included replay. Scale 0 returns theta0 exactly. Scale 1 extrapolates in the opposite
direction from forget-task fitting; it does not simply restore the checkpoint before that task
was originally learned. The existing baseline trainable scope and AdamW recipe are used. The
FM loss, model, optimizer, schedule and task data differ from the authors' CLIP/text experiments;
the weight arithmetic is the same.

`vlu/task_arithmetic.py` holds the update rule. Its CPU anchor is saved with the optimizer, scheduler
and RNG in `resume.pt`. The shared training loop calls `finalize` before saving final weights or
deleting resume state. Thus an interrupted fit/finalization cannot publish theta_f as the completed
unlearning model. Intermediate weight snapshots are refused for this method because they would
be auxiliary fits. Final probes and rollouts evaluate the negated model. `fit_forget_fm` describes
auxiliary fitting, while `task_vector_norm` and `task_edit_norm` describe the final edit.

**Cost:** all auxiliary training is charged to this method, as is final vector arithmetic and its
CPU/GPU transfer. A CPU anchor costs one extra trainable-weight copy and increases resume-file
size. Checkpoint/save time remains separately recorded under the shared instrumentation convention.
The FLOP counter omits much of the elementwise negation; measured time and energy include it.

## RMU: masked action-token representation steering

Reference: Li et al., *The WMDP Benchmark*, 2024;
[author code at c0b6c12bb0de3decf9c13bb13f9f1aa15754a132](https://github.com/centerforaisafety/wmdp/blob/c0b6c12bb0de3decf9c13bb13f9f1aa15754a132/rmu/unlearn.py).

`rmu` minimizes:

```
MSE_valid_actions(h_theta(forget), steering * u)
    + rmu_alpha * MSE_valid_actions(h_theta(retain), h_theta0(retain))
```

As in the reference, u is a fixed, normalized vector drawn from positive uniform coordinates.
It is sampled independently of the model's RNG stream and saved in resume state, not resampled
per batch. The reference activations are anchored to the checkpoint before this unlearning request.

**GR00T adaptation:** `rmu_layer` selects a DiT transformer block (default 0), and only its already
trainable parameters are updated. This is **RMU-DiT**, not a reproduction of the authors' LLM-layer
selection. The checkpoint still stores the full baseline trainable scope so descendants retain
all upstream/downstream learning. Other parameters receive no gradients and weight decay is disabled.

Only action-token hidden states participate: prefix state tokens and padded action timesteps are
excluded. Both models run without dropout. Since the selected block is the only edited part,
its frozen copy can consume the exact same incoming activations as the learner, including the
same sampled flow noise/timestep and conditioning. This is equivalent to recomputing that block
in a frozen full model with identical inputs and fixed upstream layers, while avoiding a second
full GR00T copy. The reference block, control vector and settings are checkpointed.

Defaults `rmu_alpha=100`, `rmu_steering=20` are starting values inherited from the reference code,
not calibrated robot settings. There is no FM ascent/descent term in RMU's objective: forget and
retain FM losses are logged as diagnostics alongside `forget_rmu`, `retain_rmu` and gradient norm.

## Benchmark entry point and measurements

`scripts/main_run.py --methods` accepts explicit method variants without adding uncalibrated
methods to the default main experiment. It plans the full S1 stream, including later learning,
the second unlearning request, retrain oracles, the purge/control branches when purge is selected,
and the existing feature/Fisher extraction. Reports and drift selection read the actual plan and
preserve override values rather than collapsing different rates/scales into one method name.

Example on a configured Linux GPU host, after the standard setup (rates below are provisional):

```bash
source slurm/env.sh
"$MAIN_PY" scripts/main_run.py --run ta_rmu_s0 --seed 0 \
  --methods 'purge,task_arithmetic@unlearn_lr=1e-5,rmu@unlearn_lr=1e-5'
"$MAIN_PY" scripts/submit.py --run ta_rmu_s0 --workers large:1
```

After workers complete, collect drift and reports:

```bash
"$MAIN_PY" scripts/main_drift.py --run ta_rmu_s0
"$MAIN_PY" scripts/main_report.py --run ta_rmu_s0 --out "$VLU_STORE/ta_rmu_report"
```

Use another run name with `--seed 1` for independent training seeds. `scripts/metrics.py --runs`
can aggregate them. A scale or steering sweep is expressed by comma-separated method specs with
`@task_arithmetic_scale=...` or `@rmu_steering=...` alongside explicit rates. Choose hyperparameters
on calibration tasks before reporting main benchmark results.

| Measurement group | Existing outputs used for both methods |
|---|---|
| Robot performance | Per-task SR/Wilson intervals, retain SR/AUC, forget SR per request, oracle gaps, collateral/spill, relapse, rebound, NBT split by cause, maximum forgetting, stability, combined score |
| Training/probes | Method objectives/diagnostics and fixed task-wise FM probes; TA final probes follow negation |
| Compute/cost | Method/preparation time, step latency, throughput, data wait, FLOPs/step/total, achieved TFLOP/s, peak allocated/reserved memory, GPU energy/above-idle energy/mean power when measurable |
| Drift | Parameter/task-vector/Fisher/subspace comparisons, CKA/PCA features, velocity-field and FM-loss changes, modality alignment/attention ratios and instruction sensitivity |

The headline GPU-minute column now uses the same method-time field as the detailed efficiency
table, excluding probes/checkpoints/final saves; old records without detailed instrumentation
retain their legacy session-time fallback. Time/energy are aggregated over resume sessions.
The existing partial-FLOP and GPU-energy estimation caveats still apply (`vlu/cost.py`).

## Validation and benchmark status

The CPU tests verify task arithmetic against an independently fitted AdamW vector; RMU against
an independent two-step activation objective; padding masks, exact reference inputs, update scope,
hook/mode restoration, checkpointed targets/references, stochastic resume and failure before final
checkpoint publication. Benchmark tests verify S1/oracle/feature planning, distinct variant paths,
all stream-metric calculations, report generation and all three efficiency CSV exports using
explicitly synthetic fixtures. These fixtures are not measurements of robot effectiveness or GPU cost.
Executed checks: 56 passing tests across these checks and the existing GU/Sharp/NPO/resume/safety
regressions, one skipped two-GPU RNG test, plus the hand-computed stream-metric script and Python
compilation checks. The new method tests used CPU torch 2.9.1 and the repository's GR00T-shaped fake.

The current workspace has CPU-only torch, no real checkpoints, no LIBERO data and no configured
GPU host. Per the user's instruction, real rollout performance, GPU runtime/memory/FLOPs/energy,
and real-checkpoint drift benchmarks are **deferred**. No numbers for these are claimed here.

```bash
USE_TF=0 python tests/test_task_arithmetic_rmu.py
USE_TF=0 python tests/test_method_benchmarks.py
```
