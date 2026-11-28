# Task arithmetic and RMU on GR00T

Implemented and checked on 2026-11-25. Both methods use the normal operation tree, replay exclusion,
checkpoint format, rollout evaluations, fixed FM probes, cost instrumentation and drift tools.
Neither has a calibrated GR00T learning rate: an explicit `unlearn_lr` is required.

## Task arithmetic: forget-task fitting, then vector negation

Reference: Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023;
[author code at 826a64c67082fab0f40628233287948f0f8d7fa3](https://github.com/mlfoundations/task_vectors/blob/826a64c67082fab0f40628233287948f0f8d7fa3/src/task_vectors.py).
The unlearning use here, where theta0 already knows the forget task and is fitted on it further before
the negation, is the "task vector" baseline of Eldan & Russinovich (*Who's Harry Potter?*, 2023) and
MUSE (Shi et al., 2024); cite those for the setting and Ilharco et al. for the arithmetic. Under AdamW
the fit moves every weight by about lr per step even when the forget loss is already low, so tau holds
optimizer drift as well as forget-task signal; the rate and scale are calibrated together.

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
not calibrated robot settings. The steering coefficient is the target's per-token norm, so it only
means something relative to the block's activations: each step logs `hidden_norm` (the mean per-token
norm of the forget batch's action tokens at the block; at step 1 that of theta0) and `steering_ratio`
(steering / hidden_norm). There is no FM ascent/descent term in RMU's objective. Each forward stops
right after the target block (nothing downstream enters the objective, and running it would only add
to the measured cost); the fixed per-task FM probes track the FM losses.

## Benchmark entry point and measurements

Both methods are calibrated with the other methods (`scripts/calibrate.py`, run `cal`, reusing LA and
LA/LB). Round 3: task arithmetic at `unlearn_lr` 3e-7, 1e-6, 3e-6 (scale 1) and 1e-6 at scale 2 forgot
nothing (edit norm 0.2-1.7, ~1% of tauB's 131); RMU at 1e-5 and 5e-5 on DiT blocks 0 and 7 kept A (1.00) and
partly forgot B (0.68-0.72 at 5e-5), with steering 20 about 50x below the action tokens' hidden norm
(~760 at block 0, ~1200 at block 7). Round 4: task arithmetic at 3e-6 with scales 5, 20, 50; RMU at block 7
with steering 300, 1000, 3000 at 1e-4 and 1000 at 5e-5. After the round sets their values in
`vlu/config.py`, they join `METHODS` in `scripts/main_run.py`, or are added to the same main run:

```bash
"$MAIN_PY" scripts/main_run.py --run goal_s0 --methods 'rmu@unlearn_lr=5e-5+rmu_layer=7'
```

`--methods` replaces the default method list for that call (purge is always added) and the plan only
grows, so methods already planned stay. Do not start a separate run (for example `ta_rmu_s0`) for them:
every run trains its own nodes under `runs/<run>/`, so it would retrain the whole stream (LA..LE at 10k
steps each, and the oracles) for two methods. Reports and drift selection read the actual plan and
preserve override values rather than collapsing different rates/scales into one method name. For
independent training seeds, a run per seed (`--run goal_s1 --seed 1`) is needed anyway;
`scripts/metrics.py --runs` aggregates them.

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
Executed audit rerun: 60 passing tests across these checks and the existing GU/Sharp/NPO/resume/safety
regressions, one skipped two-GPU RNG test, plus the hand-computed stream-metric script and Python
compilation checks. The new method tests used CPU torch 2.9.1 and the repository's GR00T-shaped fake.

The current workspace has CPU-only torch, no real checkpoints, no LIBERO data and no configured
GPU host. Per the user's instruction, real rollout performance, GPU runtime/memory/FLOPs/energy,
and real-checkpoint drift benchmarks are **deferred**. No numbers for these are claimed here.

```bash
USE_TF=0 python tests/test_task_arithmetic_rmu.py
USE_TF=0 python tests/test_method_benchmarks.py
```

## Correctness audit — 2026-11-25

Static review covered the objectives, optimizer scopes, masks, reference construction, method
state, snapshot/finalization paths and benchmark coverage. Executed checks below are distinct
from robot/GPU benchmarks, which remain deferred. No additional mathematical discrepancy was
found in the two stated adaptations under the tested conditions.

### Reproduced issues and fixes

- **Missing benchmark control (P2):** selecting only TA/RMU omitted the purge branch while the
  drift report still requested its incidental-forgetting comparison. `main_run.with_controls`
  now always includes vanilla purge. A regression checks that its learned endpoint and features
  are planned even when purge was not in the user's method list.
- **Auxiliary-fit snapshots through direct calls (P2):** the CLI rejected TA snapshots, but calling
  `train(..., save_at=...)` directly could write fitted, unnegated weights under a snapshot name.
  The shared loop now rejects these calls before fitting, so the invariant holds below the CLI.
- **Zero/negative smoke-step override (P2):** `run_op --steps` was applied after recipe validation,
  allowing the positive-budget guard to be bypassed. The override is now checked before loading.
- **Changed resume budget (P2):** changing the budget between attempts could continue a different
  auxiliary fit/schedule or finalize a checkpoint already beyond the requested budget. New resume
  files record `planned_steps`; incompatible budgets are rejected before copying checkpoint weights.
  Older files without that field still get the step-range check, but their original total budget
  cannot be recovered retrospectively.

The first three issues were demonstrated by failing targeted tests before the corrections. The
resume-budget test interrupts a fit, then verifies that both shorter and longer changed budgets
are rejected without modifying the fresh model. Full method/report regressions subsequently pass.

### Source-based numerical checks

The optional checks execute inspected, hash-verified reference code rather than substitutes:

- **Task arithmetic:** the original `TaskVector` constructor, negation and `apply_to` methods at
  the pinned commit above match our finalization exactly over 8 seeds and scales 0/0.25/1/2.
  Maximum parameter difference: **0**. Only checkpoint deserialization is replaced with in-memory
  models to isolate arithmetic from the old repository's pickle/PyTorch compatibility.
- **RMU:** the actual
  [GR00T DiT source at ead52833afbbf4243f8cd5e7664f48a94de03b19](https://github.com/NVIDIA/Isaac-GR00T/blob/ead52833afbbf4243f8cd5e7664f48a94de03b19/gr00t/model/modules/dit.py),
  with its pinned diffusers 0.35.1, was instantiated in a small two-block CPU model. Its real
  self/cross attention, adaptive normalization, positional embeddings, masks and keyword inputs
  were exercised. Our block-reference update matched an independent full-frozen-model objective
  over 4 configurations x 4 AdamW steps, maximum parameter difference **0**; model RNG states
  also matched. This verifies the reference shortcut in that setting, not real GR00T effectiveness.

| Claim | Audit disposition | Evidence / boundary |
|---|---|---|
| TA negates the freshly fitted vector from the operation's original anchor | Verified for tested cases | Independent AdamW fit and actual author arithmetic checks |
| TA checkpoints preserve the anchor and publish negated weights | Verified on CPU fixtures | Stochastic resume, failed-finalization recovery and direct-snapshot rejection |
| RMU implements the stated masked activation objective | Verified for tested cases | Independent two-step objective and actual DiT/full-reference trajectories |
| Only the selected RMU block changes | Verified on tested models | Full parameter comparisons with upstream/downstream weights held fixed |
| Targets, references, masks and RNG continuation survive resume | Verified on CPU fixtures | Checkpoint round trips and exact stochastic continuation |
| Benchmark variants have the required controls and reporting paths | Verified on synthetic fixtures | S1/oracle/features planning, all stream metrics and cost CSV exports |
| Author experimental setups are reproduced verbatim | Contradicted | Both are documented GR00T adaptations, especially RMU's DiT scope |
| Real policy quality and GPU/mixed-precision behavior are established | Unsupported | Full weights/data/CUDA/rollout checks remain deferred |

Reproduce the optional numerical checks after downloading the pinned sources outside the repo:

```bash
curl -L https://raw.githubusercontent.com/mlfoundations/task_vectors/826a64c67082fab0f40628233287948f0f8d7fa3/src/task_vectors.py -o /tmp/author_task_vectors.py
curl -L https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/ead52833afbbf4243f8cd5e7664f48a94de03b19/gr00t/model/modules/dit.py -o /tmp/groot_dit.py
USE_TF=0 python tests/check_task_vectors.py /tmp/author_task_vectors.py
USE_TF=0 python tests/check_groot_rmu.py /tmp/groot_dit.py
```

The second check requires diffusers 0.35.1, already part of the pinned GR00T environment. The audit
used Python 3.13.5 and CPU torch 2.9.1; it did not run GR00T's pinned CUDA torch 2.7.1 environment.

## Review fixes — 2026-11-27

- The documented `ta_rmu_s0` run would have retrained the whole stream; both methods are now in calibration
  round 3 and join the main run `goal_s0` afterwards (section above, `scripts/main_run.py`).
- RMU's forwards end after the target block (`tests/test_task_arithmetic_rmu.py` checks that nothing
  downstream runs); the FM diagnostics computed downstream are gone, `hidden_norm` and `steering_ratio` added.
- `tests/check_groot_rmu.py` defaults to the pinned GR00T checkout's `dit.py` (`$GR00T_DIR`, the same commit
  ead52833 whose hash it checks; GR00T's uv.lock pins diffusers 0.35.1) and is one of `scripts/run_local.sh`'s
  self-tests; it reports a skip where no GR00T environment exists.
- Memory (reasoning, not measured): task arithmetic's fit is a plain AdamW fine-tune, about GA's measured 33.4
  GiB; RMU computes gradients of one block only, no more than that. Neither changes weights inside a step, so
  autocast's weight cache needs no clearing.
