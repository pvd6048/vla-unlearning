# Forgetting vs. Unlearning in VLAs — Research Plan

Status: **planning only — no code yet.** Last updated 2026-10-26.
Built from the team's research doc (Google Doc "VLA Unlearning", items 1–5 plus the lit notes) and a review of the anchor papers.

## 0. One-line thesis

During continual learning on a VLA (NVIDIA GR00T), **incidental forgetting** of a task and **intentional unlearning** of that task may move the model along shared, task-specific directions. If so, those directions can be used both to *reduce* forgetting and to *unlearn cheaply and selectively*.

**Target:** ICML 2027. The deadline is likely late Jan 2027 (to be confirmed), which leaves about 13 weeks.

## Week 1 (Oct 27 – Nov 2): establish baseline numbers

**Goal:** one table of task success rates after an interleaved learn/unlearn stream, for the standard unlearning methods. No drift analysis yet.

**Setup:** GR00T N1.6, LIBERO-Goal, tasks A–E (5 goals from the suite), 1 seed, 50 rollouts per task per eval.

**Stream (S1):** `L_A → L_B → L_C → U_B → L_D → L_E → U_E`
- Learning uses ER: a 1:1 replay buffer, and an unlearned task is removed from the buffer.
- The prefix `L_A L_B L_C` is trained **once** and shared by every method, so only the tail (`U_B L_D L_E U_E`) runs per method.

**Methods:**

| Method | What it does |
|---|---|
| Retrain oracle | Re-run the stream without ever training B and E (`L_A L_C L_D`). This is the target to match. |
| Replay purge only | Drop B from the buffer and do nothing else. Expected to fail, since VLAs resist forgetting. |
| Gradient Ascent (GA) | Push the training loss *up* on B's data |
| Gradient Difference | GA on B plus normal training on the replay buffer |
| NPO | A softer GA that slows down once B is already unlikely relative to the original model |
| SSD | Retraining-free: dampen weights that matter much more for B than for the rest |
| SalUn | Update only the weights most "salient" to B |

GA, NPO, SSD and SalUn are VLA-Forget's baselines. VLA-Forget itself needs porting from OpenVLA, so it is **week 2**.

Every unlearning method gets the **same step budget per U op**. "Loss" here means GR00T's flow-matching loss; GR00T has no token cross-entropy.

**Day by day:**

| Day | Task |
|---|---|
| 1 | Install Isaac-GR00T (`n1.6-release`) + LIBERO on the GPU box. Measure training steps/sec and rollouts/hour. Fix the A–E task mapping. |
| 2 | Single-task fine-tune on task A, then evaluate. Check that SR is in the expected range (the pipeline works). Details below. |
| 3 | Train the shared prefix `L_A L_B L_C` with ER. Evaluate all 3 tasks. Start the oracle branch. |
| 4–5 | Implement the 5 unlearning methods (common interface). Run `U_B` for each, evaluate, and launch each method's tail `L_D L_E U_E`. |
| 6 | Finish the tails and run evals after every op. |
| 7 | Build the results table and write up observations. |

### Day 2 in detail: pipeline sanity check on task A

**Why:** every later number depends on data loading, action normalization, training and evaluation all being correct. A bug here, such as a flipped camera image or wrong gripper sign, would silently make every method look bad. One cheap task proves the pipeline before we spend compute on full streams.

**Steps:**

1. **Zero-shot floor.** Evaluate the base N1.6 model on task A before any fine-tuning (50 rollouts).
   - Expected: about 0%, since the base model has never seen the LIBERO robot setup (`LIBERO_PANDA`).
   - This is also the "never trained" reference the retrain oracle is compared against.
2. **Prepare task A's data.** From the LIBERO-Goal LeRobot dataset, keep only task A's demonstrations (about 50).
   - Check that the episode count, camera views (main + wrist), robot state, action dimensions and instruction text are as expected.
   - Check that the action normalization statistics look sane.
3. **Fine-tune.** Use the CL-paper recipe (lr 1e-4, batch 16, AdamW, cosine decay) for 10k steps with the default trainable scope.
   - Save checkpoints at 2k, 5k and 10k steps.
   - Log the training loss and the flow-matching loss on a few held-out demos.
4. **Evaluate each checkpoint.** Run 50 rollouts per checkpoint from LIBERO's fixed initial states, with the standard step limit.
   - Record SR and wall-clock time.
   - Save ~5 rollout videos to inspect by eye.
5. **Measure eval noise.** Re-run the 10k-step eval with a different seed. The gap between the two runs tells us how much SR moves by chance.

**Expected range and pass criterion:**
- Published GR00T results on LIBERO-Goal (multi-task) are in the 90s; confirm the exact figure from NVIDIA's N1.6 numbers.
- A single task with ~50 demos should reach **≥ 80% SR**. That is the pass bar.
- 50–80%: likely a hyperparameter or step-count issue, so investigate before moving on.
- Below 50%: almost certainly a pipeline bug.

**Debug checklist if SR is low:**
- Image orientation: LeRobot LIBERO datasets are often stored flipped 180°, and the simulator at eval time must match training.
- Gripper open/close sign convention.
- Action normalization: are train stats applied at eval, and un-normalized correctly?
- The modality config: camera keys, state keys and action keys.
- Action chunk length and how many actions are executed before re-planning.
- Instruction text at eval exactly matching the training text.

**Outputs that feed later days:**
- SR vs. steps (2k / 5k / 10k). This tells us whether 5k steps per task is enough, our compute fallback.
- Eval noise between seeds.
- Training steps/sec and rollouts/hour, to finalize the week-1 compute budget.
- A frozen config file for all later runs.

**Deliverable:** for each method, SR on A–E after each op, plus four summary numbers:
- **Forget SR:** on B and E, compared with the oracle
- **Retain SR:** on the tasks that should still be known
- **Relapse:** does B come back after `L_D L_E`?
- **Cost:** GPU-minutes per unlearn op

**If compute is short:** cut to 5k steps per task (after checking that SR has saturated) and 20 rollouts for intermediate evals.

## 1. Research agenda

| # | Item | Plan section |
|---|---|---|
| 1 | Taxonomy of drifts in CL and unlearning for VLAs, and how to measure each | §4 |
| 2 | **Core:** from a common checkpoint, run a CL branch (e.g. ER) and a CL + unlearning branch (e.g. GA). Are the drifts similar? Do they point the same way? | §5 |
| 3 | Vary the CL and unlearning methods. Are the correlations consistent? Which pairs "sync"? | §6 |
| 4 | Incidental vs. intentional *relearning* after unlearning. Can we predict which tasks recover? | §7 |
| 5 | A new VLA unlearning benchmark (L/U op streams such as `L_A L_B L_C U_B L_D L_E U_E`) | §8 |

## 2. What the literature gives us

| Paper | Takeaway for this project |
|---|---|
| **Liu et al. 2026, "Pretrained VLAs are Surprisingly Resistant to Forgetting in CL"** (arXiv 2603.03818, ICML oral) | Supplies the CL protocol: LIBERO, 10k steps/task, ER mixed 1:1, NBT metric. GR00T N1.5 recipe: lr 1e-4, bs 16, wd 1e-5. ER gives near-zero forgetting, and "forgotten" skills recover in ~6–11% of the original training steps. **Consequence for item 2: with ER there may be almost no forgetting drift to measure.** We need branches that actually forget (§5.2). |
| **VLA-Forget** (arXiv 2604.03956) | The VLA unlearning baseline: staged LoRA over vision → projector → action layers, PCGrad, forget/retain/boundary splits. It was not tested in a continual setting. It targets OpenVLA (token-CE), so it must be adapted to GR00T's flow-matching head. |
| **Unlearning Isn't Deletion** (OpenReview E5SVowO13b) | Supplies the drift toolkit for item 1: PCA similarity/shift, CKA, empirical Fisher, mean PCA distance. Its finding gives the hypothesis for item 4: large PCA shift → recovery fails, while high CKA plus a concentrated Fisher spectrum → reversible. |
| **Plasticity Collapse in Continual MU** (arXiv 2608.29513) | Sequential unlearning saturates parameter subspaces. This shows up as forward failure (worse forgetting on later requests) and backward failure (re-memorization). Supplies subspace-overlap metrics. |
| **LACU** (ECCV 2026) | In diffusion models, concepts close in representation take collateral damage first. Fixes are locality-aware target selection and replay. GR00T's DiT is in the diffusion family, so this transfers directly. |
| **UnCLe** (arXiv 2509.17530) | Interleaved learning and unlearning with a hypernetwork, on image classification only. Supplies the *task relapse* metric. |
| **Forgetting–Retention Dilemma** (PMLR v306 hu26o) | Gradient-based unlearning gets a free ride from CL's own forgetting for *old* tasks but is poor for *recent* ones. This is direct theoretical support for the core hypothesis, and it predicts a **task-age effect** (§5.4). |
| **Task-vector negation, AKLN** | If the forgetting direction ≈ −τ_B (the task vector), then task-vector negation is the unlearning method most "in sync" with forgetting. AKLN's cascading degradation in continual unlearning is a failure mode to watch for. |
| **PULSE, SineProject** | Unlearning tends to break *cross-modal alignment* (the projector) rather than erase knowledge. This makes alignment drift a key axis of the taxonomy. |
| **ICU-Bench, FIT** | Sequence-aware metrics: forgetting rebound and retain stability rate. Recent forgetting is easier than historical forgetting. |
| **CLARE, CRL-VLA, real-world VLA CL (2605.26820)** | VLA CL baselines: adapter routing, KL-regularized RL, tuned ER. CLARE is a candidate architecture-based CL method for §6. |

As far as we found, **no work compares forgetting and unlearning drift directly, and none interleaves learning and unlearning on VLAs.**

## 3. Setup

- **Model:** GR00T **N1.6-3B** (Isaac-GR00T `n1.6-release`), which pairs a Cosmos-Reason-2B VLM with a 32-layer flow-matching DiT. A Phase 0 sanity run on N1.5 reproduces the CL paper's numbers.
- **Trainable scope (needs a decision):** DiT + projector + **top-4 VLM layers**, which is what N1.6 unfreezes in pretraining. The CL paper froze the VLM, but then VLM and cross-modal drift would be zero *by construction* and taxonomy item 1 would collapse to DiT-only. A frozen-VLM run is kept as an ablation.
- **Benchmark:** LIBERO in simulation (LeRobot datasets, `LIBERO_PANDA` modality config).
  - **Primary: LIBERO-Goal.** One scene, different goals. Drift is task-specific rather than scene-specific.
  - **Secondary: LIBERO-Object**, where tasks are object-level.
  - **LIBERO-10** for long streams.
- **Hardware:** one Blackwell GPU with 96 GB. A full trainable-parameter checkpoint is a few GB, so storing every op's checkpoint is cheap. **Rollout evaluation is the likely bottleneck**, so throughput is measured in Phase 0.

## 4. Item 1 — Drift taxonomy

Every drift is measured between two checkpoints θ, θ′ on a fixed **probe set**: held-out states from every task, including the forget task, plus base-model reference states. Each is reported **globally, per module (VLM-top / projector / DiT) and per layer.**

| Drift type | What changes | Magnitude metric | Direction / similarity metric |
|---|---|---|---|
| **Parameter** | Weights | ‖Δθ‖₂, relative ‖Δθ‖/‖θ‖ | cos(Δθ₁, Δθ₂); projection onto task vector τ_X; principal angles between top-k SVD subspaces of per-layer ΔW |
| **Curvature-weighted parameter** | Weights that matter for task X | Fisher-weighted ‖Δθ‖_F(X) (diagonal empirical Fisher on X) | Fisher-weighted cosine; Fisher spectrum concentration |
| **Representation** | VLM hidden states, DiT hidden states | 1 − CKA(h, h′), PCA shift, mean PCA distance | CKA between *shifts* (h′−h); overlap of PCA subspaces of the shifts |
| **Cross-modal alignment** | How vision and language tokens bind | Change in vision↔language token similarity; change in DiT cross-attention mass on instruction vs. image tokens | **Language sensitivity**: change in action under an instruction swap ("linguistic blindness" from VLA-Forget) |
| **Policy (action)** | Predicted action chunks on fixed states | Action MSE / flow-field divergence ∫‖v_θ − v_θ′‖² over noise levels | Which policy the drifted model now resembles: distance to base model, to other tasks' experts, to no-op |
| **Behavioural** | Closed-loop success | SR, NBT, forget SR | Failure-mode labels: wrong object, wrong goal, freeze, collision |

**Statistical controls**, without which the comparisons are uninterpretable:

- **Noise floor:** cosine and CKA between two seeds of the *same* branch.
- **Random baseline:** a matched-norm random direction.
- **Generic-drift control:** the drift of an unrelated op of equal size.

In 3B-dimensional space, raw cosines are near 0, so similarity is only meaningful relative to these controls and per layer.

## 5. Item 2 (core) — Do forgetting and unlearning drift alike?

### 5.1 Branch design

All branches start from a common checkpoint **θ₀ = GR00T after `L_A L_B L_C`** (with ER). Target task: **B**.

```
            ┌─ C  (control):     L_D L_E, replay includes B         → B protected
θ₀ ─────────┼─ F  (forgetting):  L_D L_E, B dropped from replay     → B incidentally forgotten
 (L_A L_B L_C)├─ U  (unlearning):  U_B (method M_U)                   → B intentionally removed
            └─ UF (doc's branch): U_B then L_D L_E (same CL method)  → both
```

- **Incidental-forgetting drift (B-specific):** `Δ_F|B = θ_F − θ_C`. Taking the difference of the two branches cancels the shared "learn D, E" component and leaves what is attributable to *losing* B.
- **Intentional-unlearning drift:** `Δ_U = θ_U − θ₀`.
- **Task vector:** `τ_B = θ_{L_A L_B} − θ_{L_A}`.

### 5.2 Forcing measurable forgetting

ER barely forgets on VLAs, so branch F has three variants:

- **F-seq:** naive sequential FT with no replay.
- **F-ER-small:** ER with a 10-sample buffer that excludes B.
- **F-long:** more and longer subsequent tasks.

We first check that B's SR actually drops. If even F-seq barely forgets, that is a finding in its own right: forgetting in VLAs is a *readout* effect, not a weight-level one, consistent with the CL paper's "hidden knowledge".

### 5.3 Questions and measurements

1. **Same direction?** Compare cos(Δ_F|B, Δ_U), cos(Δ_F|B, −τ_B) and cos(Δ_U, −τ_B), per layer and against the noise floor.
2. **Same location?** Which layers and modules carry most of each drift, and do the profiles correlate (Spearman across layers)?
3. **Same representational effect?** CKA and PCA-shift profiles for each drift.
4. **Same behaviour?** On B's instruction, does the forgotten model and the unlearned model fail the same way? A forgotten model may execute the most recent task D (recency bias), while a GA-unlearned model may freeze or produce garbage.
5. **Gradient-level check (cheap):** at each step of L_D and L_E, the cosine between the training gradient and ∇L_B. That is the instantaneous "forgetting direction", which can then be compared with the unlearning gradient −∇L_B.

### 5.4 Factors to vary

- **Task age:** forget A (oldest) vs. C (most recent). The Forgetting–Retention Dilemma paper predicts forgetting and unlearning align more for old tasks.
- **Task similarity:** pick D to be similar or dissimilar to B.
- **Module scope:** frozen VLM vs. top-4 VLM layers trainable.

### 5.5 If aligned, how to exploit it

These feed P5:

- **Anti-forgetting:** during L ops, project updates off the estimated B-forgetting subspace, in the style of OGD/GPM, and compare with ER at an equal memory budget.
- **Cheap unlearning:** unlearn by stepping along the forgetting subspace, e.g. Fisher-masked task-vector negation. This is attractive because it needs no forget data at unlearning time.

## 6. Item 3 — Method grid

Kept small for one GPU and pruned after the item 2 pilot.

- **CL methods:**
  - Sequential FT (no replay)
  - ER (1:1, buffer 10 / 1000)
  - EWC
  - Per-task LoRA, or CLARE-lite adapters (architecture-based)
- **Unlearning methods** (all rewritten for the flow-matching loss `L_FM`):
  - GA, with clipping
  - Gradient Difference
  - NPO (with `−L_FM` as the likelihood proxy)
  - **Task-vector negation**
  - Representation-based: steer forget-set features to a fixed vector, keep retain features
  - VLA-Forget (ported)
  - LACU (ported)
- **Reference:** a retrain oracle (stream re-run without B), which gives the "true" unlearned model and a third drift to compare against.

**Output:** a CL × unlearning matrix of drift alignment (cosine and CKA versus the noise floor) next to unlearning efficacy and retained SR. This shows which pairings "sync" and whether syncing predicts good unlearning.

## 7. Item 4 — Relearning and its prediction

- **Incidental relearning (relapse):** after U_B, continue L_D and L_E. Does B's SR come back on its own, especially if D or E resemble B?
- **Intentional relearning:** fine-tune on B for a budget of k ∈ {100, 500, 1000} steps. The recovery ratio is steps-to-recover from the unlearned model divided by steps from the oracle. Near 1 means erased; much less than 1 means only suppressed.
- **Prediction:** at the moment of unlearning, compute features from §4. These include PCA shift, CKA, Fisher concentration, ‖Δθ‖ for B relative to τ_B, task age, and B's similarity to retained tasks.
  - Regress or rank-correlate these features against relapse and recovery across all (task × method × seed) points.
  - With 10 tasks × several methods × 3 seeds we get roughly 100+ points, enough for correlation, though not for a big model.
  - This tests the "Unlearning Isn't Deletion" hypothesis on VLAs.

## 8. Item 5 — Benchmark (the substrate for everything above)

The same infrastructure, packaged as **L/U op streams over LIBERO**.

| ID | Stream | Probes |
|---|---|---|
| S1 | `L_A L_B L_C U_B L_D L_E U_E` | Canonical stream; tests timing and relapse |
| S2 | `L_A … L_E U_A` | Unlearning the oldest task |
| S3 | `L_A U_A L_B U_B L_C U_C` | Immediate unlearning and plasticity |
| S4 | `L_A … L_E U_B U_C U_D` | Accumulated unlearning (plasticity collapse, cascading degradation) |
| S5 | `L_A L_B U_B L_C L_B` | Legitimate relearning |
| S6 | `L_A L_B U_B L_B'` with B' similar to B | Boundary relapse |
| S7 | Random streams of length 15–20 on LIBERO-10 | Scale |

**Metrics** come from the success matrix `c[i,k]`, built by evaluating every seen task, including unlearned ones, after every op:

- Retain SR, forget SR, and Δ-to-oracle
- NBT, split into collateral damage from L ops and from U ops
- Relapse / forgetting rebound
- Retain stability rate
- Recovery ratio
- Plasticity (SR on the next new task)
- Cost in GPU-minutes per U op
- Quantization robustness

Several drift metrics from §4 are logged as part of the benchmark too.

## 9. Phases

| Phase | Work | Exit criterion |
|---|---|---|
| **P0 Infra** | GR00T N1.6 + LIBERO environment, parallel rollout eval, op-stream runner with a checkpoint cache, probe-set builder, drift-metric library (§4) with noise-floor controls | Single-task LIBERO-Goal SR close to published; seed-noise floor measured |
| **P1 Reproduce CL** | Sequential FT vs. ER for GR00T on LIBERO-Spatial and LIBERO-Goal | NBT matches the CL paper; confirms which F-variant actually forgets |
| **P2 Core pilot (item 2)** | Branches C/F/U/UF for one target task with ER + GA, then repeated for 3 target tasks × 3 seeds | A clear yes or no on alignment versus the noise floor → **go/no-go for the "exploit" direction** |
| **P3 Grid (item 3)** | CL × unlearning matrix | Alignment matrix and efficacy |
| **P4 Relearning (item 4)** | Relapse and recovery runs, predictor analysis | Correlation results |
| **P5 Exploit / benchmark** | If aligned: forgetting-subspace-guided CL and unlearning. Either way: release the S1–S7 benchmark | Paper |

**Compute note:** the core pilot (P2) is about 4 branches × 2 tasks of 10k steps × 3 targets × 3 seeds. That is far cheaper than the full benchmark, which is why it goes first. Drift metrics need only forward passes on the probe set. If P0 throughput is low, cut to 5k steps/task (after checking that SR saturates) and use 20 rollouts for intermediate evals.

## 10. Risks

- **No measurable forgetting** (VLAs resist it): handled by the F-variants in §5.2, and a null result is still informative.
- **Cosine similarity is meaningless in raw high dimensions:** use per-layer comparisons, subspace angles and noise-floor normalization. Report *relative to seed noise* only.
- **GA on `L_FM` collapses the policy:** unlearning drift would then be dominated by generic destruction. Hence the generic-drift control and bounded methods (NPO, task-vector negation, Gradient Difference).
- **Mode connectivity and permutation symmetry:** these are not an issue, because all branches share θ₀ and fine-tuning stays in the same basin. We confirm this with linear-interpolation loss curves.
- **Zero-shot base SR on LIBERO tasks:** the oracle, not 0%, is the unlearning reference.

## 11. Open decisions

**Decided:** venue is ICML 2027. Model is **N1.6** for the main results, plus a short N1.5 run in week 2 to reproduce Liu et al.'s GR00T numbers (they used N1.5).

**Still open:**
1. Trainable scope: top-4 VLM layers + DiT (rec., so VLM and alignment drift exist) vs. frozen VLM (matches the CL paper). *Week 1 uses the NVIDIA default fine-tuning recipe.*
2. What "unlearned" behaviour should be: oracle-matching (rec.) vs. safe no-op.
3. Framing: forgetting↔unlearning analysis as the main paper with the benchmark as a contribution (rec.), or benchmark-first.
