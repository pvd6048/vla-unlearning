# GU and Sharp MinMax audit — 2026-11-24

The implemented computations pass the focused CPU checks below after two GU numerical fixes.
**They must be reported as FM adaptations. In particular, `gu` is not the authors' complete GU
algorithm.** Passing the numerical checks does not establish forgetting or retention on a robot.

## Reference basis

The GU paper links [Lemutisme/Geometric-Unlearning](https://github.com/Lemutisme/Geometric-Unlearning),
whose README directs readers to the `dev` branch of `geo-unlearning`. We inspected:

- [GU source, bbe4d5077b21f6ded0ddbff8bf49b0336260626a](https://github.com/Lemutisme/geo-unlearning/blob/bbe4d5077b21f6ded0ddbff8bf49b0336260626a/src/trainer/unlearn/geometric.py),
  particularly `RetainNullProjector` and `GeometricUnlearn.optimizer_step`;
  [paper v4](https://arxiv.org/html/2511.17100v4).
- [Sharp SAM source, 98c41d9e344bd0942421bdcdeee85878d6a657b1](https://github.com/HaoranTang/sharp-unlearn/blob/98c41d9e344bd0942421bdcdeee85878d6a657b1/sam/sam.py)
  and [outer loop](https://github.com/HaoranTang/sharp-unlearn/blob/98c41d9e344bd0942421bdcdeee85878d6a657b1/unlearn/neggrad.py);
  [paper v3](https://arxiv.org/html/2506.13715v3), especially Algorithm 2 and Appendix E.2.

Scope: the new GU/Sharp code, method planning, saliency preparation, optimizer integration,
checkpoints, configuration and documentation. NPO, generic resume and training safety tests were
regression checks, not a fresh audit of every existing method. No reported benchmark numbers or
plots were accepted as evidence of these new methods' performance.

## Findings, in order of scientific impact

### P1 — GU identity is materially different from the published algorithm

**Static evidence:** `vlu/gu.py:GeometricUpdate.step` projects the actual AdamW displacement,
including momentum and decay, using supervised retain-FM gradients and a rolling raw-gradient basis.
The reference whitens raw gradients, constructs a retain-KL anchor basis, and supports selected
layers, tangential retain updates and sign-selective components. The displacement metric here is
P^-1; the reference whitening uses W^T W. The resulting Adam updates are generally different.

**Correction:** removed wording that equated our metric projection with the authors' optimizer.
METHODS and README now require the label **GU-FM (strict proposal projection)**. `OpPlan.notes`
records `strict_fm_proposal_projection` and the pinned reference commit. This is a documented
GU-inspired constrained baseline, not a faithful reproduction claim. A faithful full GU baseline
remains unimplemented; these tests do not remove that difference.

### P1 — GU could drop an independent retain constraint under Adam scaling (fixed)

**Executed counterexample:** d=[1,1,2], U=[e1,e2], P=diag(1e8,1e-4,1). The previous Gram-matrix
pseudoinverse used relative cutoff 1e-10; its eigenvalues differed by 1e12, so it returned [0,1,2]
instead of [0,0,2]. The second retain direction was not protected.

**Fix:** `retain_null` uses column-equilibrated whitened SVD in float64, avoiding normal equations
and retaining both independent directions in this example. Tests cover that regression, dependent
and full-rank bases, and eight seeded comparisons against an independently assembled KKT linear
system. The solved problem is minimize 0.5*(delta-d)^T P^-1*(delta-d), subject to U^T delta=0.

### P2 — Finite retain gradients could normalize to zero or be skipped (fixed)

**Static and executed evidence:** a float32 gradient norm can overflow when squaring values around
1e20, or underflow around 1e-30. `GeometricUpdate.step` previously divided by that norm or skipped
the direction. Normalization now divides by the largest absolute entry before computing the norm.
`test_finite_extreme_retain_gradients_are_not_dropped` checks both scales and the resulting actual
retain-null displacement.

### P2 — Sharp reference masking and paper selection are inconsistent

The reference masks the second gradient only; the blend `g0 + sign*lambda*(g1-g0)` still contains
the original gradient. In particular the forget/max blend can update the inactive partition.
Our existing correction masks the original gradient, perturbation and final update. Weight decay
is disabled and retain/forget AdamW states are separate. This is a deliberate correction, not
bitwise reproduction of the reference outer loop.

The paper's section 3.2 and Algorithm 1 describe smallest/bottom gradients, whereas E.2 describes
the highest gradients. Our top-saliency forget split follows E.2. Sampled saliency, FM losses,
paired random draws, AdamW and the shared schedule are additional disclosed choices. Report
**Sharp MinMax-FM (partition-safe AdamW)**, recorded as `partition_safe_fm_adamw` in operation notes.

## Claim-to-evidence matrix

| Claim | Disposition | Evidence and limit |
|---|---|---|
| GU solves the stated strict displacement projection | Verified for the tested CPU cases | Closed form, independent KKT solves, degeneracy and extreme-scaling tests in `tests/test_gu.py` |
| GU's actual Adam update is locally retain-null | Partially verified | Multi-step momentum/decay tests; protects sampled current directions, with roundoff. Older basis vectors are stale, and no full-dataset guarantee follows |
| `gu` reproduces the authors' full algorithm | Contradicted | Different anchors, metric placement, basis maintenance, trainable scope and tangential behavior |
| GU shares the authors' Euclidean null-projection primitive | Verified in that restricted setting | 20 seeded orthonormal-basis cases against the actual reference class, max absolute difference 2.38e-7; Adam whitening was explicitly disabled |
| Sharp SAM/ASAM gradient blends and signs match the source | Verified for isolated partition subproblems | 32 configurations x 4 SGD-momentum steps against actual reference SAM, max absolute difference 3.33e-16; this does not test full-optimizer equivalence |
| Sharp preserves inactive coordinates and restores perturbed weights | Verified for tested models | Analytic partition checks, exception restoration, independent optimizers, paired-noise and BatchNorm-buffer checks |
| Interrupted runs preserve method state and random continuation | Verified on CPU fixtures | Exact losses and weights for deterministic and stochastic GU/Sharp runs through the shared checkpoint loop |
| Both methods integrate with the training pipeline | Partially verified | Preparation, training, probes and snapshots on the GR00T-shaped fake; frozen weights checked |
| Real GR00T/LIBERO forgetting, retention, runtime and memory are established | Unsupported | No real checkpoint training, CUDA execution or robot rollouts in this audit |

## Executed checks and reproduction

Environment: Windows, Python 3.13.5, torch 2.9.1+cpu, transformers 4.57.6, `USE_TF=0`, one torch
CPU thread. GR00T imports were replaced by the repository's existing test stubs; the fake-model
smoke test is explicitly not a real-model run. Focused suite: **44 tests, 43 passed, one skipped**.
The skipped NPO RNG test needs two CUDA devices. GPU-dependent operation tests and Linux-only
planner tests were not run here.

From the repository root, set `USE_TF=0`, then run:

```bash
python -c "import sys,unittest,torch; sys.path.insert(0,'tests'); torch.set_num_threads(1); suite=unittest.defaultTestLoader.loadTestsFromNames(['test_gu','test_sharp_minmax','test_npo','test_resume','test_training_safety']); result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())"
```

For the optional original-source checks, download the pinned files outside the repository:

```bash
curl -L https://raw.githubusercontent.com/Lemutisme/geo-unlearning/bbe4d5077b21f6ded0ddbff8bf49b0336260626a/src/trainer/unlearn/geometric.py -o /tmp/author_gu.py
curl -L https://raw.githubusercontent.com/HaoranTang/sharp-unlearn/98c41d9e344bd0942421bdcdeee85878d6a657b1/sam/sam.py -o /tmp/author_sam.py
python tests/check_author_kernels.py /tmp/author_gu.py /tmp/author_sam.py
```

`check_author_kernels.py` verifies normalized-source SHA-256 hashes before executing the inspected
SAM class and extracting the GU projector class. Sharp tests cover both signs, strengths
0/0.5/1/2, vanilla/adaptive perturbations, radii 0/0.13, with learning rate .02 and momentum .9 on
a deterministic nonlinear three-coordinate problem. GU uses seeds 0–19 and a 9x3 orthonormal basis.
Full unlearning trajectories are intentionally not claimed to match either author repository.

Before benchmarking, calibrate explicit learning rates, measure actual compute rather than equating
step counts, and run the real-model GPU smoke/resume and rollout evaluations. Sharp uses four
forward/gradient evaluations per operation step; GU requires basis and float64 projection buffers.

## Follow-up — 2026-11-26 (integration into calibration round 3)

Re-checked against the same pinned sources plus the authors' `unlearn/impl.py`, `generate_mask.py`,
`arg_parser.py`, README and `configs/trainer/GeometricUnlearn.yaml`.

- **GPU memory (measured on a toy with many small tensors, in fp32 copies of the trainable weights; one copy
  is 6.5 GB on GR00T, a training GPU has 44 GB):** a plain AdamW step 5.5; `gu` rank 2 8.3 (about 53 GB on
  GR00T); `sharp_minmax` as merged 9.8 (about 63 GB). Neither could run on the calibration machine.
- **Sharp MinMax, memory:** the authors use two AdamW optimizers over all weights (`impl.py`, "Adam
  Override"). With masked gradients and no decay their moments are disjoint, so `vlu/sharp_minmax.py` now
  keeps one pair of buffers updated per partition, which is exactly the two-optimizer update
  (`test_shared_moments_equal_two_independent_adamw`, against two torch AdamW). The blend is formed in place,
  the perturbation saves only its partition, and masks are made per tensor: 6.6 copies (about 40 GB).
- **Sharp MinMax, partition:** `generate_mask.py` sorts ascending and marks ranks below the cutoff, i.e. the
  *smallest* gradients, agreeing with section 3.2 and Algorithm 1; App. E.2 and section 4.2 say the largest.
  New recipe field `sharp_select` (`largest` default, `smallest`); the calibration runs both.
- **Sharp MinMax, alpha:** under the authors' two AdamW optimizers (and ours) a phase's constant loss scale
  cancels, so alpha only acts through eps and clipping. Kept for fidelity, not calibrated. `weight_decay` was
  removed from its recipe-change keys (it is not used).
- **GU:** in the paper and code GU is a plug-in on a base objective, with the KL-to-reference anchor, a
  Gram–Schmidt basis of up to 8 fp16 directions, the last 3 blocks plus output layers, sign-selective capped
  tangents and a retain tangent. `npo_rt_gu` (`vlu/gu_npo_rt.py`) implements that on NPO-RT, with the
  choices in its docstring and METHODS.md; `tests/test_gu_npo_rt.py` has known-answer projections, basis
  growth, NPO-RT's exact update while the basis is empty, and exact resume. Memory: NPO-RT plus 0.1–0.5
  copies (about 42 GB). `gu` (GU-FM) stays in the code, documented, not calibrated.
- **Against the authors' code** (`tests/check_author_kernels.py`, now also `check_gu_npo_rt`): `npo_rt_gu`'s
  float16 basis and its whitened perpendicular/tangent projections equal the authors' `RetainNullProjector`
  with Adam whitening bound to the optimizer, bit for bit (10 seeds, max error 0); the rewritten Sharp
  gradient still matches their SAM class (32 configurations, max error 2.8e-16). The script's hash check
  now ignores a missing final newline (raw GitHub's `sam.py` has none).
- **Mutation checks:** 14 deliberate faults in the new code (sign test, cap, retain projection, whitening,
  residual threshold, refresh schedule, saved state, basis size, per-partition moments and updates, masked
  restore, step counts, smallest-split selection and its live rows) each make a test fail.

### Second review — 2026-11-27

- **Sharp MinMax under autocast (bug, in the merged code as well):** train.py runs `rule.step` in one bf16
  autocast region. Autocast caches the bf16 copy of each trainable fp32 weight and does not notice in-place
  changes (checked on torch 2.5.1: a forward after `weight.add_(1)` in the same region gives the same output).
  So the perturbed forward ran at the unperturbed weights: g1 = g0 exactly (same replayed noise), and Sharp
  MinMax was plain split descent/ascent; the forget phase also saw the weights before the retain update. The
  CPU tests replace autocast, so they could not see it. Fixed: `perturbed` clears the cache after moving and
  after restoring the weights. Two CUDA tests (skipped without CUDA; they run on the calibration machine)
  check the perturbed forward and a full step against fresh-region gradients; removing either clear fails them.
  `npo_rt_gu` and `gu` change no weight between forwards of a step.
- **Memory, re-estimated** on a toy with a frozen backbone (half a trainable copy), bf16 autocast and
  activations sized so that NPO-RT/GA matches the measured 39.5/33.4 GiB: `npo_rt_gu` about 39 GiB,
  `sharp_minmax` about 40 GiB with the full-model perturbation below (VLA-Forget ran at 42.0 GiB). Below the
  44 GB training threshold.
- **GU basis under bf16:** with fp32 weights the anchor is nonzero from the 3rd step and the basis reaches 8
  directions by step 17 at both 1e-6 and 3e-5 with the 5% warmup (toy), so `npo_rt_gu` does not silently stay
  NPO-RT. As in the authors' code the basis is then fixed.
- **Paper/code agreement re-checked:** GU's sign rule (`a b < -tau`), cap and `P_U g_r` retain term are the
  authors' `optimizer_step`; their g_f contains `alpha/gamma (grad NLL_r - grad KL)` (retain NLL in the
  total, KL as g_r), ours follows the paper's Algorithm 1 with one retain loss (documented). Sharp's
  perturbation: the paper (Algorithm 2) and code perturb all weights along the unmasked gradient; ours did
  only the active partition (the merged code's choice, not needed for partition safety). Switched to the
  paper's: all weights move and are saved/restored; only the blended update is masked. The author check now
  runs their SAM class through Algorithm 2's phase (masked second gradient, masked base step) on a loss that
  couples the partitions: max error 2.8e-16; the old partition-only perturbation fails it and two tests.
