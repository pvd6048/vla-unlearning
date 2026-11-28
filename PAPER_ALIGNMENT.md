# GU and Sharp MinMax: current paper alignment

Updated 2026-11-27. This describes the current implementations and supersedes the earlier
partition-safe/shared-moment Sharp and separate-forget-gradient GU descriptions.

## Citation and source basis

- GU: [Geometric-disentanglement Unlearning, Algorithm 1](https://arxiv.org/html/2511.17100v4),
  [author trainer and projector at bbe4d507](https://github.com/Lemutisme/geo-unlearning/blob/bbe4d5077b21f6ded0ddbff8bf49b0336260626a/src/trainer/unlearn/geometric.py),
  [author configuration](https://github.com/Lemutisme/geo-unlearning/blob/bbe4d5077b21f6ded0ddbff8bf49b0336260626a/configs/trainer/GeometricUnlearn.yaml).
- Sharp: [Sharpness-Aware Machine Unlearning, Algorithms 1/2 and E.1/E.2](https://arxiv.org/html/2506.13715v3),
  [SAM at 98c41d9e](https://github.com/HaoranTang/sharp-unlearn/blob/98c41d9e344bd0942421bdcdeee85878d6a657b1/sam/sam.py),
  [training loop](https://github.com/HaoranTang/sharp-unlearn/blob/98c41d9e344bd0942421bdcdeee85878d6a657b1/unlearn/neggrad.py),
  [mask generation](https://github.com/HaoranTang/sharp-unlearn/blob/98c41d9e344bd0942421bdcdeee85878d6a657b1/generate_mask.py),
  [optimizer setup](https://github.com/HaoranTang/sharp-unlearn/blob/98c41d9e344bd0942421bdcdeee85878d6a657b1/unlearn/impl.py).

Use **GU on NPO-RT, adapted to flow matching** and **Sharp MinMax, adapted to flow matching**
in reports, citing the respective papers and this implementation revision. These are algorithm
adaptations to continual VLA, not reproductions of the papers' original experiments.
The older method identifier `gu` remains a legacy GU-inspired proposal projection; exclude it
when reporting the primary published-GU baseline. The primary method is `npo_rt_gu`.

## GU: literal Algorithm 1 gradient recovery

Implementation: `vlu/gu_npo_rt.py`; planning: `vlu/ops.py`.

1. Compute the joint gradient of gamma*NPO(forget) + alpha*FM(retain).
2. Compute g_r = grad(retain anchor), every step.
3. On refresh steps, grow each selected tensor's whitened anchor basis.
4. Recover g_f = (g_total - alpha*g_r)/gamma, exactly as Algorithm 1 and the author trainer.
5. Whiten using AdamW's second moment. Remove the basis component from g_f, add back only the
   opposite-signed tangent components, cap their norm, and retain the tangent projection of g_r.
6. Map back and combine gamma*forget_part + alpha*retain_part, then use the base AdamW update.

Defaults preserve the author's configuration: gamma=1.5, alpha=1, last three blocks plus output
layers, basis rank 8, refresh every two steps, fp16 basis storage, fp32 projection arithmetic,
residual threshold 1e-3, sign threshold zero, and tangent cap .5.
Unselected trainable parameters receive the weighted base gradient.

**Update after calibration round 3 (28 Nov):** this recovery (`gu_recovery=anchor`) fed back on GR00T
(anchor 0.002 -> 0.35, gradient norm 63 -> 1460, A's success rate 0.12-0.60 where NPO-RT kept 1.00).
`gu_recovery=separate` (round 4) keeps Algorithm 1's equations with one retain loss: g_f is the forget loss's
own gradient and g_r the retain FM loss's; the anchor only builds the basis. It equals the authors' rewrite
whenever the retain loss inside g_total is the anchor, Algorithm 1's own setting (`tests/check_author_kernels.py`).

The recovered g_f includes some retain-FM gradient because the anchor differs from the supervised
retain loss. This is intentional: it is the published/released recovery, rather than an independently
redefined forget gradient. With an empty basis the update equals the weighted base gradient.
To isolate projection against the existing unit-weight NPO-RT control, explicitly set
`gu_forget_weight=1`; the default gamma=1.5 comparison changes the base weight too. Calibration round 3
runs both: gamma=1.5 on NPO-RT's grid and gamma=1 at NPO-RT's working rates 3e-6 and 1e-5.

The author trainer attempts two differentiations of the same anchor graph on refresh steps,
while its basis helper frees that graph. Our implementation differentiates once with the graph
retained for the supervised retain loss, reusing the resulting gradient for basis and projection.
This fixes execution without changing the update equations. Before the first optimizer moment
exists our basis is empty and the anchor is zero, so the author's EMA fallback cannot affect the
first update; the weighted base update is used.

## Sharp: literal Algorithm 2 and exact Algorithm 1 saliency

Implementation: `vlu/sharp_minmax.py`, `vlu/saliency.py`; planning: `vlu/ops.py`.

For each retain-then-forget pair:

- Retain loss = alpha*FM(retain), sign=+1.
- Forget loss = -(1-alpha)*FM(forget), sign=-1.
- Compute full g0; perturb all trainable weights using SAM's global gradient norm.
- Compute g1 at the perturbed weights, restore the weights, mask only g1, then apply
  **g0 + sign*strength*(active_mask*g1 - g0)** with that phase's independent optimizer.

At strength=1, retain uses active_mask*g1 and forget uses 2*g0-active_mask*g1.
Thus the forget update can move the other partition. Weight decay and momentum can move entries
with zero current masked gradients too. No final mask or partition restoration is applied.
This preserves the literal paper pseudocode/released code instead of imposing stricter isolation.

Defaults: constant-rate SGD, momentum .9, alpha=.99, rho=.1, forget fraction .3, strength=1, vanilla
SAM, and weight decay **0** (an adaptation; `sharp_weight_decay=5e-4` restores the paper's value, applied
to all optimized tensors). The paper's 5e-4 is a from-scratch CIFAR regularizer: coupled SGD decay with
momentum .9 shrinks each weight by lr * wd / (1 - momentum) per optimizer, two per step, so at useful
SGD rates on GR00T it would shrink the pretrained weights by 18% (lr 0.02) to 99% (lr 0.5) over 1000
steps and dominate the result.
No extra gradient clipping or warmup/cosine scheduling is applied by default. `sharp_clip=1` clips each
phase's gradient at the repo's max_grad_norm (1.0): calibration round 3 ran without it and every variant's
forget ascent ran away to NaN (round 4 tries it on).
Two independent native optimizer states preserve per-parameter counters and phase histories.

There are two explicit paper/code ambiguities:

| Choice | Default and rationale | Explicit alternative |
|---|---|---|
| Mask direction | `sharp_select=smallest`: Algorithm 1 and released mask generator rank ascending | `largest`: Appendix E.2/section 4.2 description |
| Optimizer | `sharp_optimizer=sgd`: paper E.1's constant-rate SGD experiment recipe | `adamw`: released training script's unconditional override |

AdamW uses the released code's defaults (betas .9/.999, eps 1e-8) and the configured decay. Its two
full AdamW states need about 50 GiB on GR00T, more than a 48 GB GPU holds; SGD needs about 40 GiB
(toy measurements calibrated to the measured GA and NPO-RT peaks; see the review fixes below).
The same two-phase equations apply. Under SGD alpha affects the update scale substantially;
the old claim that alpha cancels applied approximately to AdamW, not to this default.

Saliency sums signed mean-FM gradients over **every valid forget frame once** in evaluation mode,
then takes the absolute value. It selects exactly floor(N*fraction) live trainable entries.
A CPU float32 order-statistic partition replaces the author's GPU argsort without approximating
the rank. Equal-cutoff entries are filled in parameter/flat order; source argsort tie order is
unspecified. All-zero saliency therefore still produces the requested partition.
Unused embodiment rows are excluded from this ranking so their many structural zeros cannot
dominate the split; ordinary optimizer decay can still affect those rows.
The preparation pass uses no augmentation and a fixed local RNG, restores training randomness,
and records frame count, batch count and actual fraction.

## Necessary model and continual-setting adaptations

| Adaptation | Reason and interpretation |
|---|---|
| Cross-entropy/NLL becomes supervised action flow-matching loss | GR00T predicts continuous velocities, not class/token probabilities |
| GU anchor is masked mean squared reference-velocity difference | A tractable policy KL is unavailable; this is a proxy, not a true KL or a transferred theoretical guarantee |
| NPO uses -SSE per action chunk as log-likelihood proxy | Same continuous-action adaptation as the existing NPO-RT baseline; beta needs VLA calibration |
| DiT last blocks and action output layers replace last LLM layers/head | Architectural mapping of the author's selected projection scope |
| Reference frozen at the start of each unlearning request | Continual requests start from the current policy |
| Common trainable scope, retained-task sampling and fixed request step budget | Preserve the repository's continual benchmark protocol; not the paper's original dataset/10-epoch experiment |
| Replay FM noise/timesteps for paired forwards (also replays dropout) | Evaluate changes at the same stochastic loss sample; otherwise independent FM noise contaminates the anchor/curvature |
| Clear autocast caches after in-place weight changes | Required for bf16 forwards to see the perturbed/current fp32 weights |
| Exact weight restoration, native optimizer checkpoints and finite checks | Execution/resume robustness; no extra projection or clipping added to Sharp |

Learning rates and request budgets remain explicit VLA calibration choices. The Sharp SGD grid
(2e-2, the paper's CIFAR rate, 1e-1 and 5e-1, with decay 0; derived in `scripts/calibrate.py` from the
measured gradient norms) is provisional; it is not a demonstrated stable robot-policy setting. GU retains the provisional NPO-RT rate bracket.
Fresh override tokens in `scripts/calibrate.py` prevent reuse of earlier GU/Sharp result directories.
Old GU update states, Sharp shared-moment states and sampled Sharp masks are rejected.
Start fresh operations, rather than resuming pre-correction checkpoints.

## Evidence and limits

| Claim | Disposition | Evidence |
|---|---|---|
| GU recovery, whitening, sign/cap and weighted combination follow the author | Verified for tested kernels | `tests/check_author_kernels.py`: actual author gradient rewrite, 10 seeds, maximum absolute error 2.38e-7 |
| GU basis construction follows the author | Verified for tested kernels | Same script, 10 seeds, basis and projections; maximum projection error 0 |
| Sharp's literal masked-g1 update follows released SAM | Verified for tested kernels | Same script: 32 sign/strength/rho/ASAM configurations, four SGD momentum/decay steps, error 8.88e-16 |
| Full-pass exact saliency, current weights under bf16, optimizer counters and resume work | Verified on CPU fixtures | `tests/test_saliency.py`, `tests/test_sharp_minmax.py`, `tests/test_gu_npo_rt.py` |
| Complete GR00T/LIBERO performance, CUDA peak memory, energy and cost | Indeterminate here | No GPU, dataset or model checkpoint available; skipped at user request |

The author-gradient rewrite comparison runs a non-refresh step because the unmodified released
trainer has the graph-reuse defect above; basis growth is compared separately. No author training
results were reproduced. The default SGD path and optional AdamW path preserve independent native
state, rather than the former shared-state approximation.

The pure calibration plan was also checked: all 27 variants (37 since the 27 Nov review) parse, pass recipe validation and generate
fresh GU/Sharp node paths. A Windows import stand-in for fcntl was used only for this pure check;
Linux worker locking/process behavior was not tested. The broader test_ops.py requires CUDA and
could not run in this CPU-only environment.

CPU regression environment: Python 3.13.5, torch 2.9.1+cpu, transformers 4.57.6, one torch thread,
`USE_TF=0`. Focused suite: 64 tests, 63 passed, the existing two-GPU RNG test skipped.
Reproduction commands:

```bash
USE_TF=0 python tests/test_gu_npo_rt.py
USE_TF=0 python tests/test_sharp_minmax.py
USE_TF=0 python tests/test_saliency.py
USE_TF=0 python tests/check_author_kernels.py /path/to/author_geometric.py /path/to/author_sam.py
```

Download the pinned author files linked above for the last command; the checker verifies normalized
source hashes before executing their classes. Existing performance/cost logging remains active.
Old toy GPU-memory extrapolations for the removed shared-state algorithm do not apply.
Two independent SGD momentum buffers or four AdamW moment buffers increase state/storage relative
to that shortcut. Exact ranking also requires one float32 pool over live entries in host RAM
(about 6.4 GB per running Sharp preparation). Round 3's cost meter measures the actual GR00T peaks.

## Review fixes — 2026-11-27

- **Sharp memory:** `step()` kept the retain phase's blended gradients (one weight copy) alive through the
  forget phase, whose second backward is the step's peak: about 45 GiB on GR00T, above run_local.sh's 44 GB
  training threshold and within ~2 GiB of an A6000. They are now released after the retain update, and the
  blend frees each tensor's g0/g1 as it goes: about 40 GiB (SGD), as NPO-RT. Estimates from a toy with a
  frozen backbone, bf16 autocast and activations sized so that GA and NPO-RT match their measured 33.4 and
  39.5 GiB; each kind measured in its own process.
- **Sharp decay and grid:** decay 0 and SGD rates 2e-2, 1e-1, 5e-1 (above and `scripts/calibrate.py`).
- **Sharp gradient norm:** computed without rescaling the gradients (`vlu.updates.total_norm`; `_phase` no
  longer takes an unused clipping argument).
- **GU:** the gamma=1 controls above.

