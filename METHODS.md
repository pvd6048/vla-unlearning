# How each method is implemented, and where it departs from its paper

This file lists, method by method, what the paper and the authors' code do, what our code does, and
every place the two differ. We wrote it to be checked. Each claim names the reference file and line,
and the place in our code, so it can be verified in a few minutes. All reference repositories were
read at the commits below; the comparison was redone on 2026-11-02, file by file.

Each difference is labelled:

- **Same**: the computation of the paper or its code, applied to GR00T's loss.
- **Adapted**: the paper's ingredient does not exist for GR00T, so we substitute one. GR00T has no
  tokens, no classes and no likelihood; it is trained with a flow-matching (velocity regression) loss.
  The substitute is our assumption and can be argued with.
- **Choice**: the paper or code leaves it open, or we deliberately do something different. The
  reason is given.

| Method | Paper | Reference code (commit) | Our code |
|---|---|---|---|
| Learning with experience replay | Liu et al. 2026, continual GR00T (arXiv 2603.03818) | github.com/Continual-VLAs/continual-GR00T (`4bfd66f`), `scripts/gr00t_sequential_finetune.py` | `vlu/ops.py` `_plan_learn`, `vlu/data.py` `ReplayBuffer`, `vlu/train.py` |
| Gradient ascent (GA), Gradient difference (GradDiff) | standard baselines, as in TOFU (Maini et al. 2024) | NPO repo below, `TOFU/dataloader.py:141-147, 165-175` | `vlu/ops.py` `_plan_ga`, `_plan_graddiff` |
| NPO | Zhang et al. 2024 | github.com/licong-lin/negative-preference-optimization (`1207bbe`), `TOFU/dataloader.py:255-270`, `TOFU/data_module.py:242-250` | `vlu/ops.py` `_plan_npo` |
| SSD | Foster et al. 2024 (AAAI) | github.com/if-loops/selective-synaptic-dampening (`cdfdc0e`), `src/ssd.py:151-225`, `src/forget_full_class_main.py:161-177` | `vlu/ops.py` `_plan_ssd` |
| SalUn | Fan et al. 2024 (ICLR) | github.com/OPTML-Group/Unlearn-Saliency (`a6e89fd`), `Classification/generate_mask.py:14-80`, `Classification/unlearn/RL.py` | `vlu/ops.py` `_plan_salun` |
| VLA-Forget | Ranjan & Polyzou 2026 (arXiv 2604.03956) | github.com/raviranjan-ai/VLA-Forget (`ccb3c98`), `VLA-Forget-main.py` (the only script with the full staged method), `VLA-Forget-OpenVLA.py` (LoRA settings) | `vlu/vlaforget.py`, `vlu/lora.py` |
| The model's loss | GR00T N1.6 | github.com/NVIDIA/Isaac-GR00T (`ead5283`), `gr00t/model/gr00t_n1d6/gr00t_n1d6.py:242-252` | called, not reimplemented: `vlu/ops.py` `fm_loss`, `fm_sse_per_sample` |

## 0. What applies to every method

**The loss is GR00T's own.**
- **What it computes:** every method calls the model's forward pass. That pass samples a timestep t
  and noise ε, and returns `loss` = the mean of (predicted velocity − (a − ε))² over the valid action
  entries. It also returns the per-entry `action_loss` (`gr00t_n1d6.py:242-252`).
- **No loss of our own:** we never re-derive the loss. "L_FM" below means this value.
- **Shared randomness:** where a method compares the model with a frozen copy (NPO, VLA-Forget), both
  see the same t and ε, because the random number generator's state is replayed for the second
  forward pass.

**Weights that can change.**
- **GR00T N1.6's fine-tuning scope:** the action head's state and action projectors, the DiT, and the
  top 4 LLM layers. The vision tower and the vision–language projector are frozen. This is NVIDIA's
  default, set in `vlu/model.py:43`.
- **Choice:** GA, GradDiff, NPO, SSD and SalUn act only on this scope.
- **VLA-Forget is the exception (Adapted):** it is defined by edits to the vision encoder and the
  vision–language projector, so its LoRA stages may change frozen layers. Those changes are saved as
  "frozen overrides" (`vlu/model.py`).
- **For comparisons:** this means VLA-Forget may change more of the model than the baselines.
  Comparisons should state it.

**Same optimizer and budget for all methods (Choice).**
- **What we use:** AdamW with linear warm-up and cosine decay, restarted for each operation. Every
  unlearning operation gets 1000 steps.
- **What the papers used:**
  - NPO: paged AdamW;
  - SalUn: SGD with momentum;
  - VLA-Forget: Adam without a schedule, 210 steps;
  - SSD: no training.
- **Why we differ:** one optimizer and one step budget keep the comparison about the methods, not the
  training setup.
- **Learning rates:** these are not shared. Each method's learning rate (or strength, for SSD) is
  calibrated separately by `scripts/calibrate.py`.
- **VLA-Forget keeps its own setup:** it uses the authors' Adam without a schedule (Same), inside the
  shared step budget.

**Samples instead of full passes over the data (Choice).**
- **The papers:** SSD and SalUn compute importances over the whole forget and training sets.
- **We:** use a fixed number of batches (`importance_batches` = 64 batches of 16 frames). A full pass
  over every frame of every LIBERO demonstration would cost more than the unlearning itself.
- **Effect:** the estimate is noisier, not biased in a known direction.

## 1. Learning a task with experience replay (Liu et al.)

| | Liu et al. code | Ours | |
|---|---|---|---|
| Batch | 16 frames of the new task + 16 from the replay buffer (`ERDataCollator`, l. 160-173; `BATCH_SIZE=16` in `bash_scripts/generate_train_commands.sh:11`) | 16 + 16 | Same |
| Buffer | first `n_memories` = 1000 entries of a fixed seeded permutation per earlier task (l. 227-238) | 1000 random frames per retained task, fixed per (run seed, task) | Same |
| Optimizer | lr 1e-4, weight decay 1e-5, 5% warm-up, cosine (l. 90-96, 410-413) | same | Same |
| Steps per task | set per experiment | 10 000, to be set from the calibration's 2k/5k/10k snapshots | Choice |
| Trained weights | Liu et al. trained GR00T **N1.5** (`tune_llm=False`, `tune_visual=False`) | GR00T **N1.6** with NVIDIA's N1.6 default (adds the top 4 LLM layers) | Choice: N1.6 is the current model; an N1.5 run to reproduce their numbers is planned (PLAN.md) |
| Unlearned tasks | no unlearning | an unlearned task is never replayed again | (our setting) |

## 2. Gradient ascent and gradient difference

- GA minimizes −L_FM(forget). TOFU's GA uses −(mean token cross-entropy) (`dataloader.py:141-147`).
  **Adapted:** the token loss becomes L_FM.
- GradDiff minimizes L_FM(retain) − L_FM(forget), with both weights 1 as in TOFU
  (`dataloader.py:165-175`). **Same**, apart from the loss.
- Retain data are frames of all tasks the parent model still keeps (Choice; see "Open points").

## 3. NPO

| | Official code | Ours | |
|---|---|---|---|
| Loss | `-logsigmoid(β·(NLL_θ − NLL_ref)).mean() · 2/β` (`dataloader.py:270`) | `softplus(β·(log p_θ − log p_ref)).mean() · 2/β` | **Same** (the two expressions are identical) |
| Per-sample log-likelihood | − sum of token cross-entropies of the sequence (`data_module.py:242-250`) | − sum of squared flow-matching errors of the action chunk at one sampled (t, ε) | **Adapted** (see below) |
| Reference model | the model before unlearning (`ref_policy='fine_tuned'`) | the parent model, frozen | Same |
| β | 0.1 (`config/forget.yaml:20`) | 0.1; β = 1 also in the calibration | Same, plus a calibration check |
| Retain term | the repo's default config is NPO + GradDiff retain term (`forget_loss: npo_grad_diff`, `forget.yaml:14`); the paper reports NPO, NPO-RT and NPO-KL | NPO alone (forget term only) | **Choice**, see "Open points" |

**Why −SSE is only a stand-in.** A flow-matching model has no tractable per-sample likelihood. We use
−SSE because:
- −SSE is the log-density of a Gaussian with variance ½ around the predicted velocity;
- it falls when the model fits a chunk worse, like a log-likelihood would;
- it is computed exactly as GR00T's training loss is, only summed per sample.

Its limits:
- **One sampled timestep:** it is a single-timestep, unweighted estimate. It is not a bound on the
  log-likelihood (a diffusion ELBO would weight and integrate over t).
- **Scale is coupled to β:** the "variance ½" sets the scale of the log-ratio. Any other variance is
  equivalent to rescaling β. This is why β is calibrated rather than copied from the paper.

## 4. SSD (Selective Synaptic Dampening)

| | Official code | Ours | |
|---|---|---|---|
| Importance | Σ over batches of grad², ÷ number of batches, gradients of the batch-mean loss (`ssd.py:161-180`) | Σ over 64 batches of grad² of the batch-mean L_FM (same number of batches for both sets, so the ÷ cancels in the ratio) | Same |
| Data | forget set; full training set including the forget data (`forget_full_class_main.py:152-156`) | forget task; all tasks the parent keeps, including the forget task | Same |
| Selection | weights with I_forget > α · I_full (`ssd.py:206-207`) | same | Same |
| Dampening | multiply by min(λ · I_full / I_forget, 1) (`ssd.py:210-218`) | same | Same |
| α, λ | α = 10 · `model_size_scaler`, with scaler 0.5 for ViT, so α = 5 for transformers; λ = 1 (`forget_full_class_main.py:161-177`) | α = 5, λ = 1; α = 10 and 20 also in the calibration | Same, plus a calibration check |
| Model mode | `model.eval()` | eval mode | Same |
| Which weights | all parameters of the classifier | the trainable scope only (section 0) | **Choice** |

## 5. SalUn

| | Official code | Ours | |
|---|---|---|---|
| Saliency | \|Σ over forget batches of grad(−CE)\| in eval mode (`generate_mask.py:24-48`) | \|Σ over 64 forget batches of grad(L_FM)\| in eval mode (the sign does not matter under \|·\|) | Same, with L_FM |
| Mask | global top fraction over all weights (`generate_mask.py:50-79`); 50% is the paper's default | global top 50% of the weights that can change. LIBERO's row of the per-embodiment projectors counts; the 31 rows that never get a gradient do not. Zero saliency is never selected. | Same, applied to the weights that can move |
| Masked weights | gradient × mask, then restored to their old values after every step (`RL.py:11-34, 84-90`) | gradient × mask, no weight decay, so masked weights never change (checked by a test) | Same |
| "Random label" | a uniformly random class (`RL.py:53-55`) | the action chunk of a random frame of a retained task | **Adapted** (see below) |
| Data mix | relabelled forget set + retain set concatenated and shuffled, so the mix follows the data sizes (`RL.py:57-59`) | each batch splits forget and retain samples in proportion to their frame counts; the loss is the mean over the mixed batch | Same |

**Random labels.** Action chunks are continuous, so there are no classes to randomize. We needed a
stand-in that is still a valid action for the robot but wrong for the observation. We take a retained
task's action chunk: it is valid for the robot but belongs to another observation. Other stand-ins
would also be defensible, such as Gaussian noise chunks or chunks from the same task at another time,
and they could behave differently. This is an assumption to state in the paper.

## 6. VLA-Forget

**What we followed.**
- **The script:** the authors' `VLA-Forget-main.py`, step by step. It is the only script that
  implements the staged method.
- **From the paper:** the parts that script simplifies: LoRA, gradient clipping and the
  nearest-neighbour boundary set.
- **The full list of choices:** it is in the docstring of `vlu/vlaforget.py`, and the values used are
  written to `vlaforget.json` in every run.

| | Authors' code / paper | Ours | |
|---|---|---|---|
| Stages and losses | three stages; per step four gradients g1–g4: retain CE + β·KL(base‖θ), −λ_f·forget CE, −λ_m·KL(base‖θ) on forget, λ_feat·(feature MSE on retain + boundary) (`main.py:656-664`) | same four gradients with L_FM in place of CE | Same, with L_FM |
| Combining | PCGrad in a random order, each gradient projected onto the *already projected* others, then the **mean** (`main.py:498-512`) | identical (`pcgrad`); hand-computed test cases | Same |
| KL between policies | KL of action-token distributions (`main.py:335-343`) | mean squared difference of the two models' predicted velocities at the same (t, ε) | **Adapted** (see below) |
| Features h_V, h_P | vision features and projector output (`main.py:346-349`) | SigLIP2 output after its final layer norm, and the output of `mlp1` (the vision → LLM projector) | Adapted (closest GR00T modules; names checked in GR00T's source) |
| Layer selection | vision and projector stages: ‖g_f‖/‖θ‖ · (1 − cos(g_f, g_r))^α; reasoning stage: ‖g_f‖/‖g_r‖. g_r uses the retain CE + 0.5·KL (`main.py:518-560`) | same formulas, same 0.5 (β_kl) | Same |
| Selection gradients | 1 batch (code); the whole set (paper) | mean over 16 batches | Choice (between the two) |
| Candidate layers | vision blocks, projector, reasoning blocks and action head of the authors' small model | linear layers of the last 4 SigLIP2 blocks, of `mlp1`, and of the trainable LLM layers and the DiT | Adapted (the paper does not say how many vision blocks are "late") |
| top-k | 2 / 2 / 3 (`main.py:602-604`) | 2 / 2 / 3 | Same |
| λ_f, λ_m, λ_feat, β_kl, α | code: 1.2, 0.8, 0.7, 0.5, 1 (`main.py:609-613`); paper (OpenVLA): λ_f = 0.7 | 0.7, 0.8, 0.7, 0.5, 1 | Same as the paper (the code's 1.2 is for its small model) |
| lr, optimizer | 2e-4, Adam, no schedule (`main.py:608, 631`); appendix also mentions 5e-5 | 2e-4 (5e-5 also calibrated), Adam, no schedule, clipping at 1.0 (paper) | Same |
| LoRA | r 16, α 16, dropout 0.05, `init_lora_weights="gaussian"` (`OpenVLA.py:419-425`) | r 16, α 16, Gaussian init (A ~ N(0, 1/r²), B = 0), no dropout | Same, except dropout (below) |
| Train/eval mode | `model.train()` | eval mode for both models (no dropout) | **Choice** (below) |
| Stage lengths | 60 / 60 / 90 steps | the op's step budget split 60 : 60 : 90 | Choice (same proportions, our budget) |
| Boundary set | random retain frames (code); nearest neighbours to the forget data (paper) | for each of 256 forget frames, the most similar of 2048 retain frames by the frozen model's pooled vision features | Same as the paper |
| End | adapters stay as separate modules | adapters merged into the weights; the run fails if merging changes a probe loss by more than 1% | Choice (a node is one set of weights) |

**Velocity difference as the KL.** For two Gaussian policies with the same variance, KL equals the
squared difference of their means up to a constant factor. The predicted velocity at a given (t, ε)
determines the flow model's update at that point. So the squared velocity difference is the natural
analogue, but it is not a KL between action distributions. It measures one point of the flow (one t
and one ε per sample), not the whole generated distribution.

**No dropout.** GR00T's DiT has dropout 0.2. In training mode the adapted model and the frozen copy
draw different dropout masks. The KL and feature terms would then be non-zero even for identical
weights, so they would measure dropout noise rather than drift. We therefore run both models in eval
mode. LoRA dropout (0.05 in the authors' OpenVLA script) is off for the same reason.

## 7. Found and fixed during this audit

- **LoRA initialization.** We used peft's default (Kaiming uniform). The authors' OpenVLA script uses
  `init_lora_weights="gaussian"` (A ~ N(0, 1/r²)). Since B = 0 at the start, both leave the model
  unchanged, but they change the first steps' updates. We switched to the authors' initialization;
  `tests/test_vlaforget.py` checks it.
- **Known-answer tests against the official code.** Our NPO and SSD tests only checked properties
  (the log-ratio is 0 at the start; SSD only shrinks weights). Two tests now compare against the
  official code:
  - **NPO:** our loss equals the official formula (`-logsigmoid(β·(NLL_θ − NLL_ref)).mean()·2/β`) on a
    perturbed model.
  - **SSD:** the official `modify_weight`, copied verbatim, and ours give bit-identical weights from
    the same importances.

  Both are in `tests/test_ops.py`.
- **README.** It still said SSD α = 10 (changed to the ViT value 5 earlier) and "lr 1e-5 for every
  method" (VLA-Forget uses 2e-4). Both corrected.

Earlier rounds of the same comparison changed:
- the replay batch to 16 + 16 from fixed 1000-frame buffers;
- SalUn's data mix to follow the data sizes;
- SSD α from 10 to 5;
- VLA-Forget to run without dropout.

## 8. What is verified, and what is not

**Verified on the laptop (`tests/`):**
- **The formulas:** checked on small fake models that have GR00T N1.6's module names and layout, with
  known answers:
  - **NPO:** equals the official loss formula on a perturbed model. θ and the reference see the same
    noise, so the log-ratio is exactly 0 before the first step.
  - **SSD:** bit-identical to the official `modify_weight` on the same importances. It only shrinks
    weights, and never touches the 31 embodiment rows that cannot change.
  - **SalUn:** the mask keeps 50% of the weights that can change. Masked weights stay bit-identical
    through training, and a resumed run reuses the same mask. The forget/retain split follows the
    data sizes.
  - **VLA-Forget:**
    - PCGrad equals hand-computed answers;
    - the velocity KL and feature losses are exactly 0 before the first step;
    - the candidate layers are the right ones;
    - LoRA starts with the Gaussian initialization and leaves the model unchanged;
    - merging is exact in fp32, and a lost update is reported in bf16;
    - a merge beyond tolerance fails the run;
    - the boundary set is reproducible;
    - the stage lengths and logging steps are as specified.
  - **Experience replay:** batches split equally between the new task and the replay buffer, with
    replay drawn only from the fixed per-task buffers.
- **Checked by reading only:** GA and GradDiff (a sign and a sum), the SalUn saliency formula and
  VLA-Forget's layer-selection scores. The comparison is in the tables above, with line numbers.
- **Names in GR00T's source:** `loss`, `action_loss`, `action_mask`, `action_decoder`, `mlp1` and
  `post_layernorm` were read in Isaac-GR00T at the pinned commit.

**Not verified yet:**
- **Not run on the real model:** nothing has run on the real GR00T weights, the LIBERO data or the
  simulator. The calibration run (`CALIBRATION.md`) is the first time. It checks that every method
  runs and that its losses move in the expected direction, and it sets each method's learning rate or
  strength.
- **No reproduction of published numbers:** we have not reproduced any paper's results, so a port
  could run and still differ from the original in an effect we have not thought of. Two checks are
  planned:
  - Liu et al.'s GR00T N1.5 + LIBERO numbers, for the replay recipe;
  - VLA-Forget on OpenVLA (PLAN.md, "Bridge check"), for our port of VLA-Forget.

## Open points where we would value your view

1. **NPO with or without a retain term.** Our NPO uses only the forget data, while GradDiff, SalUn and
   VLA-Forget also train on retained data. The NPO authors' default configuration is NPO + retain
   loss. Adding NPO-RT is a one-line loss and would make the comparison fairer.
2. **The likelihood stand-ins:**
   - −SSE at one timestep, for NPO;
   - the velocity difference, for VLA-Forget's KL;
   - a retained task's action chunk, as SalUn's "random label".

   Each is a defensible choice, but each is ours, not the authors'.
3. **Which weights the baselines may change.** They change only GR00T's fine-tuning scope; VLA-Forget
   also changes vision layers. SSD and SalUn could be run on the whole model instead.
4. **Retain data for unlearning:** all demonstrations of the kept tasks (current), or only the
   1000-frame replay buffer that learning used.
