"""VLA-Forget (Ranjan & Polyzou 2026, arXiv 2604.03956) ported from OpenVLA to GR00T N1.6.

References: the paper, and the authors' code github.com/raviranjan-ai/VLA-Forget (commit ccb3c98),
whose VLA-Forget-main.py is the only script implementing the full staged method. This port follows
that script step by step and takes from the paper what the script simplifies (LoRA, gradient
clipping, nearest-neighbour boundary set).

Three stages, each: (1) score the candidate linear layers on the current model with forget and
retain gradients, keep the top-k; (2) put LoRA adapters on them (earlier adapters stay, frozen);
(3) train the adapters for the stage's step budget with four gradients combined by PCGrad:
    g1 = grad[ L_FM(retain) + beta * KL(theta0 || theta)(retain) ]
    g2 = grad[ -lambda_f * L_FM(forget) ]
    g3 = grad[ -lambda_m * KL(theta || theta0)(forget) ]                       (mismatch)
    g4 = grad[ lambda_feat * (F(retain) + F(boundary)) ],  F = MSE(h_V, h_V0) + gamma * MSE(h_P, h_P0)
At the end every adapter is merged into its base weight. theta0 = the model before the op (frozen copy).

Mapping to GR00T (OpenVLA -> GR00T):
    stage 1 "late vision blocks"  -> linear layers of the last `vision_blocks` SigLIP2 encoder layers
           score phi = ||g_f|| / (||theta|| + eps) * (1 - cos(g_f, g_r))^alpha, top_k_vision
    stage 2 projector             -> linear layers of the vision-language projector (backbone mlp1)
           same score, top_k_projector
    stage 3 upper reasoning/action blocks -> linear layers of the trainable LLM layers and of the DiT
           (GR00T generates actions with the DiT); score Sig = ||g_f|| / (||g_r|| + eps), top_k_reasoning
    token cross-entropy  -> flow-matching loss L_FM
    KL between policies  -> mean squared difference of the two models' predicted velocities at the same
                            noisy action and timestep (the flow-model analogue of a KL between policies)
    h_V, h_P             -> vision tower output (post-layernorm), projector (mlp1) output

Choices where the paper and the authors' code differ or say nothing (recorded in vlaforget.json):
    - LoRA r=16, alpha=16 on the selected layers (paper; the small script updates them directly)
    - both models run in eval mode (no dropout): GR00T's DiT has dropout 0.2, and with dropout the
      adapted model and the frozen copy draw different masks, so the KL terms would measure dropout
      noise rather than drift. Hence also no LoRA dropout (paper: 0.05).
    - PCGrad: shuffled order, projections onto the already-projected gradients, then the MEAN (code)
    - Adam without schedule (code), gradient clipping at 1.0 (paper)
    - selection gradients: averaged over `selection_batches` batches (paper: the whole set; code: 1 batch)
    - boundary set: for each frame of a random pool of forget frames, the retain frame with the most
      similar pooled vision features under theta0 (paper; the code uses random retain frames)
    - fixed stage budgets in the ratio 60/60/90 of the op's step budget; no early stopping and no
      layer expansion (the code has neither; the paper gives no thresholds)
    - no LM-head/action-token rows (OpenVLA-specific)
    - not resumable mid-op: an interrupted op restarts from the parent (it is short)
"""

from dataclasses import asdict
import copy
import json
import math
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from vlu import lora
from vlu.config import Recipe
from vlu.data import Frames, OpBatches, make_sample
from vlu.model import add_frozen_overrides
from vlu.ops import OpContext
from vlu.train import jsonl_logger, probe_losses

EPS = 1e-8
STAGES = ("vision", "projector", "reasoning")


# --- Where things are in GR00T N1.6 -------------------------------------------------------------

_VISION_LAYER = re.compile(r"^backbone\.model\.vision_model\.vision_model\.encoder\.layers\.(\d+)\.")
_LLM_LAYER = re.compile(r"^backbone\.model\.language_model\.model\.layers\.(\d+)\.")


def candidates(model, trainable_names: set[str], vision_blocks: int) -> dict[str, list[str]]:
    """Linear layers each stage may select (names as in model.named_modules())."""
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    n_vision = 1 + max(int(m[1]) for n in linears if (m := _VISION_LAYER.match(n)))
    trainable_llm = {int(m[1]) for n in trainable_names if (m := _LLM_LAYER.match(n))}
    out = {
        "vision": [n for n in linears if (m := _VISION_LAYER.match(n)) and int(m[1]) >= n_vision - vision_blocks],
        "projector": [n for n in linears if n.startswith("backbone.model.mlp1.")],
        "reasoning": [n for n in linears if ((m := _LLM_LAYER.match(n)) and int(m[1]) in trainable_llm)
                      or n.startswith("action_head.model.")],
    }
    for stage, names in out.items():
        if not names:
            raise RuntimeError(f"VLA-Forget found no candidate layers for stage {stage!r}")
    return out


class Taps:
    """Forward hooks keeping the outputs VLA-Forget's losses need: vision features (h_V), projector
    features (h_P) and the predicted velocity."""

    def __init__(self, model):
        self.out = {}
        modules = {
            "vision": model.backbone.model.vision_model.vision_model.post_layernorm,
            "projector": model.backbone.model.mlp1,
            "velocity": model.action_head.action_decoder,
        }
        for name, module in modules.items():
            module.register_forward_hook(lambda _m, _i, output, name=name: self.out.__setitem__(name, output))


def paired_forward(model, ref, taps: Taps, ref_taps: Taps, inputs: dict):
    """Forward theta (with graph) and theta0 (no graph) on the same inputs with the same flow-matching
    noise and timesteps (the RNG state is replayed). Returns (theta's output, theta's taps, theta0's taps)."""
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inputs)
    mine = dict(taps.out)
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1])
    ref.train(model.training)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref(inputs)
    return out, mine, dict(ref_taps.out)


def velocity_kl(out: dict, mine: dict, theirs: dict) -> torch.Tensor:
    """KL stand-in between the two flow policies: masked mean squared velocity difference."""
    mask = out["action_mask"].float()
    horizon = mask.shape[1]
    diff = mine["velocity"][:, -horizon:].float() - theirs["velocity"][:, -horizon:].float()
    return (diff.square() * mask).sum() / mask.sum()


def feature_loss(mine: dict, theirs: dict, gamma: float) -> torch.Tensor:
    mse = torch.nn.functional.mse_loss
    return mse(mine["vision"].float(), theirs["vision"].float()) + gamma * mse(mine["projector"].float(), theirs["projector"].float())


# --- Gradients -------------------------------------------------------------------------------------


def flat_grad(loss: torch.Tensor, params: list, retain_graph: bool = False) -> torch.Tensor:
    """d loss / d params as one flat fp32 vector (zeros where the loss does not depend on a param)."""
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in params), device=params[0].device)
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).float().flatten() for p, g in zip(params, grads)])


def pcgrad(grads: list[torch.Tensor], generator: torch.Generator) -> tuple[torch.Tensor, int]:
    """PCGrad exactly as the authors' code: in a random order, project each gradient off every other
    (already projected) gradient it conflicts with; return the mean and the number of projections."""
    proj = [g.clone() for g in grads]
    order = torch.randperm(len(proj), generator=generator).tolist()
    conflicts = 0
    for i in order:
        for j in order:
            if i == j:
                continue
            dot = proj[i] @ proj[j]
            if dot < 0:
                proj[i] = proj[i] - dot / (proj[j] @ proj[j]).clamp_min(1e-12) * proj[j]
                conflicts += 1
    return torch.stack(proj).mean(dim=0), conflicts


def set_grads(params: list, flat: torch.Tensor) -> None:
    i = 0
    for p in params:
        p.grad = flat[i : i + p.numel()].view_as(p).to(p.dtype).clone()
        i += p.numel()


# --- Selection and boundary set -------------------------------------------------------------------


def selection_scores(ctx: OpContext, ref, taps, ref_taps, names: list[str], score: str, cfg, forget_slots, retain_slots,
                     stream: int) -> dict[str, float]:
    """Score candidate layers with the gradients of L_forget = L_FM(forget) and
    L_retain = L_FM(retain) + beta * KL (as the authors' code), averaged over cfg.selection_batches batches."""
    model = ctx.model
    params = {n: list(model.get_submodule(n).parameters()) for n in names}
    flat_params = [p for ps in params.values() for p in ps]
    for p in flat_params:
        p.requires_grad_(True)
    sums = {}
    model.eval()  # see the module docstring: no dropout, so theta and theta0 compute alike
    for kind, slots in (("forget", forget_slots), ("retain", retain_slots)):
        total = [torch.zeros_like(p, dtype=torch.float32) for p in flat_params]
        for inputs in ctx.batches(slots, cfg.selection_batches, stream=stream + (kind == "retain")):
            out, mine, theirs = paired_forward(model, ref, taps, ref_taps, inputs)
            loss = out["loss"] if kind == "forget" else out["loss"] + cfg.beta_kl * velocity_kl(out, mine, theirs)
            grads = torch.autograd.grad(loss, flat_params, allow_unused=True)
            for t, g in zip(total, grads):
                if g is not None:
                    t += g.float()
        sums[kind] = total
    for p in flat_params:
        p.requires_grad_(False)

    scores, i = {}, 0
    for name, ps in params.items():
        k = len(ps)
        g_f = torch.cat([t.flatten() for t in sums["forget"][i : i + k]]) / cfg.selection_batches
        g_r = torch.cat([t.flatten() for t in sums["retain"][i : i + k]]) / cfg.selection_batches
        i += k
        if score == "ratio":
            theta = torch.cat([p.detach().float().flatten() for p in ps])
            cos = float(g_f @ g_r / (g_f.norm() * g_r.norm()).clamp_min(1e-12))
            scores[name] = float(g_f.norm() / (theta.norm() + EPS)) * (1.0 - cos) ** cfg.alpha_ratio
        else:
            scores[name] = float(g_f.norm() / (g_r.norm() + EPS))
    return scores


@torch.no_grad()
def boundary_set(ctx: OpContext, ref, ref_taps, forget_task: str, retain_tasks: tuple, cfg) -> tuple[list, dict]:
    """For each frame of a random pool of forget frames, the retain frame whose pooled vision features
    (theta0) are most similar (cosine). Saved in the node directory and reused by a rerun."""
    path = ctx.node.dir / "vlaforget_boundary.json"
    if path.exists():
        saved = json.loads(path.read_text())
        return [tuple(f) for f in saved["frames"]], saved["summary"]
    rng = np.random.default_rng([ctx.seed, 77])
    f_demos = ctx.demos[forget_task]
    n_forget = min(cfg.boundary_forget_pool, len(f_demos.index))
    forget_pool = [(forget_task, *f_demos.index[i]) for i in rng.choice(len(f_demos.index), n_forget, replace=False)]
    retain_pool = []
    for _ in range(cfg.boundary_retain_pool):
        letter = retain_tasks[rng.integers(len(retain_tasks))]
        retain_pool.append((letter, *ctx.demos[letter].index[rng.integers(len(ctx.demos[letter].index))]))

    was_training = ctx.processor.training
    ctx.processor.eval()
    ref.eval()

    def pooled(frames):
        feats = []
        for start in range(0, len(frames), 32):
            chunk = frames[start : start + 32]
            samples = [make_sample(ctx.processor, ctx.modality_configs, ctx.demos[l], e, s) for l, e, s in chunk]
            torch.manual_seed(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ref(dict(ctx.processor.collator(samples)["inputs"]))
            h = ref_taps.out["vision"].float()  # (images, patches, C); images of a sample are contiguous
            if h.shape[0] % len(chunk):
                raise RuntimeError(f"{h.shape[0]} vision rows for {len(chunk)} samples")
            feats.append(h.reshape(len(chunk), -1, h.shape[-1]).mean(dim=1))
        return torch.nn.functional.normalize(torch.cat(feats), dim=1)

    f_feats, r_feats = pooled(forget_pool), pooled(retain_pool)
    if was_training:
        ctx.processor.train()
    sims = f_feats @ r_feats.T
    best = sims.max(dim=1)
    frames = [retain_pool[j] for j in best.indices.tolist()]
    summary = {"forget_pool": len(forget_pool), "retain_pool": len(retain_pool), "distinct_frames": len(set(frames)),
               "mean_cosine": float(best.values.mean()), "tasks": {t: sum(f[0] == t for f in frames) for t in retain_tasks}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "frames": frames}) + "\n")
    return frames, summary


# --- The op ---------------------------------------------------------------------------------------


def stage_steps(total: int, ratio: tuple) -> list[int]:
    steps = [math.floor(total * r / sum(ratio)) for r in ratio]
    steps[-1] += total - sum(steps)
    return steps


def run(ctx: OpContext, probes: dict, recipe: Recipe, forget_task: str, retain_tasks: tuple, total_steps: int) -> dict:
    """Run VLA-Forget on ctx.model in place; afterwards the model has no adapters, its original
    requires_grad flags, and frozen overrides registered for the edited frozen weights."""
    cfg = recipe.vlaforget
    model, node = ctx.model, ctx.node
    log = jsonl_logger(node.dir / "train_log.jsonl")
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    trainable = {n for n, f in flags.items() if f}
    ref = copy.deepcopy(model).eval()
    ref.requires_grad_(False)
    taps, ref_taps = Taps(model), Taps(ref)
    model.requires_grad_(False)

    forget_slots = [(forget_task,)] * recipe.batch_size
    retain_slots = [retain_tasks] * recipe.batch_size
    boundary, boundary_summary = boundary_set(ctx, ref, ref_taps, forget_task, retain_tasks, cfg)
    cands = candidates(model, trainable, cfg.vision_blocks)
    top_k = {"vision": cfg.top_k_vision, "projector": cfg.top_k_projector, "reasoning": cfg.top_k_reasoning}
    report = {"config": asdict(cfg), "boundary": boundary_summary, "stages": {}}
    log({"step": 0, "stage": "start", "probe": probe_losses(model, probes)})

    adapters, done = {}, 0
    for s_index, (stage, steps) in enumerate(zip(STAGES, stage_steps(total_steps, cfg.stage_ratio))):
        started = time.time()
        scores = selection_scores(ctx, ref, taps, ref_taps, cands[stage], "significance" if stage == "reasoning" else "ratio",
                                  cfg, forget_slots, retain_slots, stream=10 * s_index)
        chosen = sorted(scores, key=scores.get, reverse=True)[: top_k[stage]]
        for a in adapters.values():
            a.lora_A.requires_grad_(False)
            a.lora_B.requires_grad_(False)
        new = lora.attach(model, chosen, cfg.lora_r, cfg.lora_alpha, dropout=0.0)
        adapters.update(new)
        params = [p for a in new.values() for p in (a.lora_A, a.lora_B)]
        optimizer = torch.optim.Adam(params, lr=cfg.lr)
        data = OpBatches(ctx.demos, {"forget": forget_slots, "retain": retain_slots, "boundary": Frames(boundary, recipe.batch_size)},
                         ctx.processor, ctx.modality_configs, seed=ctx.seed * 1000 + 500 + s_index, num_steps=steps)
        loader = DataLoader(data, batch_size=None, sampler=range(steps), num_workers=recipe.num_workers,
                            prefetch_factor=4, pin_memory=True, multiprocessing_context="fork")
        generator = torch.Generator().manual_seed(ctx.seed * 1000 + s_index)
        model.eval()  # no dropout (module docstring); gradients still flow
        window = {}
        for step, batch in zip(range(steps), loader):
            out_f, mine_f, ref_f = paired_forward(model, ref, taps, ref_taps, batch["forget"])
            forget_fm, mismatch = out_f["loss"], velocity_kl(out_f, mine_f, ref_f)
            g2 = flat_grad(-cfg.lambda_forget * forget_fm, params, retain_graph=True)
            g3 = flat_grad(-cfg.lambda_mismatch * mismatch, params)
            out_r, mine_r, ref_r = paired_forward(model, ref, taps, ref_taps, batch["retain"])
            retain_fm, retain_kl = out_r["loss"], velocity_kl(out_r, mine_r, ref_r)
            feat_r = feature_loss(mine_r, ref_r, cfg.gamma)
            g1 = flat_grad(retain_fm + cfg.beta_kl * retain_kl, params, retain_graph=True)
            g4 = flat_grad(cfg.lambda_feat * feat_r, params)
            _, mine_m, ref_m = paired_forward(model, ref, taps, ref_taps, batch["boundary"])
            feat_m = feature_loss(mine_m, ref_m, cfg.gamma)
            g4 = g4 + flat_grad(cfg.lambda_feat * feat_m, params)
            g, conflicts = pcgrad([g1, g2, g3, g4], generator)
            set_grads(params, g)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            metrics = {"forget_fm": forget_fm.item(), "mismatch": mismatch.item(), "retain_fm": retain_fm.item(),
                       "retain_kl": retain_kl.item(), "feat": (feat_r + feat_m).item(), "conflicts": conflicts, "grad_norm": grad_norm}
            for k, v in metrics.items():
                window[k] = window.get(k, 0.0) + v
            done += 1
            if (step + 1) % recipe.log_every == 0 or step + 1 == steps:
                n = recipe.log_every if (step + 1) % recipe.log_every == 0 else (step + 1) % recipe.log_every
                record = {"step": done, "stage": stage, **{k: v / n for k, v in window.items()}}
                if step + 1 == steps:
                    record["probe"] = probe_losses(model, probes)
                log(record)
                window = {}
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
        report["stages"][stage] = {"steps": steps, "candidates": len(scores), "selected": chosen, "top_scores": top,
                                   "minutes": (time.time() - started) / 60}
        print(f"VLA-Forget stage {stage}: {steps} steps on {chosen}", flush=True)

    # Merge, and check that the merged model computes what the adapted model computed.
    before = probe_losses(model, probes)
    changed = lora.merge(model, adapters)
    after = probe_losses(model, probes)
    for name, p in model.named_parameters():
        p.requires_grad_(flags[name])
    overrides = sorted(set(changed) - trainable)
    add_frozen_overrides(model, overrides)
    rel = {t: abs(after[t] - before[t]) / max(abs(before[t]), 1e-12) for t in before}
    report["merge"] = {"changed_weights": changed, "frozen_overrides": overrides, "probe_before": before,
                       "probe_after": after, "max_relative_change": max(rel.values())}
    if max(rel.values()) > 0.01:
        print(f"WARNING: merging the adapters changed a probe loss by {max(rel.values()):.2%} (bf16 rounding?)", flush=True)
    log({"step": done, "stage": "merged", "probe": after})
    (node.dir / "vlaforget.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report
