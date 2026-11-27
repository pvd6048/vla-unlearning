"""The training loop shared by every op.

By default, AdamW with linear warmup + cosine decay (fresh for each op), bf16 autocast over fp32 trainable
weights, and gradient clipping. Update rules can supply their published optimizer and schedule;
Sharp uses constant-rate SGD (or explicitly selected AdamW), with no clipping. The loop is
resumable: every `save_every` steps it writes resume.pt (weights, optimizer, scheduler, RNG, step),
and a rerun of the same op continues from there.
"""

import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from vlu import cost
from vlu.config import Recipe
from vlu.model import check_finite, save_weights, trainable_named_parameters
from vlu.nodes import Node
from vlu.ops import OpPlan, fm_loss


def jsonl_logger(path, keep_through: int = -1):
    """log(record): append record (plus a timestamp) to a JSON-lines file and print it. Records already
    in the file for steps after `keep_through` (left by an interrupted attempt) are dropped first, so
    a resumed op logs each step once; the default drops them all (a fresh start)."""

    def keep(line: str) -> bool:
        try:
            return json.loads(line)["step"] <= keep_through
        except (ValueError, KeyError):  # e.g. a line cut off by a crash
            return False

    if path.exists():
        kept = [line for line in path.read_text().splitlines() if keep(line)]
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(line + "\n" for line in kept))
        tmp.replace(path)

    def log(record: dict):
        record["time"] = time.time()
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    return log


def _optimizer(model, lr: float, weight_decay: float):
    # No weight decay on biases and norm weights (1-D tensors), like the HF Trainer.
    params = list(trainable_named_parameters(model).values())
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def _check_optimizer_finite(optimizer):
    check_finite({f"{i}.{name}": value for i, state in enumerate(optimizer.state.values())
                  for name, value in state.items() if isinstance(value, torch.Tensor)}, "optimizer state")


@torch.no_grad()
def probe_losses(model, probes: dict[str, dict]) -> dict[str, float]:
    """L_FM of each task's fixed probe batch, with fixed noise and no dropout, so values are
    comparable across steps, ops and branches. A "probe" phase of the op's cost (vlu/cost.py): our
    measurement, not the method's work."""
    was_training = model.training
    model.eval()
    out = {}
    with cost.phase("probe"):
        for task, inputs in probes.items():
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out[task] = fm_loss(model, inputs).item()
    model.train(was_training)
    return out


def train(node: Node, model, plan: OpPlan, batches, probes: dict[str, dict], recipe: Recipe, save_at: set[int]):
    named = trainable_named_parameters(model)
    params = list(named.values())
    rule = getattr(plan, "update_rule", None)
    finalize = getattr(rule, "finalize", None)
    if finalize is not None and save_at:
        raise ValueError("snapshots of an auxiliary fit are not completed unlearning checkpoints")
    # With a gradient mask (SalUn) only the selected weights may change, so no weight decay either
    # (AdamW's decay would shrink the masked-out weights).
    no_decay = plan.grad_mask is not None or (rule is not None and rule.disable_weight_decay)
    optimizer = (rule.make_optimizer(model, plan.lr) if hasattr(rule, "make_optimizer")
                 else _optimizer(model, plan.lr, 0.0 if no_decay else recipe.weight_decay))
    if rule is not None:
        rule.initialize(optimizer)
    masked = [(named[n], m) for n, m in plan.grad_mask.items()] if plan.grad_mask is not None else []
    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                 if getattr(rule, "constant_lr", False) else
                 get_cosine_schedule_with_warmup(optimizer, math.ceil(recipe.warmup_ratio * plan.steps), plan.steps))

    resume_path = node.dir / "resume.pt"
    # Short ops (including the default 1000-step U ops) need a checkpoint before the end too.
    save_every = min(recipe.save_every, max(1, plan.steps // 2))
    start = 0
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not 0 <= state["step"] < plan.steps or state.get("planned_steps", plan.steps) != plan.steps:
            raise ValueError("resume checkpoint has a different training budget; use a fresh operation")
        check_finite(state["weights"], "resume weights")
        with torch.no_grad():
            for name, p in trainable_named_parameters(model).items():
                p.copy_(state["weights"][name])
        optimizer.load_state_dict(state["optimizer"])
        _check_optimizer_finite(optimizer)
        scheduler.load_state_dict(state["scheduler"])
        if rule is not None:
            if "update_rule" not in state:
                raise ValueError("checkpoint is missing method-specific update state")
            rule.load_state_dict(state["update_rule"])
            if hasattr(rule, "to"):
                rule.to(named)
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        if "python_rng" in state:  # accept checkpoints written before these RNGs were recorded
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
        start = state["step"]
        print(f"resuming {node} at step {start}", flush=True)

    loader = DataLoader(
        batches,
        batch_size=None,  # each item is already a full batch
        sampler=range(start, plan.steps),
        num_workers=recipe.num_workers,
        prefetch_factor=4,
        pin_memory=True,
        multiprocessing_context="fork",
        generator=torch.Generator().manual_seed(0),  # worker seeds must not consume the model's CPU RNG
    )
    log = jsonl_logger(node.dir / "train_log.jsonl", keep_through=start if start > 0 else -1)
    if start == 0:
        log({"step": 0, "probe": probe_losses(model, probes)})

    model.train()
    window_start, window_steps, window_metrics = time.time(), 0, {}
    step_end = time.monotonic()  # when the loop last asked for a batch (vlu/cost.py: the wait for the next one)
    for step, batch in zip(range(start, plan.steps), loader):
        step_start = time.monotonic()  # the step's own work, after its batch has arrived (vlu/cost.py)
        with cost.flops("train"):
            if rule is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = rule.step(model, batch, optimizer, recipe.max_grad_norm)
                grad_norm = metrics.pop("grad_norm")
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = plan.loss_fn(model, batch)
                check_finite({"loss": loss}, f"training step {step + 1}")
                loss.backward()
                for p, mask in masked:  # SalUn: only the selected weights get a gradient
                    if p.grad is not None:
                        p.grad.mul_(mask)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, recipe.max_grad_norm, error_if_nonfinite=True).item()
                optimizer.step()
            check_finite(named, f"updated weights at step {step + 1}")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        cost.step_done(step_start, "train", cost.frames(batch), step_start - step_end)

        done = step + 1  # number of completed updates
        window_steps += 1
        for k, v in {"loss": loss.item(), "grad_norm": grad_norm, **metrics}.items():
            window_metrics[k] = window_metrics.get(k, 0.0) + v

        if done % recipe.log_every == 0 or done == plan.steps:
            elapsed = time.time() - window_start
            record = {"step": done, "lr": scheduler.get_last_lr()[0]}
            record.update({k: v / window_steps for k, v in window_metrics.items()})
            record["steps_per_s"] = window_steps / elapsed
            record["peak_mem_gb"] = cost.peak_allocated_gb()
            if done % (plan.probe_every or recipe.probe_every) == 0 or done == plan.steps:
                record["probe"] = probe_losses(model, probes)
            log(record)
            window_start, window_steps, window_metrics = time.time(), 0, {}

        if done in save_at:
            with cost.phase("checkpoint"):
                save_weights(model, node.dir / f"step_{done:06d}.safetensors")
        if done % save_every == 0 and done < plan.steps:
            with cost.phase("checkpoint"):
                _check_optimizer_finite(optimizer)
                state = {
                    "step": done,
                    "planned_steps": plan.steps,
                    "weights": {n: p.detach().cpu() for n, p in trainable_named_parameters(model).items()},
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                }
                if rule is not None:
                    state["update_rule"] = rule.state_dict()
                tmp = resume_path.with_name("resume.pt.tmp")
                torch.save(state, tmp)
                tmp.replace(resume_path)  # atomic: a crash leaves either the old or the new state
        step_end = time.monotonic()

    # Some methods publish a transformation of their fitted weights, rather than the fit itself.
    # Keep resume.pt until the transformed checkpoint is safely saved: an interruption must never
    # mark an auxiliary fit as a completed unlearning operation.
    if finalize is not None:
        with cost.phase("finalize"), cost.flops("finalize"):
            final_metrics = finalize(model)
        check_finite(named, "finalized weights")
        log({"step": plan.steps, "stage": "finalize", **final_metrics, "probe": probe_losses(model, probes)})
    with cost.phase("save"):
        save_weights(model, node.weights_path)
    resume_path.unlink(missing_ok=True)
