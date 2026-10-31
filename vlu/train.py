"""The training loop shared by every op.

AdamW with linear warmup + cosine decay (fresh for each op), bf16 autocast over fp32 trainable
weights, gradient clipping: the same optimization as NVIDIA's HF-Trainer fine-tune. The loop is
resumable: every `save_every` steps it writes resume.pt (weights, optimizer, scheduler, RNG, step),
and a rerun of the same op continues from there.
"""

import json
import math
import time

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from vlu.config import Recipe
from vlu.model import save_weights, trainable_named_parameters
from vlu.nodes import Node
from vlu.ops import OpPlan, fm_loss


def jsonl_logger(path):
    """log(record): append record (plus a timestamp) to a JSON-lines file and print it."""

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


@torch.no_grad()
def probe_losses(model, probes: dict[str, dict]) -> dict[str, float]:
    """L_FM of each task's fixed probe batch, with fixed noise and no dropout, so values are
    comparable across steps, ops and branches."""
    was_training = model.training
    model.eval()
    out = {}
    for task, inputs in probes.items():
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[task] = fm_loss(model, inputs).item()
    model.train(was_training)
    return out


def train(node: Node, model, plan: OpPlan, batches, probes: dict[str, dict], recipe: Recipe, save_at: set[int]):
    params = list(trainable_named_parameters(model).values())
    optimizer = _optimizer(model, plan.lr, recipe.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(recipe.warmup_ratio * plan.steps), plan.steps)

    resume_path = node.dir / "resume.pt"
    start = 0
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, p in trainable_named_parameters(model).items():
                p.copy_(state["weights"][name])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
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
    )
    log = jsonl_logger(node.dir / "train_log.jsonl")
    if start == 0:
        log({"step": 0, "probe": probe_losses(model, probes)})

    model.train()
    window_start, window_steps, window_metrics = time.time(), 0, {}
    for step, batch in zip(range(start, plan.steps), loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = plan.loss_fn(model, batch)
        loss.backward()
        if plan.grad_mask is not None:  # e.g. SalUn: only the selected weights may change
            for name, p in trainable_named_parameters(model).items():
                if p.grad is not None:
                    p.grad.mul_(plan.grad_mask[name])
        grad_norm = torch.nn.utils.clip_grad_norm_(params, recipe.max_grad_norm).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        done = step + 1  # number of completed updates
        window_steps += 1
        for k, v in {"loss": loss.item(), "grad_norm": grad_norm, **metrics}.items():
            window_metrics[k] = window_metrics.get(k, 0.0) + v

        if done % recipe.log_every == 0 or done == plan.steps:
            elapsed = time.time() - window_start
            record = {"step": done, "lr": scheduler.get_last_lr()[0]}
            record.update({k: v / window_steps for k, v in window_metrics.items()})
            record["steps_per_s"] = window_steps / elapsed
            record["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            if done % recipe.probe_every == 0 or done == plan.steps:
                record["probe"] = probe_losses(model, probes)
            log(record)
            window_start, window_steps, window_metrics = time.time(), 0, {}

        if done in save_at:
            save_weights(model, node.dir / f"step_{done:06d}.safetensors")
        if done % recipe.save_every == 0 and done < plan.steps:
            state = {
                "step": done,
                "weights": {n: p.detach().cpu() for n, p in trainable_named_parameters(model).items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            }
            tmp = resume_path.with_name("resume.pt.tmp")
            torch.save(state, tmp)
            tmp.replace(resume_path)  # atomic: a crash leaves either the old or the new state

    save_weights(model, node.weights_path)
    resume_path.unlink(missing_ok=True)
