"""Probe-set features of one checkpoint (for vlu.drift.compare) and, optionally, its diagonal Fisher
information on some tasks (for vlu.drift.params --fisher). GPU job, main environment.

    python -m vlu.drift.extract --run goal_s0 --node LA/LB/LC [--weights final] [--fisher B,C]
    (normally a "features" item of the run's plan: scripts/submit.py --features / --fisher)

Probe set: PROBE_PER_TASK evenly spread frames of every task A–E (the same frames for every
checkpoint, including tasks it never saw), processed without augmentation. The flow-matching noise
and timesteps are fixed by seed, so every checkpoint gets exactly the same inputs. Written to
<node dir>/drift/features_v<FEATURES_VERSION>.pt (vlu/drift/paths.py):
    tasks      list of N task letters (one per probe sample)
    features   {name: (N, D) float32}, mean-pooled hidden states (not float16: it overflows to inf above
               65504, which diverged checkpoints reach). Their precision is set by the forward pass, which
               runs under bf16 autocast as in training and evaluation (matrix products rounded to 8
               significant bits; the backbone's flash attention needs bf16 anyway), not by this storage:
        vlm.layerXX.image / .text   each trainable LLM layer, pooled over the image / text tokens
        vl.image / vl.text          the VL embeddings the DiT attends to (after the VL layer norm)
        dit.blockXX.<kind>          each DiT block (kind: x-text, x-image or self), pooled over the action tokens
    attn_ratio {DiT block name: (N,)} size of the block's attention update relative to its input,
               ||attn(norm(h))|| / ||h|| averaged over the action tokens: in x-text / x-image blocks,
               how much the action head draws on the instruction / the images
    velocity   (N, 16, 7) predicted flow-matching velocity, normalized action space, valid dims only
    velocity_swapped  (N, 16, 7) the same with the instruction replaced by the next task's
               (A->B, ..., E->A), same noise and timestep: the language sensitivity of the policy
    sse, n_valid  (N,) per-sample squared flow-matching error and number of valid action entries

Fisher (--fisher B): mean over RECIPE.importance_batches batches of the squared L_FM gradient of a
batch of task B (batch Fisher, not per-sample), saved like a weights file to drift/fisher_B.safetensors,
with a summary (share of the mass in the top 1% of weights, per-module shares) in fisher_B.json.
"""

import argparse
from collections import defaultdict
import json

import numpy as np
from safetensors.torch import save_file
import torch

from vlu.config import RECIPE
from vlu.data import load_demos, make_sample
from vlu.drift.paths import FEATURES_VERSION, features_path, fisher_path
from vlu.drift.weights import DIT_BLOCK_KIND, WeightReader, module_of
from vlu.model import build_processor, gr00t_config, load_model, modality_configs
from vlu.nodes import Node
from vlu.ops import OpContext, accumulate_grads, top_fraction_threshold
from vlu.tasks import TASKS, instruction

PROBE_PER_TASK = 64
CHUNK = 16  # probe samples per forward pass
SEED = 12345


class Recorder:
    """Forward hooks that keep the outputs of the layers we pool."""

    def __init__(self, model):
        self.outputs = {}
        self.block_inputs = {}
        self.attn_outputs = {}
        self.handles = []
        layers = model.backbone.model.language_model.model.layers
        for i, layer in enumerate(layers):
            if any(p.requires_grad for p in layer.parameters()):
                self._hook(layer, f"vlm.layer{i:02d}")
        self._hook(model.action_head.vlln, "vl")
        for i, block in enumerate(model.action_head.model.transformer_blocks):
            name = f"dit.block{i:02d}.{DIT_BLOCK_KIND[i % 4]}"
            self._hook(block, name)
            self.handles.append(block.register_forward_pre_hook(
                lambda _m, args, name=name: self.block_inputs.__setitem__(name, args[0].detach())))
            self.handles.append(block.attn1.register_forward_hook(
                lambda _m, _a, out, name=name: self.attn_outputs.__setitem__(name, out.detach())))
        self._hook(model.action_head.action_decoder, "velocity")

    def _hook(self, module, name: str):
        def keep(_module, _inputs, output):
            self.outputs[name] = (output[0] if isinstance(output, tuple) else output).detach()

        self.handles.append(module.register_forward_hook(keep))

    def remove(self):
        for handle in self.handles:
            handle.remove()


def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(h.device, h.dtype)[..., None]
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)


@torch.no_grad()
def extract_features(model, processor, mcfg) -> dict:
    recorder = Recorder(model)
    image_token = model.backbone.model.config.image_token_index
    horizon = len(mcfg["action"].delta_indices)
    letters = list(TASKS)
    tasks, features, velocity, sse, n_valid = [], defaultdict(list), [], [], []
    attn_ratio, velocity_swapped = defaultdict(list), []
    for t_index, letter in enumerate(letters):
        demos = load_demos(mcfg, letters=(letter,))[letter]
        picks = np.linspace(0, len(demos.index) - 1, PROBE_PER_TASK).round().astype(int)
        samples = [make_sample(processor, mcfg, demos, *demos.index[i]) for i in picks]
        swapped_text = instruction(letters[(t_index + 1) % len(letters)])
        swapped = [make_sample(processor, mcfg, demos, *demos.index[i], text=swapped_text) for i in picks]
        del demos
        for start in range(0, len(samples), CHUNK):
            seed = SEED + 1000 * t_index + start  # same noise/timesteps for every checkpoint
            inputs = dict(processor.collator(samples[start : start + CHUNK])["inputs"])
            torch.manual_seed(seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(inputs)
            valid = inputs["attention_mask"] == 1
            image = (inputs["input_ids"] == image_token) & valid
            for name, h in recorder.outputs.items():
                if name.startswith("vlm.") or name == "vl":
                    features[f"{name}.image"].append(_masked_mean(h.float(), image).cpu())
                    features[f"{name}.text"].append(_masked_mean(h.float(), valid & ~image).cpu())
                elif name.startswith("dit."):
                    features[name].append(h[:, -horizon:].float().mean(dim=1).cpu())
            for name, update in recorder.attn_outputs.items():
                ratio = update[:, -horizon:].float().norm(dim=-1) / recorder.block_inputs[name][:, -horizon:].float().norm(dim=-1).clamp_min(1e-6)
                attn_ratio[name].append(ratio.mean(dim=1).cpu())
            action_mask = out["action_mask"].bool()
            dims = int(action_mask[0, 0].sum())  # valid action dims (7 for LIBERO)
            velocity.append(recorder.outputs["velocity"][:, -horizon:, :dims].float().cpu())
            sse.append(out["action_loss"].float().sum(dim=(1, 2)).cpu())
            n_valid.append(action_mask.sum(dim=(1, 2)).float().cpu())
            tasks += [letter] * len(inputs["input_ids"])
            # The same frames and noise with another task's instruction (recorder outputs are not used).
            torch.manual_seed(seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(dict(processor.collator(swapped[start : start + CHUNK])["inputs"]))
            velocity_swapped.append(recorder.outputs["velocity"][:, -horizon:, :dims].float().cpu())
        print(f"probe features of task {letter}: {len(samples)} samples", flush=True)
    recorder.remove()  # no hooks (and no kept activations) during later passes, e.g. the Fisher
    return {
        "tasks": tasks,
        "features": {k: torch.cat(v) for k, v in features.items()},
        "attn_ratio": {k: torch.cat(v) for k, v in attn_ratio.items()},
        "velocity": torch.cat(velocity),
        "velocity_swapped": torch.cat(velocity_swapped),
        "sse": torch.cat(sse),
        "n_valid": torch.cat(n_valid),
        "probe": {"per_task": PROBE_PER_TASK, "seed": SEED, "version": FEATURES_VERSION, "swap": "instruction of the next task"},
    }


def extract_fisher(node: Node, model, processor, mcfg, task: str) -> tuple[dict, dict]:
    demos = load_demos(mcfg, letters=(task,))
    ctx = OpContext(node, model, demos, processor, mcfg, seed=SEED)
    n = RECIPE.importance_batches
    fisher = {k: v / n for k, v in accumulate_grads(ctx, [(task,)] * RECIPE.batch_size, n, stream=7, square=True).items()}
    # Summary over the weights that can change (LIBERO's row of the per-embodiment projectors).
    reader = WeightReader(device="cuda")
    live = {k: reader.reduce(k, v) for k, v in fisher.items()}
    threshold = top_fraction_threshold(live, 0.01)
    total = sum(float(v.sum()) for v in live.values())
    top = sum(float(v[v >= threshold].sum()) for v in live.values())
    per_module = defaultdict(float)
    for k, v in live.items():
        per_module[module_of(k)] += float(v.sum()) / total
    summary = {"task": task, "batches": n, "batch_size": RECIPE.batch_size, "total_mass": total,
               "top1pct_mass_share": top / total, "module_mass_share": dict(per_module)}
    return fisher, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--node", required=True, help='op path; "" = base model')
    parser.add_argument("--weights", default="final")
    parser.add_argument("--fisher", default="", help="tasks to compute the diagonal Fisher on, e.g. B,C")
    args = parser.parse_args()

    node = Node.from_path(args.run, args.node)
    feats = features_path(node, args.weights)
    fisher_tasks = [t for t in args.fisher.split(",") if t and not fisher_path(node, t, args.weights).exists()]
    if feats.exists() and not fisher_tasks:
        print(f"nothing to do for {node}: {feats} and the requested Fisher files exist")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    config = gr00t_config()
    processor = build_processor(config)
    processor.eval()
    model = load_model(config, node.weights_file(args.weights)).eval()
    mcfg = modality_configs(config)
    feats.parent.mkdir(parents=True, exist_ok=True)

    if not feats.exists():
        data = extract_features(model, processor, mcfg)
        data.update(node=str(node), weights=args.weights)
        tmp = feats.with_name(feats.name + ".tmp")
        torch.save(data, tmp)
        tmp.replace(feats)
        print(f"wrote {feats}")
    for task in fisher_tasks:
        fisher, summary = extract_fisher(node, model, processor, mcfg, task)
        path = fisher_path(node, task, args.weights)
        tmp = path.with_name(path.name + ".tmp")
        save_file({k: v.cpu().contiguous() for k, v in fisher.items()}, str(tmp))
        tmp.replace(path)
        summary["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30  # so far in this process
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {path}: {summary}")
    # The launcher's memory threshold for features-only workers (scripts/run_local.sh) is checked against this.
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
