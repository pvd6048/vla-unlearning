"""Train one node: apply its last op to its parent.

    python -m vlu.run_op --run goal_s0 --node LA/LB/LC/UB-ga

Safe to rerun: a finished node is skipped, an interrupted one resumes from its last checkpoint.
The parent must be finished first (scripts/submit.py chains the jobs).
"""

import argparse
from dataclasses import asdict
import json
import os
import shutil
import subprocess
import time

import torch

from vlu.config import PROJECT_DIR, RECIPE
from vlu.data import OpBatches, fixed_batch, load_demos
from vlu.model import build_processor, describe, gr00t_config, load_model, modality_configs, save_weights
from vlu.nodes import Node, run_seed
from vlu.ops import ZERO_STEP_METHODS, OpContext, plan_op, recipe_for
from vlu.train import train


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(PROJECT_DIR), "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run name, e.g. goal_s0")
    parser.add_argument("--node", required=True, help="op path, e.g. LA/LB/LC/UB-ga")
    parser.add_argument("--seed", type=int, help="only needed for the first op of a new run")
    parser.add_argument("--steps", type=int, help="override the op's step budget (smoke tests only)")
    parser.add_argument("--save-at", default="", help="extra weight snapshots, e.g. 2000,5000")
    args = parser.parse_args()

    node = Node.from_path(args.run, args.node)
    seed = run_seed(args.run, args.seed)
    if node.is_base:
        raise ValueError("--node must contain at least one op")
    if node.done:
        print(f"{node} is already done ({node.weights_path})")
        return
    if not node.parent.done:
        raise RuntimeError(f"parent {node.parent} is not done; train it first")

    op = node.last_op
    if op.kind == "U" and op.task not in node.parent.retained_tasks():  # as every method's plan checks
        raise ValueError(f"{node}: cannot unlearn {op.task}, it is not retained by the parent")
    if op.kind == "U" and op.method in ZERO_STEP_METHODS:
        node.write_json("op.json", {"op": str(op), "parent": node.parent.path, "steps": 0, "git": git_commit()})
        try:  # hard link: same bytes, no extra disk
            os.link(node.parent.weights_path, node.weights_path)
        except OSError:  # via a temporary file: a partial weights file would mark the node done
            tmp = node.weights_path.with_name(node.weights_path.name + ".tmp")
            shutil.copyfile(node.parent.weights_path, tmp)
            tmp.replace(node.weights_path)
        print(f"{node}: {op.method} trains nothing; copied the parent's weights")
        return

    recipe = recipe_for(op, RECIPE)  # the frozen recipe plus this op's overrides (if any)
    plan = plan_op(node, RECIPE)  # applies the same overrides itself
    if args.steps is not None and plan.steps > 0:  # zero-step methods (SSD) stay zero-step
        plan.steps = args.steps
    save_at = {int(s) for s in args.save_at.split(",") if s}
    if save_at and (plan.custom_train is not None or plan.steps == 0 or not all(0 < s <= plan.steps for s in save_at)):
        # Snapshots are written only by the standard loop, at steps 1..plan.steps; fail now rather
        # than leave evals of snapshots that will never exist.
        raise ValueError(f"--save-at {sorted(save_at)} is not possible for {op} ({plan.steps} steps"
                         + (", own training loop" if plan.custom_train is not None else "") + ")")

    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = gr00t_config()
    processor = build_processor(config)
    model = load_model(config, node.parent.weights_file())
    mcfg = modality_configs(config)
    demos = load_demos(mcfg)
    probes = {t: fixed_batch(d, recipe.probe_size, processor, mcfg) for t, d in demos.items()}
    # The op's cost (PLAN.md: GPU-minutes per op) counts its own work: the method's preparation (all of SSD,
    # SalUn's mask, NPO's reference), training and probe losses; not loading the model and the demos.
    start, resumed = time.time(), (node.dir / "resume.pt").exists()
    if plan.prepare is not None:  # method-specific setup on the parent model (see vlu/ops.py)
        plan.prepare(OpContext(node, model, demos, processor, mcfg, seed))

    node.write_json(
        "op.json",
        {
            "op": str(op),
            "parent": node.parent.path,
            "run_seed": seed,
            "steps": plan.steps,
            "lr": plan.lr,
            "parts": {name: [list(s) if isinstance(s, tuple) else str(s) for s in slots] for name, slots in plan.parts.items()},
            "notes": plan.notes,
            "recipe": asdict(recipe),
            "overrides": dict(op.overrides),
            "model": describe(model),
            "git": git_commit(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        },
    )
    # Batches depend on (run seed, depth, step): sibling branches at the same depth draw the same
    # random numbers, which makes them easier to compare.
    if plan.custom_train is not None:  # a method with its own loop (VLA-Forget)
        plan.custom_train(OpContext(node, model, demos, processor, mcfg, seed), probes)
        save_weights(model, node.weights_path)
    else:
        batches = OpBatches(demos, plan.parts, processor, mcfg, seed=seed * 1000 + node.depth, num_steps=plan.steps,
                            run_seed=seed)
        train(node, model, plan, batches, probes, recipe, save_at)
    # resumed: the seconds are only those of the session that finished the op
    node.write_json("cost.json", {"seconds": time.time() - start, "gpu": torch.cuda.get_device_name(), "resumed": resumed})
    print(f"{node} done: {node.weights_path}")


if __name__ == "__main__":
    main()
