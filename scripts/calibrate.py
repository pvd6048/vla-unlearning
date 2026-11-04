"""Write the calibration plan: the runs that set the provisional hyperparameters before any long run.
Normally called by scripts/run_local.sh (see CALIBRATION.md).

    python scripts/calibrate.py [--run cal]

Run `cal` (seed 0), tasks A and B of LIBERO-Goal:
    base        zero-shot eval of A and B                         -> the floor
    LA          10k steps, snapshots at 2k and 5k; A and B evaluated at 2k / 5k / 10k with eval seeds 0
                and 1                                             -> steps per task, pass bar on A,
                                                                     eval noise, B never trained
    LA/LB       learn B with A replayed; A, B with seeds 0 and 1   -> theta0 of the unlearning variants
    LA/LB/UB-v  every unlearning variant v below (1000 steps; probe losses every 50 steps, which show
                how many steps a method needs); A and B evaluated (seed 0). LA is their retrain oracle
                (it never saw B).

Disk: a node's weights are ~6.5 GB (1.62B trainable parameters in fp32), so only weights that are
evaluated are saved: LA's 2k/5k snapshots and every node's final weights (about 200 GB in all).
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import merge, warn_recipe_changes  # noqa: E402
from vlu.nodes import Node, check_op, run_seed  # noqa: E402
from vlu.worker import item_key, load_plan, plan_path, write_plan  # noqa: E402

# Each variant is <method>[@<recipe overrides>] (vlu/nodes.py). Grids around each paper's default.
# Round 2 (after the first calibration, see METHODS.md): GA and GradDiff collapsed the model at every lr of
# round 1, so lower ones are added; NPO 1e-6 likewise. SSD's round-1 variants (ssd@ssd_alpha=5/10/20) used
# the old reference (all tasks) and are dropped; the new ones name ssd_reference explicitly, so they never
# reuse a round-1 node directory.
VARIANTS = [
    "purge",
    *[f"ga@unlearn_lr={lr}" for lr in ("1e-7", "3e-7", "1e-6", "1e-5", "3e-5", "1e-4")],
    *[f"graddiff@unlearn_lr={lr}" for lr in ("1e-7", "3e-7", "1e-6", "1e-5", "3e-5", "1e-4")],
    *[f"npo@unlearn_lr={lr}" for lr in ("1e-6", "1e-5", "3e-5", "1e-4")],
    "npo@unlearn_lr=3e-5+npo_beta=1",
    "ssd@ssd_reference=retain",  # the paper's ViT alpha = 5, lambda = 1
    "ssd@ssd_alpha=10+ssd_reference=retain",  # the paper's ResNet alpha
    "ssd@ssd_alpha=50+ssd_lambda=0.1+ssd_reference=retain",  # the paper's setting for very similar classes
    "ssd@ssd_adaptive=1+ssd_reference=retain",  # ASSD's tuning-free alpha
    "ssd@ssd_alpha=5+ssd_reference=all",  # the paper's literal D, for comparison
    *[f"salun@unlearn_lr={lr}" for lr in ("1e-5", "3e-5", "1e-4")],
    *[f"vlaforget@vlaforget.lr={lr}" for lr in ("5e-5", "2e-4")],  # the paper's two values
]


def eval_item(node: str, weights: str, seed: int, videos: int = 0) -> dict:
    return {"kind": "eval", "node": node, "weights": weights, "seed": seed, "tasks": ["A", "B"], "videos": videos}


def items() -> list[dict]:
    out = [eval_item("", "final", 0, videos=3)]
    out.append({"kind": "train", "node": "LA", "save_at": [2000, 5000]})
    for weights in ("step_002000", "step_005000", "final"):
        for seed in (0, 1):
            out.append(eval_item("LA", weights, seed, videos=5 if (weights, seed) == ("final", 0) else 0))
    out.append({"kind": "train", "node": "LA/LB", "save_at": []})
    out += [eval_item("LA/LB", "final", seed, videos=3 if seed == 0 else 0) for seed in (0, 1)]
    for variant in VARIANTS:
        path = f"LA/LB/UB-{variant}"
        check_op(Node.from_path("cal", path))  # validates the token (method, override keys) and the op
        out.append({"kind": "train", "node": path, "save_at": []})
        if variant != "purge":  # purge does not change the weights: its results are LA/LB's
            out.append(eval_item(path, "final", 0, videos=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cal")
    args = parser.parse_args()
    run_seed(args.run, 0)
    # The plan is exactly the current calibration: items of earlier rounds that are no longer planned
    # (e.g. round 1's SSD variants, computed with the old reference) are dropped, so no worker can run
    # them again under the current code. Their node directories are left as they are.
    wanted = items()
    keys = {item_key(i) for i in wanted}
    old = load_plan(args.run)
    for item in old:
        if item_key(item) not in keys:
            print(f"no longer planned (kept on disk, not run again): {item_key(item)}")
    plan = merge([i for i in old if item_key(i) in keys], wanted)
    write_plan(args.run, plan)
    warn_recipe_changes(args.run, plan)
    print(f"calibration plan {plan_path(args.run)}: {len(plan)} items, {len(VARIANTS)} unlearning variants")


if __name__ == "__main__":
    main()
