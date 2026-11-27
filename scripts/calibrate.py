"""Write the calibration plan: the runs that set the provisional hyperparameters before any long run.
Normally called by scripts/run_local.sh (see CALIBRATION.md).

    python scripts/calibrate.py [--run cal]
    python scripts/calibrate.py --prune-weights      only free disk (below), when no worker is running

Run `cal` (seed 0), tasks A and B of LIBERO-Goal:
    base        zero-shot eval of A and B                         -> the floor
    LA          10k steps, snapshots at 2k and 5k; A and B evaluated at 2k / 5k / 10k with eval seeds 0
                and 1                                             -> steps per task, pass bar on A,
                                                                     eval noise, B never trained
    LA/LB       learn B with A replayed; A, B with seeds 0 and 1   -> theta0 of the unlearning variants
    LA/LB/UB-v  every unlearning variant v below (1000 steps; probe losses every 50 steps, which show
                how many steps a method needs); A and B evaluated (seed 0). LA is their retrain oracle
                (it never saw B).
    drift       probe features (vlu.drift.extract) of base, LA, LA/LB and every variant, and the diagonal
                Fisher of A and B at LA/LB; scripts/calibration_drift.py turns them and the weights into
                drift metrics after the runs (PLAN.md section 4).

Every variant's cost is measured (vlu/cost.py: time, compute, memory, energy). A variant trained before
that measurement existed (rounds 1-2 and the 24 November run on the guide's machine) is moved to
RUNS_DIR/_archive/<run>/before_cost_meter/ (outside the run, so no report or tool sees it; nothing is
deleted) and trained, evaluated and measured again, so every variant of a round is measured by the same
code on the same machine. LA and LA/LB are kept.

Disk: a node's weights are ~6.5 GB (1.62B trainable parameters in fp32), so only weights that are
evaluated are saved: LA's 2k/5k snapshots and every node's final weights (about 130 GB in all), plus
the two Fisher files (6.5 GB each) and the probe features (~100 MB per node, float32). --prune-weights
deletes the weights (and resume checkpoints) of nodes the plan no longer has (earlier rounds' variants,
e.g. GA and GradDiff) and of archived ones; their op.json, logs, eval results and drift features stay.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import merge, warn_recipe_changes  # noqa: E402
from vlu.config import RUNS_DIR  # noqa: E402
from vlu.nodes import Node, check_op, run_seed  # noqa: E402
from vlu.worker import item_key, load_plan, plan_path, someone_working, write_plan  # noqa: E402

# Each variant is <method>[@<recipe overrides>] (vlu/nodes.py). Grids around each paper's default.
# Round 3 (on the guide's machine, reusing LA and LA/LB and retraining every variant with the cost meter; or on
# the cluster from scratch), from rounds 1-2 on the guide's machine (METHODS.md, CALIBRATION.md):
# - GA and GradDiff are dropped: no rate both forgot B and kept A (GA: 1e-7 forgot nothing, 3e-7 lost A;
#   GradDiff 3e-7 was halfway on both). Their code stays (vlu/ops.py), as do the round-2 results.
# - NPO: no rate both forgot B and kept A. 1e-7 kept B (SR 1.00) and lowered A to 0.70; 3e-7 lost A (0.00)
#   with B still at 0.66; 1e-6 and up lost A. Plain NPO has nothing that holds A, so only the gap below 3e-7
#   can still hold a trade-off: 1e-7, 2e-7, 3e-7 (6e-7 and 1e-6 lie in the range that lost A).
# - NPO-RT (NPO plus GradDiff's retain term, the NPO authors' default; first run 24 November on the old plan):
#   3e-6 and 1e-5 forgot B and kept A (1e-5: B's probe loss at the oracle's), 1e-6 halfway (B 0.50). 3e-7
#   behaved as plain NPO 3e-7 (A 0.38): NPO's loss (~14 at the start) outweighs the retain loss (~0.02) until
#   it saturates, which at 3e-7 the cosine schedule ends first, so 3e-7 is dropped. 3e-5 is added as the
#   upper bracket above the authors' 1e-5: 1e-6, 3e-6, 1e-5, 3e-5.
# - SalUn: all three rates kept A and forgot B; repeated here to confirm 3e-5 on this hardware.
# - SSD: alpha 5 and 10 (retain reference) forgot B and kept A, ASSD halfway; alpha 50 / lambda 0.1 forgot
#   nothing and the paper's literal reference (all tasks) dampens nothing at our forget share (vlu/ops.py
#   _plan_ssd), so both are dropped.
# - VLA-Forget (first run 24 November, no OOM): both paper values forgot B (SR 0.00) but hurt A: 5e-5 to
#   0.66, 2e-4 to 0.00. A's probe loss rose in the vision and projector stages (about equally), hardly in the
#   reasoning stage. 2e-4 is dropped and two lower rates added: 1e-5, 2e-5, 5e-5. (A's probe loss ended at
#   0.08 for both paper values: it does not show the difference in SR A.)
# - GU now follows Algorithm 1's anchor-gradient recovery, gamma=1.5, alpha=1. NPO-RT's grid is a
#   provisional bracket, not a matched-weight projection ablation. The explicit new weight override
#   gives fresh node paths so results from the earlier algorithm cannot be reused.
# - Sharp now follows literal Algorithm 2 (mask g1 only), two independent optimizers, full-forget-pass
#   saliency, and the paper's constant-rate SGD. Its former AdamW rates are not transferable. The new
#   PROVISIONAL bracket is 2e-4, 2e-3, 2e-2 (the paper's CIFAR rate), with both mask orientations because
#   Algorithm 1/code say smallest and Appendix E.2 says largest. Alpha matters under SGD.
# - The collaborator's GU-FM (method "gu", a GU-inspired projection of the AdamW proposal) is not calibrated:
#   it is not the authors' algorithm (GU_SHARP_AUDIT.md) and at rank 2 over all trainable weights needs
#   about 53 GB, more than a training GPU has.
# SSD variants name ssd_reference explicitly, so they never reuse a round-1 node directory (old reference).
VARIANTS = [
    "purge",
    *[f"npo@unlearn_lr={lr}" for lr in ("1e-7", "2e-7", "3e-7")],
    *[f"npo_rt@unlearn_lr={lr}" for lr in ("1e-6", "3e-6", "1e-5", "3e-5")],
    "ssd@ssd_reference=retain",  # the paper's ViT alpha = 5, lambda = 1
    "ssd@ssd_alpha=10+ssd_reference=retain",  # the paper's ResNet alpha
    "ssd@ssd_adaptive=1+ssd_reference=retain",  # ASSD's tuning-free alpha
    *[f"salun@unlearn_lr={lr}" for lr in ("1e-5", "3e-5", "1e-4")],
    *[f"vlaforget@vlaforget.lr={lr}" for lr in ("1e-5", "2e-5", "5e-5")],  # 5e-5: the paper's lower value
    *[f"npo_rt_gu@unlearn_lr={lr}+gu_forget_weight=1.5" for lr in ("1e-6", "3e-6", "1e-5", "3e-5")],
    *[f"sharp_minmax@unlearn_lr={lr}+sharp_optimizer=sgd+sharp_select={select}"
      for select in ("smallest", "largest") for lr in ("2e-4", "2e-3", "2e-2")],
]


def eval_item(node: str, weights: str, seed: int, videos: int = 0) -> dict:
    return {"kind": "eval", "node": node, "weights": weights, "seed": seed, "tasks": ["A", "B"], "videos": videos}


def features_item(node: str, fisher: tuple[str, ...] = ()) -> dict:
    """Optional (vlu/worker.py): drift metrics are extra data, so a failure is reported, not fatal."""
    return {"kind": "features", "node": node, "weights": "final", "fisher": list(fisher), "optional": True}


def items() -> list[dict]:
    """In run order: a node's evals and drift features come right after its training."""
    out = [eval_item("", "final", 0, videos=3), features_item("")]
    out.append({"kind": "train", "node": "LA", "save_at": [2000, 5000]})
    for weights in ("step_002000", "step_005000", "final"):
        for seed in (0, 1):
            out.append(eval_item("LA", weights, seed, videos=5 if (weights, seed) == ("final", 0) else 0))
    out.append(features_item("LA"))
    out.append({"kind": "train", "node": "LA/LB", "save_at": []})
    out += [eval_item("LA/LB", "final", seed, videos=3 if seed == 0 else 0) for seed in (0, 1)]
    out.append(features_item("LA/LB", fisher=("A", "B")))  # theta0 of every variant: its curvature on A and B
    for variant in VARIANTS:
        path = f"LA/LB/UB-{variant}"
        check_op(Node.from_path("cal", path))  # validates the token (method, override keys) and the op
        out.append({"kind": "train", "node": path, "save_at": []})
        if not Node.from_path("cal", path).last_op.shares_parent_weights:  # purge: its results are LA/LB's
            out.append(eval_item(path, "final", 0, videos=2))
            out.append(features_item(path))
    return out


ARCHIVE = "before_cost_meter"


def archive_dir(run: str) -> Path:
    return RUNS_DIR / "_archive" / run / ARCHIVE


def archive_unmeasured(run: str) -> list[str]:
    """Move every trained variant whose cost.json has no "efficiency" (trained before vlu/cost.py) to
    archive_dir(run), so that it is trained and measured again. A variant trained with the meter is kept, so a
    rerun archives nothing new. Purge trains nothing (no cost to measure)."""
    moved = []
    for variant in VARIANTS:
        node = Node.from_path(run, f"LA/LB/UB-{variant}")
        if node.last_op.shares_parent_weights or not node.done:
            continue
        cost = node.dir / "cost.json"
        if cost.exists() and "efficiency" in json.loads(cost.read_text()):
            continue
        dest = archive_dir(run) / node.path
        if dest.exists():
            raise SystemExit(f"cannot archive {node}: {dest} exists already (move it away and rerun)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(node.dir), str(dest))
        moved.append(node.path)
        print(f"archived (trained before the cost meter; trained again now): {node.path} -> {dest}")
    return moved


def prune_weights(run: str, plan: list[dict]) -> int:
    """Delete the weights and resume checkpoints of nodes the plan does not train (earlier rounds' variants)
    and of archived nodes. Only files directly in such a node's folder: a planned node below it (every
    variant is below LA/LB's folder) keeps its own. Returns the bytes freed."""
    planned = {Node.from_path(run, i["node"]).dir.resolve() for i in plan if i["kind"] == "train"}
    planned.add((RUNS_DIR / run).resolve())  # the base: nothing of its own to delete
    candidates = [d.parent for d in (RUNS_DIR / run).rglob("op.json") if d.parent.resolve() not in planned]
    if archive_dir(run).exists():
        candidates += [d.parent for d in archive_dir(run).rglob("op.json")]
    freed = 0
    for node_dir in sorted(set(candidates)):
        for f in [*node_dir.glob("*.safetensors"), *node_dir.glob("resume.pt")]:
            size = f.stat().st_size if f.stat().st_nlink == 1 else 0  # a hard link (purge) frees nothing
            f.unlink()
            freed += size
            print(f"deleted {f} ({size / 1e9:.1f} GB)")
    print(f"freed {freed / 1e9:.1f} GB")
    return freed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cal")
    parser.add_argument("--prune-weights", action="store_true",
                        help="only delete the weights of nodes no longer planned and of archived ones (frees disk)")
    args = parser.parse_args()
    # Only claims known to be live block (a claim whose job cannot be checked, e.g. squeue failing, does not stop
    # a cluster successor job from planning); run_local.sh and run.sbatch plan before their workers start.
    if someone_working(args.run, verified=True):
        raise SystemExit(f"a worker is running on run {args.run}: stop it first (the plan and its nodes must not change under it)")
    if args.prune_weights:
        plan = load_plan(args.run)
        if not plan:
            raise SystemExit(f"run {args.run} has no plan yet: nothing to compare with")
        prune_weights(args.run, plan)
        return
    run_seed(args.run, 0)
    archive_unmeasured(args.run)
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
