"""Write the calibration plan: the runs that set the provisional hyperparameters before any long run.
Normally called by scripts/run_local.sh (see CALIBRATION.md).

    python scripts/calibrate.py [--run cal]

Run `cal` (seed 0), tasks A and B of LIBERO-Goal:
    base        zero-shot eval of A and B                         -> the floor
    LA          10k steps, snapshots at 2k and 5k; A and B evaluated at 2k / 5k / 10k with eval seeds 0
                and 1                                             -> steps per task, pass bar on A,
                                                                     eval noise, B never trained
    LA/LB       learn B with A replayed; A, B with seeds 0 and 1   -> theta0 of the unlearning variants
    LA/LB/UB-v  every unlearning variant v below (1000 steps; probe losses every 50 steps; snapshots at
                250 / 500 where the method trains with the standard loop); A and B evaluated (seed 0).
                LA is their retrain oracle (it never saw B).
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit import merge  # noqa: E402
from vlu.nodes import Node, run_seed  # noqa: E402
from vlu.worker import load_plan, plan_path  # noqa: E402

# Each variant is <method>[@<recipe overrides>] (vlu/nodes.py). Grids around each paper's default.
VARIANTS = [
    "purge",
    *[f"ga@unlearn_lr={lr}" for lr in ("1e-5", "3e-5", "1e-4")],
    *[f"graddiff@unlearn_lr={lr}" for lr in ("1e-5", "3e-5", "1e-4")],
    *[f"npo@unlearn_lr={lr}" for lr in ("1e-5", "3e-5", "1e-4")],
    "npo@unlearn_lr=3e-5+npo_beta=1",
    *[f"ssd@ssd_alpha={a}" for a in ("5", "10", "20")],
    *[f"salun@unlearn_lr={lr}" for lr in ("1e-5", "3e-5", "1e-4")],
    *[f"vlaforget@vlaforget.lr={lr}" for lr in ("5e-5", "2e-4")],  # the paper's two values
]
SNAPSHOT_METHODS = {"ga", "graddiff", "npo", "salun"}  # trained by the standard loop


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
        Node.from_path("cal", path)  # validates the token (method, override keys)
        method = variant.split("@")[0]
        out.append({"kind": "train", "node": path, "save_at": [250, 500] if method in SNAPSHOT_METHODS else []})
        if method != "purge":  # purge does not change the weights: its results are LA/LB's
            out.append(eval_item(path, "final", 0, videos=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cal")
    args = parser.parse_args()
    run_seed(args.run, 0)
    plan = merge(load_plan(args.run), items())
    tmp = plan_path(args.run).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n")
    tmp.replace(plan_path(args.run))
    print(f"calibration plan {plan_path(args.run)}: {len(plan)} items, {len(VARIANTS)} unlearning variants")


if __name__ == "__main__":
    main()
