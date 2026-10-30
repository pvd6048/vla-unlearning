"""L op: fine-tune on one task with NVIDIA's own training code, unchanged.

Experience replay = a 1:1 dataset mixture of the new task and the replay folders of kept tasks.

    python -m vla_unlearning.learn --init <ckpt> --task data/goal/tasks/task_0 \
        --replay data/goal/replay/task_1 ... --out <op_dir> --steps 10000
"""

import argparse
from pathlib import Path
import shutil

from gr00t.experiment.experiment import run

from vla_unlearning import groot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="checkpoint dir or HF id")
    ap.add_argument("--task", required=True)
    ap.add_argument("--replay", nargs="*", default=[])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mix = [([args.task], 1.0)] + ([(args.replay, 1.0)] if args.replay else [])
    train_dir = args.out / "train"
    cfg = groot.make_config(args.init, mix, train_dir, args.steps, args.lr, args.batch, args.seed)
    run(cfg)

    # Keep only the final NVIDIA-format checkpoint (weights + processor + experiment_cfg).
    final = train_dir / f"checkpoint-{args.steps}"
    for f in final.glob("optimizer*"):
        f.unlink()
    final.rename(args.out / "model")
    shutil.rmtree(train_dir)


if __name__ == "__main__":
    main()
