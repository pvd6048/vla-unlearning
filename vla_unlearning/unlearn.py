"""U op: load a checkpoint, apply one unlearning method, save an NVIDIA-format checkpoint.

    python -m vla_unlearning.unlearn --init <ckpt> --forget data/goal/tasks/task_1 \
        --retain data/goal/replay/task_0 ... --method '{"method": "ga", "steps": 500, "lr": 1e-5}' \
        --out <op_dir>
"""

import argparse
import json
from pathlib import Path
import time

import torch

from vla_unlearning import groot
from vla_unlearning.methods import METHODS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--forget", required=True)
    ap.add_argument("--retain", nargs="*", default=[])
    ap.add_argument("--method", required=True, help="method config as JSON")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = json.loads(args.method)
    torch.manual_seed(args.seed)

    model, data = groot.load(args.init, [args.forget], args.retain, args.out / "work", cfg.get("batch", 16), args.seed)
    ref = groot.frozen_copy(model)  # the model right before this U op

    start = time.time()
    METHODS[cfg["method"]](model, ref, data, cfg)
    minutes = (time.time() - start) / 60

    groot.save(model, args.init, args.out / "model")
    (args.out / "cost.json").write_text(json.dumps({"gpu_minutes": minutes}))


if __name__ == "__main__":
    main()
