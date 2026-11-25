"""Optional numerical comparison with the original TaskVector class; see TASK_ARITHMETIC_RMU.md.

Only checkpoint deserialization is replaced with in-memory models. The author's subtraction,
negation and apply_to arithmetic execute unchanged, independently of our finalization code.
"""
import argparse
import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

import stubs  # noqa: F401
import torch
from torch import nn

from vlu.task_arithmetic import TaskArithmeticUpdate


def check(source):
    namespace = {}
    exec(compile(source, "author_task_vectors.py", "exec"), namespace)
    maximum = 0.
    for seed in range(8):
        for scale in (0., .25, 1., 2.):
            torch.manual_seed(seed)
            anchor = nn.Linear(5, 3)
            fitted = copy.deepcopy(anchor)
            with torch.no_grad():
                for p in fitted.parameters():
                    p.add_(.1 * torch.randn_like(p))
            checkpoints = {"anchor": anchor, "fitted": fitted}
            with patch("torch.load", side_effect=lambda name: copy.deepcopy(checkpoints[name])):
                vector = namespace["TaskVector"]("anchor", "fitted")
                expected = (-vector).apply_to("anchor", scaling_coef=scale)
            rule = TaskArithmeticUpdate(scale)
            rule.anchor = {n: p.detach().clone() for n, p in anchor.named_parameters()}
            rule.finalize(fitted)
            for p, q in zip(fitted.parameters(), expected.parameters()):
                maximum = max(maximum, (p-q).abs().max().item())
                torch.testing.assert_close(p, q, atol=0, rtol=0)
    print(f"Author TaskVector: 8 seeds x 4 scales; max parameter difference {maximum:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_vector_source", type=Path)
    args = parser.parse_args()
    source = args.task_vector_source.read_text(encoding="utf-8").rstrip() + "\n"
    if hashlib.sha256(source.encode()).hexdigest() != "582425128b96d5ea1648cbdd86ddea064cf7802516a417734ee676417ce4e10e":
        parser.error("source does not match the inspected task_vectors revision")
    torch.set_num_threads(1)
    check(source)
