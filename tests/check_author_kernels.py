"""Optional CPU comparison with pinned author sources; see GU_SHARP_AUDIT.md for downloads.

This checks isolated update kernels, not equivalence of the complete unlearning algorithms.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re

import stubs  # noqa: F401
import torch
from torch import nn

from vlu.gu import retain_null
from vlu.sharp_minmax import SharpMinMaxUpdate


def checked_source(path, expected_hash):
    source = path.read_text(encoding="utf-8")  # normalize CRLF before hashing
    if hashlib.sha256(source.encode()).hexdigest() != expected_hash:
        raise ValueError(f"{path}: source differs from the inspected author revision")
    return source


class Nonlinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.7, -1.2, .4], dtype=torch.float64))

    def forward(self, inputs):
        # Give the reference the same restricted subproblem: inactive entries have zero derivative.
        x = torch.where(inputs["mask"], self.weight, self.weight.detach())
        return {"loss": (x[0] + 2*x[1] - x[2]).square() + .2*x.pow(4).sum()}


def check_sharp(source):
    namespace = {}
    exec(compile(source, "author_sam.py", "exec"), namespace)
    cases, max_error = 0, 0.
    for forget in (False, True):
        for strength in (0., .5, 1., 2.):
            for adaptive in (False, True):
                for rho in (0., .13):
                    own, reference = Nonlinear(), Nonlinear()
                    mask = torch.tensor([True, False, True])
                    inputs = {"mask": mask if forget else ~mask}
                    coefficient = -.3 if forget else .7
                    rule = SharpMinMaxUpdate({"weight": mask}, .7, rho, strength, adaptive)
                    optimizer = torch.optim.SGD(own.parameters(), lr=.02, momentum=.9)
                    other = namespace["SAM"](reference.parameters(), torch.optim.SGD,
                                             sign="max" if forget else "min", rho=rho, lamb=strength,
                                             adaptive=adaptive, lr=.02, momentum=.9)
                    for _ in range(4):
                        rule._phase(own, inputs, {"weight": own.weight}, optimizer, forget, float("inf"))
                        (coefficient * reference(inputs)["loss"]).backward()
                        other.first_step(zero_grad=True)
                        (coefficient * reference(inputs)["loss"]).backward()
                        other.second_step(zero_grad=True)
                        max_error = max(max_error, (own.weight-reference.weight).abs().max().item())
                        torch.testing.assert_close(own.weight, reference.weight, atol=1e-12, rtol=1e-12)
                    cases += 1
    print(f"Sharp author SAM: {cases} configurations x 4 momentum steps; max abs error {max_error:.3g}")


def check_gu(source):
    # Extract only the projector class; the full trainer needs unrelated LLM dependencies.
    tree = ast.parse(source)
    projector = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RetainNullProjector")
    namespace = {"torch": torch, "re": re}
    exec(compile(ast.Module(body=[projector], type_ignores=[]), "author_gu.py", "exec"), namespace)
    reference = namespace["RetainNullProjector"](nn.Linear(9, 1), ["weight"], use_adam_diag=False)
    max_error = 0.
    for seed in range(20):
        torch.manual_seed(seed)
        q = torch.linalg.qr(torch.randn(9, 3)).Q
        d = torch.randn(9)
        expected = reference._project_tensor(d.clone(), list(q.T), mode="perp")
        actual = retain_null(d, list(q.T), torch.ones_like(d))
        max_error = max(max_error, (expected-actual).abs().max().item())
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    print(f"GU author Euclidean projection primitive: 20 seeded cases; max abs error {max_error:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gu_source", type=Path)
    parser.add_argument("sharp_source", type=Path)
    args = parser.parse_args()
    gu = checked_source(args.gu_source, "e9878ec852083bb86704f320eaf5882da872b7d17b28a4e054dc6d6152cb5c55")
    sharp = checked_source(args.sharp_source, "437de1860ddcf4d8c0db08696dfc576ac2e29f5f45bbd0f82d8568e13af381ae")
    torch.set_num_threads(1)
    check_sharp(sharp)
    check_gu(gu)
    print("Kernel checks only; GU optimizer/anchors and Sharp partition handling differ from the authors.")
