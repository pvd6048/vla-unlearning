"""Known-answer GU projections and actual shared-loop checkpoint continuation on CPU."""
import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import stubs  # noqa: F401
import torch
from torch import nn

from vlu.config import RECIPE
from vlu.gu import GeometricUpdate, retain_null
from vlu.nodes import Node, check_op
from vlu.ops import plan_op
from vlu.train import train
from test_training_safety import cpu_runtime


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., 2., -1.]))
        self.frozen = nn.Parameter(torch.tensor(4.), requires_grad=False)

    def forward(self, inputs):
        return {"loss": ((self.weight * inputs["x"]).sum() - inputs["y"]).square()}


def batch():
    return {"retain": {"x": torch.tensor([1., 2., 0.]), "y": torch.tensor(1.)},
            "forget": {"x": torch.tensor([0., 1., 1.]), "y": torch.tensor(-2.)}}


class GUTests(unittest.TestCase):
    def test_metric_projection_matches_closed_form(self):
        d = torch.tensor([1., 3., 2.])
        u = torch.tensor([1., 2., 0.])
        p = torch.tensor([2., .5, 3.])
        expected = d - p * u * (u @ d) / (u @ (p * u))
        actual = retain_null(d, [u], p)
        torch.testing.assert_close(actual, expected)
        self.assertLess(abs(float(u @ actual)), 1e-6)
        self.assertFalse(torch.allclose(actual, retain_null(d, [u], torch.ones(3))))

    def test_degenerate_and_full_rank_subspaces(self):
        d = torch.tensor([1., 3.])
        p = torch.tensor([2., .5])
        u = torch.tensor([1., 1.])
        torch.testing.assert_close(retain_null(d, [u, u * 2], p), retain_null(d, [u], p))
        torch.testing.assert_close(retain_null(d, list(torch.eye(2)), p), torch.zeros(2), atol=1e-6, rtol=0.)
        torch.testing.assert_close(retain_null(d, [torch.zeros(2)], p), d)
        self.assertIs(retain_null(d, [], p), d)

    def test_extreme_adam_scaling_preserves_independent_constraints(self):
        # The former normal-equation pseudoinverse discarded the second constraint and returned
        # [0, 1, 2], because the two Gram eigenvalues differed by 1e12.
        d = torch.tensor([1., 1., 2.], dtype=torch.float64)
        u = list(torch.eye(3, dtype=torch.float64)[:2])
        p = torch.tensor([1e8, 1e-4, 1.], dtype=torch.float64)
        torch.testing.assert_close(retain_null(d, u, p), torch.tensor([0., 0., 2.], dtype=d.dtype),
                                   atol=1e-12, rtol=0.)

    def test_projection_matches_independent_kkt_solve(self):
        generator = torch.Generator().manual_seed(2026)
        for _ in range(8):
            u = torch.randn(7, 3, generator=generator, dtype=torch.float64)
            d = torch.randn(7, generator=generator, dtype=torch.float64)
            p = torch.exp(torch.randn(7, generator=generator, dtype=torch.float64) * 3)
            inverse_metric = torch.diag(p.reciprocal())
            kkt = torch.cat([torch.cat([inverse_metric, u], dim=1),
                             torch.cat([u.T, torch.zeros(3, 3, dtype=u.dtype)], dim=1)], dim=0)
            rhs = torch.cat([inverse_metric @ d, torch.zeros(3, dtype=u.dtype)])
            expected = torch.linalg.solve(kkt, rhs)[:7]
            actual = retain_null(d, list(u.T), p)
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(u.T @ actual, torch.zeros(3, dtype=u.dtype), atol=1e-10, rtol=0.)

    def test_invalid_metric_is_rejected(self):
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "preconditioner"):
                retain_null(torch.ones(2), [torch.ones(2)], torch.tensor([1., value]))

    def test_actual_adam_displacement_is_retain_null_with_momentum_and_decay(self):
        model = Quadratic()
        optimizer = torch.optim.AdamW([model.weight], lr=.02, weight_decay=.1)
        rule = GeometricUpdate(2)
        rule.initialize(optimizer)
        for _ in range(4):
            old = model.weight.detach().clone()
            retain = torch.autograd.grad(model(batch()["retain"])["loss"], model.weight)[0]
            rule.step(model, batch(), optimizer, 1.)
            delta = model.weight.detach() - old
            self.assertGreater(float(delta.norm()), 0.)
            self.assertLess(abs(float(retain @ delta)), 2e-6)
            optimizer.zero_grad(set_to_none=True)
        self.assertEqual(model.frozen.item(), 4.)
        self.assertEqual(len(rule.basis["weight"]), 2)

    def test_planning_validation_and_no_calibrated_default(self):
        recipe = dataclasses.replace(RECIPE, unlearn_lr=.01)
        node = Node.from_path("gu", "LA/LB/UB-gu")
        self.assertEqual(plan_op(node, recipe).parts.keys(), {"forget", "retain"})
        with self.assertRaisesRegex(ValueError, "needs a retained task"):
            plan_op(Node.from_path("gu", "LA/UA-gu"), recipe)
        for rank in (0, -1, None):
            with self.assertRaisesRegex(ValueError, "gu_rank"):
                check_op(node, dataclasses.replace(recipe, gu_rank=rank))
        with self.assertRaisesRegex(ValueError, "no learning rate yet"):
            check_op(node)

    def test_resume_matches_full_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial = Quadratic().state_dict()
            def run(name, interrupt=False):
                path = Path(tmp) / name
                path.mkdir(exist_ok=True)
                node = SimpleNamespace(dir=path, weights_path=path / "weights.safetensors")
                model = Quadratic()
                model.load_state_dict(initial)
                recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, unlearn_steps=6, save_every=3, log_every=6)
                plan = plan_op(Node.from_path("gu", "LA/LB/UB-gu"), recipe)
                original = plan.update_rule.step
                calls = []
                def step(*args):
                    if interrupt and len(calls) == 3:
                        raise InterruptedError()
                    result = original(*args)
                    calls.append(result[0].item())
                    return result
                plan.update_rule.step = step
                with cpu_runtime():
                    try:
                        train(node, model, plan, [batch()] * 6, {}, recipe, set())
                    except InterruptedError:
                        if not interrupt:
                            raise
                return model, calls
            full, expected = run("full")
            _, first = run("resume", True)
            resumed, last = run("resume")
            self.assertEqual(expected, first + last)
            torch.testing.assert_close(full.weight, resumed.weight, rtol=0., atol=0.)

    def test_checkpoint_rank_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "different method or basis rank"):
            GeometricUpdate(3).load_state_dict(GeometricUpdate(2).state_dict())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
