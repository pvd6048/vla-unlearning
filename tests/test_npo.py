"""CPU checks of the actual NPO/NPO-RT plans; no GR00T checkpoint or CUDA shims for the loss.

The integration check reuses the shared training loop's CPU test plumbing. Optional CUDA checks
exercise a non-current model device when the machine has two GPUs.
"""
import copy
import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import stubs  # noqa: F401 (GR00T import stubs, not NPO/model replacements)
import torch
from torch import nn
from torch.nn import functional as F

from vlu.config import RECIPE
from vlu.nodes import Node, check_op
from vlu.ops import OpContext, plan_op
import vlu.train as training
from test_training_safety import cpu_runtime


class StochasticFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.4))
        self.frozen = nn.Parameter(torch.tensor(.2), requires_grad=False)
        self.drop = nn.Dropout(.25)
        self.extra_draws = False
        self.fail = False

    def forward(self, inputs):
        x, mask = inputs["action"], inputs["action_mask"]
        noise = torch.randn_like(x)
        t = torch.rand(x.shape[0], 1, 1, device=x.device)
        pred = self.drop(self.scale * (t * x + (1 - t) * noise) + self.frozen)
        errors = (pred - (x - noise)).square() * mask
        # Some models use different no-grad paths; the reference must not advance the training RNG.
        if self.extra_draws and not torch.is_grad_enabled():
            torch.rand(13, device=x.device)
        if self.fail:
            raise RuntimeError("reference failed")
        return {"loss": errors.sum() / mask.sum(), "action_loss": errors}


def batch(device="cpu"):
    x = torch.tensor([[[1., 2., 8.], [2., 1., 9.]], [[-2., 3., 7.], [4., -1., 6.]]], device=device)
    mask = torch.ones_like(x)
    mask[..., -1] = 0
    return {"action": x, "action_mask": mask}


def reference(plan):
    return next(c.cell_contents["ref"] for c in plan.loss_fn.__closure__
                if isinstance(c.cell_contents, dict) and "ref" in c.cell_contents)


def make_plan(method, model, beta=.1):
    recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, npo_beta=beta, batch_size=2, unlearn_steps=6)
    node = Node.from_path("npo_test", f"LA/LB/UB-{method}")
    plan = plan_op(node, recipe)
    plan.prepare(OpContext(node, model, {}, None, {}, 0))
    return plan, recipe


class NPOTests(unittest.TestCase):
    def test_invalid_beta_rejected_before_preparation(self):
        for method in ("npo", "npo_rt"):
            for value in (0., -1., float("nan"), float("inf"), -float("inf"), None):
                with self.subTest(method=method, beta=value):
                    node = Node.from_path("test", f"LA/LB/UB-{method}")
                    recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, npo_beta=value)
                    with self.assertRaisesRegex(ValueError, "npo_beta must be finite and strictly positive"):
                        check_op(node, recipe)
                    with self.assertRaisesRegex(ValueError, "npo_beta"):
                        plan_op(node, recipe)

    def test_beta_validation_uses_op_override(self):
        for method in ("npo", "npo_rt"):
            for value in ("0", "-1", "nan", "inf", "None"):
                node = Node.from_path("test", f"LA/LB/UB-{method}@npo_beta={value}+unlearn_lr=1e-6")
                with self.subTest(op=node.path), self.assertRaisesRegex(ValueError, "npo_beta"):
                    check_op(node)
            node = Node.from_path("test", f"LA/LB/UB-{method}@npo_beta=1+unlearn_lr=1e-6")
            check_op(node, dataclasses.replace(RECIPE, npo_beta=0.))

    def test_npo_rt_is_registered_and_requires_retain_data(self):
        model = StochasticFlow()
        plan, recipe = make_plan("npo_rt", model)
        self.assertEqual(plan.parts, {"forget": [("B",)] * 2, "retain": [("A",)] * 2})
        self.assertEqual(plan.notes["retain_coeff"], 1.)
        with self.assertRaisesRegex(ValueError, "needs a retained task"):
            plan_op(Node.from_path("test", "LA/UA-npo_rt"), recipe)
        plan_op(Node.from_path("test", "LA/UA-npo"), recipe)

    def test_initial_ratio_and_reference_is_fixed(self):
        model = StochasticFlow().train()
        plan, _ = make_plan("npo", model)
        ref = reference(plan)
        self.assertIs(ref.frozen, model.frozen)
        self.assertIsNot(ref.scale, model.scale)
        self.assertFalse(any(p.requires_grad for p in ref.parameters()))
        old_ref = {n: v.clone() for n, v in ref.state_dict().items()}
        loss, metrics = plan.loss_fn(model, {"forget": batch()})
        self.assertEqual(metrics["log_ratio"], 0.)
        self.assertEqual(metrics["npo_weight"], 1.)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in ref.parameters()))
        with torch.no_grad():
            model.scale.add_(.5)
        self.assertTrue(all(torch.equal(v, old_ref[n]) for n, v in ref.state_dict().items()))

    def test_loss_matches_official_formula_per_sample(self):
        model = StochasticFlow()
        plan, recipe = make_plan("npo", model)
        with torch.no_grad():
            model.scale.add_(.3)
        torch.manual_seed(23)
        cur = model(batch())["action_loss"].sum((1, 2))
        torch.manual_seed(23)
        with torch.no_grad():
            ref = reference(plan)(batch())["action_loss"].sum((1, 2))
        expected = -2 / recipe.npo_beta * F.logsigmoid(recipe.npo_beta * (cur - ref)).mean()
        torch.manual_seed(23)
        actual, metrics = plan.loss_fn(model, {"forget": batch()})
        torch.testing.assert_close(actual, expected)
        self.assertNotEqual(metrics["log_ratio"], 0.)
        g_expected, = torch.autograd.grad(expected, model.scale)
        g_actual, = torch.autograd.grad(actual, model.scale)
        torch.testing.assert_close(g_actual, g_expected)

    def test_gradient_starts_as_ascent_and_fades_after_forgetting(self):
        model = StochasticFlow()
        plan, _ = make_plan("npo", model)
        torch.manual_seed(7)
        loss, _ = plan.loss_fn(model, {"forget": batch()})
        grad, = torch.autograd.grad(loss, model.scale)
        torch.manual_seed(7)
        ga = -model(batch())["action_loss"].sum((1, 2)).mean()
        ga_grad, = torch.autograd.grad(ga, model.scale)
        torch.testing.assert_close(grad, ga_grad)
        with torch.no_grad():
            model.scale.fill_(100.)
        torch.manual_seed(7)
        loss, metrics = plan.loss_fn(model, {"forget": batch()})
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(metrics["npo_weight"], 1e-5)

    def test_npo_rt_loss_and_gradient_include_retain_descent(self):
        model = StochasticFlow()
        npo, _ = make_plan("npo", model)
        rt, _ = make_plan("npo_rt", model)
        with torch.no_grad():
            model.scale.add_(.2)
        inputs = {"forget": batch(), "retain": batch()}
        inputs["retain"]["action"] = -inputs["retain"]["action"]
        torch.manual_seed(7)
        forget, _ = npo.loss_fn(model, inputs)
        retain = model(inputs["retain"])["loss"]
        expected = forget + retain
        expected_grad, = torch.autograd.grad(expected, model.scale)
        torch.manual_seed(7)
        actual, metrics = rt.loss_fn(model, inputs)
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual, model.scale)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertEqual(metrics["retain_fm"], retain.item())

    def test_cpu_loss_does_not_access_cuda(self):
        plan, _ = make_plan("npo", StochasticFlow())
        with patch("torch.cuda.get_rng_state", side_effect=AssertionError("CPU loss accessed CUDA")):
            plan.loss_fn(StochasticFlow(), {"forget": batch()})

    def test_reference_preserves_post_student_rng_even_with_extra_draws(self):
        for method in ("npo", "npo_rt"):
            model = StochasticFlow()
            plan, _ = make_plan(method, model)
            reference(plan).extra_draws = True
            torch.manual_seed(11)
            model(batch())
            if method == "npo_rt":
                model(batch())
            expected = torch.get_rng_state()
            torch.manual_seed(11)
            _, metrics = plan.loss_fn(model, {"forget": batch(), "retain": batch()})
            self.assertEqual(metrics["log_ratio"], 0.)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected), method)

    def test_reference_exception_does_not_rewind_rng(self):
        model = StochasticFlow()
        plan, _ = make_plan("npo", model)
        reference(plan).extra_draws = reference(plan).fail = True
        torch.manual_seed(11)
        model(batch())
        expected = torch.get_rng_state()
        torch.manual_seed(11)
        with self.assertRaisesRegex(RuntimeError, "reference failed"):
            plan.loss_fn(model, {"forget": batch()})
        self.assertTrue(torch.equal(torch.get_rng_state(), expected))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two CUDA devices")
    def test_rng_uses_model_device_not_current_device(self):
        with torch.cuda.device(0):
            model = StochasticFlow().to("cuda:1")
            plan, _ = make_plan("npo", model)
            torch.manual_seed(11)
            model(batch("cuda:1"))
            cpu_expected = torch.get_rng_state()
            expected = torch.cuda.get_rng_state(1)
            torch.manual_seed(11)
            unrelated = torch.cuda.get_rng_state(0)
            _, metrics = plan.loss_fn(model, {"forget": batch("cuda:1")})
            self.assertEqual(metrics["log_ratio"], 0.)
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_expected))
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(1), expected))
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(0), unrelated))

    def test_resume_keeps_original_reference_for_both_methods(self):
        torch.manual_seed(5)
        parent = copy.deepcopy(StochasticFlow().state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            for method in ("npo", "npo_rt"):
                def run(name, interrupt_at=None):
                    path = Path(tmp) / (method + name)
                    path.mkdir(exist_ok=True)
                    node = SimpleNamespace(dir=path, weights_path=path / "weights.safetensors")
                    model = StochasticFlow()
                    model.load_state_dict(parent)
                    plan, recipe = make_plan(method, model)  # preparation is on the parent, before resume
                    recipe = dataclasses.replace(recipe, save_every=3, log_every=6)
                    losses = []
                    actual_loss = plan.loss_fn
                    def loss_fn(m, b):
                        if interrupt_at is not None and len(losses) == interrupt_at:
                            raise InterruptedError("simulated interruption")
                        loss, metrics = actual_loss(m, b)
                        losses.append(loss.item())
                        return loss, metrics
                    plan.loss_fn = loss_fn
                    torch.manual_seed(23)
                    with cpu_runtime():
                        try:
                            training.train(node, model, plan, [{"forget": batch(), "retain": batch()}] * 6,
                                           {}, recipe, set())
                        except InterruptedError:
                            if interrupt_at is None:
                                raise
                    return model, losses
                full, expected = run("full")
                _, first = run("resume", interrupt_at=3)
                resumed, last = run("resume")
                self.assertEqual(expected, first + last, method)
                for name, value in full.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.state_dict()[name]), (method, name))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
