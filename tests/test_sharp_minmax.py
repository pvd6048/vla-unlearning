"""Sharp MinMax known-answer signs, partitions, temporary perturbations and resume on CPU."""
import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import stubs  # noqa: F401
import torch
from torch import nn

from vlu.config import RECIPE
from vlu.nodes import Node, check_op
from vlu.ops import OpContext, plan_op
from vlu.sharp_minmax import SharpMinMaxUpdate, perturbed, replay_randomness
from vlu.train import train
from test_training_safety import cpu_runtime


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., 2.]))
        self.frozen = nn.Parameter(torch.tensor(3.), requires_grad=False)

    def forward(self, inputs):
        return {"loss": ((self.weight - inputs["target"]).square()).sum()}


class NoisyQuadratic(Quadratic):
    def forward(self, inputs):
        return {"loss": ((self.weight - inputs["target"] - torch.randn_like(self.weight)).square()).sum()}


def make_rule(rho=.1, adaptive=False):
    return SharpMinMaxUpdate({"weight": torch.tensor([False, True])}, .5, rho, 1., adaptive)


def batches():
    return {"retain": {"target": torch.tensor([0., 0.])},
            "forget": {"target": torch.tensor([0., 0.])}}


class SharpTests(unittest.TestCase):
    def test_phase_matches_author_gradient_blend_and_partition(self):
        for forget in (False, True):
            for strength in (.5, 1., 2.):
                model = Quadratic()
                rule = make_rule()
                rule.strength = strength
                old = model.weight.detach().clone()
                active = torch.tensor([False, True]) if forget else torch.tensor([True, False])
                coefficient = -.5 if forget else .5
                g0 = coefficient * 2 * old * active
                e = .1 * g0 / g0.norm()
                g1 = coefficient * 2 * (old + e) * active
                sign = -1 if forget else 1
                expected = old - .01 * (g0 + sign * strength * (g1 - g0))
                opt = torch.optim.SGD([model.weight], lr=.01)
                rule._phase(model, batches()["forget" if forget else "retain"], {"weight": model.weight},
                            opt, forget, float("inf"))
                torch.testing.assert_close(model.weight, expected)
                self.assertTrue(torch.equal(model.weight[~active], old[~active]))
                self.assertEqual(model.frozen.item(), 3.)

    def test_adaptive_perturbation_matches_asam(self):
        p = nn.Parameter(torch.tensor([2., -3.]))
        old = p.detach().clone()
        g = torch.tensor([1., 2.])
        expected = old + .1 * old.square() * g / (old.abs() * g).norm()
        with perturbed([p], [g], .1, True):
            torch.testing.assert_close(p, expected)
        self.assertTrue(torch.equal(p, old))

    def test_perturbation_restores_weights_on_failure(self):
        p = nn.Parameter(torch.tensor([1., 2.]))
        old = p.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "second pass"):
            with perturbed([p], [torch.ones(2)], .1, False):
                raise RuntimeError("second pass")
        self.assertTrue(torch.equal(p, old))

    def test_zero_radius_is_split_descent_and_ascent(self):
        model = Quadratic()
        rule = make_rule(rho=0.)
        optimizer = torch.optim.AdamW([model.weight], lr=.01, weight_decay=0.)
        rule.initialize(optimizer)
        for _ in range(4):
            old = model.weight.detach().clone()
            rule.step(model, batches(), optimizer, 1.)
            self.assertLess(model.weight[0].item(), old[0].item())
            self.assertGreater(model.weight[1].item(), old[1].item())
        self.assertIsNot(optimizer.state[model.weight]["exp_avg"],
                         rule.forget_optimizer.state[model.weight]["exp_avg"])

    def test_two_passes_replay_noise_and_preserve_rng(self):
        draws = []
        class Noisy(Quadratic):
            def forward(self, inputs):
                noise = torch.randn(2)
                draws.append(noise.clone())
                return {"loss": ((self.weight - noise).square()).sum()}
        model = Noisy()
        rule = make_rule()
        optimizer = torch.optim.AdamW([model.weight], lr=.01, weight_decay=0.)
        rule.initialize(optimizer)
        torch.manual_seed(10)
        torch.randn(2); torch.randn(2)
        expected = torch.get_rng_state()
        torch.manual_seed(10)
        rule.step(model, batches(), optimizer, 1.)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected))
        self.assertEqual(len(draws), 4)
        self.assertTrue(torch.equal(draws[0], draws[1]))
        self.assertTrue(torch.equal(draws[2], draws[3]))
        self.assertFalse(torch.equal(draws[0], draws[2]))

    def test_second_pass_restores_all_batchnorm_statistics(self):
        model = nn.BatchNorm1d(2, momentum=None).train()
        x = torch.tensor([[1., 3.], [5., 7.]])
        model(x)
        before = {n: b.clone() for n, b in model.named_buffers()}
        with replay_randomness(model, torch.get_rng_state(), {}):
            model(x * 2)
        self.assertIsNone(model.momentum)
        for name, buffer in model.named_buffers():
            self.assertTrue(torch.equal(buffer, before[name]), name)

    def test_planning_rejects_invalid_settings_and_missing_retain(self):
        recipe = dataclasses.replace(RECIPE, unlearn_lr=.01)
        for key, values in {"sharp_alpha": (0., 1., float("nan")), "sharp_forget_fraction": (0., 1., None),
                            "sharp_rho": (-1., float("inf")), "sharp_strength": (-1., None),
                            "importance_batches": (0, -1)}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    check_op(Node.from_path("sharp", "LA/LB/UB-sharp_minmax"),
                             dataclasses.replace(recipe, **{key: value}))
        with self.assertRaisesRegex(ValueError, "needs a retained task"):
            plan_op(Node.from_path("sharp", "LA/UA-sharp_minmax"), recipe)

    def test_mask_preparation_is_saved_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = Node.from_path("sharp", "LA/LB/UB-sharp_minmax")
            recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, sharp_forget_fraction=.5)
            model = Quadratic()
            ctx = OpContext(node, model, {}, None, {}, 0)
            directory = Path(tmp)
            with patch.object(Node, "dir", new_callable=lambda: property(lambda self: directory)), \
                 patch("vlu.ops.accumulate_grads", return_value={"weight": torch.tensor([1., 3.])}) as importance:
                plan = plan_op(node, recipe)
                plan.prepare(ctx)
                self.assertTrue(torch.equal(plan.update_rule.masks["weight"], torch.tensor([False, True])))
                second = plan_op(node, recipe)
                second.prepare(ctx)
                self.assertEqual(importance.call_count, 1)
                self.assertTrue((directory / "sharp_minmax_mask.pt").exists())

    def test_empty_saliency_partition_is_rejected(self):
        node = Node.from_path("sharp", "LA/LB/UB-sharp_minmax")
        recipe = dataclasses.replace(RECIPE, unlearn_lr=.01)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Node, "dir", new_callable=lambda: property(lambda self: Path(tmp))), \
                 patch("vlu.ops.accumulate_grads", return_value={"weight": torch.zeros(2)}):
                plan = plan_op(node, recipe)
                with self.assertRaisesRegex(ValueError, "empty partition"):
                    plan.prepare(OpContext(node, Quadratic(), {}, None, {}, 0))

    def test_resume_saves_both_optimizers_and_masks(self):
        self.check_resume(Quadratic)

    def test_resume_matches_with_stochastic_sharpness_evaluations(self):
        self.check_resume(NoisyQuadratic)

    def check_resume(self, model_type):
        with tempfile.TemporaryDirectory() as tmp:
            def run(name, interrupt=False):
                torch.manual_seed(41)
                directory = Path(tmp) / name
                directory.mkdir(exist_ok=True)
                node = SimpleNamespace(dir=directory, weights_path=directory / "weights.safetensors")
                recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, unlearn_steps=6, save_every=3, log_every=6)
                plan = plan_op(Node.from_path("sharp", "LA/LB/UB-sharp_minmax"), recipe)
                plan.update_rule = make_rule()
                model = model_type()
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
                        train(node, model, plan, [batches()] * 6, {}, recipe, set())
                    except InterruptedError:
                        if not interrupt:
                            raise
                return model, calls
            full, expected = run("full")
            _, first = run("resume", True)
            resumed, last = run("resume")
            self.assertEqual(expected, first + last)
            self.assertTrue(torch.equal(full.weight, resumed.weight))

    def test_both_methods_integrate_with_groot_shaped_fake(self):
        from fakes import Fake, Demos, Proc, collate, make_sample
        from vlu.data import OpBatches
        for method in ("gu", "sharp_minmax"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, cpu_runtime():
                directory = Path(tmp)
                with patch.object(Node, "dir", new_callable=lambda: property(lambda self: directory)), \
                     patch("vlu.data.make_sample", make_sample):
                    node = Node.from_path("smoke", f"LA/LB/UB-{method}")
                    model, processor = Fake(), Proc()
                    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
                    demos = {t: Demos(t) for t in "AB"}
                    recipe = dataclasses.replace(RECIPE, unlearn_lr=1e-5, unlearn_steps=4, batch_size=2,
                                                 importance_batches=2, log_every=4, save_every=2)
                    plan = plan_op(node, recipe)
                    if plan.prepare:
                        plan.prepare(OpContext(node, model, demos, processor, {}, 0))
                    data = OpBatches(demos, plan.parts, processor, {}, seed=0, num_steps=4)
                    probes = {t: collate([make_sample(None, None, d, 0, 0)])["inputs"] for t, d in demos.items()}
                    train(node, model, plan, data, probes, recipe, {2})
                    self.assertTrue(node.weights_path.exists())
                    self.assertTrue((directory / "step_000002.safetensors").exists())
                    self.assertFalse((directory / "resume.pt").exists())
                    for n, p in model.named_parameters():
                        if n in frozen:
                            self.assertTrue(torch.equal(p, frozen[n]), n)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
