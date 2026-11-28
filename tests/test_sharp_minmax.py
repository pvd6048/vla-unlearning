"""Sharp MinMax known-answer signs, partitions, temporary perturbations and resume on CPU."""
import dataclasses

from fakes import Demos
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
                g0 = coefficient * 2 * old
                e = .1 * g0 / g0.norm()  # all weights move, as the authors' first_step
                g1 = coefficient * 2 * (old + e)
                sign = -1 if forget else 1
                expected = g0 + sign * strength * (g1 * active - g0)
                _, grads, _ = rule._phase(model, batches()["forget" if forget else "retain"], {"weight": model.weight},
                                          forget)
                torch.testing.assert_close(grads[0], expected)
                if forget:
                    self.assertGreater(grads[0][~active].abs().sum().item(), 0.)
                self.assertTrue(torch.equal(model.weight, old))  # the perturbation is undone exactly
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

    def test_perturbed_forward_sees_moved_weights_under_autocast(self):
        """train.py runs a whole rule.step in one autocast region, whose cache of bf16 weight copies does not
        notice in-place changes: without clearing it the perturbed forward used the original weights."""
        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        layer = nn.Linear(16, 4).to(device)
        x = torch.randn(8, 16, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            before = layer(x).detach()
            with perturbed([layer.weight], [torch.ones_like(layer.weight)], 1., False):
                moved = layer(x).detach()
            after = layer(x).detach()
        self.assertGreater((moved - before).abs().max().item(), .1)
        self.assertTrue(torch.equal(after, before))

    def test_step_under_autocast_uses_current_weights(self):
        """Under train.py's autocast: the retain phase's second gradient is taken at w + e and the forget
        phase's first gradient at the weights after the retain update."""
        class Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(16, 1, bias=False)

            def forward(self, inputs):
                return {"loss": (self.layer(inputs["x"]).float() - inputs["y"]).square().mean()}

        def grad_at(model, weight, inputs, scale):
            with torch.no_grad():
                model.layer.weight.copy_(weight)
            with torch.autocast(device, dtype=torch.bfloat16):  # a fresh region: no cached copies
                loss = model(inputs)["loss"]
            return torch.autograd.grad(scale * loss, [model.layer.weight])[0]

        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Linear().to(device)
        data = {k: {"x": torch.randn(32, 16, device=device), "y": torch.randn(32, 1, device=device)}
                for k in ("retain", "forget")}
        forget_mask = torch.arange(16, device=device).view(1, 16) % 2 == 0
        rule = SharpMinMaxUpdate({"layer.weight": forget_mask}, .5, 1., 1.)
        optimizer = rule.make_optimizer(model, .1)
        rule.initialize(optimizer)
        seen = []
        original = rule._phase

        def recording(*args, **kwargs):
            result = original(*args, **kwargs)
            seen.append((model.layer.weight.detach().clone(), result[1][0].clone()))
            return result
        rule._phase = recording
        w0 = model.layer.weight.detach().clone()
        with torch.autocast(device, dtype=torch.bfloat16):
            rule.step(model, data, optimizer, 1e9)
        w_end = model.layer.weight.detach().clone()
        (w_retain, retain_grad), (w_forget, forget_grad) = seen
        probe = Linear().to(device)
        # Retain, strength 1: the blend is the gradient at w + e alone (inside the retain partition); e moves
        # all weights along the full gradient.
        g0 = grad_at(probe, w0, data["retain"], .5)
        expected = grad_at(probe, w0 + g0 / g0.norm(), data["retain"], .5) * ~forget_mask
        g0 = g0 * ~forget_mask
        self.assertGreater((expected - g0).abs().max().item(), 1e-3)  # the perturbation matters here
        torch.testing.assert_close(retain_grad, expected, atol=1e-5, rtol=1e-3)
        # Forget: at the weights after the retain update (moved), blend 2 g0 - g1.
        self.assertGreater((w_forget - w0).abs().max().item(), 1e-3)
        g0 = grad_at(probe, w_forget, data["forget"], -.5)
        g1 = grad_at(probe, w_forget + g0 / g0.norm(), data["forget"], -.5)
        torch.testing.assert_close(forget_grad, 2 * g0 - g1 * forget_mask, atol=1e-5, rtol=1e-3)
        self.assertFalse(torch.equal(w_end, w_forget))

    def test_optional_clipping_bounds_each_phase(self):
        """sharp_clip: each phase's blended gradient reaches its optimizer clipped at max_grad_norm; the logged
        norm is the one before clipping. Without it the gradient is passed on unchanged (the paper)."""
        for clip in (False, True):
            model = Quadratic()
            rule = make_rule()
            rule.clip = clip
            optimizer = rule.make_optimizer(model, .01)
            rule.initialize(optimizer)
            seen = []
            for opt in (optimizer, rule.forget_optimizer):
                original = opt.step
                def step(*args, original=original, **kwargs):
                    seen.append(model.weight.grad.norm().item())
                    return original(*args, **kwargs)
                opt.step = step
            _, metrics = rule.step(model, batches(), optimizer, .01)
            self.assertEqual(len(seen), 2)
            if clip:
                self.assertTrue(all(n <= .01 + 1e-6 for n in seen), seen)
                self.assertGreater(metrics["grad_norm"], .01)
            else:
                self.assertTrue(any(n > .01 for n in seen), seen)

    def test_zero_radius_still_updates_outside_forget_selection(self):
        model = Quadratic()
        rule = make_rule(rho=0.)
        _, grads, _ = rule._phase(model, batches()["forget"], {"weight": model.weight}, True)
        # g0=-w, g1=g0: 2*g0 - mask*g0. No final mask or clipping.
        torch.testing.assert_close(grads[0], torch.tensor([-2., -2.]))

    def test_retain_gradients_are_freed_before_the_forget_phase(self):
        """Memory: the retain phase's gradients (one weight copy) must not be alive during the forget phase,
        whose second backward is the step's peak (on GR00T about 45 instead of 40 GiB)."""
        import weakref
        model = Quadratic()
        rule = make_rule()
        optimizer = rule.make_optimizer(model, .01)
        rule.initialize(optimizer)
        original, alive, retain_refs = rule._phase, [], []

        def phase(model_, inputs, named, forget):
            if forget:
                alive.extend(ref() is not None for ref in retain_refs)
            loss, grads, norm = original(model_, inputs, named, forget)
            if not forget:
                retain_refs.extend(weakref.ref(g) for g in grads if g is not None)
            return loss, grads, norm
        rule._phase = phase
        rule.step(model, batches(), optimizer, 1.)
        self.assertTrue(retain_refs)
        self.assertEqual(alive, [False] * len(retain_refs))

    def test_native_optimizers_keep_independent_per_parameter_counters(self):
        class OptionalWeight(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.tensor([1., 2.]))
                self.b = nn.Parameter(torch.tensor([3., 4.]))
            def forward(self, inputs):
                loss = self.a.square().sum()
                return {"loss": loss + self.b.square().sum() if inputs["use_b"] else loss}

        model = OptionalWeight()
        rule = SharpMinMaxUpdate({n: torch.tensor([True, False]) for n, _ in model.named_parameters()},
                                 .9, .1, 1., optimizer_name="adamw")
        optimizer = rule.make_optimizer(model, .001)
        rule.initialize(optimizer)
        for use_retain, use_forget in ((False, False), (False, True), (True, True)):
            rule.step(model, {"retain": {"use_b": use_retain}, "forget": {"use_b": use_forget}}, optimizer, .01)
        self.assertEqual(optimizer.state[model.a]["step"].item(), 3)
        self.assertEqual(optimizer.state[model.b]["step"].item(), 1)
        self.assertEqual(rule.forget_optimizer.state[model.b]["step"].item(), 2)
        self.assertNotEqual(optimizer.state[model.a]["exp_avg"].data_ptr(),
                            rule.forget_optimizer.state[model.a]["exp_avg"].data_ptr())

    def test_two_passes_replay_noise_and_preserve_rng(self):
        draws = []
        class Noisy(Quadratic):
            def forward(self, inputs):
                noise = torch.randn(2)
                draws.append(noise.clone())
                return {"loss": ((self.weight - noise).square()).sum()}
        model = Noisy()
        rule = make_rule()
        optimizer = rule.make_optimizer(model, .01)
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
        for key, values in {"sharp_select": ("top", None), "sharp_alpha": (0., 1., float("nan")), "sharp_forget_fraction": (0., 1., None),
                            "sharp_rho": (-1., float("inf")), "sharp_strength": (-1., None),
                            "sharp_optimizer": ("adam", None), "sharp_momentum": (-1., float("nan")),
                            "sharp_weight_decay": (-1., None)}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    check_op(Node.from_path("sharp", "LA/LB/UB-sharp_minmax"),
                             dataclasses.replace(recipe, **{key: value}))
        with self.assertRaisesRegex(ValueError, "needs a retained task"):
            plan_op(Node.from_path("sharp", "LA/UA-sharp_minmax"), recipe)

    def test_mask_preparation_is_saved_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = Node.from_path("sharp", "LA/LB/UB-sharp_minmax")
            recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, sharp_forget_fraction=.5, sharp_select="largest")
            model = Quadratic()
            ctx = OpContext(node, model, {"B": Demos("B")}, None, {}, 0)
            directory = Path(tmp)
            with patch.object(Node, "dir", new_callable=lambda: property(lambda self: directory)), \
                 patch("vlu.ops.accumulate_grads", return_value={"weight": torch.tensor([1., 3.])}) as importance:
                plan = plan_op(node, recipe)
                rng_before = torch.get_rng_state()
                plan.prepare(ctx)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
                self.assertTrue(torch.equal(plan.update_rule.masks["weight"], torch.tensor([False, True])))
                second = plan_op(node, recipe)
                second.prepare(ctx)
                self.assertEqual(importance.call_count, 1)
                self.assertTrue((directory / "sharp_minmax_mask.pt").exists())
                with self.assertRaisesRegex(ValueError, "different recipe"):
                    plan_op(node, dataclasses.replace(recipe, sharp_select="smallest")).prepare(ctx)
                (directory / "sharp_minmax_mask.pt").unlink()
                smallest = plan_op(node, dataclasses.replace(recipe, sharp_select="smallest"))
                smallest.prepare(ctx)  # the authors' generate_mask.py: the lowest-ranked half
                self.assertTrue(torch.equal(smallest.update_rule.masks["weight"], torch.tensor([True, False])))
                torch.save({"masks": smallest.update_rule.masks}, directory / "sharp_minmax_mask.pt")
                with self.assertRaisesRegex(ValueError, "older algorithm"):
                    smallest.prepare(ctx)

    def test_smallest_split_counts_only_live_entries(self):
        """Per-embodiment projector stacks: only LIBERO's row can change, so the other rows (zero saliency)
        are never in the forget partition, and the fraction refers to the live entries."""
        from vlu.drift.weights import live_part
        from vlu.ops import embodiment_index
        node = Node.from_path("sharp", "LA/LB/UB-sharp_minmax")
        recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, sharp_forget_fraction=.25, sharp_select="smallest")
        name = "action_head.state_encoder.W"
        row = embodiment_index()
        saliency = torch.zeros(32, 2, 4)
        saliency[row] = torch.arange(1., 9.).reshape(2, 4)
        class Stack(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(32, 2, 4))
        model = Stack()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(Node, "dir", new_callable=lambda: property(lambda self: Path(tmp))), \
                patch("vlu.ops.trainable_named_parameters", return_value={name: model.p}), \
                patch("vlu.ops.accumulate_grads", return_value={name: saliency}):
            plan = plan_op(node, recipe)
            plan.prepare(OpContext(node, model, {"B": Demos("B")}, None, {}, 0))
            mask = plan.update_rule.masks[name]
        self.assertEqual(int(mask.sum()), 2)  # the 2 smallest of the 8 live entries
        self.assertTrue(torch.equal(live_part(name, mask, row).flatten()[:2], torch.tensor([True, True])))
        self.assertEqual(plan.notes["actual_forget_fraction"], .25)

    def test_empty_saliency_partition_is_rejected(self):
        node = Node.from_path("sharp", "LA/LB/UB-sharp_minmax")
        recipe = dataclasses.replace(RECIPE, unlearn_lr=.01)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Node, "dir", new_callable=lambda: property(lambda self: Path(tmp))), \
                 patch("vlu.ops.accumulate_grads", return_value={"weight": torch.zeros(2)}):
                plan = plan_op(node, recipe)
                with self.assertRaisesRegex(ValueError, "nonempty partitions"):
                    plan.prepare(OpContext(node, Quadratic(), {"B": Demos("B")}, None, {}, 0))

    def test_resume_saves_both_optimizers_and_masks(self):
        self.check_resume(Quadratic)

    def test_optional_author_code_adamw_resume_is_exact(self):
        self.check_resume(NoisyQuadratic, optimizer_name="adamw")

    def test_legacy_shared_moment_checkpoints_are_rejected(self):
        model = Quadratic()
        rule = make_rule()
        rule.initialize(rule.make_optimizer(model, .01))
        with self.assertRaisesRegex(ValueError, "predates"):
            rule.load_state_dict({"kind": "sharp_minmax", "moments": {}, "steps": {}})

    def test_resume_matches_with_stochastic_sharpness_evaluations(self):
        self.check_resume(NoisyQuadratic)

    def check_resume(self, model_type, optimizer_name="sgd"):
        with tempfile.TemporaryDirectory() as tmp:
            def run(name, interrupt=False):
                torch.manual_seed(41)
                directory = Path(tmp) / name
                directory.mkdir(exist_ok=True)
                node = SimpleNamespace(dir=directory, weights_path=directory / "weights.safetensors")
                recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, unlearn_steps=6, save_every=3, log_every=6)
                plan = plan_op(Node.from_path("sharp", "LA/LB/UB-sharp_minmax"), recipe)
                plan.update_rule = make_rule()
                plan.update_rule.optimizer_name = optimizer_name
                model = model_type()
                original = plan.update_rule.step
                calls = []
                def step(*args):
                    if interrupt and len(calls) == 3:
                        raise InterruptedError()
                    self.assertEqual(args[2].param_groups[0]["lr"], .01)  # constant from the first step
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
        for method in ("gu", "sharp_minmax", "npo_rt_gu"):
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
