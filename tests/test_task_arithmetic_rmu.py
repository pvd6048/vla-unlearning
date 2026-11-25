"""CPU known-answer, partition, masking and real shared-loop resume checks for TA and RMU."""
import copy
import dataclasses
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import stubs  # noqa: F401
import torch
from torch import nn

from fakes import Fake, Proc, collate, make_sample, Demos
from vlu.config import RECIPE
from vlu.model import trainable_named_parameters
from vlu.nodes import Node, check_op
from vlu.ops import OpContext, plan_op
from vlu.rmu import RMUUpdate, action_mse
from vlu.task_arithmetic import TaskArithmeticUpdate
from vlu.train import train
from test_gu import Quadratic, batch
from test_training_safety import cpu_runtime


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = nn.Parameter(torch.eye(2))
        self.action_head = nn.Module()
        self.action_head.model = nn.Module()
        self.action_head.model.transformer_blocks = nn.ModuleList([nn.Linear(2, 2, bias=False)])

    def forward(self, inputs):
        hidden = self.action_head.model.transformer_blocks[0](inputs["x"] @ self.upstream)
        return {"loss": hidden.square().mean()}


def mini_batch():
    inputs = {"x": torch.tensor([[[2., 3.], [1., -1.], [.2, .5]]]),
              "action_mask": torch.ones(1, 2, 1)}
    return {"forget": inputs, "retain": {**inputs, "x": inputs["x"] * .5}}


class TaskArithmeticTests(unittest.TestCase):
    def test_negates_independently_fitted_adamw_vector(self):
        for scale in (0., .5, 1., 2.):
            model, fitted = Quadratic(), Quadratic()
            anchor = model.weight.detach().clone()
            opt = torch.optim.AdamW([model.weight], lr=.01, weight_decay=.1)
            other = torch.optim.AdamW([fitted.weight], lr=.01, weight_decay=.1)
            rule = TaskArithmeticUpdate(scale)
            for _ in range(4):
                opt.zero_grad(set_to_none=True)
                rule.step(model, batch(), opt, float("inf"))
                other.zero_grad(set_to_none=True)
                fitted(batch()["forget"])["loss"].backward()
                other.step()
            rule.finalize(model)
            expected = anchor - scale * (fitted.weight.detach() - anchor)
            torch.testing.assert_close(model.weight, expected, atol=0., rtol=0.)
            self.assertEqual(model.frozen.item(), 4.)

    def test_checkpoint_rejects_wrong_scale_and_missing_anchor(self):
        with self.assertRaisesRegex(ValueError, "settings"):
            TaskArithmeticUpdate(2.).load_state_dict(TaskArithmeticUpdate(1.).state_dict())
        with self.assertRaisesRegex(ValueError, "anchor"):
            TaskArithmeticUpdate(1.).load_state_dict(TaskArithmeticUpdate(1.).state_dict())


class RMUTests(unittest.TestCase):
    def test_two_steps_match_independent_rmu_objective(self):
        torch.manual_seed(8)
        model = Mini()
        original, expected = copy.deepcopy(model), copy.deepcopy(model)
        rule = RMUUpdate(model, 0, 3., 2., 19)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        block = expected.action_head.model.transformer_blocks[0]
        other = torch.optim.SGD(block.parameters(), lr=.01)
        vector = torch.rand(2, generator=torch.Generator().manual_seed(19))
        control = vector / vector.norm() * 2.
        data = mini_batch()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            _, metrics = rule.step(model, data, optimizer, float("inf"))
            hf = block(data["forget"]["x"] @ expected.upstream)[:, -2:]
            hr = block(data["retain"]["x"] @ expected.upstream)[:, -2:]
            with torch.no_grad():
                h0 = original.action_head.model.transformer_blocks[0](
                    data["retain"]["x"] @ original.upstream)[:, -2:]
            retain = (hr - h0).square().mean()
            loss = (hf - control).square().mean() + 3. * retain
            other.zero_grad(set_to_none=True)
            loss.backward()
            other.step()
            torch.testing.assert_close(next(rule.module.parameters()), next(block.parameters()))
            self.assertAlmostEqual(metrics["retain_rmu"], retain.item(), places=7)
        self.assertTrue(torch.equal(model.upstream, original.upstream))
        self.assertTrue(all(not p.requires_grad for p in rule.reference.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_mask_ignores_state_and_padded_action_tokens(self):
        hidden = torch.tensor([[[1e6, 1e6], [1., 3.], [1e6, 1e6]]])
        mask = torch.tensor([[[1.], [0.]]])
        self.assertEqual(action_mse(hidden, torch.zeros(2), mask).item(), 5.)
        hidden = hidden.requires_grad_()
        with torch.no_grad():
            hidden[:, -1] = float("nan")
        action_mse(hidden, torch.zeros(2), mask).backward()
        torch.testing.assert_close(hidden.grad, torch.tensor([[[0., 0.], [1., 3.], [0., 0.]]]))
        with self.assertRaisesRegex(ValueError, "valid action"):
            action_mse(hidden, torch.zeros(2), torch.zeros_like(mask))

    def test_reference_uses_identical_noisy_incoming_activations(self):
        class Noisy(Mini):
            def forward(self, inputs):
                return super().forward({**inputs, "x": inputs["x"] + torch.randn_like(inputs["x"])})
        model = Noisy()
        rule = RMUUpdate(model, 0, 3., 2., 19)
        with torch.no_grad():
            _, captured = rule.forward(model, mini_batch()["retain"], with_reference=True)
        torch.testing.assert_close(captured["hidden"], captured["reference"], atol=0., rtol=0.)

    def test_hooks_and_modes_are_restored_on_failure(self):
        model = Mini().train()
        rule = RMUUpdate(model, 0, 3., 2., 19)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with patch.object(model, "forward", side_effect=RuntimeError("failed forward")):
            with self.assertRaisesRegex(RuntimeError, "failed forward"):
                rule.step(model, mini_batch(), optimizer, 1.)
        self.assertTrue(all(m.training for m in model.modules()))
        self.assertFalse(rule.module._forward_hooks)
        self.assertFalse(rule.module._forward_pre_hooks)

    def test_checkpoint_restores_reference_and_control(self):
        model = Mini()
        rule = RMUUpdate(model, 0, 3., 2., 19)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        rule.step(model, mini_batch(), optimizer, 1.)
        resumed = RMUUpdate(model, 0, 3., 2., 19)
        resumed.load_state_dict(rule.state_dict())
        self.assertTrue(torch.equal(resumed.control, rule.control))
        for name, p in rule.reference.state_dict().items():
            self.assertTrue(torch.equal(p, resumed.reference.state_dict()[name]))
        with self.assertRaisesRegex(ValueError, "settings"):
            RMUUpdate(model, 0, 4., 2., 19).load_state_dict(rule.state_dict())


class IntegrationTests(unittest.TestCase):
    def test_resume_rejects_a_changed_fit_budget_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp, cpu_runtime():
            directory = Path(tmp)
            node = SimpleNamespace(dir=directory, weights_path=directory / "weights.safetensors")
            recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, unlearn_steps=2, log_every=2)
            path = Node.from_path("test", "LA/LB/UB-task_arithmetic")
            plan = plan_op(path, recipe)
            original = plan.update_rule.step
            calls = []

            def stop_after_one(*args):
                if calls:
                    raise InterruptedError()
                calls.append(True)
                return original(*args)
            plan.update_rule.step = stop_after_one
            with self.assertRaises(InterruptedError):
                train(node, Quadratic(), plan, [batch()] * 2, {}, recipe, set())
            for steps in (1, 3):
                model = Quadratic()
                before = model.weight.detach().clone()
                changed = dataclasses.replace(recipe, unlearn_steps=steps)
                with self.assertRaisesRegex(ValueError, "training budget"):
                    train(node, model, plan_op(path, changed), [batch()] * steps, {}, changed, set())
                self.assertTrue(torch.equal(before, model.weight))
            self.assertFalse(node.weights_path.exists())

    def test_shared_loop_refuses_auxiliary_fit_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp, cpu_runtime():
            directory = Path(tmp)
            node = SimpleNamespace(dir=directory, weights_path=directory / "weights.safetensors")
            recipe = dataclasses.replace(RECIPE, unlearn_lr=.01, unlearn_steps=2, log_every=2)
            plan = plan_op(Node.from_path("test", "LA/LB/UB-task_arithmetic"), recipe)
            with self.assertRaisesRegex(ValueError, "snapshot"):
                train(node, Quadratic(), plan, [batch()] * 2, {}, recipe, {1})
            self.assertFalse((directory / "step_000001.safetensors").exists())
            self.assertFalse(node.weights_path.exists())

    def test_cli_rejects_zero_or_negative_smoke_step_budgets(self):
        from vlu.run_op import main
        for method in ("rmu", "task_arithmetic"):
            for steps in (0, -1):
                args = ["run_op.py", "--run", "test", "--node", f"LA/LB/UB-{method}@unlearn_lr=1e-5",
                        "--steps", str(steps)]
                with patch.object(sys, "argv", args), patch("vlu.run_op.run_seed", return_value=0), \
                     patch.object(Node, "done", new=property(lambda n: n.last_op.kind == "L")):
                    with self.assertRaisesRegex(ValueError, "positive"):
                        main()

    def test_auxiliary_fit_cannot_be_requested_as_a_finished_snapshot(self):
        from vlu.run_op import main
        args = ["run_op.py", "--run", "test", "--node", "LA/LB/UB-task_arithmetic@unlearn_lr=1e-5",
                "--save-at", "1"]
        with patch.object(sys, "argv", args), patch("vlu.run_op.run_seed", return_value=0), \
             patch.object(Node, "done", new=property(lambda n: n.last_op.kind == "L")):
            with self.assertRaisesRegex(ValueError, "not possible"):
                main()  # rejects before loading a model or accessing CUDA

    def test_configuration_validation(self):
        recipe = dataclasses.replace(RECIPE, unlearn_lr=.01)
        for method, keys in {"task_arithmetic": ["task_arithmetic_scale"],
                             "rmu": ["rmu_alpha", "rmu_steering"]}.items():
            for key in keys:
                for bad in (-1., None, float("nan"), float("inf")):
                    with self.assertRaisesRegex(ValueError, key):
                        check_op(Node.from_path("test", f"LA/LB/UB-{method}"),
                                 dataclasses.replace(recipe, **{key: bad}))
            with self.assertRaisesRegex(ValueError, "no learning rate"):
                check_op(Node.from_path("test", f"LA/LB/UB-{method}"))
        for layer in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "rmu_layer"):
                check_op(Node.from_path("test", "LA/LB/UB-rmu"), dataclasses.replace(recipe, rmu_layer=layer))
        with self.assertRaisesRegex(ValueError, "retained task"):
            plan_op(Node.from_path("test", "LA/UA-rmu"), recipe)
        with self.assertRaisesRegex(ValueError, "outside"):
            RMUUpdate(Mini(), 4, 3., 2., 19)

    def test_stochastic_shared_loop_resume_and_checkpoint_scope(self):
        for method in ("task_arithmetic", "rmu"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, cpu_runtime():
                torch.manual_seed(7)
                initial = copy.deepcopy(Fake().state_dict())
                data = {part: collate([make_sample(None, None, Demos(task), 0, 2)])["inputs"]
                        for part, task in (("forget", "B"), ("retain", "A"))}

                def run(name, interrupted=False, fail_finalize=False):
                    directory = Path(tmp) / name
                    directory.mkdir(exist_ok=True)
                    node = SimpleNamespace(dir=directory, weights_path=directory / "weights.safetensors", depth=4)
                    model = Fake()
                    model.load_state_dict(initial)
                    torch.manual_seed(31)
                    recipe = dataclasses.replace(RECIPE, unlearn_lr=1e-5, unlearn_steps=6,
                                                 save_every=3, log_every=6, rmu_steering=2.)
                    plan = plan_op(Node.from_path("test", f"LA/LB/UB-{method}"), recipe)
                    if plan.prepare:
                        plan.prepare(OpContext(node, model, {}, Proc(), {}, seed=0))
                    rule = plan.update_rule
                    original = rule.step
                    losses = []

                    def step(*args):
                        if interrupted and len(losses) == 3:
                            raise InterruptedError()
                        result = original(*args)
                        losses.append(result[0].item())
                        return result
                    rule.step = step
                    if fail_finalize:
                        rule.finalize = lambda model: (_ for _ in ()).throw(InterruptedError())
                    try:
                        train(node, model, plan, [data] * 6, {"A": data["retain"]}, recipe, set())
                    except InterruptedError:
                        if not (interrupted or fail_finalize):
                            raise
                        self.assertFalse(node.weights_path.exists())
                        self.assertTrue((directory / "resume.pt").exists())
                    return model, losses

                full, expected = run("full")
                _, first = run("resume", interrupted=True)
                resumed, last = run("resume")
                self.assertEqual(expected, first + last)
                for name, value in full.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)
                    editable = name.startswith("action_head.model.transformer_blocks.0.") if method == "rmu" else name in trainable_named_parameters(full)
                    if not editable:
                        self.assertTrue(torch.equal(value, initial[name]), name)
                if method == "task_arithmetic":
                    run("finalize", fail_finalize=True)
                    recovered, _ = run("finalize")
                    for name, value in full.state_dict().items():
                        self.assertTrue(torch.equal(value, recovered.state_dict()[name]), name)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
