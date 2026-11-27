"""GU on NPO-RT (vlu/gu_npo_rt.py): known-answer projections, basis growth, NPO-RT's update while the basis
is empty, exact resume and planning, on CPU with the GR00T-shaped fake."""
import dataclasses
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import stubs  # noqa: F401
import torch
from torch import nn

from fakes import Demos, Fake, Proc, collate, make_sample
from test_training_safety import cpu_runtime
from vlu.config import RECIPE
from vlu.data import OpBatches
from vlu.gu_npo_rt import EPS, GuNpoRt, projected_names
from vlu.nodes import Node, check_op
from vlu.ops import OpContext, plan_op
from vlu.train import train


def rule_for(n, basis, v, cap=.5, k=8):
    """A rule projecting one tensor "w" of n entries, with the given whitened basis and Adam second moment."""
    p = nn.Parameter(torch.zeros(n))
    optimizer = torch.optim.AdamW([p], lr=1e-3)
    optimizer.state[p] = {"exp_avg_sq": v.clone()}
    rule = GuNpoRt(.1, ["w"], k, 2, cap)
    rule.initialize(optimizer)
    rule.basis["w"] = [u.half() for u in basis]
    return rule, p


class ProjectionTests(unittest.TestCase):
    def test_known_answer_with_axis_basis(self):
        """Basis e0, e1 (exact in float16). Whitened: f~ = g_f / s, r~ = g_r / s, s = sqrt(v) + eps.
        Forget part: f~ without e0, e1, plus a0 e0 where a0 b0 < 0 (opposite signs; e1 has the same sign),
        capped at 0.5 ||perp||. Retain part: r~ on e0, e1. Both times s."""
        v = torch.tensor([4., 1., 9., .25, 16.])
        s = v.sqrt() + EPS
        e = torch.eye(5)
        g_f = torch.tensor([1., 2., 3., -1., .5])
        g_r = torch.tensor([-2., 4., 1., 1., -3.])
        for cap in (10., .1):
            with self.subTest(cap=cap):
                rule, p = rule_for(5, [e[0], e[1]], v, cap=cap)
                out = rule._project("w", p, g_f.clone(), g_r.clone())
                f, r = g_f / s, g_r / s
                perp = f.clone(); perp[:2] = 0
                tangent = torch.zeros(5); tangent[0] = f[0]  # f0 * r0 < 0; f1 * r1 > 0 is dropped
                if tangent.norm() > cap * perp.norm():
                    tangent = tangent * cap * perp.norm() / tangent.norm()
                retain = torch.zeros(5); retain[:2] = r[:2]
                torch.testing.assert_close(out, (perp + tangent) * s + retain * s)
        rule, p = rule_for(5, [e[0], e[1]], v)
        perp_only = rule._project("w", p, g_f.clone(), None)  # no retain gradient: no sign test, no retain part
        expected = g_f / s; expected[:2] = 0
        torch.testing.assert_close(perp_only, expected * s)

    def test_empty_basis_or_no_moment_is_npo_rt(self):
        g_f, g_r = torch.randn(6), torch.randn(6)
        rule, p = rule_for(6, [], torch.rand(6))
        self.assertTrue(torch.equal(rule._project("w", p, g_f, g_r), g_f + g_r))
        rule, p = rule_for(6, [torch.eye(6)[0]], torch.rand(6))
        rule.optimizer.state.clear()  # before AdamW's first step
        self.assertTrue(torch.equal(rule._project("w", p, g_f, g_r), g_f + g_r))

    def test_forget_part_is_retain_null_for_a_general_basis(self):
        """With no kept tangent (all signs equal), the whitened forget part is orthogonal to the basis (up
        to its float16 rounding), and the retain part lies in the basis' span."""
        torch.manual_seed(0)
        n = 40
        v = torch.rand(n) + .1
        q, _ = torch.linalg.qr(torch.randn(n, 3, dtype=torch.float64))
        basis = [q[:, j].float() for j in range(3)]
        s = v.sqrt() + EPS
        coeffs = torch.tensor([1., -2., .5])
        g_f = (torch.randn(n) + sum(c * u for c, u in zip(coeffs, basis))) * s
        g_r = sum(c * u for c, u in zip(coeffs, basis)) * s  # a_j b_j > 0 for every j: nothing kept
        rule, p = rule_for(n, basis, v)
        stored = [u.float() for u in rule.basis["w"]]
        forget_only = rule._project("w", p, g_f, None) / s
        for u in stored:
            self.assertLess(abs((forget_only * u).sum().item()), 2e-3 * forget_only.norm().item())
        out = rule._project("w", p, g_f, g_r) / s
        retain_part = out - forget_only
        span = torch.stack(stored, 1)
        least_squares = torch.linalg.lstsq(span, retain_part[:, None]).solution
        torch.testing.assert_close(span @ least_squares[:, 0], retain_part, atol=1e-5, rtol=0)

    def test_basis_growth_gram_schmidt_threshold_and_cap(self):
        v = torch.tensor([1., 4., 9., 16.])
        rule, p = rule_for(4, [], v, k=2)
        named = {"w": p}
        s = v.sqrt() + EPS
        g1 = torch.tensor([1., 0., 3., 0.])
        rule._grow_basis(named, [g1])
        self.assertEqual(len(rule.basis["w"]), 1)
        self.assertEqual(rule.basis["w"][0].dtype, torch.float16)
        torch.testing.assert_close(rule.basis["w"][0].float(), ((g1 / s) / (g1 / s).norm()), atol=1e-3, rtol=0)
        # Nearly parallel: a residual (5e-6) above 1e-12 but below 1e-3 x the gradient's norm, not kept.
        rule._grow_basis(named, [2 * g1 + torch.tensor([0., 1e-5, 0., 0.])])
        self.assertEqual(len(rule.basis["w"]), 1)
        rule._grow_basis(named, [torch.tensor([0., 2., 0., 1.])])
        u0, u1 = (u.float() for u in rule.basis["w"])
        self.assertLess(abs((u0 * u1).sum().item()), 1e-3)
        rule._grow_basis(named, [torch.tensor([5., -1., 2., 7.])])  # k = 2 reached
        self.assertEqual(len(rule.basis["w"]), 2)
        empty, q = rule_for(4, [], v)
        empty.optimizer.state.clear()
        empty._grow_basis({"w": q}, [g1])  # no second moment yet: no whitening, no vector
        self.assertEqual(empty.basis["w"], [])

    def test_projected_tensors(self):
        names = ["backbone.model.language_model.model.layers.2.self_attn.q_proj.weight",
                 *[f"action_head.model.transformer_blocks.{i}.ff.weight" for i in range(5)],
                 "action_head.model.proj_out_2.weight", "action_head.action_decoder.0.weight",
                 "action_head.action_encoder.weight"]
        self.assertEqual(projected_names(names, 3), [
            "action_head.model.transformer_blocks.2.ff.weight", "action_head.model.transformer_blocks.3.ff.weight",
            "action_head.model.transformer_blocks.4.ff.weight", "action_head.model.proj_out_2.weight",
            "action_head.action_decoder.0.weight"])
        with self.assertRaisesRegex(ValueError, "no trainable DiT"):
            projected_names(names[:1], 3)


def run(directory, method, steps, recipe_changes=None, interrupt_at=None, seed=41):
    """Train UB-<method> on the fake for `steps` steps; returns (model, losses, rule)."""
    with patch.object(Node, "dir", new_callable=lambda: property(lambda self: directory)), \
         patch("vlu.data.make_sample", make_sample), cpu_runtime():
        node = Node.from_path("gu", f"LA/LB/UB-{method}")
        recipe = dataclasses.replace(RECIPE, unlearn_lr=3e-3, unlearn_steps=steps, batch_size=2, log_every=steps,
                                     save_every=2, **(recipe_changes or {}))
        torch.manual_seed(seed)
        model, processor = Fake(), Proc()
        demos = {t: Demos(t) for t in "AB"}
        plan = plan_op(node, recipe)
        plan.prepare(OpContext(node, model, demos, processor, {}, 0))
        losses = []
        if plan.update_rule is not None:
            original = plan.update_rule.step
            def step(*args):
                if interrupt_at is not None and len(losses) == interrupt_at:
                    raise InterruptedError()
                result = original(*args)
                losses.append(result[0].item())
                return result
            plan.update_rule.step = step
        data = OpBatches(demos, plan.parts, processor, {}, seed=0, num_steps=steps)
        probes = {t: collate([make_sample(None, None, d, 0, 0)])["inputs"] for t, d in demos.items()}
        torch.manual_seed(seed + 1)
        try:
            train(node, model, plan, data, probes, recipe, set())
        except InterruptedError:
            if interrupt_at is None:
                raise
        return model, losses, plan.update_rule


class TrainingTests(unittest.TestCase):
    def test_empty_basis_trains_exactly_as_npo_rt(self):
        """With the basis never refreshed after the first step (where the anchor is zero), every step is
        NPO-RT's: same forwards, random draws, gradients (up to summation order) and AdamW."""
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            npo_rt, _, _ = run(Path(a), "npo_rt", 4)
            gu, _, rule = run(Path(b), "npo_rt_gu", 4, {"gu_refresh_every": 10 ** 6})
        self.assertEqual(sum(map(len, rule.basis.values())), 0)
        moved = 0
        for (n, p), q in zip(npo_rt.named_parameters(), gu.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=2e-6, msg=n)
            moved += not torch.equal(p, Fake().state_dict()[n]) if p.requires_grad else 0
        self.assertGreater(moved, 0)

    def test_projection_changes_the_projected_tensors_only_through_the_basis(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            npo_rt, _, _ = run(Path(a), "npo_rt", 4)
            gu, _, rule = run(Path(b), "npo_rt_gu", 4)
        self.assertGreater(sum(map(len, rule.basis.values())), 0)  # filled from the 3rd step
        self.assertTrue(all(len(u) <= RECIPE.gu_basis_k for u in rule.basis.values()))
        self.assertGreater(rule.last_anchor, 0.)
        differs = [n for (n, p), q in zip(npo_rt.named_parameters(), gu.parameters()) if not torch.equal(p, q)]
        self.assertTrue(any(n in rule.names for n in differs), differs)
        for (n, p), q in zip(npo_rt.named_parameters(), gu.parameters()):
            if not p.requires_grad:
                self.assertTrue(torch.equal(p, q), n)

    def test_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "full").mkdir()
            full, expected, _ = run(Path(tmp) / "full", "npo_rt_gu", 6)
            (Path(tmp) / "resume").mkdir()
            _, first, _ = run(Path(tmp) / "resume", "npo_rt_gu", 6, interrupt_at=4)
            self.assertTrue((Path(tmp) / "resume" / "resume.pt").exists())
            resumed, last, rule = run(Path(tmp) / "resume", "npo_rt_gu", 6)
        self.assertEqual(expected, first + last)
        self.assertEqual(rule.calls, 6)
        for (n, p), q in zip(full.named_parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(p, q), n)

    def test_planning_checks(self):
        recipe = dataclasses.replace(RECIPE, unlearn_lr=1e-5)
        node = Node.from_path("gu", "LA/LB/UB-npo_rt_gu")
        check_op(node, recipe)
        for key, value in (("gu_basis_k", 0), ("gu_last_blocks", 0), ("gu_refresh_every", 0), ("gu_cap", 1.5),
                           ("gu_cap", float("nan")), ("npo_beta", 0.)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                check_op(node, dataclasses.replace(recipe, **{key: value}))
        with self.assertRaisesRegex(ValueError, "no learning rate"):
            check_op(node, RECIPE)
        with self.assertRaisesRegex(ValueError, "needs a retained task"):
            check_op(Node.from_path("gu", "LA/UA-npo_rt_gu"), recipe)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
