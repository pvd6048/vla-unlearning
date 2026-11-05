"""CPU regression checks for rejecting non-finite training and checkpoint state."""
import contextlib
import dataclasses
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import stubs  # noqa: F401
import torch
from safetensors.torch import save_file

from fakes import Demos, Fake, Proc, collate, make_sample
from vlu.config import RECIPE
from vlu.model import check_finite, load_weights, save_weights
from vlu.ops import OpContext
import vlu.train as training
import vlu.vlaforget as vf


def serial_loader(data, sampler, **kwargs):
    return (data[i] for i in sampler)


@contextlib.contextmanager
def cpu_runtime():
    """Keep the real objectives/optimizers; replace only CUDA and worker plumbing."""
    rand, fork_rng = torch.rand, torch.random.fork_rng
    def cpu_rand(*args, **kwargs):
        kwargs.pop("device", None)
        return rand(*args, **kwargs)
    with contextlib.ExitStack() as stack:
        for target, replacement in [
            ("torch.Tensor.cuda", lambda self, *a, **k: self),
            ("torch.rand", cpu_rand),
            ("torch.autocast", lambda *a, **k: contextlib.nullcontext()),
            ("torch.cuda.get_rng_state", torch.get_rng_state),
            ("torch.cuda.set_rng_state", torch.set_rng_state),
            ("torch.cuda.current_device", lambda: 0),
            ("torch.cuda.max_memory_allocated", lambda: 0),
            ("torch.random.fork_rng", lambda **k: fork_rng(devices=[])),
            ("vlu.train.DataLoader", serial_loader),
            ("vlu.vlaforget.DataLoader", serial_loader),
            ("vlu.data.make_sample", make_sample),
            ("vlu.vlaforget.make_sample", make_sample),
        ]:
            stack.enter_context(patch(target, replacement))
        yield


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.node = SimpleNamespace(dir=Path(self.tmp.name), weights_path=Path(self.tmp.name) / "weights.safetensors")

    def standard(self, model, loss_fn):
        plan = SimpleNamespace(lr=.01, grad_mask=None, steps=1, probe_every=None, loss_fn=loss_fn)
        recipe = dataclasses.replace(RECIPE, log_every=1)
        with cpu_runtime():
            training.train(self.node, model, plan, [{}], {}, recipe, set())

    def test_nonfinite_losses_and_gradients_are_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                model = torch.nn.Linear(1, 1)
                old = model.weight.detach().clone()
                with self.assertRaises(FloatingPointError):
                    self.standard(model, lambda m, b: (m.weight.sum() * bad, {}))
                self.assertTrue(torch.equal(model.weight, old))
                self.assertFalse(self.node.weights_path.exists())
        model = torch.nn.Linear(1, 1)
        model.weight.register_hook(lambda grad: grad * float("nan"))
        with self.assertRaises(RuntimeError):
            self.standard(model, lambda m, b: (m.weight.square().sum(), {}))
        self.assertFalse(self.node.weights_path.exists())

    def test_nonfinite_optimizer_update_is_rejected(self):
        original = training._optimizer
        def broken_optimizer(model, *args):
            optimizer = original(model, *args)
            def step(*a, **k):
                with torch.no_grad():
                    model.weight.fill_(float("nan"))
            optimizer.step = MethodType(step, optimizer)
            return optimizer
        with patch.object(training, "_optimizer", broken_optimizer), self.assertRaises(FloatingPointError):
            self.standard(torch.nn.Linear(1, 1), lambda m, b: (m.weight.square().sum(), {}))
        self.assertFalse(self.node.weights_path.exists())

    def test_invalid_checkpoint_cannot_replace_existing_weights(self):
        model = torch.nn.Linear(1, 1)
        save_weights(model, self.node.weights_path)
        old = self.node.weights_path.read_bytes()
        with torch.no_grad():
            model.weight.fill_(float("inf"))
        with self.assertRaises(FloatingPointError):
            save_weights(model, self.node.weights_path)
        self.assertEqual(self.node.weights_path.read_bytes(), old)
        invalid = self.node.dir / "invalid.safetensors"
        save_file({"weight": model.weight}, str(invalid))
        with self.assertRaises(FloatingPointError):
            load_weights(torch.nn.Linear(1, 1), invalid)

    def vf_run(self):
        torch.manual_seed(17)
        model, proc = Fake(), Proc()
        demos = {t: Demos(t) for t in "ABC"}
        probes = {t: collate([make_sample(None, None, demos[t], 0, 0)])["inputs"] for t in demos}
        cfg = dataclasses.replace(RECIPE.vlaforget, selection_batches=1, boundary_forget_pool=2,
                                  boundary_retain_pool=4, top_k_vision=1, top_k_projector=1,
                                  top_k_reasoning=1, lora_r=2, lora_alpha=2)
        recipe = dataclasses.replace(RECIPE, batch_size=1, vlaforget=cfg)
        return vf.run(OpContext(self.node, model, demos, proc, {}, seed=0), probes, recipe, "B", ("A", "C"), 7)

    def test_vlaforget_rejects_nonfinite_combined_gradient(self):
        original = vf.set_grads
        with cpu_runtime(), patch.object(vf, "set_grads", lambda ps, g: original(ps, g * float("nan"))):
            with self.assertRaises(RuntimeError):
                self.vf_run()
        self.assertFalse(self.node.weights_path.exists())

    def test_vlaforget_rejects_nonfinite_merge_probes(self):
        count = 0
        def probe(model, probes):
            nonlocal count
            count += 1
            # Start + three stage ends + pre-merge + post-merge.
            return {t: (float("nan") if count == 6 else 1.) for t in probes}
        with cpu_runtime(), patch.object(vf, "probe_losses", probe):
            with self.assertRaisesRegex(FloatingPointError, "merge"):
                self.vf_run()

    def test_finite_state_is_accepted(self):
        check_finite({"x": torch.tensor([0., 1.])}, "test")
        self.standard(torch.nn.Linear(1, 1), lambda m, b: (m.weight.square().sum(), {}))
        self.assertTrue(self.node.weights_path.exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
