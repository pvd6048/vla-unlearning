"""CPU checks for deterministic preprocessing and interrupted optimizer continuation."""
import copy
import dataclasses
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import stubs  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

import vlu.data as data
from vlu.config import RECIPE
import vlu.train as training
from test_training_safety import cpu_runtime


class SeededTransform:
    def set_random_seed(self, seed):
        self.rng = np.random.default_rng(seed)


class Processor:
    def __init__(self):
        self.train_image_transform = SeededTransform()

    def collator(self, samples):
        return {"inputs": {"x": torch.tensor(samples).reshape(-1, 1)}}


def augmented_sample(processor, mcfg, demos, episode, step):
    return (step + random.random() + np.random.random() + torch.rand(()).item()
            + processor.train_image_transform.rng.random())


def cpu_loader(dataset, **kwargs):
    kwargs.update(num_workers=0, pin_memory=False)
    kwargs.pop("prefetch_factor")
    kwargs.pop("multiprocessing_context")
    return DataLoader(dataset, **kwargs)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def batches(self, steps):
        demos = {"A": SimpleNamespace(index=[(0, i) for i in range(10)])}
        return data.OpBatches(demos, {"train": [("A",)] * 2}, Processor(), {}, seed=17, num_steps=steps)

    def test_batches_match_across_worker_histories_and_leave_rng_unchanged(self):
        with patch.object(data, "make_sample", augmented_sample):
            a, b = self.batches(10), self.batches(10)
            random.seed(2); np.random.seed(2); torch.manual_seed(2)
            py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
            expected = a[7]["train"]["x"]
            self.assertEqual(py, random.getstate())
            self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
            self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
            for step in (1, 9, 0, 4):
                b[step]
            random.seed(99); np.random.seed(99); torch.manual_seed(99)
            self.assertTrue(torch.equal(expected, b[7]["train"]["x"]))

    def run_training(self, name, initial, steps, interrupt_at=None):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        node = SimpleNamespace(dir=directory, weights_path=directory / "weights.safetensors")
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.2))
        model.load_state_dict(initial)
        random.seed(23); np.random.seed(23); torch.manual_seed(23)
        losses = []
        def loss_fn(model, batch):
            if len(losses) == interrupt_at:
                raise InterruptedError("simulated process interruption")
            x = batch["train"]["x"]
            target = torch.randn_like(x) + random.random() + np.random.random()
            loss = (model(x) - target).square().mean()
            losses.append(loss.item())
            return loss, {}
        plan = SimpleNamespace(steps=steps, lr=.01, grad_mask=None, probe_every=None, loss_fn=loss_fn)
        recipe = dataclasses.replace(RECIPE, log_every=1000, save_every=2500)
        with cpu_runtime(), patch.object(training, "DataLoader", cpu_loader), patch.object(data, "make_sample", augmented_sample):
            try:
                training.train(node, model, plan, self.batches(steps), {}, recipe, set())
            except InterruptedError:
                if interrupt_at is None:
                    raise
        return model, losses, node

    def test_interrupted_run_matches_uninterrupted_losses_and_weights(self):
        torch.manual_seed(5)
        initial = copy.deepcopy(torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.2)).state_dict())
        full, expected, _ = self.run_training("full", initial, 6)
        _, first, node = self.run_training("resumed", initial, 6, interrupt_at=3)
        state = torch.load(node.dir / "resume.pt", weights_only=False)
        self.assertEqual(state["step"], 3)
        resumed, last, _ = self.run_training("resumed", initial, 6)
        self.assertEqual(expected, first + last)
        for name, value in full.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)

    def test_default_unlearning_budget_writes_a_midrun_resume_checkpoint(self):
        initial = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.2)).state_dict()
        _, _, node = self.run_training("short_u", initial, RECIPE.unlearn_steps, interrupt_at=500)
        state = torch.load(node.dir / "resume.pt", weights_only=False)
        self.assertEqual(state["step"], 500)
        self.assertIn("python_rng", state)
        self.assertIn("numpy_rng", state)
        self.assertFalse(node.weights_path.exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
