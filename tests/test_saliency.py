"""Sharp Algorithm 1: exact ranks, ties, embodiment rows and complete data coverage."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import stubs  # noqa: F401
import torch

from vlu.ops import OpContext
from vlu.saliency import ranked_masks


class SaliencyTests(unittest.TestCase):
    def test_exact_ranks_match_sort_across_tensors(self):
        for seed in range(5):
            torch.manual_seed(seed)
            values = torch.randperm(101).float()
            scores = {"a": values[:40].reshape(5, 8), "b": values[40:]}
            for fraction in (.01, .3, .5, .99):
                for select in ("smallest", "largest"):
                    masks = ranked_masks(scores, fraction, select, 2)
                    actual = torch.cat([m.flatten() for m in masks.values()])
                    expected = torch.zeros(101, dtype=torch.bool)
                    expected[values.argsort(descending=select == "largest")[:int(101 * fraction)]] = True
                    self.assertTrue(torch.equal(actual, expected))

    def test_ties_fill_exact_count_including_zero_saliency(self):
        scores = {"a": torch.zeros(7), "b": torch.zeros(3)}
        for select in ("smallest", "largest"):
            mask = ranked_masks(scores, .3, select, 2)
            self.assertEqual(sum(int(m.sum()) for m in mask.values()), 3)
            self.assertTrue(torch.equal(mask["a"], torch.tensor([1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)))

    def test_unused_embodiment_rows_do_not_affect_rank(self):
        name = "action_head.state_encoder.W"
        scores = torch.zeros(32, 2, 4)
        scores[2] = torch.arange(8).reshape(2, 4)
        mask = ranked_masks({name: scores}, .25, "smallest", 2)[name]
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(mask[2].flatten()[:2].all())

    def test_full_forget_pass_visits_each_frame_and_restores_processor(self):
        visited = []
        class Processor:
            training = True
            def eval(self): self.training = False
            def train(self): self.training = True
            def collator(self, samples): return {"inputs": samples}
        processor = Processor()
        index = [(episode, step) for episode in range(2) for step in range(5)]
        ctx = OpContext(None, None, {"B": SimpleNamespace(index=index)}, processor, {}, 0)

        def sample(proc, config, demos, episode, step):
            self.assertFalse(proc.training)
            visited.append((episode, step))
            return episode, step

        with patch("vlu.data.make_sample", side_effect=sample):
            batches = list(ctx.full_task_batches("B", 4))
        self.assertEqual(visited, index)
        self.assertEqual([len(b) for b in batches], [4, 4, 2])
        self.assertTrue(processor.training)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
