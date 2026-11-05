"""Known-answer subspace checks, including rank-deficient tall/wide matrices."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu.drift.metrics import subspace_overlap, top_singular_subspace


class RankSubspaceTests(unittest.TestCase):
    def assert_basis(self, w, k, rank):
        basis = top_singular_subspace(w, k)
        self.assertEqual(basis.shape, (min(k, rank), w.shape[0]))
        self.assertTrue(torch.allclose(basis @ basis.T, torch.eye(len(basis)), atol=1e-5))
        self.assertAlmostEqual(subspace_overlap(basis, basis), 1., places=5)
        return basis

    def test_reported_tall_rank_one_counterexample(self):
        w = torch.tensor([[1., 0.], [0., 0.], [0., 0.]])
        basis = self.assert_basis(w, 2, 1)
        self.assertAlmostEqual(abs(basis[0, 0].item()), 1.)

    def test_wide_and_square_rank_deficiency(self):
        self.assert_basis(torch.tensor([[1., 0., 0.], [0., 0., 0.]]), 3, 1)
        self.assert_basis(torch.diag(torch.tensor([3., 2., 0.])), 3, 2)

    def test_rotated_low_rank_subspace(self):
        torch.manual_seed(7)
        left, _ = torch.linalg.qr(torch.randn(12, 2))
        right, _ = torch.linalg.qr(torch.randn(8, 2))
        w = left @ torch.diag(torch.tensor([3., 1.])) @ right.T
        basis = self.assert_basis(w, 8, 2)
        self.assertAlmostEqual(subspace_overlap(basis, left.T), 1., places=5)

    def test_full_rank_and_top_k_order_are_preserved(self):
        for shape in ((8, 12), (12, 8)):
            torch.manual_seed(11)
            w = torch.randn(*shape)
            u, _, _ = torch.linalg.svd(w, full_matrices=False)
            basis = self.assert_basis(w, 3, min(shape))
            self.assertAlmostEqual(subspace_overlap(basis, u[:, :3].T), 1., places=5)

    def test_zero_matrix_has_no_drift_subspace(self):
        self.assertIsNone(top_singular_subspace(torch.zeros(5, 4), 2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
