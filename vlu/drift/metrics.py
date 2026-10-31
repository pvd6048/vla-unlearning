"""The math of the drift metrics, on plain tensors (no GR00T, no files), so it is easy to check.

Representations are (N, D) matrices: N probe samples (the same samples for every checkpoint), D
features. All functions compute in float64.
"""

import torch


def _centered(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x - x.mean(dim=0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019) between two representations of the same N samples:
    1 = identical up to rotation and isotropic scaling, 0 = unrelated. Computed with N x N Gram
    matrices, so D may be large. NaN if either representation does not vary across the samples
    (e.g. the shift of a drift that changes nothing): CKA is undefined there, not 0."""
    k, l = _centered(x) @ _centered(x).T, _centered(y) @ _centered(y).T
    if not (k.norm() > 0 and l.norm() > 0):
        return float("nan")
    return float((k * l).sum() / (k.norm() * l.norm()))


def pca_basis(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mean (D,), top-k principal directions (k, D) as orthonormal rows, their variances (k,))."""
    mean = x.double().mean(dim=0)
    _, s, vh = torch.linalg.svd(x.double() - mean, full_matrices=False)
    k = min(k, vh.shape[0])
    return mean, vh[:k], s[:k] ** 2 / max(x.shape[0] - 1, 1)


def subspace_overlap(u: torch.Tensor, v: torch.Tensor) -> float:
    """Overlap of the subspaces spanned by the orthonormal rows of u (k, D) and v (k', D):
    ||u v^T||_F^2 / min(k, k') = mean squared cosine of the principal angles. 1 = same subspace;
    two random k-dim subspaces of R^D give about k / D."""
    return float((u.double() @ v.double().T).square().sum() / min(u.shape[0], v.shape[0]))


def pca_shift(ref: torch.Tensor, new: torch.Tensor, k: int = 10) -> dict[str, float]:
    """How the representation `new` moved away from `ref` (same samples), in ref's top-k PCA space:
      pca_overlap   subspace_overlap of the two top-k PCA subspaces (1 = same principal directions)
      pca_distance  mean distance between each sample's ref and new projection, divided by the RMS
                    norm of ref's projections (0 = no movement, ~1 = moved as much as the data spread)"""
    mean, basis, _ = pca_basis(ref, k)
    _, new_basis, _ = pca_basis(new, k)
    z_ref = (ref.double() - mean) @ basis.T
    z_new = (new.double() - mean) @ basis.T
    spread = z_ref.square().sum(dim=1).mean().sqrt().clamp_min(1e-300)
    return {
        "pca_overlap": subspace_overlap(basis, new_basis),
        "pca_distance": float((z_new - z_ref).norm(dim=1).mean() / spread),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))


def top_singular_subspace(w: torch.Tensor, k: int) -> torch.Tensor | None:
    """Top-k left singular vectors of a 2-D matrix, as orthonormal rows (k, rows); None if w = 0.
    Exact, via the eigenvectors of the smaller Gram matrix (at most 2048 x 2048 for GR00T's weights)."""
    w = w.float()
    if not w.abs().max() > 0:
        return None
    m, n = w.shape
    k = min(k, m, n)
    if m <= n:
        _, vectors = torch.linalg.eigh(w @ w.T)  # ascending eigenvalues
        return vectors[:, -k:].flip(-1).T
    _, vectors = torch.linalg.eigh(w.T @ w)
    u = w @ vectors[:, -k:].flip(-1)  # (m, k): left singular vectors up to scale
    return (u / u.norm(dim=0, keepdim=True).clamp_min(1e-30)).T
