import numpy as np


def normalize(X):
    """Zero-mean, unit-variance normalization per coordinate."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def inpainting_corruption(X, p=0.2, prevent_zero=True, rng=None):
    """
    Per-coordinate inpainting corruption.

    Each coordinate is independently masked with probability p.
    If prevent_zero=True, ensures at least one coordinate remains visible.

    Args:
        X            : clean data, shape (N, D)
        p            : probability of masking each coordinate
        prevent_zero : if True, forbids the all-zero mask
        rng          : numpy Generator for reproducibility

    Returns:
        Y : corrupted data (masked coords set to 0), shape (N, D)
        A : binary mask (1 = visible, 0 = masked), shape (N, D)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N, D = X.shape
    A = (rng.uniform(size=(N, D)) > p).astype(np.float32)

    if prevent_zero:
        all_zero = A.sum(axis=1) == 0
        n_zero = all_zero.sum()
        if n_zero > 0:
            forced_dim = rng.integers(0, D, size=n_zero)
            A[all_zero, forced_dim] = 1.0

    Y = X * A
    return Y, A


def inpainting_corruption_pointwise(X, p=0.2, rng=None):
    """
    Point-level inpainting corruption.

    Each point is entirely masked (all coordinates) with probability p.

    Args:
        X   : clean data, shape (N, D)
        p   : probability of masking each point
        rng : numpy Generator for reproducibility

    Returns:
        Y : corrupted data, shape (N, D)
        A : binary mask, shape (N, D)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, D = X.shape
    mask = (rng.uniform(0, 1, size=N) >= p).astype(np.float32)

    Y = X * mask[:, None]
    A = np.stack([mask] * D, axis=1)
    return Y, A


def compressed_sensing_corruption(X, m=2, rng=None):
    """
    Compressed sensing corruption.

    A in R^(m x d) with iid N(0, I_d) rows. y = A @ x.

    Args:
        X   : clean data, shape (N, d)
        m   : number of measurements
        rng : numpy Generator

    Returns:
        Y          : measurements, shape (N, m)
        A_matrices : measurement matrices, shape (N, m, d)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, d = X.shape
    A_matrices = rng.standard_normal(size=(N, m, d))
    Y = np.einsum('nmd,nd->nm', A_matrices, X)
    return Y, A_matrices
