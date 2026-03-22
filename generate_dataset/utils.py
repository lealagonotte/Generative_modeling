import numpy as np


def normalize(X):
    """Zero-mean, unit-variance normalization per coordinate."""
    return ((X - X.mean(axis=0)) / X.std(axis=0)).astype(np.float32)


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
    Corruption Compressed Sensing (Ambient Diffusion, Corollary A.2).
    A ∈ R^(m x d) : m lignes iid N(0, I_d) → y = A·x ∈ R^m
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if len(X.shape) == 2:
        N, d = X.shape
    else:
        # Nx2D case
        N, n, d = X.shape
        X = X.reshape(N, n*d)
        d = n*d

    A_matrices = rng.standard_normal(size=(N, m, d), dtype=np.float32)          # (N, m, d)
    Y = np.einsum('nmd,nd->nm', A_matrices, X, dtype=np.float32)                # (N, m)
    
    return Y, A_matrices


# ======================= N×2D Corruption Functions =======================

def inpainting_corruption_Nx2D(X, p=0.2, prevent_zero=True, rng=None):
    """
    Per-coordinate inpainting corruption for batched point clouds.

    Each coordinate of each point is independently masked with probability p.

    Args:
        X            : clean data, shape (n_clouds, N, D)
        p            : probability of masking each coordinate
        prevent_zero : if True, forbids the all-zero mask per point
        rng          : numpy Generator for reproducibility

    Returns:
        Y : corrupted data, shape (n_clouds, N, D)
        A : binary mask,     shape (n_clouds, N, D)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_clouds, N, D = X.shape
    A = (rng.uniform(size=(n_clouds, N, D)) > p).astype(np.float32)

    if prevent_zero:
        all_zero = A.sum(axis=2) == 0  # (n_clouds, N)
        idx_cloud, idx_point = np.where(all_zero)
        if len(idx_cloud) > 0:
            forced_dim = rng.integers(0, D, size=len(idx_cloud))
            A[idx_cloud, idx_point, forced_dim] = 1.0

    Y = X * A
    return Y, A


def inpainting_corruption_pointwise_Nx2D(X, p=0.2, rng=None):
    """
    Point-level inpainting corruption for batched point clouds.

    Each point is entirely masked (all coordinates) with probability p.

    Args:
        X   : clean data, shape (n_clouds, N, D)
        p   : probability of masking each point
        rng : numpy Generator for reproducibility

    Returns:
        Y : corrupted data, shape (n_clouds, N, D)
        A : binary mask,     shape (n_clouds, N, D)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_clouds, N, D = X.shape
    mask = (rng.uniform(0, 1, size=(n_clouds, N)) >= p).astype(np.float32)

    Y = X * mask[:, :, None]
    A = np.repeat(mask[:, :, None], D, axis=2)
    return Y, A


# ======================= Geometric Augmentations for N×2D =======================

def random_rotation_2D(X, rng=None):
    """
    Apply a random rotation to each point cloud.

    Args:
        X   : (n_clouds, N, 2)
        rng : numpy Generator

    Returns:
        X_rot : (n_clouds, N, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_clouds = X.shape[0]
    angles = rng.uniform(0, 2 * np.pi, size=n_clouds)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Rotation matrices: (n_clouds, 2, 2)
    R = np.stack([
        np.stack([cos_a, -sin_a], axis=1),
        np.stack([sin_a,  cos_a], axis=1),
    ], axis=1)

    # Apply: X_rot[i] = X[i] @ R[i].T  (equiv to each point rotated)
    X_rot = np.einsum('cnj,cij->cni', X, R)
    return X_rot


def random_translation_2D(X, shift_std=0.5, rng=None):
    """
    Apply a random translation to each point cloud.

    Args:
        X         : (n_clouds, N, 2)
        shift_std : standard deviation of the Gaussian shift
        rng       : numpy Generator

    Returns:
        X_shifted : (n_clouds, N, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_clouds = X.shape[0]
    shifts = rng.normal(0, shift_std, size=(n_clouds, 1, 2))
    return X + shifts


def random_scale_2D(X, scale_min=0.5, scale_max=2.0, rng=None):
    """
    Apply a random isotropic scaling to each point cloud.

    Args:
        X         : (n_clouds, N, 2)
        scale_min : minimum scale factor
        scale_max : maximum scale factor
        rng       : numpy Generator

    Returns:
        X_scaled : (n_clouds, N, 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_clouds = X.shape[0]
    scales = rng.uniform(scale_min, scale_max, size=(n_clouds, 1, 1))
    return X * scales
