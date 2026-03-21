"""Flexible dataset generation script for Ambient Diffusion experiments.

Supports two modes:
  - 2D:    One sample = one 2D point.       Output shape: (n_samples, 2)
  - Nx2D:  One sample = a full point cloud. Output shape: (n_clouds, N, 2)
"""

import argparse
import os, sys
import pickle
from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll

BASE_DIR = str(Path(__file__).resolve().parent)
sys.path.append(BASE_DIR)

from utils import (normalize,
                   inpainting_corruption, inpainting_corruption_pointwise,
                   compressed_sensing_corruption,
                   inpainting_corruption_Nx2D, inpainting_corruption_pointwise_Nx2D,
                   random_rotation_2D, random_translation_2D, random_scale_2D)


def generate_data(dataset, n_samples, noise, seed):
    """Generate raw 2D data from the chosen distribution."""
    if dataset == "two_moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif dataset == "swiss_roll":
        X, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
        X = X[:, [0, 2]]  # project to 2D
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X = normalize(X)
    return X


def generate_Nx2D_data(dataset, n_clouds, n_points_per_cloud,
                        noise_min, noise_max,
                        augment_rotation, augment_translation, shift_std,
                        augment_scale, scale_min, scale_max,
                        seed):
    """Generate N×2D point cloud dataset with geometric augmentations."""
    rng = np.random.default_rng(seed)
    noise_levels = rng.uniform(noise_min, noise_max, size=n_clouds)

    clouds = []
    for i, noise_val in enumerate(noise_levels):
        if dataset == "two_moons":
            pts, _ = make_moons(n_samples=n_points_per_cloud, noise=noise_val, random_state=i)
        elif dataset == "swiss_roll":
            pts, _ = make_swiss_roll(n_samples=n_points_per_cloud, noise=noise_val, random_state=i)
            pts = pts[:, [0, 2]]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        pts = normalize(pts)
        clouds.append(pts)

    X = np.stack(clouds, axis=0)  # (n_clouds, N, 2)

    rng_aug = np.random.default_rng(seed + 100)
    if augment_rotation:
        X = random_rotation_2D(X, rng=rng_aug)
    if augment_translation:
        X = random_translation_2D(X, shift_std=shift_std, rng=rng_aug)
    if augment_scale:
        X = random_scale_2D(X, scale_min=scale_min, scale_max=scale_max, rng=rng_aug)

    return X.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate corrupted 2D / N×2D datasets")

    # Common args
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["two_moons", "swiss_roll"],
                        help="Type of 2D distribution")
    parser.add_argument("--mode", type=str, default="2D",
                        choices=["2D", "Nx2D"],
                        help="2D (one point = one sample) or Nx2D (one cloud = one sample)")
    parser.add_argument("--corruption", type=str, default="inpainting",
                        choices=["inpainting", "inpainting_pw", "gaussian"],
                        help="Corruption type")
    parser.add_argument("--p", type=float, default=0.2,
                        help="Corruption probability")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pkl file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--prevent_zero", action="store_true",
                        help="Prevent all-zero masks (inpainting only)")

    # 2D-specific args
    parser.add_argument("--n_samples", type=int, default=100000,
                        help="[2D] Number of points to generate")
    parser.add_argument("--noise", type=float, default=None,
                        help="[2D] Dataset noise level (default: 0.1 for two_moons, 0.5 for swiss_roll)")
    parser.add_argument("--m", type=float, default=2,
                        help="[2D] Number of observed measurements (gaussian corruption)")

    # Nx2D-specific args
    parser.add_argument("--n_clouds", type=int, default=1000,
                        help="[Nx2D] Number of point clouds to generate")
    parser.add_argument("--n_points", type=int, default=200,
                        help="[Nx2D] Number of points per cloud (N)")
    parser.add_argument("--noise_min", type=float, default=0.01,
                        help="[Nx2D] Minimum noise level")
    parser.add_argument("--noise_max", type=float, default=0.9,
                        help="[Nx2D] Maximum noise level")
    parser.add_argument("--no_rotation", action="store_true",
                        help="[Nx2D] Disable random rotation augmentation")
    parser.add_argument("--no_translation", action="store_true",
                        help="[Nx2D] Disable random translation augmentation")
    parser.add_argument("--shift_std", type=float, default=0.5,
                        help="[Nx2D] Std of Gaussian translation")
    parser.add_argument("--no_scale", action="store_true",
                        help="[Nx2D] Disable random scale augmentation")
    parser.add_argument("--scale_min", type=float, default=0.5,
                        help="[Nx2D] Minimum scale factor")
    parser.add_argument("--scale_max", type=float, default=2.0,
                        help="[Nx2D] Maximum scale factor")

    args = parser.parse_args()

    if args.mode == "2D":
        # ---- 2D mode ----
        if args.noise is None:
            args.noise = 0.1 if args.dataset == "two_moons" else 0.5

        X = generate_data(args.dataset, args.n_samples, args.noise, args.seed)

        rng = np.random.default_rng(args.seed + 1)
        corruption = args.corruption
        if corruption == "inpainting":
            Y, A = inpainting_corruption(X, p=args.p, prevent_zero=args.prevent_zero, rng=rng)
        elif corruption == "inpainting_pw":
            Y, A = inpainting_corruption_pointwise(X, p=args.p, rng=rng)
        elif corruption == "gaussian":
            Y, A = compressed_sensing_corruption(X, m=int(args.m), rng=rng)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        data = {"mode": "2D", "type": corruption, "X": X, "A": A}
        with open(args.output, "wb") as f:
            pickle.dump(data, f)

        N, D = A.shape
        visible_rate = A.mean()
        print(f"[2D] Dataset: {args.dataset} | Corruption: {args.corruption} (p={args.p})")
        print(f"Samples: {N} | Dimensions: {D} | Visible rate: {100*visible_rate:.1f}%")
        print(f"Saved to {args.output}")

    else:
        # ---- Nx2D mode ----
        X = generate_Nx2D_data(
            dataset=args.dataset,
            n_clouds=args.n_clouds,
            n_points_per_cloud=args.n_points,
            noise_min=args.noise_min,
            noise_max=args.noise_max,
            augment_rotation=not args.no_rotation,
            augment_translation=not args.no_translation,
            shift_std=args.shift_std,
            augment_scale=not args.no_scale,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            seed=args.seed,
        )

        rng = np.random.default_rng(args.seed + 1)
        corruption = args.corruption
        if corruption == "inpainting":
            Y, A = inpainting_corruption_Nx2D(X, p=args.p, prevent_zero=args.prevent_zero, rng=rng)
        elif corruption == "inpainting_pw":
            Y, A = inpainting_corruption_pointwise_Nx2D(X, p=args.p, rng=rng)
        elif corruption == "gaussian":
            raise NotImplementedError("Compressed sensing corruption not yet supported for Nx2D mode.")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        data = {"mode": "Nx2D", "type": corruption, "X": X, "A": A,
                "n_points_per_cloud": args.n_points}
        with open(args.output, "wb") as f:
            pickle.dump(data, f)

        n_clouds, N, D = X.shape
        visible_rate = A.mean()
        print(f"[Nx2D] Dataset: {args.dataset} | Corruption: {args.corruption} (p={args.p})")
        print(f"Clouds: {n_clouds} | Points/cloud: {N} | Dim: {D} | Visible rate: {100*visible_rate:.1f}%")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
