"""Flexible dataset generation script for Ambient Diffusion experiments."""

import argparse
import os
import pickle
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll

from utils import normalize, inpainting_corruption, inpainting_corruption_pointwise, compressed_sensing_corruption


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


def main():
    parser = argparse.ArgumentParser(description="Generate corrupted 2D datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["two_moons", "swiss_roll"],
                        help="Type of 2D distribution")
    parser.add_argument("--n_samples", type=int, default=100000,
                        help="Number of points to generate")
    parser.add_argument("--corruption", type=str, default="inpainting",
                        choices=["inpainting", "inpainting_pw", "gaussian"],
                        help="Corruption type")
    parser.add_argument("--m", type=float, default=2,
                        help="Number of observed measurements")
    parser.add_argument("--p", type=float, default=0.2,
                        help="Corruption probability")
    parser.add_argument("--noise", type=float, default=None,
                        help="Dataset noise level (default: 0.1 for two_moons, 0.5 for swiss_roll)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pkl file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--prevent_zero", action="store_true",
                        help="Prevent all-zero masks (inpainting only)")

    args = parser.parse_args()

    # Default noise per dataset
    if args.noise is None:
        args.noise = 0.1 if args.dataset == "two_moons" else 0.5

    # Generate clean data
    X = generate_data(args.dataset, args.n_samples, args.noise, args.seed)

    # Apply corruption
    rng = np.random.default_rng(args.seed + 1)
    corruption = args.corruption
    if corruption == "inpainting":
        Y, A = inpainting_corruption(X, p=args.p, prevent_zero=args.prevent_zero, rng=rng)
    elif corruption == "inpainting_pw":
        Y, A = inpainting_corruption_pointwise(X, p=args.p, rng=rng)
    elif corruption == "gaussian":
        Y, A = compressed_sensing_corruption(X, m=args.m, rng=rng)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    data = {"type": corruption,"X": X, "A": A}
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    # Summary
    N, D = A.shape
    visible_rate = A.mean()
    print(f"Dataset: {args.dataset} | Corruption: {args.corruption} (p={args.p})")
    print(f"Samples: {N} | Dimensions: {D} | Visible rate: {100*visible_rate:.1f}%")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
