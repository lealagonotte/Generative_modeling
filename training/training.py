import numpy as np
import os, sys
import pickle
import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from pathlib import Path

# Ensure sibling modules are importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ambient_diffusion import NoiseScheduler, FurtherCorrupter, AmbientLoss, Sampler
from module import Denoiser, FlatDenoiserNx2D, PointNetDenoiserNx2D
from viz import viz_sample_2D

try:
    from training.utils import TqdmToLogger
except Exception:
    from utils import TqdmToLogger

def load_dataset(path, batch_size=256, val_split=0.1):
    with open(path, "rb") as f:
        data = pickle.load(f)

    X = torch.tensor(data["X"], dtype=torch.float32)
    A = torch.tensor(data["A"], dtype=torch.float32)
    dataset_type = data["type"]

    n = X.shape[0]
    perm = torch.randperm(n)
    X, A = X[perm], A[perm]

    n_val = int(n * val_split)
    n_train = n - n_val

    train_dataset = TensorDataset(X[:n_train], A[:n_train])
    val_dataset = TensorDataset(X[n_train:], A[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset_type


def batch_step(module: nn.Module,
                    batch: tuple,
                    loss: nn.Module,
                    noise_scheduler: NoiseScheduler,
                    further_corrupter: FurtherCorrupter,
                    method: str):
    x0, A = batch
    device = next(module.parameters()).device
    x0, A = x0.to(device), A.to(device)
    batch_size = x0.size(0)

    # sample random time steps uniformly in [0, 1]
    t = torch.rand(batch_size, device=device)

    # sample noise
    eps = torch.randn_like(x0)

    # obtaining x_t (handles VP/VE/interpolation internally)
    x_t = noise_scheduler.apply_noise(x0, t, eps)

    # obtain the further corrupted operator
    if method == "naive":
        further_A = A # the naive approach described in ambient diffusion, Eq. 3.1
    else:
        further_A = further_corrupter.get_operator(A)

    # applying the further corruption to x_t
    further_x_t = further_corrupter.apply_operator(further_A, x_t)

    # pass through the module
    predicted_x0 = module(further_A, further_x_t, t)

    loss_value = loss(x0, predicted_x0, A)

    return loss_value

def train(train_dataloader: DataLoader, val_dataloader: DataLoader,
            epochs: int, patience: int,
            loss: nn.Module, optimizer,
            module: nn.Module,
            noise_scheduler: NoiseScheduler,
            further_corrupter: FurtherCorrupter,
            method: str = "ambient",
            logger: None | logging.Logger = None
            ):
    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    pbar = tqdm(range(epochs), desc="Epoch")
    for epoch in pbar:
        # Training loop
        module.train()
        epoch_train_loss = []
        for i, batch in enumerate(train_dataloader):
            loss_value = batch_step(module, batch, loss, noise_scheduler, 
            further_corrupter, method)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            epoch_train_loss.append(loss_value.detach().cpu().numpy())

        # Validation loop
        module.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                loss_value = batch_step(module, batch, loss, noise_scheduler, further_corrupter, method)

                epoch_val_loss.append(loss_value.detach().cpu().numpy())

        train_mean = np.mean(epoch_train_loss)
        val_mean = np.mean(epoch_val_loss)
        epoch_train_losses.append(train_mean)
        epoch_val_losses.append(val_mean)

        pbar.set_postfix(train=f"{train_mean:.4e}", val=f"{val_mean:.4e}")

        if logger:
            logger.info(f"Epoch {epoch+1}: train={train_mean:.4e}, val={val_mean:.4e}")

        # Early stopping
        if val_mean < best_val_loss:
            best_val_loss = val_mean
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                else:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    return module, epoch_train_losses, epoch_val_losses

def sample(shape, n_steps, device, module, further_corrupter, noise_scheduler):
    # Use Fixed Mask Sampling as the authors
    sampler = Sampler("fms")

    # For sampling, we first need to sample a further corrupted mask
    A = further_corrupter.init_operator(shape, device) # sample A
    A_sample = further_corrupter.get_operator(A) # sample A_further

    start = time.time()
    samples = sampler.sample(shape, n_steps, A_sample, module, noise_scheduler)
    elapsed = time.time() - start
    return samples, elapsed

def main():
    parser = argparse.ArgumentParser(description="Ambient Diffusion Training")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .pkl dataset")
    parser.add_argument("--method", type=str, default="ambient", help="Type of training framework")
    parser.add_argument("--schedule", type=str, default="interpolation", choices=["interpolation", "ve", "vp"])
    parser.add_argument("--sigma_max", type=float, default=1.0, help="Parameter used in Interpolation and Variance Exploding schemes.")
    parser.add_argument("--sigma_min", type=float, default=0.01, help="Variance Exploding parameter")
    parser.add_argument("--beta_max", type=float, default=20.0, help="Variance Preserving parameter")
    parser.add_argument("--beta_min", type=float, default=0.1, help="Variance Preserving parameter")
    parser.add_argument("--further_p", type=float, default=0.1, help="Further corruption probability")
    
    # Model args
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "flat_nx2d", "pointnet_nx2d"])
    parser.add_argument("--n_points_per_cloud", type=int, default=None, help="Required for Nx2D models")
    parser.add_argument("--data_dim", type=int, default=2, help="Data dimension (usually 2)")
    
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--format", type=str, default="png", help="plot format", choices=['png', 'pdf'])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, dataset_type = load_dataset(args.dataset, args.batch_size)

    # Setup components
    if args.schedule == "interpolation":
        noise_scheduler = NoiseScheduler(args.schedule, sigma_max=args.sigma_max)
    elif args.schedule == "vp":
        noise_scheduler = NoiseScheduler(args.schedule, beta_min=args.beta_min, beta_max=args.beta_max)
    elif args.schedule == "ve":
        noise_scheduler = NoiseScheduler(args.schedule, sigma_max=args.sigma_max, sigma_min=args.sigma_min)
    else:
        raise ValueError(f"Unknown noise schedule {args.schedule}. Must be eitehr 'interpolation', 'vp' or 've'.")
    
    if dataset_type == "inpainting":
        further_corrupter = FurtherCorrupter(dataset_type, p=args.further_p)
    elif dataset_type == "inpainting_pw":
        further_corrupter = FurtherCorrupter(dataset_type, p=args.further_p)
    elif dataset_type == "gaussian":
        further_corrupter = FurtherCorrupter(dataset_type) # we always use m_prime=1
    else:
        raise ValueError(f"Unknown dataset corruption type {dataset_type}. Must be eitehr 'inpainting' or 'gaussian'.")
    
    if args.model == "mlp":
        module = Denoiser(data_dim=args.data_dim).to(device)
    elif args.model == "flat_nx2d":
        assert args.n_points_per_cloud is not None, "n_points_per_cloud must be set for flat_nx2d"
        module = FlatDenoiserNx2D(n_points=args.n_points_per_cloud, data_dim=args.data_dim).to(device)
    elif args.model == "pointnet_nx2d":
        module = PointNetDenoiserNx2D(data_dim=args.data_dim).to(device)
    else:
        raise ValueError(f"Unknown model type {args.model}")
    
    ambient_loss = AmbientLoss(further_corrupter.apply_operator_func)
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    # Train
    module, train_losses, val_losses = train(
        train_loader, val_loader, args.epochs, args.patience,
        ambient_loss, optimizer, module, noise_scheduler, further_corrupter,
        args.method,
    )

    # Load clean reference data for plotting
    with open(args.dataset, "rb") as f:
        ref_data = pickle.load(f)
    X_ref = ref_data["X"]

    # Generate sampling GIF with reference overlay
    gif_path = os.path.join(args.output, "sampling.gif")
    viz_sample_2D(
        module, noise_scheduler, further_corrupter,
        n_samples=args.n_samples, n_steps=args.n_steps,
        output_path=gif_path,
        ref_data=X_ref,
    )

    # Plot training curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="Train", color=sns.color_palette("mako", 3)[1])
    ax.plot(val_losses, label="Val", color=sns.color_palette("flare", 3)[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Training curves")
    plt.tight_layout()
    curves_path = os.path.join(args.output, f"training_curves.{args.format}")
    plt.savefig(curves_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {curves_path}")

if __name__ == "__main__":
    main()
