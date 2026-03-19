import numpy as np
import os, sys
import pickle
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Ensure sibling modules are importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ambient_diffusion import NoiseScheduler, FurtherCorrupter, AmbientLoss, Sampler
from module import Denoiser

def load_dataset(path, batch_size=256, val_split=0.1):
    with open(path, "rb") as f:
        data = pickle.load(f)

    X = torch.tensor(data["X"], dtype=torch.float32)
    A = torch.tensor(data["A"], dtype=torch.float32)

    n = X.shape[0]
    perm = torch.randperm(n)
    X, A = X[perm], A[perm]

    n_val = int(n * val_split)
    n_train = n - n_val

    train_dataset = TensorDataset(X[:n_train], A[:n_train])
    val_dataset = TensorDataset(X[n_train:], A[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


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
            ):
    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training loop
        module.train()
        epoch_train_loss = []
        for i, batch in enumerate(train_dataloader):
            loss_value = batch_step(module, batch, loss, noise_scheduler, further_corrupter, method)

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

        print(f"Epoch {epoch+1}: train={train_mean:.4e}, val={val_mean:.4e}")

        # Early stopping
        if val_mean < best_val_loss:
            best_val_loss = val_mean
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return module, epoch_train_losses, epoch_val_losses

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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--output", type=str, default="output", help="Output directory")
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
    elif dataset_type == "gaussian":
        further_corrupter = FurtherCorrupter(dataset_type) # we always use m_prime=1
    else:
        raise ValueError(f"Unknown dataset corruption type {dataset_type}. Must be eitehr 'inpainting' or 'gaussian'.")
    
    module = Denoiser(data_dim=2).to(device)

    ambient_loss = AmbientLoss(further_corrupter.apply_operator_func)
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    # Train
    module, train_losses, val_losses = train(
        train_loader, val_loader, args.epochs, args.patience,
        ambient_loss, optimizer, module, noise_scheduler, further_corrupter,
        args.method,
    )

    # Sample
    sampler = Sampler("fms")
    # For sampling, use identity mask (all ones = fully observed)
    A_sample = torch.ones(args.n_samples, 2, device=device)

    start = time.time()
    samples = sampler.sample((args.n_samples, 2), args.n_steps, A_sample, module, noise_scheduler)
    elapsed = time.time() - start
    print(f"Generated {args.n_samples} samples in {elapsed:.2f}s")

    samples_np = samples.detach().cpu().numpy()

    # Load clean reference data for plotting
    with open(args.dataset, "rb") as f:
        ref_data = pickle.load(f)
    X_ref = ref_data["X"]

    # Plot generated vs reference
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X_ref[:, 0], X_ref[:, 1], s=1, alpha=0.2, c='lightgray', label='Reference')
    axes[0].scatter(samples_np[:, 0], samples_np[:, 1], s=5, alpha=0.6, c='steelblue', label='Generated')
    axes[0].legend()
    axes[0].set_title(f"Generated samples ({elapsed:.2f}s)")
    axes[0].set_aspect('equal')

    axes[1].plot(train_losses, label='Train')
    axes[1].plot(val_losses, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].set_title('Training curves')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "results.pdf"), bbox_inches='tight')
    print(f"Saved results to {args.output}/results.pdf")

if __name__ == "__main__":
    main()
