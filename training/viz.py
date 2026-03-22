"""Module to define visualizations and animations for 2D diffusion sampling."""

import os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

BASE_DIR = str(Path(__file__).resolve().parent)
sys.path.append(BASE_DIR)

from ambient_diffusion import Sampler


sns.set_theme(style="whitegrid")


def viz_sample_2D(
    module,
    noise_scheduler,
    further_corrupter,
    n_samples=1000,
    n_steps=100,
    output_path="sampling.gif",
    fps=16,
    n_frames=None,
    ref_data=None,
    xlim=None,
    ylim=None,
    addon=None,
):
    """
    Create a GIF showing the reverse diffusion sampling process for 2D points.

    Args:
        module          : trained Denoiser
        noise_scheduler : NoiseScheduler instance
        further_corrupter: FurtherCorrupter instance
        n_samples       : number of points to sample
        n_steps         : number of reverse SDE steps
        output_path     : path to save the .gif
        fps             : frames per second
        n_frames        : number of frames in the GIF (subsampled from n_steps)
        ref_data        : optional (N, 2) numpy array of clean reference points
        xlim            : optional (xmin, xmax) tuple
        ylim            : optional (ymin, ymax) tuple
    """
    if n_frames is None:
        n_frames = min(int(0.5*n_steps), 120)

    device = next(module.parameters()).device
    shape = (n_samples, 2)

    # Sample mask (same logic as training.py sample())
    sampler = Sampler("fms")
    A = further_corrupter.init_operator(shape, device)
    A_sample = further_corrupter.get_operator(A)

    # Run sampling with trajectory
    module.eval()
    trajectory, timesteps = sampler.sample_with_trajectory(
        shape, n_steps, A_sample, module, noise_scheduler,
        apply_operator=further_corrupter.apply_operator_func
    )
    # trajectory: (n_steps, n_samples, 2), timesteps: (n_steps,)
    trajectory = trajectory.numpy()
    timesteps = timesteps.numpy()

    # Subsample frames evenly
    total_steps = trajectory.shape[0]
    frame_indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)
    frame_indices = np.unique(frame_indices)  # remove duplicates

    # Compute axis limits from trajectory if not provided
    if xlim is None:
        all_x = trajectory[:, :, 0]
        margin = 0.1 * (all_x.max() - all_x.min())
        xlim = (all_x.min() - margin, all_x.max() + margin)
    if ylim is None:
        all_y = trajectory[:, :, 1]
        margin = 0.1 * (all_y.max() - all_y.min())
        ylim = (all_y.min() - margin, all_y.max() + margin)

    # Color palette
    sample_color = sns.color_palette("mako", as_cmap=False, n_colors=3)[1]
    ref_color = "#8b8989"

    # Build animation
    fig, ax = plt.subplots(figsize=(6, 6))

    if addon is not None:
        addon += " | "
    else:
        addon = ""

    def update(frame_idx):
        ax.clear()
        step = frame_indices[frame_idx]
        points = trajectory[step]
        t_val = timesteps[min(step, len(timesteps) - 1)]

        # Reference data in background
        if ref_data is not None:
            ax.scatter(
                ref_data[:, 0], ref_data[:, 1],
                s=1, alpha=0.15, c=ref_color, rasterized=True,
                label="Reference"
            )

        # Current sample points
        ax.scatter(
            points[:, 0], points[:, 1],
            s=8, alpha=0.5, c=[sample_color], edgecolors="none",
            label="Sampled"
        )

        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"{addon}Step {step}/{total_steps - 1}  |  t = {t_val:.4f}", fontsize=12)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 // fps,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved sampling GIF to {output_path}")

    # Generate static plot with 4 subplots
    static_steps = [0, int((total_steps - 1) * 0.33), int((total_steps - 1) * 0.66), total_steps - 1]
    fig_static, axes_static = plt.subplots(1, 4, figsize=(16, 4))
    if addon:
        fig_static.suptitle(addon.replace(" | ", ""), fontsize=16)

    for idx, step in enumerate(static_steps):
        ax_static = axes_static[idx]
        points = trajectory[step]
        t_val = timesteps[min(step, len(timesteps) - 1)]

        if ref_data is not None:
            ax_static.scatter(ref_data[:, 0], ref_data[:, 1], s=1, alpha=0.15, c=ref_color, rasterized=True, label="Reference" if idx == 0 else "")
            
        ax_static.scatter(points[:, 0], points[:, 1], s=8, alpha=0.5, c=[sample_color], edgecolors="none", label="Sampled" if idx == 0 else "")
        
        if idx == 0:
            ax_static.legend()
        
        ax_static.set_xlim(xlim)
        ax_static.set_ylim(ylim)
        ax_static.set_aspect("equal")
        ax_static.set_title(f"Step {step}/{total_steps - 1}  |  t = {t_val:.4f}", fontsize=12)
        ax_static.set_xlabel("$x_1$")
        if idx == 0:
            ax_static.set_ylabel("$x_2$")

    fig_static.tight_layout()
    static_path = str(output_path).replace(".gif", ".png")
    fig_static.savefig(static_path, dpi=150)
    plt.close(fig_static)
    print(f"Saved static sampling plot to {static_path}")


def viz_sample_Nx2D(
    module,
    noise_scheduler,
    further_corrupter,
    n_clouds=9,         # Will arrange in a sqrt(n_clouds) x sqrt(n_clouds) grid
    n_points=1000,       # Pts per cloud
    n_steps=100,
    output_path="sampling_Nx2D.gif",
    fps=16,
    n_frames=None,
    ref_data=None,      # (n_ref_clouds, N, 2)
    xlim=None,
    ylim=None,
    addon=None,
):
    """
    Create a GIF showing the reverse diffusion sampling process for Nx2D point clouds.
    Plots multiple clouds in a small grid.

    Args:
        module          : trained Denoiser (FlatDenoiserNx2D or PointNetDenoiserNx2D)
        noise_scheduler : NoiseScheduler instance
        further_corrupter: FurtherCorrupter instance
        n_clouds        : number of clouds to sample (ideally a perfect square like 4, 9, 16)
        n_points        : points per cloud (N)
        n_steps         : number of reverse SDE steps
        output_path     : path to save the .gif
        fps             : frames per second
        n_frames        : number of frames in the GIF (subsampled from n_steps)
        ref_data        : optional reference data array
        xlim, ylim      : axis limits limits
    """
    if n_frames is None:
        n_frames = min(int(0.5*n_steps), 120)

    device = next(module.parameters()).device
    shape = (n_clouds, n_points, 2)

    sampler = Sampler("fms")
    A = further_corrupter.init_operator(shape, device)
    A_sample = further_corrupter.get_operator(A)

    module.eval()
    trajectory, timesteps = sampler.sample_with_trajectory(
        shape, n_steps, A_sample, module, noise_scheduler,
        apply_operator=further_corrupter.apply_operator_func
    )
    # trajectory: (n_steps, n_clouds, N, 2)
    trajectory = trajectory.numpy()
    timesteps = timesteps.numpy()

    total_steps = trajectory.shape[0]
    frame_indices = np.unique(np.linspace(0, total_steps - 1, n_frames, dtype=int))

    if xlim is None:
        all_x = trajectory[:, :, :, 0]
        margin = 0.1 * (all_x.max() - all_x.min())
        xlim = (all_x.min() - margin, all_x.max() + margin)
    if ylim is None:
        all_y = trajectory[:, :, :, 1]
        margin = 0.1 * (all_y.max() - all_y.min())
        ylim = (all_y.min() - margin, all_y.max() + margin)

    sample_color = sns.color_palette("mako", as_cmap=False, n_colors=3)[1]
    ref_color = "#8b8989"

    grid_size = int(np.ceil(np.sqrt(n_clouds)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
    if grid_size == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(grid_size, grid_size)

    if addon is not None:
        addon += " | "
    else:
        addon = ""

    def update(frame_idx):
        step = frame_indices[frame_idx]
        clouds = trajectory[step]
        t_val = timesteps[min(step, len(timesteps) - 1)]

        fig.suptitle(f"{addon}Step {step}/{total_steps - 1}  |  t = {t_val:.4f}", fontsize=14)

        for i in range(grid_size * grid_size):
            r, c = divmod(i, grid_size)
            ax = axes[r, c]
            ax.clear()

            if i < n_clouds:
                points = clouds[i]
                if ref_data is not None and i < len(ref_data):
                    ax.scatter(ref_data[i, :, 0], ref_data[i, :, 1], 
                               s=1, alpha=0.15, c=ref_color, rasterized=True)

                ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.5, c=[sample_color], edgecolors="none")

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 // fps,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved Nx2D sampling GIF to {output_path}")

    # Generate static plots for each cloud
    static_steps = [0, int((total_steps - 1) * 0.33), int((total_steps - 1) * 0.66), total_steps - 1]
    for i in range(n_clouds):
        fig_static, axes_static = plt.subplots(1, 4, figsize=(16, 4))
        if addon:
            fig_static.suptitle(addon.replace(" | ", ""), fontsize=16)
        for idx, step in enumerate(static_steps):
            ax_static = axes_static[idx]
            points = trajectory[step, i]
            t_val = timesteps[min(step, len(timesteps) - 1)]

            if ref_data is not None and i < len(ref_data):
                ax_static.scatter(ref_data[i, :, 0], ref_data[i, :, 1], s=1, alpha=0.15, c=ref_color, rasterized=True)
                
            ax_static.scatter(points[:, 0], points[:, 1], s=4, alpha=0.5, c=[sample_color], edgecolors="none")
            
            ax_static.set_xlim(xlim)
            ax_static.set_ylim(ylim)
            ax_static.set_aspect("equal")
            ax_static.set_title(f"Step {step}/{total_steps - 1}  |  t = {t_val:.4f}", fontsize=12)
            ax_static.set_xticks([])
            ax_static.set_yticks([])

        fig_static.tight_layout()
        static_path = str(output_path).replace(".gif", f"_cloud_{i}.png")
        fig_static.savefig(static_path, dpi=150)
        plt.close(fig_static)
    print(f"Saved {n_clouds} static sampling plots for Nx2D.")


def _plot_mean_std(ax, epochs, mean, std, color, label):
    ax.plot(epochs, mean, color=color, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)


def _group_loss_curves(loss_curves):
    grouped = defaultdict(list)
    for entry in loss_curves:
        if "p" in entry:
            key = (entry["method"], entry["p"], entry["delta"])
        else:
            key = (entry["method"], entry["m"], entry["m_prime"])
        grouped[key].append(entry)
    return grouped


def _compute_mean_std(entries, key):
    arrays = [np.array(e[key]) for e in entries]
    min_len = min(len(a) for a in arrays)
    stacked = np.stack([a[:min_len] for a in arrays], axis=0)
    return np.arange(min_len), stacked.mean(axis=0), stacked.std(axis=0)


def viz_loss_curves(loss_curves, output_folder):
    viz_dir = Path(output_folder)
    os.makedirs(str(viz_dir), exist_ok=True)

    if not loss_curves:
        return

    grouped = _group_loss_curves(loss_curves)
    method_colors = {"ambient": sns.color_palette()[0], "naive": sns.color_palette()[3]}

    is_inpainting = "p" in loss_curves[0]

    # --- (a) Ambient vs Naive per (p, delta) or (m, m_prime) setting ---
    settings = set()
    for key in grouped:
        settings.add((key[1], key[2]))

    for val1, val2 in sorted(settings):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for split_idx, split in enumerate(["train", "val"]):
            ax = axes[split_idx]
            for method in ["ambient", "naive"]:
                key = (method, val1, val2)
                if key not in grouped:
                    continue
                epochs, mean, std = _compute_mean_std(grouped[key], split)
                _plot_mean_std(ax, epochs, mean, std, method_colors[method], method)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            if is_inpainting:
                ax.set_title(f"{split.capitalize()} loss  ($p={val1:.2f}$, $\\delta={val2:.2f}$)")
            else:
                ax.set_title(f"{split.capitalize()} loss  ($m={val1}$, $m'={val2}$)")
            ax.legend()
        fig.tight_layout()
        if is_inpainting:
            fname = viz_dir / f"loss_ambient_vs_naive_p{val1:.4f}_delta{val2:.4f}.png"
        else:
            fname = viz_dir / f"loss_ambient_vs_naive_m{val1}_m_prime{val2}.png"
        fig.savefig(str(fname), dpi=150)
        plt.close(fig)

    # --- (b) All ambient / (c) All naive ---
    cmap = sns.color_palette("viridis", n_colors=max(len(settings), 1))

    for method in ["ambient", "naive"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        color_idx = 0
        for val1, val2 in sorted(settings):
            key = (method, val1, val2)
            if key not in grouped:
                continue
            color = cmap[color_idx % len(cmap)]
            if is_inpainting:
                label = f"$p={val1:.2f}$, $\\delta={val2:.2f}$"
            else:
                label = f"$m={val1}$, $m'={val2}$"
            for split_idx, split in enumerate(["train", "val"]):
                epochs, mean, std = _compute_mean_std(grouped[key], split)
                _plot_mean_std(axes[split_idx], epochs, mean, std, color, label)
            color_idx += 1
        for split_idx, split in enumerate(["train", "val"]):
            axes[split_idx].set_xlabel("Epoch")
            axes[split_idx].set_ylabel("Loss")
            axes[split_idx].set_title(f"{method.capitalize()} \u2014 {split.capitalize()} loss")
            axes[split_idx].legend(fontsize=8)
        fig.tight_layout()
        fname = viz_dir / f"loss_all_{method}.png"
        fig.savefig(str(fname), dpi=150)
        plt.close(fig)
