"""Module to define visualizations and animations for 2D diffusion sampling."""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

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
        shape, n_steps, A_sample, module, noise_scheduler
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
        ax.set_title(f"Step {step}/{total_steps - 1}  |  t = {t_val:.4f}", fontsize=12)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), interval=1000 // fps,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved sampling GIF to {output_path}")
