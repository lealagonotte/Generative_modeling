"""Module to define components to train a diffusion model following the AmbientDiffusion framework"""
import pickle as pkl
import numpy as np
import os, sys
from pathlib import Path

import torch
import torch.nn as nn

BASE_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.append(BASE_DIR)

from generate_dataset.utils import (inpainting_corruption, inpainting_corruption_pointwise, 
                                    compressed_sensing_corruption, inpainting_corruption_Nx2D, 
                                    inpainting_corruption_pointwise_Nx2D)



class NoiseScheduler(object):
    """
    Defines for t ~ U[0,1] the schedule corresponding to the application
    t -> sigma_t
    """
    def __init__(self, mode, **kwargs):
        # mode is the type of noise schedule to use
        # different noise schedules might need differnet params
        # => if mode == "smthng" we call the

        mode = mode.lower()
        if mode == "interpolation":
            self._noise_func, self._apply_noise_func = self._init_interpolation(**kwargs)
        elif mode == "vp":
            self._noise_func, self._apply_noise_func = self._init_vp(**kwargs)
        elif mode == "ve":
            self._noise_func, self._apply_noise_func = self._init_ve(**kwargs)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Must be either 'interpolation', 'vp' or 've'.")

    def _init_interpolation(self, sigma_max=1.0, **kwargs):
        """
        simple interpolation schedule (uniform t => uniform sigma between 0 and a max_sigma value)
        returns the noise function and the function to apply noise on batched inputs
        """
        def noise_func(t):
            return t * sigma_max

        def apply_noise(x0, t, eps):
            sigma_t = noise_func(t)
            # reshape for broadcasting: (batch, 1) or (batch, 1, 1)
            shape = [sigma_t.shape[0]] + [1] * (x0.dim() - 1)
            sigma_t = sigma_t.view(shape)
            return x0 + sigma_t * eps

        return noise_func, apply_noise

    def _init_vp(self, beta_min=0.1, beta_max=20.0, **kwargs):
        """
        variance preserving schedule
        x_t = alpha_t * x0 + sigma_t * eps
        where log(alpha_t) = -1/2 * integral_0^t beta(s) ds
        and sigma_t = sqrt(1 - alpha_t^2)
        """
        def noise_func(t):
            log_mean_coeff = -0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2)
            alpha_t = torch.exp(log_mean_coeff)
            sigma_t = torch.sqrt(1.0 - alpha_t ** 2)
            return sigma_t

        def apply_noise(x0, t, eps):
            log_mean_coeff = -0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2)
            alpha_t = torch.exp(log_mean_coeff)
            sigma_t = torch.sqrt(1.0 - alpha_t ** 2)
            
            shape = [alpha_t.shape[0]] + [1] * (x0.dim() - 1)
            alpha_t = alpha_t.view(shape)
            sigma_t = sigma_t.view(shape)
            
            return alpha_t * x0 + sigma_t * eps

        return noise_func, apply_noise

    def _init_ve(self, sigma_min=0.01, sigma_max=1.0, **kwargs):
        """
        variance exploding schedule
        sigma(t) = sigma_min * (sigma_max / sigma_min)^t
        x_t = x0 + sigma_t * eps
        """
        def noise_func(t):
            return sigma_min * (sigma_max / sigma_min) ** t

        def apply_noise(x0, t, eps):
            sigma_t = noise_func(t)
            shape = [sigma_t.shape[0]] + [1] * (x0.dim() - 1)
            sigma_t = sigma_t.view(shape)
            return x0 + sigma_t * eps

        return noise_func, apply_noise

    def __call__(self, t):
        return self._noise_func(t)

    def apply_noise(self, x0, t, eps):
        return self._apply_noise_func(x0, t, eps)

class FurtherCorrupter(object):
    """
    Defines for a corruption matrix A the corruption A' = BA
    """
    def __init__(self, mode, **kwargs):
        # mode is the type of corruption, either inpainting or compressed_sensing measurements (see ..generate_dataset/generated_dataset.py)
        # different noise schedules might need differnet params
        # => if mode == "smthng" we call the

        mode = mode.lower()
        if mode == "inpainting":
            self.init_operator_func, self.operator_func, self.apply_operator_func = self._init_inpainting(**kwargs)
        elif mode == "inpainting_pw":
            self.init_operator_func, self.operator_func, self.apply_operator_func = self._init_inpainting_pw(**kwargs)
        elif mode == "compressed_sensing":
            self.init_operator_func, self.operator_func, self.apply_operator_func = self._init_compressed_sensing(**kwargs)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Must be either 'inpainting' or 'compressed_sensing'.")

    def _init_inpainting(self, p=0.5, **kwargs):
        """
        inpainting further corruption: B is a random binary mask with probability p of zeroing each entry.
        A' = A * B (element-wise). apply: A' * x (element-wise).
        """
        if 'seed' in kwargs.keys():
            seed = kwargs['seed']
        else:
            seed = 1234
        rng = np.random.default_rng(seed + 1) 

        if 'prevent_zero' in kwargs.keys():
            prevent_zero = kwargs['prevent_zero']
        else:
            prevent_zero = True

        def init_operator_func(shape, device):
            X = np.zeros(shape)
            if len(shape) == 3:
                _, A = inpainting_corruption_Nx2D(X, 
                                         p=p, 
                                         prevent_zero=prevent_zero, 
                                         rng=rng)
            else:
                _, A = inpainting_corruption(X, 
                                         p=p, 
                                         prevent_zero=prevent_zero, 
                                         rng=rng)  
            return torch.from_numpy(A).to(device)

        def operator_func(A):
            # A: binary mask
            B = (torch.rand_like(A) > p).float()
            return A * B

        def apply_operator_func(further_A, x):
            # further_A: binary mask, x: data
            return further_A * x

        return init_operator_func, operator_func, apply_operator_func

    def _init_inpainting_pw(self, p=0.5, **kwargs):
        """
        inpainting further corruption: B is a random binary mask with probability p of zeroing each entry.
        A' = A * B (element-wise). apply: A' * x (element-wise).
        """
        if 'seed' in kwargs.keys():
            seed = kwargs['seed']
        else:
            seed = 1234
        rng = np.random.default_rng(seed + 1) 

        def init_operator_func(shape, device):
            X = np.zeros(shape)

            if len(shape) == 3:
                _, A = inpainting_corruption_pointwise_Nx2D(X, 
                                         p=p, 
                                         rng=rng)
            else:
                _, A = inpainting_corruption_pointwise(X, 
                                         p=p, 
                                         rng=rng)  
            return torch.from_numpy(A).to(device)

        def operator_func(A):
            # A: (batch, D) binary mask
            B = (torch.rand_like(A) > p).float()
            return A * B

        def apply_operator_func(further_A, x):
            # further_A: (batch, ...) binary mask, x: (batch, ...)
            return further_A * x

        return init_operator_func, operator_func, apply_operator_func

    def _init_compressed_sensing(self, m_prime=1, **kwargs):
        """
        compressed_sensing measurements further corruption: B ~ N(0,I) of shape (batch, m', m).
        A' = B @ A. apply: A' @ x.
        """
        if 'seed' in kwargs.keys():
            seed = kwargs['seed']
        else:
            seed = 1234
        rng = np.random.default_rng(seed + 1) 

        if 'm' in kwargs.keys():
            m = kwargs["m"]
        else:
            m = 2

        def init_operator_func(shape, device):
            X = np.zeros(shape)
            _, A = compressed_sensing_corruption(X, 
                                         m=m, 
                                         rng=rng)  
            return torch.from_numpy(A).to(device)
        
        def operator_func(A):
            # A: (batch, m, d)
            batch, m, d = A.shape
            B = torch.randn(batch, m_prime, m, device=A.device)
            return torch.bmm(B, A)  # (batch, m_prime, d)

        def apply_operator_func(further_A, x):
            # further_A: (batch, m', d), x: (batch, d)
            # Currently only supported for 2D inputs, not Nx2D
            if x.dim() > 2:
                raise NotImplementedError("Gaussian corruption not supported for Nx2D inputs yet.")
            return torch.bmm(further_A, x.unsqueeze(-1)).squeeze(-1)

        return init_operator_func, operator_func, apply_operator_func

    def init_operator(self, shape, device):
        return self.init_operator_func(shape, device)
    
    def get_operator(self, A):
        return self.operator_func(A)

    def apply_operator(self, further_A, x):
        return self.apply_operator_func(further_A, x)

class Sampler(object):
    """
    Sampling scheme to solve the SDE and sample from p0
    """
    def __init__(self, mode, **kwargs):
        mode = mode.lower()
        if mode == "fms": # fixed mask sampling
            self.sampling_step, self.define_steps = self._init_fixed_mask_sampling(**kwargs)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Must be 'fms'.")

    def _init_fixed_mask_sampling(self, **kwargs):

        def define_steps(n_steps):
            """
            Defines the list of time steps from T=1 down to near 0
            """
            return torch.linspace(1.0, 1e-4, n_steps)

        def sampling_step(x_t, A, t_curr, t_next, module, noise_scheduler, eps=1e-8):
            """
            Reverse SDE step following Eq. 3.3.

            1. Predict x0: pred_x0 = module(A, x_t, t_curr)
            2. Step: x_next = (sigma_next / sigma_t) * x_t + (1 - sigma_next / sigma_t) * pred_x0
            """
            sigma_t = noise_scheduler(t_curr)
            sigma_next = noise_scheduler(t_next)

            # Ensure shapes for broadcasting: (batch,) -> (batch, 1)
            if sigma_t.dim() == 0:
                sigma_t = sigma_t.unsqueeze(0)
            if sigma_next.dim() == 0:
                sigma_next = sigma_next.unsqueeze(0)

            with torch.no_grad():
                pred_x0 = module(A, x_t, t_curr)

            # Dynamic reshaping for broadcasting to x_t shape
            shape = [sigma_t.size(0)] + [1] * (x_t.dim() - 1)
            sigma_t = sigma_t.view(shape)
            sigma_next = sigma_next.view(shape)

            ratio_sigma = sigma_next / (sigma_t + eps) # stablize ratio for low sigma values

            x_next = ratio_sigma * x_t + (1 - ratio_sigma) * pred_x0

            return x_next

        return sampling_step, define_steps

    def step(self, *args):
        return self.sampling_step(*args)

    def sample(self, shape, n_steps, A, module, noise_scheduler):
        """
        Generate samples by running the reverse SDE.

        Args:
            shape: (n_samples, D) shape of the output
            n_steps: number of discretization steps
            A: corruption operator for sampling (e.g. identity mask = all ones)
            module: trained denoiser
            noise_scheduler: NoiseScheduler instance
        """
        device = next(module.parameters()).device
        timesteps = self.define_steps(n_steps).to(device)

        x_t = torch.randn(shape, device=device)

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i].expand(shape[0])
            t_next = timesteps[i + 1].expand(shape[0])
            x_t = self.sampling_step(x_t, A, t_curr, t_next, module, noise_scheduler)

        return x_t

    def sample_with_trajectory(self, shape, n_steps, A, module, noise_scheduler):
        """
        Same as sample(), but returns the full trajectory of x_t at each step.

        Returns:
            trajectory: tensor of shape (n_steps, n_samples, D) with x_t at each timestep
        """
        device = next(module.parameters()).device
        timesteps = self.define_steps(n_steps).to(device)

        x_t = torch.randn(shape, device=device)
        trajectory = [x_t.detach().cpu()]

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i].expand(shape[0])
            t_next = timesteps[i + 1].expand(shape[0])
            x_t = self.sampling_step(x_t, A, t_curr, t_next, module, noise_scheduler)
            trajectory.append(x_t.detach().cpu())

        return torch.stack(trajectory, dim=0), timesteps.cpu()

class AmbientLoss(nn.Module):
    """
    Defines the ambient diffusion objective
    Jcorr = 1/2 * E[||Ax0 - Ah(A', A'x_t, t)||^2]

    => Naive if A' = A
    => Eq.2 if A' = BA
    """
    def __init__(self, apply_opertor_func):
        super().__init__()
        self.apply_opertor_func = apply_opertor_func

    def forward(self, x0, predicted_x0, A):
        Ax0 = self.apply_opertor_func(A, x0)

        A_predx0 = self.apply_opertor_func(A, predicted_x0)

        diff = Ax0 - A_predx0
        
        # Flatten and sum all dimensions except batch
        diff = diff.view(diff.size(0), -1)
        per_sample_loss = torch.sum(diff ** 2, dim=-1)  # (batch,)
        return 0.5 * torch.mean(per_sample_loss)
