import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion time steps."""

    def __init__(self, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        # t: (batch,) in [0, 1]
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * 1000.0 * freqs[None, :]  # scale t to [0, 1000]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, embed_dim)


class Denoiser(nn.Module):
    """
    Simple MLP denoiser for ambient diffusion.

    Inputs:
        A   : (batch, data_dim) corruption mask / operator
        x_t : (batch, data_dim) noisy corrupted observations
        t   : (batch,)          diffusion time in [0, 1]

    Output:
        predicted x0 : (batch, data_dim)
    """

    def __init__(self, data_dim=2, hidden_dim=128, n_layers=4, time_embed_dim=32):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        input_dim = 2 * data_dim + time_embed_dim

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, A, x_t, t):
        t_emb = self.time_embed(t)
        x = torch.cat([A, x_t, t_emb], dim=-1)
        return self.net(x)
