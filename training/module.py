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

    For inpainting:
        A   : (batch, data_dim) corruption mask
        x_t : (batch, data_dim) noisy corrupted observations
        => input_dim = 2 * data_dim + time_embed_dim

    For compressed sensing:
        A   : (batch, m, data_dim) measurement matrix
        x_t : (batch, m)           noisy measurements
        => input_dim = m * data_dim + m + time_embed_dim
        Set measurement_dim=m at construction time.

    Output:
        predicted x0 : (batch, data_dim)
    """

    def __init__(self, data_dim=2, hidden_dim=128, n_layers=4, time_embed_dim=32,
                 measurement_dim=None):
        super().__init__()
        self.data_dim = data_dim
        self.measurement_dim = measurement_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        if measurement_dim is not None:
            # Compressed sensing: flatten A (m*d) + x_t (m) + t_emb
            input_dim = measurement_dim * data_dim + measurement_dim + time_embed_dim
        else:
            # Inpainting: A (d) + x_t (d) + t_emb
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

        if A.dim() == 3:
            # Compressed sensing: A is (batch, m, d), x_t is (batch, m)
            # Flatten A to (batch, m*d), keep x_t as (batch, m)
            A_flat = A.view(A.size(0), -1)
            x = torch.cat([A_flat, x_t, t_emb], dim=-1)
        else:
            # Inpainting: A is (batch, d), x_t is (batch, d)
            x = torch.cat([A, x_t, t_emb], dim=-1)

        return self.net(x)


class FlatDenoiserNx2D(nn.Module):
    """
    Flat MLP denoiser for the Nx2D point cloud setting.
    Flattens the point cloud to a 1D vector (N * 2).
    WARNING: Does not preserve permutation invariance.

    For inpainting:
        A   : (batch, N, data_dim) corruption mask
        x_t : (batch, N, data_dim) noisy corrupted observations
        => input_dim = 2 * N * data_dim + time_embed_dim

    For compressed sensing:
        A   : (batch, m, N*data_dim) measurement matrix
        x_t : (batch, m)             noisy measurements
        => input_dim = m * N * data_dim + m + time_embed_dim
        Set measurement_dim=m at construction time.

    Output:
        predicted x0 : (batch, N, data_dim)
    """
    def __init__(self, n_points, data_dim=2, hidden_dim=256, n_layers=4, time_embed_dim=64,
                 measurement_dim=None):
        super().__init__()
        self.n_points = n_points
        self.data_dim = data_dim
        self.flat_dim = n_points * data_dim
        self.measurement_dim = measurement_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        if measurement_dim is not None:
            # Compressed sensing: flatten A (m * N*d) + x_t (m) + t_emb
            input_dim = measurement_dim * self.flat_dim + measurement_dim + time_embed_dim
        else:
            # Inpainting: 2 * N*d + t_emb
            input_dim = 2 * self.flat_dim + time_embed_dim

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.flat_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, A, x_t, t):
        batch = x_t.size(0)
        t_emb = self.time_embed(t)

        if A.dim() == 3 and x_t.dim() == 2 and self.measurement_dim is not None:
            # Compressed sensing: A is (batch, m, N*d), x_t is (batch, m)
            A_flat = A.view(batch, -1)
            x = torch.cat([A_flat, x_t, t_emb], dim=-1)
        else:
            # Inpainting: flatten (batch, N, 2) to (batch, N*2)
            A_flat = A.view(batch, -1)
            x_t_flat = x_t.view(batch, -1)
            x = torch.cat([A_flat, x_t_flat, t_emb], dim=-1)

        out_flat = self.net(x)
        return out_flat.view(batch, self.n_points, self.data_dim)


class PointNetDenoiserNx2D(nn.Module):
    """
    Permutation-equivariant PointNet denoiser for the Nx2D setting.

    For compressed sensing, this architecture is NOT directly applicable
    because measurements are global (not per-point). Use FlatDenoiserNx2D
    for compressed sensing Nx2D experiments instead.

    For inpainting:
        A   : (batch, N, data_dim) corruption mask
        x_t : (batch, N, data_dim) noisy corrupted observations
        t   : (batch,)             diffusion time

    Output:
        predicted x0 : (batch, N, data_dim)
    """
    def __init__(self, data_dim=2, time_embed_dim=32, **kwargs):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        in_channels = 2 * data_dim + time_embed_dim

        # --- PointNet Encoder ---
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # --- PointNet Decoder ---
        self.conv4 = nn.Conv1d(1024 + 64, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, data_dim, 1)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def forward(self, A, x_t, t):
        batch_size = x_t.size(0)

        if A.dim() == 3 and x_t.dim() == 2:
            raise ValueError(
                "PointNetDenoiserNx2D does not support compressed sensing inputs. "
                "Use FlatDenoiserNx2D for compressed sensing Nx2D experiments."
            )

        num_points = x_t.shape[1]
        point_inputs = torch.cat([A, x_t], dim=-1)

        t_emb = self.time_embed(t)  # (batch, time_embed_dim)
        t_emb_expanded = t_emb.unsqueeze(1).expand(batch_size, num_points, -1)

        point_inputs = torch.cat([point_inputs, t_emb_expanded], dim=-1)

        x = point_inputs.transpose(1, 2)

        # PointNet Encoder
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))

        # Global Max Pooling
        global_feat = torch.max(x3, 2, keepdim=True)[0]
        global_feat_expanded = global_feat.expand(-1, -1, num_points)

        combined_features = torch.cat([x1, global_feat_expanded], dim=1)

        # PointNet Decoder
        x4 = self.relu(self.bn4(self.conv4(combined_features)))
        x5 = self.relu(self.bn5(self.conv5(x4)))
        x6 = self.relu(self.bn6(self.conv6(x5)))
        out = self.conv7(x6)

        return out.transpose(1, 2)