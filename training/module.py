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
        if A.dim() == 3 and x_t.dim() == 2:
            # Compressed sensing: project from measurement space to data space
            # Use pseudoinverse-style normalization: A^+ y = A^T y / ||A||^2_F
            x_t_proj = torch.bmm(A.transpose(1, 2), x_t.unsqueeze(-1)).squeeze(-1)  # (batch, d)
            A_norm_sq = (A ** 2).sum(dim=(1, 2), keepdim=False).unsqueeze(-1).clamp(min=1e-8)
            x_t_proj = x_t_proj / A_norm_sq
            A_proj = (A ** 2).sum(dim=1)  # (batch, d)
            A = A_proj
            x_t = x_t_proj
        t_emb = self.time_embed(t)
        x = torch.cat([A, x_t, t_emb], dim=-1)
        return self.net(x)


class FlatDenoiserNx2D(nn.Module):
    """
    Flat MLP denoiser for the Nx2D point cloud setting.
    Flattens the point cloud to a 1D vector (N * 2).
    WARNING: Does not preserve permutation invariance.

    Inputs:
        A   : (batch, N, data_dim) corruption mask
        x_t : (batch, N, data_dim) noisy corrupted observations
        t   : (batch,)             diffusion time

    Output:
        predicted x0 : (batch, N, data_dim)
    """
    def __init__(self, n_points, data_dim=2, hidden_dim=256, n_layers=4, time_embed_dim=64):
        super().__init__()
        self.n_points = n_points
        self.data_dim = data_dim
        self.flat_dim = n_points * data_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

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
        
        if A.dim() == 3 and x_t.dim() == 2:
            # Handle Compressed Sensing
            # Use pseudoinverse-style normalization: A^+ y = A^T y / ||A||^2_F
            x_t_proj = torch.bmm(A.transpose(1, 2), x_t.unsqueeze(-1)).squeeze(-1) # (batch, N*2)
            A_norm_sq = (A ** 2).sum(dim=(1, 2), keepdim=False).unsqueeze(-1).clamp(min=1e-8)
            x_t_proj = x_t_proj / A_norm_sq
            A_proj = (A ** 2).sum(dim=1) # (batch, N*2)

            A_flat = A_proj
            x_t_flat = x_t_proj
        else:
            # Flatten (batch, N, 2) to (batch, N*2)
            A_flat = A.view(batch, -1)
            x_t_flat = x_t.view(batch, -1)

        t_emb = self.time_embed(t)

        x = torch.cat([A_flat, x_t_flat, t_emb], dim=-1)
        out_flat = self.net(x)

        return out_flat.view(batch, self.n_points, self.data_dim)


class PointNetDenoiserNx2D(nn.Module):
    """
    Permutation-equivariant PointNet denoiser for the Nx2D setting.
    
    PointNet Explanation:
    Point clouds are unstructured, unordered sets of points. Standard network 
    architectures (like CNNs or RNNs) expect ordered grids or sequences and 
    cannot natively handle permutation invariance. PointNet solves this by:
    1. Local Processing: Processing each point identically and independently 
       through shared Multi-Layer Perceptrons (implemented as 1D Convolutions).
    2. Global Aggregation: Applying a symmetric mathematical function (Max Pooling)
       across all processed points to capture a single, global geometric feature 
       signature representing the entire point cloud.
    3. Per-Point Decoding: For dense tasks (like segmentation or denoising), the 
       global feature is concatenated back with local point features. This gives 
       each point both global context and local geometry to make its prediction.
       
    Implementation adapted from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    Specifically inspired by `models/pointnet_utils.py` and `models/pointnet_sem_seg.py`.

    Inputs:
        A   : (batch, N, data_dim) corruption mask
        x_t : (batch, N, data_dim) noisy corrupted observations
        t   : (batch,)             diffusion time

    Output:
        predicted x0 : (batch, N, data_dim)
    """
    def __init__(self, data_dim=2, time_embed_dim=32):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        in_channels = 2 * data_dim + time_embed_dim
        
        # --- PointNet Encoder ---
        # Using Conv1d with kernel_size=1 is equivalent to shared MLPs per point
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # --- PointNet Decoder (Segmentation Head style because we want to reconstruct the original point cloud) ---
        # Concatenates global feature (1024) + local feature from conv1 (64)
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
            # Handle Compressed Sensing (Measurements to Data Space)
            # Use pseudoinverse-style normalization: A^+ y = A^T y / ||A||^2_F
            x_t_proj = torch.bmm(A.transpose(1, 2), x_t.unsqueeze(-1)).squeeze(-1) # (batch, N*2)
            A_norm_sq = (A ** 2).sum(dim=(1, 2), keepdim=False).unsqueeze(-1).clamp(min=1e-8)
            x_t_proj = x_t_proj / A_norm_sq
            A_proj = (A ** 2).sum(dim=1) # (batch, N*2)

            x_t_reshaped = x_t_proj.view(batch_size, -1, self.data_dim)
            A_reshaped = A_proj.view(batch_size, -1, self.data_dim)
            num_points = x_t_reshaped.shape[1]
            
            point_inputs = torch.cat([A_reshaped, x_t_reshaped], dim=-1)
        else:
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
