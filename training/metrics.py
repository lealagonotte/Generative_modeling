import numpy as np
import torch
import ot
from typing import Optional


def _normalize_weights(weights: Optional[torch.Tensor], n: int, dtype=None, device=None) -> np.ndarray:
    """Helper partagé : retourne un tableau numpy de poids normalisés."""
    if weights is not None:
        w = weights.detach().cpu().numpy().astype(np.float64)
    else:
        w = np.ones(n, dtype=np.float64) / n
    w = w / w.sum()
    return w


def _validate_p(p: int) -> None:
    """Valide que p est un entier strictement positif."""
    if not isinstance(p, int) or p < 1:
        raise ValueError(f"p doit être un entier >= 1, reçu : {p!r}")


################################ Distance de Wasserstein ################################

def wasserstein_distance(
    P: torch.Tensor,
    Q: torch.Tensor,
    p: int = 2,
    weights_P: Optional[torch.Tensor] = None,
    weights_Q: Optional[torch.Tensor] = None,
) -> float:
    """
    Distance de Wasserstein W_p(P, Q).

    Args:
        P          : (N, d) échantillons de la distribution source
        Q          : (M, d) échantillons de la distribution cible
        p          : exposant entier >= 1 (défaut 2)
        weights_P  : (N,) poids de P, uniformes par défaut
        weights_Q  : (M,) poids de Q, uniformes par défaut

    Returns:
        W_p (float) : distance de Wasserstein
    """
    _validate_p(p)

    if P.dim() == 1:
        P = P.unsqueeze(1)
    if Q.dim() == 1:
        Q = Q.unsqueeze(1)
    
    if len(P.size()) == 3:
        N, n, d = P.size()
        P = P.view(N, n*d)
    if len(Q.size()) == 3:
        N, n, d = Q.size()
        Q = Q.view(N, n*d)
    
    if P.shape[1] != Q.shape[1]:
        raise ValueError(
            f"P et Q doivent avoir la même dimension d. "
            f"Reçu: {P.shape[1]} vs {Q.shape[1]}"
        )

    N = P.shape[0]
    M_samples = Q.shape[0]  

    a = _normalize_weights(weights_P, N)
    b = _normalize_weights(weights_Q, M_samples)

    # Matrice de coût : ‖x_i − y_j‖^p
    # On calcule toujours les distances euclidiennes et on élève à la puissance p
    M_cost = ot.dist(
        P.detach().cpu().numpy(),
        Q.detach().cpu().numpy(),
        metric="euclidean",
    ) ** p  # (N, M)

    # Transport optimal exact → W_p^p
    wp_p = ot.emd2(a, b, M_cost)

    return float(wp_p) ** (1.0 / p)


################################ Sliced Wasserstein Distance (SWD) ################################

def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 100,
    p: int = 2,
    seed: Optional[int] = 0,
    weights_x: Optional[torch.Tensor] = None,
    weights_y: Optional[torch.Tensor] = None,
) -> float:
    """
    Sliced Wasserstein distance entre deux tenseurs PyTorch avec POT.

    Nécessite POT >= 0.8.0.

    Paramètres
    ----------
    x : torch.Tensor
        Forme (n, d) ou (n,)
    y : torch.Tensor
        Forme (m, d) ou (m,)
    n_projections : int
        Nombre de projections aléatoires
    p : int
        Ordre de la distance (>= 1, souvent 2)
    seed : int | None
        Graine aléatoire
    weights_x : torch.Tensor | None
        Poids source, forme (n,)
    weights_y : torch.Tensor | None
        Poids cible, forme (m,)

    Retour
    ------
    float : distance de Sliced Wasserstein
    """
    _validate_p(p)

    x = x.float().detach().cpu()
    y = y.float().detach().cpu()

    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    if len(x.size()) == 3:
        N, n, d = x.size()
        x = x.view(N, n*d)
    if len(y.size()) == 3:
        N, n, d = y.size()
        y = y.view(N, n*d)
        
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x et y doivent avoir la même dimension d'espace, "
            f"reçu {x.shape[1]} et {y.shape[1]}"
        )

    n = x.shape[0]
    m = y.shape[0]

    # On utilise le helper commun (numpy), puis on repasse en tenseur float32
    a_np = _normalize_weights(weights_x, n)
    b_np = _normalize_weights(weights_y, m)

    a = torch.tensor(a_np, dtype=x.dtype, device=x.device)
    b = torch.tensor(b_np, dtype=y.dtype, device=y.device)

    sw = ot.sliced.sliced_wasserstein_distance(
        X_s=x,
        X_t=y,
        a=a,
        b=b,
        n_projections=n_projections,
        p=p,
        seed=seed,
    )
    return float(sw)


################################ Chamfer Distance ################################

def chamfer_distance(
    P1: torch.Tensor,
    P2: torch.Tensor,
) -> float:
    """
    Computes the Chamfer distance between two batches of point clouds.

    Args:
        P1 : (batch, N, d) point clouds
        P2 : (batch, M, d) point clouds

    Returns:
        float : Average Chamfer distance across the batch.
    """
    if P1.dim() != 3:
        P1 = P1.unsqueeze(0)
    if P2.dim() != 3:
        P2 = P2.unsqueeze(0)

    if P1.shape[2] != P2.shape[2]:
        raise ValueError(f"P1 and P2 must have the same dimension d. Got {P1.shape[2]} and {P2.shape[2]}.")

    if P1.shape[0] < P2.shape[0]:
        mult = (P2.shape[0] // P1.shape[0]) + 1
        P1 = P1.repeat(mult, 1, 1)[:P2.shape[0]]
    elif P2.shape[0] < P1.shape[0]:
        P1 = P1[:P2.shape[0]]
    
    device = P2.device
    P1 = P1.detach().to(device)

    # (batch, N, M) pairwise squared distances
    diff = P1.unsqueeze(2) - P2.unsqueeze(1)
    dist = torch.sum(diff ** 2, dim=-1)

    # For each point in P1, min distance to P2
    min_dist_1_to_2 = torch.min(dist, dim=2)[0]  # (batch, N)
    
    # For each point in P2, min distance to P1
    min_dist_2_to_1 = torch.min(dist, dim=1)[0]  # (batch, M)

    # Average over points, then over batch
    chamfer_loss = torch.mean(min_dist_1_to_2, dim=1) + torch.mean(min_dist_2_to_1, dim=1)
    return float(torch.mean(chamfer_loss).item())