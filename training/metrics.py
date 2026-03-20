import torch
import torch.nn.functional as F
from typing import Optional
try:
    import ot
except ImportError:
    raise ImportError(
        "La librairie POT est requise : pip install POT"
    )

################################ Distance de Wasserstein ################################
def wasserstein_distance(
    P: torch.Tensor,
    Q: torch.Tensor,
    p: int = 2,
    weights_P: torch.Tensor = None,
    weights_Q: torch.Tensor = None,
) -> float:
    """
    Distance de Wasserstein W_p(P, Q) .
 
    Args:
        P          : (N, d) échantillons de la distribution source
        Q          : (M, d) échantillons de la distribution cible
        p          : exposant (1 ou 2, défaut 2)
        weights_P  : (N,) poids de P, uniformes par défaut
        weights_Q  : (M,) poids de Q, uniformes par défaut
 
    Returns:
        W_p (float) : distance de Wasserstein
 
    """
    if P.dim() == 1:
        P = P.unsqueeze(1)
    if Q.dim() == 1:
        Q = Q.unsqueeze(1)
 
    if P.shape[1] != Q.shape[1]:
        raise ValueError(
            f"P et Q doivent avoir la même dimension d. "
            f"Reçu: {P.shape[1]} vs {Q.shape[1]}"
        )
 
    N, M = P.shape[0], Q.shape[0]
 
    # Poids uniformes par défaut
    a = weights_P.cpu().numpy() if weights_P is not None else np.ones(N) / N
    b = weights_Q.cpu().numpy() if weights_Q is not None else np.ones(M) / M
 
    # Normalisation des poids
    a = a / a.sum()
    b = b / b.sum()
 
    # Matrice de coût : ‖x_i − y_j‖^p
    metric = "sqeuclidean" if p == 2 else "euclidean"
    M_cost = ot.dist(
        P.detach().cpu().numpy(),
        Q.detach().cpu().numpy(),
        metric=metric,
    )  # (N, M)
 
    # Transport optimal exact 
    wp_p = ot.emd2(a, b, M_cost)   # W_p^p
 
    return float(wp_p) ** (1.0 / p)


################################ Sliced Wasserstein Distance (SWD) ################################

def sliced_wasserstein_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 100,
    p: int = 2,
    seed: int | None = 0,
    weights_x: torch.Tensor | None = None,
    weights_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sliced Wasserstein distance entre deux tenseurs PyTorch avec POT.

    Paramètres
    ----------
    x : torch.Tensor
        Forme (n, d) ou (n,)
    y : torch.Tensor
        Forme (m, d) ou (m,)
    n_projections : int
        Nombre de projections aléatoires
    p : int
        Ordre de la distance (souvent 2)
    seed : int | None
        Graine aléatoire
    weights_x : torch.Tensor | None
        Poids source, forme (n,)
    weights_y : torch.Tensor | None
        Poids cible, forme (m,)

    Retour
    ------
    torch.Tensor scalaire
    """
    x = x.float()
    y = y.float()

    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x et y doivent avoir la même dimension d'espace, reçu {x.shape[1]} et {y.shape[1]}"
        )

    n = x.shape[0]
    m = y.shape[0]

    if weights_x is None:
        a = torch.full((n,), 1.0 / n, dtype=x.dtype, device=x.device)
    else:
        a = weights_x.to(device=x.device, dtype=x.dtype)
        a = a / a.sum()

    if weights_y is None:
        b = torch.full((m,), 1.0 / m, dtype=y.dtype, device=y.device)
    else:
        b = weights_y.to(device=y.device, dtype=y.dtype)
        b = b / b.sum()

    sw = ot.sliced.sliced_wasserstein_distance(
        X_s=x,
        X_t=y,
        a=a,
        b=b,
        n_projections=n_projections,
        p=p,
        seed=seed,
    )

    return sw