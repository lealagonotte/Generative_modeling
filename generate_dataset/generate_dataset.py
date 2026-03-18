import pandas as pd
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
import pickle
 
#################################### INPAINTING CORRUPTION ##############################################
def inpainting_corruption(X: np.ndarray, p: float = 0.2, prevent_zero: bool = True,
                           rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Corruption de type inpainting.
 
    Pour chaque point et chaque coordonnée, masque indépendamment avec proba p.
    Si prevent_zero=True, on s'assure qu'au moins une coordonnée reste observable
    (interdit les points (0, 0)).
 
    Args:
        X            : données propres, shape (N, 2)
        p            : probabilité de masquer chaque coordonnée (défaut 0.2)
        prevent_zero : si True, interdit le masque (0, 0)
        rng          : générateur numpy pour la reproductibilité
 
    Returns:
        Y      : données corrompues, shape (N, 2)  — coords masquées = 0
        A      : masque binaire,     shape (N, 2)  — 1 = observable, 0 = masqué
    """
    if rng is None:
        rng = np.random.default_rng(42)
 
    N, D = X.shape
 
    # Masque i.i.d. : chaque dim masquée avec proba p
    A = (rng.uniform(size=(N, D)) > p).astype(np.float32)   # 1 = visible
 
    if prevent_zero:
        # Points où tout est masqué : forcer une dim aléatoire à 1
        all_zero = A.sum(axis=1) == 0
        n_zero = all_zero.sum()
        if n_zero > 0:
            forced_dim = rng.integers(0, D, size=n_zero)
            A[all_zero, forced_dim] = 1.0
 
    Y = X * A
    return Y, A
 
def compressed_sensing_corruption(X, m=2, rng=None):
    """
    Corruption Compressed Sensing (Ambient Diffusion, Corollary A.2).
    A ∈ R^(m x d) : m lignes iid N(0, I_d) → y = A·x ∈ R^m
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N, d = X.shape
    
    A_matrices = rng.standard_normal(size=(N, m, d))          # (N, m, d)
    Y = np.einsum('nmd,nd->nm', A_matrices, X)                # (N, m)
    
    return Y, A_matrices
 
def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)
 
 
# RNG dédié à la corruption (indépendant de la génération)
rng_corrupt = np.random.default_rng(123)
 
 
#################################### Two moons ##############################################
X, _ = make_moons(n_samples=100000, noise=0.1, random_state=42)
X = normalize(X)

p=0.2
 
Y, A = inpainting_corruption(X, p, prevent_zero=False, rng=rng_corrupt)
data = {"X": X, "A": A}

with open(f"two_moons_{p}.pkl", "wb") as f:
    pickle.dump(data, f)

 
 
####################################### Swiss roll 2D ################################################
x_roll, _ = make_swiss_roll(n_samples=100000, noise=0.5, random_state=42)
x_roll = normalize(x_roll[:, [0, 2]])

p=0.2
Y_roll, A_roll = inpainting_corruption(x_roll,  prevent_zero=False, rng=rng_corrupt)
data = {"X": x_roll, "A": A_roll}

with open(f"swiss_roll_p{p}.pkl", "wb") as f:
    pickle.dump(data, f)
 
 ################################# Corruption pour les datasets de spirale ###########################################
def inpainting_corruption_pointwise(X, p=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    N = X.shape[0]
    mask = (rng.uniform(0, 1, size=N) >= p).astype(int)  # 1=intact, 0=corrompu
    
    Y = X * mask[:, None]  # points corrompus mis à zéro
    A = np.stack([mask, mask], axis=1)  # a1=a2=mask
    
    return Y, A

################################### DATASET (N,2) ###############################################

print("Génération two moons...")

N_DATASETS = 1000
N_SAMPLES  = 1000
NOISE_MIN  = 0.01
NOISE_MAX  = 0.9

p = 0.2

rng_noise = np.random.default_rng(42)
noise_levels = rng_noise.uniform(NOISE_MIN, NOISE_MAX, size=N_DATASETS)

chunks = []
for i, noise in enumerate(noise_levels):
    X, _ = make_moons(n_samples=N_SAMPLES, noise=noise, random_state=i)
    X = normalize(X)
    chunks.append(X)

    if (i + 1) % 10000 == 0:
        print(f"  {i+1}/{N_DATASETS} datasets générés...")

# Concatenation
X_all = np.concatenate(chunks, axis=0)  # (N_DATASETS * N_SAMPLES, 2)

# Corruption sur le dataset complet
rng_corrupt = np.random.default_rng(456)
Y_all, A_all = inpainting_corruption_pointwise(X_all, p=p, rng=rng_corrupt)

data = {"X": X_all, "A": A_all}

with open(f"two_moons_all_p{p}.pkl", "wb") as f:
    pickle.dump(data, f)

#################################### DATASET swiss roll (N,2) ###############################################
print("Génération swiss roll...")

p = 0.2

chunks = []
for i, noise in enumerate(noise_levels):
    X, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=noise * 10, random_state=i)
    X = normalize(X[:, [0, 2]])
    chunks.append(X)

    if (i + 1) % 10000 == 0:
        print(f"  {i+1}/{N_DATASETS}...")

X_all = np.concatenate(chunks, axis=0)  # (N_DATASETS * N_SAMPLES, 2)

rng_corrupt = np.random.default_rng(456)
Y_all, A_all = inpainting_corruption_pointwise(X_all, p=p, rng=rng_corrupt)

data = {"X": X_all, "A": A_all}

with open(f"swiss_roll_all_p{p}.pkl", "wb") as f:
    pickle.dump(data, f)
