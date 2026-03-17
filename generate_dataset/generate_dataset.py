import pandas as pd
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
 
 
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
 
 
def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)
 
 
# RNG dédié à la corruption (indépendant de la génération)
rng_corrupt = np.random.default_rng(123)
 
 
#################################### Two moons ##############################################
X, _ = make_moons(n_samples=100000, noise=0.1, random_state=42)
X = normalize(X)
pd.DataFrame(X, columns=['x1', 'x2']).to_csv("two_moons.csv", index=False)
 
Y, A = inpainting_corruption(X, p=0.2, prevent_zero=False, rng=rng_corrupt)
pd.DataFrame({
    'x1': Y[:, 0], 'x2': Y[:, 1],
    'a1': A[:, 0], 'a2': A[:, 1]
}).to_csv("two_moons_corrupted.csv", index=False)
 
 
####################################### Swiss roll 2D ################################################
x_roll, _ = make_swiss_roll(n_samples=100000, noise=0.5, random_state=42)
x_roll = normalize(x_roll[:, [0, 2]])
pd.DataFrame(x_roll, columns=['x1', 'x2']).to_csv("swiss_roll.csv", index=False)
 
Y_roll, A_roll = inpainting_corruption(x_roll, p=0.2, prevent_zero=False, rng=rng_corrupt)
pd.DataFrame({
    'x1': Y_roll[:, 0], 'x2': Y_roll[:, 1],
    'a1': A_roll[:, 0], 'a2': A_roll[:, 1]
}).to_csv("swiss_roll_corrupted.csv", index=False)
 
 
########################################### N two moons datasets with different noise levels ########################################
print("Génération two moons...")
 
N_DATASETS = 10000
N_SAMPLES  = 10000
NOISE_MIN  = 0.01
NOISE_MAX  = 0.9
 
rng_noise = np.random.default_rng(42)
noise_levels = rng_noise.uniform(NOISE_MIN, NOISE_MAX, size=N_DATASETS)
 
# RNG dédié à la corruption des boucles (indépendant de noise_levels)
rng_corrupt_loop = np.random.default_rng(456)
 
chunks, chunks_corrupted = [], []
for i, noise in enumerate(noise_levels):
    X, _ = make_moons(n_samples=N_SAMPLES, noise=noise, random_state=i)
    X = normalize(X)
 
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['dataset_id'] = i
    df['noise']      = round(noise, 6)
    chunks.append(df)
 
    Y, A = inpainting_corruption(X, p=0.2, prevent_zero=False, rng=rng_corrupt_loop)
    df_corrupted = pd.DataFrame({
        'x1': Y[:, 0], 'x2': Y[:, 1],
        'a1': A[:, 0], 'a2': A[:, 1]
    })
    df_corrupted['dataset_id'] = i
    df_corrupted['noise']      = round(noise, 6)
    chunks_corrupted.append(df_corrupted)
 
    if (i + 1) % 10000 == 0:
        print(f"  {i+1}/{N_DATASETS} datasets générés...")
 
pd.concat(chunks, ignore_index=True).to_csv("two_moons_all.csv", index=False)
pd.concat(chunks_corrupted, ignore_index=True).to_csv("two_moons_all_corrupted.csv", index=False)
print("two_moons_all.csv / two_moons_all_corrupted.csv\n")
 
 
############################################ N swiss roll datasets with different noise levels ########################################
print("Génération swiss roll...")
 
chunks, chunks_corrupted = [], []
for i, noise in enumerate(noise_levels):
    X, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=noise * 10, random_state=i)
    X = normalize(X[:, [0, 2]])
 
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['dataset_id'] = i
    df['noise']      = round(noise, 6)
    chunks.append(df)
 
    Y, A = inpainting_corruption(X, p=0.2, prevent_zero=False, rng=rng_corrupt_loop)
    df_corrupted = pd.DataFrame({
        'x1': Y[:, 0], 'x2': Y[:, 1],
        'a1': A[:, 0], 'a2': A[:, 1]
    })
    df_corrupted['dataset_id'] = i
    df_corrupted['noise']      = round(noise, 6)
    chunks_corrupted.append(df_corrupted)
 
    if (i + 1) % 10000 == 0:
        print(f"  {i+1}/{N_DATASETS}...")
 
pd.concat(chunks, ignore_index=True).to_csv("swiss_roll_all.csv", index=False)
pd.concat(chunks_corrupted, ignore_index=True).to_csv("swiss_roll_all_corrupted.csv", index=False)
print("swiss_roll_all.csv / swiss_roll_all_corrupted.csv")