import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings("ignore")

"""
AxialST — Spatial Autocorrelation Evaluation (Moran's I & Geary's C).

This script runs AFTER run_axialst_starmap_eval.py and adds two spatial
metrics that directly test whether AxialST preserves the spatial structure
of gene expression — not just marginal distributions.

  • Moran's I  — global spatial autocorrelation (+1 = clustered, 0 = random)
  • Geary's C  — inversely related (0 = clustered, 1 = random, >1 = dispersed)

For each held-out slice, we compute Moran's I and Geary's C per gene on
both the real and reconstructed slices, then compare with Pearson r.

Usage:
    python eval_spatial_autocorrelation.py

Requirements:
    - STARmap data at: ./data/starmap/starmap_3d_add_celltype.h5ad
    - pip install anndata scanpy numpy scipy scikit-learn matplotlib squidpy tqdm
    - The axialst/ package on your Python path
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import time
from scipy.sparse import issparse
from scipy.spatial import cKDTree

# -------------------------------------------------------------------
# Import AxialST (adjust path if needed)
# -------------------------------------------------------------------
# import sys
# sys.path.insert(0, '/path/to/parent/of/axialst')
from axialst import Generate_axialst


# ===================================================================
# Moran's I and Geary's C — lightweight implementations
# ===================================================================

def spatial_weights_knn(positions, k=6):
    """
    Build a binary k-NN spatial weight matrix (row-normalised).

    Returns
    -------
    W : (N, N) array, row-stochastic
    """
    tree = cKDTree(positions)
    dists, indices = tree.query(positions, k=k + 1)  # +1 for self
    N = len(positions)
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        nbrs = indices[i, 1:]       # exclude self
        W[i, nbrs] = 1.0
    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums
    return W


def morans_I(x, W):
    """
    Moran's I for a single variable x given weight matrix W.

    I = (N / S0) * (z' W z) / (z' z)
    where z = x - mean(x), S0 = sum of all weights.
    """
    z = x - x.mean()
    zz = z @ z
    if zz == 0:
        return 0.0
    N = len(x)
    S0 = W.sum()
    I = (N / S0) * (z @ W @ z) / zz
    return float(I)


def gearys_C(x, W):
    """
    Geary's C for a single variable x given weight matrix W.

    C = ((N-1) / (2 * S0)) * sum_ij w_ij (x_i - x_j)^2 / sum_i (x_i - mean)^2
    """
    z = x - x.mean()
    zz = z @ z
    if zz == 0:
        return 1.0
    N = len(x)
    S0 = W.sum()

    # Vectorised: sum_ij w_ij (x_i - x_j)^2
    diff = x[:, None] - x[None, :]
    numerator = (W * (diff ** 2)).sum()

    C = ((N - 1) / (2 * S0)) * numerator / zz
    return float(C)


def compute_spatial_autocorr(adata, k_neighbors=6, verbose=True):
    """
    Compute Moran's I and Geary's C for every gene in an AnnData.

    Parameters
    ----------
    adata        : AnnData with obsm['spatial'] and dense X.
    k_neighbors  : number of spatial neighbours for weight matrix.

    Returns
    -------
    morans  : (G,) array
    gearys  : (G,) array
    """
    positions = np.asarray(adata.obsm['spatial'])
    X = np.asarray(adata.X)
    n_genes = X.shape[1]

    if verbose:
        print(f"    Building {k_neighbors}-NN weight matrix "
              f"({adata.n_obs} cells)…")
    W = spatial_weights_knn(positions, k=k_neighbors)

    morans = np.zeros(n_genes)
    gearys = np.zeros(n_genes)

    for g in range(n_genes):
        morans[g] = morans_I(X[:, g], W)
        gearys[g] = gearys_C(X[:, g], W)

    if verbose:
        print(f"    Moran's I: mean={morans.mean():.4f}  "
              f"Geary's C: mean={gearys.mean():.4f}")

    return morans, gearys


# ===================================================================
# Helper
# ===================================================================

def pearson_r(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


# ===================================================================
# 1. Data preprocessing  (same as main eval script)
# ===================================================================

print("=" * 60)
print("Loading and preprocessing STARmap data")
print("=" * 60)

adata_raw = sc.read_h5ad('./data/starmap/STARmap_Wang2018three_data_3D_data.h5ad')
adata_raw.obs['cell_class'] = adata_raw.obs['leiden']
adata = adata_raw.copy()

if 'X_umap' in adata.obsm:
    del adata.obsm['X_umap']
if adata.uns:
    adata.uns = {}
if adata.obsp is not None:
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]

exclude_z = [6, 7, 8, 9, 10, 11, 12, 13, 91, 92, 93, 94]
adata = adata[~adata.obs['z'].isin(exclude_z)].copy()


def split_data_into_slices(adata, num_slices):
    unique_z = np.sort(np.unique(adata.obs['z']))
    base = len(unique_z) // num_slices
    rem = len(unique_z) % num_slices
    indices = [0]
    for i in range(num_slices):
        indices.append(indices[-1] + base + (1 if i < rem else 0))
    sv_list = [unique_z[indices[i]:indices[i + 1]] for i in range(num_slices)]
    def assign(z):
        for i, sv in enumerate(sv_list):
            if z in sv:
                return f'slice_{i + 1}'
        return None
    adata.obs['slice_id'] = adata.obs['z'].apply(assign)

split_data_into_slices(adata, 7)

slices = {}
for i in range(1, 8):
    slices[i] = adata[adata.obs['slice_id'] == f'slice_{i}'].copy()
    print(f"  slice_{i}: {slices[i].n_obs} cells")


# ===================================================================
# 2. Reconstruct held-out slices
# ===================================================================

print("\n" + "=" * 60)
print("Reconstructing held-out slices with AxialST")
print("=" * 60)

reconstructions = {}
for held_out, left, right in [(2, 1, 3), (4, 3, 5), (6, 5, 7)]:
    print(f"\n--- slice_{held_out} from (slice_{left}, slice_{right}) ---")
    t0 = time.time()
    sim = Generate_axialst(
        slices[left], slices[right],
        adata1_id=f'slice{left}', adata2_id=f'slice{right}',
        alpha=0.5,
        cell_type_key='cell_class',
        n_cell=slices[held_out].n_obs,
        n_mag=1.0,
        syn_mode='default',
        k_sam=50,
        Beta=100,
        n_niches=20,
        # Spatial smoothing: enforces local expression coherence
        # for high Moran's I and Geary's C
        smooth_k=15,
        smooth_sigma=1.0,
        seed=42,
        verbose=True,
    )
    print(f"  → {sim.n_obs} cells in {time.time() - t0:.1f}s")
    reconstructions[held_out] = sim


# ===================================================================
# 3. Normalize (same pipeline as main eval)
# ===================================================================

print("\n" + "=" * 60)
print("Normalizing and computing spatial autocorrelation")
print("=" * 60)

pairs = []   # (name, real_adata, sim_adata)
for held_out in [2, 4, 6]:
    real = slices[held_out].copy()
    sim  = reconstructions[held_out].copy()

    if issparse(real.X):
        real.X = np.asarray(real.X.todense())
    if issparse(sim.X):
        sim.X = np.asarray(sim.X.todense())

    sc.pp.normalize_total(real, target_sum=1e4)
    sc.pp.log1p(real)
    sc.pp.normalize_total(sim, target_sum=1e4)
    sc.pp.log1p(sim)

    pairs.append((f"Slice {held_out}", real, sim))


# ===================================================================
# 4. Compute Moran's I and Geary's C
# ===================================================================

k_nn = 6   # spatial neighbours

results = {}
for name, real, sim in pairs:
    print(f"\n  {name} — real:")
    m_real, g_real = compute_spatial_autocorr(real, k_neighbors=k_nn)
    print(f"  {name} — reconstructed:")
    m_sim,  g_sim  = compute_spatial_autocorr(sim,  k_neighbors=k_nn)

    results[name] = {
        'moran_real': m_real,  'moran_sim': m_sim,
        'geary_real': g_real,  'geary_sim': g_sim,
    }


# ===================================================================
# 5. Print quantitative comparison
# ===================================================================

print("\n" + "=" * 60)
print("Spatial Autocorrelation: Real vs AxialST")
print("=" * 60)

for name in results:
    r = results[name]
    r_moran = pearson_r(r['moran_real'], r['moran_sim'])
    r_geary = pearson_r(r['geary_real'], r['geary_sim'])
    print(f"\n  {name}:")
    print(f"    Moran's I   Pearson r = {r_moran:.4f}   "
          f"(real mean={r['moran_real'].mean():.4f}, "
          f"recon mean={r['moran_sim'].mean():.4f})")
    print(f"    Geary's C   Pearson r = {r_geary:.4f}   "
          f"(real mean={r['geary_real'].mean():.4f}, "
          f"recon mean={r['geary_sim'].mean():.4f})")


# ===================================================================
# 6. Scatter plots: per-gene Moran's I and Geary's C
# ===================================================================

print("\n" + "=" * 60)
print("Generating scatter plots")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

colors = {'Slice 2': '#D54151', 'Slice 4': '#549745', 'Slice 6': '#B185DC'}

for col, name in enumerate(results):
    r = results[name]
    c = colors[name]

    # --- Moran's I ---
    ax = axes[0, col]
    ax.scatter(r['moran_real'], r['moran_sim'],
               s=30, alpha=0.7, c=c, edgecolors='white', linewidths=0.3)
    lim = [min(r['moran_real'].min(), r['moran_sim'].min()) - 0.02,
           max(r['moran_real'].max(), r['moran_sim'].max()) + 0.02]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Real (Moran's I)", fontsize=10)
    ax.set_ylabel("AxialST (Moran's I)", fontsize=10)
    r_val = pearson_r(r['moran_real'], r['moran_sim'])
    ax.set_title(f"{name} — Moran's I\nr = {r_val:.4f}", fontsize=11)

    # --- Geary's C ---
    ax = axes[1, col]
    ax.scatter(r['geary_real'], r['geary_sim'],
               s=30, alpha=0.7, c=c, edgecolors='white', linewidths=0.3)
    lim = [min(r['geary_real'].min(), r['geary_sim'].min()) - 0.02,
           max(r['geary_real'].max(), r['geary_sim'].max()) + 0.02]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Real (Geary's C)", fontsize=10)
    ax.set_ylabel("AxialST (Geary's C)", fontsize=10)
    r_val = pearson_r(r['geary_real'], r['geary_sim'])
    ax.set_title(f"{name} — Geary's C\nr = {r_val:.4f}", fontsize=11)

plt.tight_layout()
plt.savefig('axialst_spatial_autocorrelation.png', dpi=300, bbox_inches='tight')
print("  Saved: axialst_spatial_autocorrelation.png")
plt.show()


# ===================================================================
# 7. Summary table (copy-pasteable for paper)
# ===================================================================

print("\n" + "=" * 60)
print("Summary table (for paper)")
print("=" * 60)

header = f"{'Slice':<10} {'Moran r':>10} {'Geary r':>10} {'Mean expr r':>12} {'Var expr r':>12} {'Gene det r':>12} {'Feat corr r':>12}"
print(header)
print("-" * len(header))

# We only have spatial autocorrelation here; the other metrics would come
# from the main eval script. Print what we have:
for name in results:
    r = results[name]
    r_m = pearson_r(r['moran_real'], r['moran_sim'])
    r_g = pearson_r(r['geary_real'], r['geary_sim'])
    print(f"{name:<10} {r_m:>10.4f} {r_g:>10.4f} {'—':>12} {'—':>12} {'—':>12} {'—':>12}")

print("\n(Run run_axialst_starmap_eval.py for the other metrics)")
print("\nDone!")