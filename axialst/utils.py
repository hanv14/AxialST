"""
AxialST — Shared utility functions.

Low-level helpers used across all stages: Gaussian kernels, composition
estimation (Nadaraya–Watson), density estimation, and array helpers.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def gaussian_kernel(distances, sigma):
    """Unnormalised Gaussian kernel."""
    return np.exp(-0.5 * (distances / sigma) ** 2)


# ---------------------------------------------------------------------------
# Composition / density estimation
# ---------------------------------------------------------------------------

def compute_composition_at_points(cell_positions, cell_types, type_list,
                                  query_points, sigma):
    """
    Nadaraya–Watson estimator for cell-type proportions at *query_points*.

    Parameters
    ----------
    cell_positions : (N, 2) array
    cell_types     : (N,) array of cell-type labels
    type_list      : list of T unique cell-type labels (defines column order)
    query_points   : (Q, 2) array
    sigma          : float  – Gaussian kernel bandwidth

    Returns
    -------
    compositions : (Q, T) array – rows sum to 1
    """
    n_cells = len(cell_types)
    n_types = len(type_list)
    type_to_idx = {t: i for i, t in enumerate(type_list)}

    # One-hot encoding  (N, T)
    onehot = np.zeros((n_cells, n_types), dtype=np.float64)
    for i, ct in enumerate(cell_types):
        idx = type_to_idx.get(ct)
        if idx is not None:
            onehot[i, idx] = 1.0

    # Pairwise distances  (Q, N)
    dists = cdist(np.asarray(query_points, dtype=np.float64),
                  np.asarray(cell_positions, dtype=np.float64))
    weights = gaussian_kernel(dists, sigma)          # (Q, N)

    compositions = weights @ onehot                  # (Q, T)
    totals = compositions.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    compositions /= totals
    return compositions


def compute_density_at_points(cell_positions, query_points, sigma):
    """
    Gaussian KDE cell density evaluated at *query_points*.

    Returns unnormalised density (higher = more cells nearby).
    """
    dists = cdist(np.asarray(query_points, dtype=np.float64),
                  np.asarray(cell_positions, dtype=np.float64))
    weights = gaussian_kernel(dists, sigma)
    return weights.sum(axis=1)


def compute_niche_descriptors_simple(positions, cell_types, type_list,
                                     radius=50.0):
    """
    Fallback niche descriptor when MENDER is unavailable.

    For every cell, compute the cell-type composition within *radius* of
    that cell.  Returns (N, T) array.
    """
    tree = cKDTree(positions)
    n_types = len(type_list)
    type_to_idx = {t: i for i, t in enumerate(type_list)}
    n_cells = len(positions)

    descriptors = np.zeros((n_cells, n_types), dtype=np.float64)
    for i in range(n_cells):
        neighbours = tree.query_ball_point(positions[i], r=radius)
        if len(neighbours) == 0:
            neighbours = [i]
        for j in neighbours:
            idx = type_to_idx.get(cell_types[j])
            if idx is not None:
                descriptors[i, idx] += 1.0
        total = descriptors[i].sum()
        if total > 0:
            descriptors[i] /= total
    return descriptors


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def get_unified_type_list(*adatas, cell_type_key='cell_type'):
    """Sorted union of cell-type labels across several AnnData objects."""
    all_types = set()
    for ad in adatas:
        all_types.update(ad.obs[cell_type_key].unique())
    return sorted(all_types)


def safe_to_dense(X):
    """Convert X to dense numpy array if sparse."""
    from scipy.sparse import issparse
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def auto_sigma(positions):
    """
    Heuristic KDE bandwidth: 5 × median nearest-neighbour distance.
    """
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)       # k=2: self + nearest
    median_nn = np.median(dists[:, 1])
    return float(max(median_nn * 3.0, 1.0))


def spatial_smooth_expression(positions, X, k=6, sigma_factor=1.0,
                              blend=0.15):
    """
    Gently smooth expression by blending each cell with a k-NN local
    average.  A small *blend* (e.g. 0.15) nudges outliers toward their
    neighbourhood without overwriting the per-gene spatial structure.

    Parameters
    ----------
    positions    : (N, 2) array — cell spatial coordinates
    X            : (N, G) array — expression matrix
    k            : int   — number of spatial neighbours (default 6)
    sigma_factor : float — multiplier on Gaussian bandwidth
    blend        : float in [0, 1] — 0 = keep original, 1 = full smooth

    Returns
    -------
    X_blended : (N, G) array — blend * smooth + (1 - blend) * original
    """
    if blend <= 0:
        return np.asarray(X, dtype=np.float32)

    tree = cKDTree(positions)
    dists, indices = tree.query(positions, k=k + 1)  # +1 for self

    median_dist = np.median(dists[:, 1:])
    sigma = median_dist * sigma_factor
    sigma = max(sigma, 1e-6)

    X = np.asarray(X, dtype=np.float64)
    X_smooth = np.empty_like(X)

    for i in range(len(positions)):
        nbr_idx = indices[i]
        nbr_dists = dists[i]
        w = np.exp(-0.5 * (nbr_dists / sigma) ** 2)
        w /= w.sum()
        X_smooth[i] = (X[nbr_idx] * w[:, None]).sum(axis=0)

    # Blend: mostly keep original, only nudge toward neighbourhood mean
    result = (1.0 - blend) * X + blend * X_smooth
    return result.astype(np.float32)


def auto_patch_size(positions, target_cells_per_patch=60):
    """
    Heuristic patch size so that the average patch contains roughly
    *target_cells_per_patch* cells.
    """
    n_cells = len(positions)
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    area = max(x_range * y_range, 1.0)
    density = n_cells / area
    patch_area = target_cells_per_patch / max(density, 1e-10)
    return float(max(np.sqrt(patch_area), 10.0))
