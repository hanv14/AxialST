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
    n_query = len(query_points)
    type_to_idx = {t: i for i, t in enumerate(type_list)}

    # One-hot encoding  (N, T)
    onehot = np.zeros((n_cells, n_types), dtype=np.float64)
    for i, ct in enumerate(cell_types):
        idx = type_to_idx.get(ct)
        if idx is not None:
            onehot[i, idx] = 1.0

    cell_positions = np.asarray(cell_positions, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)

    # Use chunked computation for large datasets to avoid O(Q*N) memory
    # (threshold: ~500MB at float64)
    max_elements = 64_000_000  # ~500MB
    if n_query * n_cells > max_elements:
        chunk_size = max(1, max_elements // n_cells)
        compositions = np.zeros((n_query, n_types), dtype=np.float64)
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            dists = cdist(query_points[start:end], cell_positions)
            weights = gaussian_kernel(dists, sigma)
            compositions[start:end] = weights @ onehot
    else:
        dists = cdist(query_points, cell_positions)
        weights = gaussian_kernel(dists, sigma)
        compositions = weights @ onehot

    totals = compositions.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    compositions /= totals
    return compositions


def compute_density_at_points(cell_positions, query_points, sigma):
    """
    Gaussian KDE cell density evaluated at *query_points*.

    Returns unnormalised density (higher = more cells nearby).
    """
    cell_positions = np.asarray(cell_positions, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)
    n_query = len(query_points)
    n_cells = len(cell_positions)

    # Chunked for large datasets
    max_elements = 64_000_000
    if n_query * n_cells > max_elements:
        chunk_size = max(1, max_elements // n_cells)
        density = np.zeros(n_query, dtype=np.float64)
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            dists = cdist(query_points[start:end], cell_positions)
            density[start:end] = gaussian_kernel(dists, sigma).sum(axis=1)
        return density
    else:
        dists = cdist(query_points, cell_positions)
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
    Heuristic KDE bandwidth: 3 × median nearest-neighbour distance.

    Adapts to any coordinate scale (microns, pixels, arbitrary units).
    The minimum is clamped to half the median NN distance to avoid
    degenerate zero bandwidth, rather than a hardcoded constant.
    """
    n = len(positions)
    if n < 2:
        return 1.0
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=min(2, n))
    median_nn = float(np.median(dists[:, -1]))
    if median_nn <= 0:
        return 1.0
    return float(max(median_nn * 3.0, median_nn * 0.5))


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

    n = len(positions)
    actual_k = min(k, n - 1)  # can't have more neighbours than cells
    if actual_k < 1:
        return np.asarray(X, dtype=np.float32)

    tree = cKDTree(positions)
    dists, indices = tree.query(positions, k=actual_k + 1)  # +1 for self

    median_dist = np.median(dists[:, 1:])
    sigma = median_dist * sigma_factor
    sigma = max(sigma, median_dist * 0.1 if median_dist > 0 else 1e-6)

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

    Adapts to any coordinate scale.  The minimum patch size is clamped
    to 2× median NN distance (data-driven) rather than a fixed constant.
    """
    n_cells = len(positions)
    if n_cells < 2:
        return 100.0
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    area = max(x_range * y_range, 1e-10)
    density = n_cells / area
    patch_area = target_cells_per_patch / max(density, 1e-10)
    patch_size = float(np.sqrt(patch_area))

    # Data-driven minimum: at least 2× median NN distance
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=min(2, n_cells))
    median_nn = float(np.median(dists[:, -1]))
    min_size = max(median_nn * 2.0, 1e-6)

    return float(max(patch_size, min_size))
