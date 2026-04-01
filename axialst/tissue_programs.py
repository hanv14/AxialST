"""
AxialST — Stage 1: Tissue Program Extraction and Interpolation.

A *tissue program* describes the organisational blueprint of one section:
  • Regional composition map  C_k(x, y)   — Nadaraya–Watson KDE
  • Niche catalogue           N_k          — cluster centroids of niche descriptors
  • Expression programme      E_k[t, n]    — mean expression per (type, niche)

Two tissue programs are interpolated at parameter α to produce the blueprint
for a virtual slice.
"""

import numpy as np
import contextlib
import io
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist as scipy_cdist

from .utils import (compute_composition_at_points, compute_density_at_points,
                    compute_niche_descriptors_simple, safe_to_dense)


# ===================================================================
# TissueProgram — container for a single real section
# ===================================================================

class TissueProgram:
    """Tissue programme descriptor extracted from one real section."""

    def __init__(self):
        # Per-cell data
        self.positions        = None   # (N, 2)
        self.cell_types       = None   # (N,) str labels
        self.type_list        = None   # sorted list of T type names
        self.sigma            = None   # KDE bandwidth

        # Niche catalogue
        self.niche_labels     = None   # (N,) int  cluster id per cell
        self.niche_centroids  = None   # (K, D) archetype centroids
        self.niche_compositions = None # (K, T) cell-type proportions per niche
        self.n_niches         = 0

        # Expression programmes  {(type_idx, niche_idx): (G,) array}
        self.expression_programs = {}

        # Niche descriptors (MENDER or fallback)
        self.niche_descriptors = None  # (N, D)

        # Global stats
        self.cell_density     = 0.0    # cells / unit²

        # Back-reference (kept for donor-cell look-up in Stage 3)
        self.adata            = None

    # ---- composition / density evaluation ----------------------------

    def eval_composition(self, query_points):
        """Cell-type proportions at arbitrary (x, y) locations."""
        return compute_composition_at_points(
            self.positions, self.cell_types, self.type_list,
            query_points, self.sigma)

    def eval_density(self, query_points):
        """KDE cell density at arbitrary (x, y) locations."""
        return compute_density_at_points(
            self.positions, query_points, self.sigma)


# ===================================================================
# InterpolatedTissueProgram — virtual-slice blueprint
# ===================================================================

class InterpolatedTissueProgram:
    """Blueprint for a virtual slice: interpolation of two TissuePrograms."""

    def __init__(self):
        self.type_list            = None
        self.alpha                = None
        self.tp1                  = None   # TissueProgram (section k)
        self.tp2                  = None   # TissueProgram (section k+1)

        # Niche matching  [(i_k, j_{k+1}), …]
        self.niche_matching       = []
        self.n_niches             = 0
        self.niche_centroids      = None   # (M, D) interpolated centroids
        self.niche_compositions   = None   # (M, T) interpolated compositions

        # Expression programmes  {(type_idx, niche_idx): (G,) array}
        self.expression_programs  = {}

    # ---- evaluate at query points ------------------------------------

    def eval_composition(self, query_points):
        """Interpolated cell-type proportions."""
        c1 = self.tp1.eval_composition(query_points)
        c2 = self.tp2.eval_composition(query_points)
        return self.alpha * c1 + (1 - self.alpha) * c2

    def eval_density(self, query_points):
        """Interpolated cell density."""
        d1 = self.tp1.eval_density(query_points)
        d2 = self.tp2.eval_density(query_points)
        return self.alpha * d1 + (1 - self.alpha) * d2

    def assign_niche(self, compositions):
        """
        Assign niche archetypes to patches from their local composition.

        Returns
        -------
        assignments  : (P,) int   — niche index per patch
        confidences  : (P,) float — max cosine-similarity (higher = more certain)
        """
        if self.niche_compositions is None or len(self.niche_compositions) == 0:
            return np.zeros(len(compositions), dtype=int), np.ones(len(compositions))
        sims = cosine_similarity(compositions, self.niche_compositions)
        assignments = sims.argmax(axis=1)
        confidences = sims.max(axis=1)
        return assignments, confidences


# ===================================================================
# Extraction
# ===================================================================

def run_mender(adata_list, cell_type_key, batch_labels,
               scale=6, radius=15, micro_env_key='mender'):
    """
    Run MENDER on a list of AnnData objects (concatenated internally).

    Returns a list of niche-descriptor arrays, one per input adata,
    in the same order as *adata_list*.
    """
    import MENDER as _MENDER

    import anndata as _anndata

    # Tag each slice
    for ad, tag in zip(adata_list, batch_labels):
        ad.obs['_axialst_batch'] = tag

    combined = _anndata.concat(adata_list)
    combined.obs['_axialst_batch'] = combined.obs['_axialst_batch'].astype('category')
    combined.obs[cell_type_key] = combined.obs[cell_type_key].astype('category')

    msm = _MENDER.MENDER(combined,
                          batch_obs='_axialst_batch',
                          ct_obs=cell_type_key,
                          random_seed=0)
    msm.prepare()
    msm.set_MENDER_para(n_scales=scale, nn_mode='radius', nn_para=radius)

    with contextlib.redirect_stdout(io.StringIO()):
        msm.run_representation_mp(200)

    combined.obsm[micro_env_key] = msm.adata_MENDER.X

    # Split back
    results = []
    for tag in batch_labels:
        mask = combined.obs['_axialst_batch'] == tag
        results.append(combined[mask].obsm[micro_env_key].copy())

    # Clean up temp columns
    for ad in adata_list:
        if '_axialst_batch' in ad.obs.columns:
            del ad.obs['_axialst_batch']

    return results


def extract_tissue_program(adata, cell_type_key, type_list, sigma,
                           n_niches=20, niche_radius=50.0,
                           niche_descriptors=None, verbose=True):
    """
    Build a TissueProgram from one real section.

    Parameters
    ----------
    adata             : AnnData with obsm['spatial'] and obs[cell_type_key].
    cell_type_key     : column name in obs for cell types.
    type_list         : unified list of all cell types (same across sections).
    sigma             : Gaussian kernel bandwidth for composition KDE.
    n_niches          : number of niche archetypes to discover.
    niche_radius      : radius for fallback niche descriptors (if MENDER
                        embeddings are not provided).
    niche_descriptors : (N, D) pre-computed MENDER embeddings, or None.
    verbose           : print progress messages.

    Returns
    -------
    TissueProgram
    """
    tp = TissueProgram()
    tp.positions   = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    tp.cell_types  = np.asarray(adata.obs[cell_type_key].values)
    tp.type_list   = type_list
    tp.sigma       = sigma
    tp.adata       = adata

    # Global density (adaptive floor: avoid hardcoded 1.0)
    x_span = tp.positions[:, 0].max() - tp.positions[:, 0].min()
    y_span = tp.positions[:, 1].max() - tp.positions[:, 1].min()
    area = x_span * y_span
    if area <= 0:
        # Degenerate: collinear or single-point data
        area = max(x_span + y_span, 1e-10) ** 2
    tp.cell_density = len(tp.positions) / area

    # ---- niche descriptors -------------------------------------------
    if niche_descriptors is not None:
        tp.niche_descriptors = niche_descriptors
    else:
        if verbose:
            print("  Computing fallback niche descriptors (composition in radius)…")
        # Adaptive radius: use data-driven default if niche_radius seems
        # unreasonable for this dataset's coordinate scale
        tree = cKDTree(tp.positions)
        nn_dists, _ = tree.query(tp.positions, k=min(2, len(tp.positions)))
        median_nn = float(np.median(nn_dists[:, -1])) if len(tp.positions) > 1 else 1.0
        # Use provided niche_radius, but warn if it's very different
        # from the data scale
        if niche_radius < median_nn * 0.5 or niche_radius > median_nn * 100:
            effective_radius = median_nn * 10.0
            if verbose:
                print(f"  niche_radius={niche_radius:.1f} seems mismatched with "
                      f"data scale (median NN={median_nn:.1f}), "
                      f"using {effective_radius:.1f} instead")
        else:
            effective_radius = niche_radius
        tp.niche_descriptors = compute_niche_descriptors_simple(
            tp.positions, tp.cell_types, type_list, radius=effective_radius)

    # ---- cluster into niche archetypes -------------------------------
    # Need at least 3 cells per cluster for meaningful niche definition
    actual_k = min(n_niches, max(len(tp.positions) // 3, 2))
    actual_k = max(actual_k, 2)

    km = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    tp.niche_labels = km.fit_predict(tp.niche_descriptors)
    tp.niche_centroids = km.cluster_centers_
    tp.n_niches = actual_k

    # ---- niche compositions ------------------------------------------
    type_to_idx = {t: i for i, t in enumerate(type_list)}
    tp.niche_compositions = np.zeros((actual_k, len(type_list)))
    for ni in range(actual_k):
        mask = tp.niche_labels == ni
        for ct in tp.cell_types[mask]:
            idx = type_to_idx.get(ct)
            if idx is not None:
                tp.niche_compositions[ni, idx] += 1
        total = tp.niche_compositions[ni].sum()
        if total > 0:
            tp.niche_compositions[ni] /= total

    # ---- expression programmes  E_k[t, n] ---------------------------
    X = safe_to_dense(adata.X)
    tp.expression_programs = {}
    for ni in range(actual_k):
        niche_mask = tp.niche_labels == ni
        for t_idx, t_name in enumerate(type_list):
            type_mask = tp.cell_types == t_name
            both = niche_mask & type_mask
            if both.sum() > 0:
                tp.expression_programs[(t_idx, ni)] = X[both].mean(axis=0)

    if verbose:
        print(f"  Tissue program extracted: {len(tp.positions)} cells, "
              f"{actual_k} niches, "
              f"{len(tp.expression_programs)} (type, niche) expression entries.")
    return tp


# ===================================================================
# Interpolation
# ===================================================================

def interpolate_tissue_programs(tp1, tp2, alpha, verbose=True):
    """
    Interpolate two tissue programmes at blending parameter *alpha*.

    Niche archetypes are matched using the Hungarian algorithm on
    cosine distance of their centroids (or compositions as fallback).

    Returns an InterpolatedTissueProgram.
    """
    vtp = InterpolatedTissueProgram()
    vtp.type_list = tp1.type_list
    vtp.alpha     = alpha
    vtp.tp1       = tp1
    vtp.tp2       = tp2

    n1, n2 = tp1.n_niches, tp2.n_niches

    # ---- match niche archetypes --------------------------------------
    # Prefer centroids; fall back to compositions when dims differ
    if (tp1.niche_centroids.shape[1] == tp2.niche_centroids.shape[1]):
        sim = cosine_similarity(tp1.niche_centroids, tp2.niche_centroids)
    else:
        sim = cosine_similarity(tp1.niche_compositions, tp2.niche_compositions)

    cost = 1.0 - sim                    # (n1, n2) cost matrix

    # Pad to square for Hungarian
    n_max = max(n1, n2)
    padded = np.full((n_max, n_max), cost.max() + 1.0)
    padded[:n1, :n2] = cost
    row_idx, col_idx = linear_sum_assignment(padded)

    # Keep only valid matches
    matching = [(r, c) for r, c in zip(row_idx, col_idx)
                if r < n1 and c < n2]
    vtp.niche_matching = matching
    n_matched = len(matching)
    vtp.n_niches = n_matched

    # ---- interpolated centroids & compositions -----------------------
    dim_c = tp1.niche_centroids.shape[1]
    vtp.niche_centroids = np.zeros((n_matched, dim_c))
    vtp.niche_compositions = np.zeros((n_matched, len(tp1.type_list)))

    for k, (i, j) in enumerate(matching):
        vtp.niche_centroids[k]    = alpha * tp1.niche_centroids[i] \
                                  + (1 - alpha) * tp2.niche_centroids[j]
        vtp.niche_compositions[k] = alpha * tp1.niche_compositions[i] \
                                  + (1 - alpha) * tp2.niche_compositions[j]

    # ---- interpolated expression programmes --------------------------
    vtp.expression_programs = {}
    for k, (i, j) in enumerate(matching):
        for t_idx in range(len(tp1.type_list)):
            e1 = tp1.expression_programs.get((t_idx, i))
            e2 = tp2.expression_programs.get((t_idx, j))
            if e1 is not None and e2 is not None:
                vtp.expression_programs[(t_idx, k)] = alpha * e1 + (1 - alpha) * e2
            elif e1 is not None:
                vtp.expression_programs[(t_idx, k)] = e1.copy()
            elif e2 is not None:
                vtp.expression_programs[(t_idx, k)] = e2.copy()

    if verbose:
        print(f"  Interpolated tissue program (alpha={alpha:.2f}): "
              f"{n_matched} matched niches, "
              f"{len(vtp.expression_programs)} expression entries.")
    return vtp
