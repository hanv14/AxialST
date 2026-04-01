"""
AxialST — Stage 3: Expression Materialisation.

Assign gene expression to every virtual cell.  Unlike SpatialZ (which matches
by type *or* niche independently), we require joint matching: a donor cell
must be the same cell type **and** from a similar niche archetype.

Two modes:
  • 'default'  – niche-aware sampling with z-weighted donor selection
  • 'fast'     – nearest same-type cell averaging (simpler, faster)
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from scipy.sparse import issparse


def _dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


# ===================================================================
# Default mode  (niche-aware, joint matching)
# ===================================================================

def synthesize_expression_default(ref_adatas, virtual_adata,
                                  cell_type_key,
                                  niche_labels_refs,
                                  niche_labels_virtual,
                                  niche_desc_refs,
                                  niche_desc_virtual,
                                  alpha, k_sam=1, Beta=0,
                                  smooth_k=0, smooth_sigma=1.0,
                                  smooth_alpha=0.0,
                                  verbose=True):
    """
    Niche-coherent expression synthesis (Stage 3 — default mode).

    For best spatial autocorrelation preservation, use k_sam=1 (single-
    donor copy).  This directly inherits per-gene variance structure
    from real data.  Higher k_sam averages donors, which inflates
    Moran's I uniformly and degrades the per-gene correlation.

    Parameters
    ----------
    ref_adatas          : list of 2 AnnData  (flanking real sections)
    virtual_adata       : AnnData with obs[cell_type_key] and obsm['spatial']
    cell_type_key       : column in obs
    niche_labels_refs   : list of 2 int-arrays  (per-cell niche label, real)
    niche_labels_virtual: (V,) int array        (per-cell niche label, virtual)
    niche_desc_refs     : list of 2  (N_k, D) arrays  (MENDER / fallback)
    niche_desc_virtual  : (V, D) array
    alpha               : float  interpolation parameter
    k_sam               : donors per virtual cell (1 = copy, >1 = average)
    Beta                : temperature for niche-similarity weighting
                          (keep low, e.g. 5, so spatial proximity dominates)
    smooth_k            : spatial smoothing neighbours (0 = disabled)
    smooth_sigma        : bandwidth multiplier for spatial smoothing
    smooth_alpha        : blend factor for smoothing (0 = disabled)
    verbose             : print progress

    Returns
    -------
    virtual_adata  : (X is filled in)
    donor_counts   : (V,) int — number of matched donor candidates per cell
    donor_variances: (V,) float — mean per-gene variance among donor expression
    """
    from .utils import spatial_smooth_expression

    # ---- combine references -----------------------------------------
    import anndata
    combined = anndata.concat(ref_adatas, label='_ref_batch',
                              keys=[f'ref_{i}' for i in range(len(ref_adatas))])
    comb_X     = _dense(combined.X).astype(np.float32)
    comb_types = np.asarray(combined.obs[cell_type_key].values)
    comb_pos   = np.asarray(combined.obsm['spatial'])

    # z-weights: alpha for section k, (1-alpha) for section k+1
    batches = combined.obs['_ref_batch'].values
    z_weights = np.where(batches == 'ref_0', alpha, 1.0 - alpha)

    # Concatenated niche info
    comb_niche_labels = np.concatenate(niche_labels_refs)
    comb_niche_desc   = np.vstack(niche_desc_refs)

    virt_types  = np.asarray(virtual_adata.obs[cell_type_key].values)
    virt_pos    = np.asarray(virtual_adata.obsm['spatial'])
    n_virtual   = len(virtual_adata)
    n_genes     = comb_X.shape[1]

    # Spatial KD-tree
    kdtree = cKDTree(comb_pos)

    # Spatial bandwidth: tight to strongly favour nearest donors
    _nn_dists, _ = kdtree.query(comb_pos, k=2)
    sigma_spatial = float(np.median(_nn_dists[:, 1])) * 1.5
    sigma_spatial = max(sigma_spatial, 1.0)

    # Pre-compute cosine similarities  (combined × virtual)
    if niche_desc_virtual is not None and comb_niche_desc is not None:
        cos_sims = cos_sim(comb_niche_desc, niche_desc_virtual)  # (R, V)
    else:
        cos_sims = None

    virt_X = np.zeros((n_virtual, n_genes), dtype=np.float32)
    donor_counts = np.zeros(n_virtual, dtype=int)
    donor_variances = np.zeros(n_virtual, dtype=np.float32)

    rng = np.random.RandomState(42)

    # Pre-query spatial neighbours at base radius
    max_radius = sigma_spatial * 4.0
    virt_nbr_lists = kdtree.query_ball_point(virt_pos, r=max_radius)

    # Pre-compute per-type indices for efficient global lookup
    unique_types = np.unique(comb_types)
    type_to_indices = {t: np.where(comb_types == t)[0] for t in unique_types}

    n_local = 0       # matched within base radius
    n_expanded = 0    # matched via expanded radius
    n_global = 0      # matched via global fallback

    for i in range(n_virtual):
        ct = virt_types[i]

        # --- 1) Base radius: same type within sigma*4 ---
        nearby = np.array(virt_nbr_lists[i], dtype=int) if virt_nbr_lists[i] else np.array([], dtype=int)
        cands = nearby[comb_types[nearby] == ct] if len(nearby) > 0 else np.array([], dtype=int)

        if len(cands) >= 1:
            n_local += 1
        else:
            # --- 2) Adaptive expansion: 8σ then 16σ ---
            for exp_mult in (8.0, 16.0):
                nearby_exp = np.array(
                    kdtree.query_ball_point(virt_pos[i], r=sigma_spatial * exp_mult),
                    dtype=int)
                if len(nearby_exp) > 0:
                    cands = nearby_exp[comb_types[nearby_exp] == ct]
                if len(cands) >= 1:
                    break
            if len(cands) >= 1:
                n_expanded += 1
            else:
                # --- 3) Global fallback ---
                cands = type_to_indices.get(ct, np.array([], dtype=int))
                n_global += 1

        if len(cands) == 0:
            # Last resort: nearest cell regardless of type
            _, nn_idx = kdtree.query(virt_pos[i], k=1)
            nn_idx = np.atleast_1d(nn_idx)
            virt_X[i] = comb_X[nn_idx[0]]
            donor_counts[i] = 1
            continue

        donor_counts[i] = len(cands)

        # --- For k_sam=1, pick the nearest same-type cell by distance ---
        # Using raw distance (argmin) instead of Gaussian weight (argmax)
        # avoids the saturation problem where all distant candidates get
        # the same clipped weight, making selection effectively random.
        if k_sam <= 1:
            xy_dists = np.linalg.norm(comb_pos[cands] - virt_pos[i], axis=1)
            nearest = np.argmin(xy_dists)
            virt_X[i] = comb_X[cands[nearest]]
            donor_variances[i] = 0.0
            continue

        # --- k_sam > 1: weighted averaging (unchanged) ---
        w_z = z_weights[cands]

        xy_dists = np.linalg.norm(comb_pos[cands] - virt_pos[i], axis=1)
        w_spatial = np.exp(-0.5 * (xy_dists / sigma_spatial) ** 2)
        w_spatial = np.maximum(w_spatial, 1e-6)

        w = w_z * w_spatial

        if Beta > 0 and cos_sims is not None:
            w_niche = np.exp(Beta * cos_sims[cands, i])
            w *= w_niche
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(len(cands)) / len(cands)

        k_use = min(k_sam, len(cands))
        if len(cands) <= k_use:
            virt_X[i] = (comb_X[cands] * w[:, None]).sum(axis=0)
            if len(cands) > 1:
                donor_variances[i] = float(comb_X[cands].var(axis=0).mean())
        else:
            top_k = np.argpartition(w, -k_use)[-k_use:]
            sel = cands[top_k]
            sel_w = w[top_k]
            sel_w /= sel_w.sum()
            virt_X[i] = (comb_X[sel] * sel_w[:, None]).sum(axis=0)
            donor_variances[i] = float(comb_X[sel].var(axis=0).mean())

    # --- Optional spatial smoothing pass ---
    if smooth_k > 0 and smooth_alpha > 0 and n_virtual > smooth_k:
        if verbose:
            print(f"  Applying spatial smoothing (k={smooth_k}, "
                  f"blend={smooth_alpha:.2f}) …")
        virt_X = spatial_smooth_expression(
            virt_pos, virt_X, k=smooth_k, sigma_factor=smooth_sigma,
            blend=smooth_alpha)

    virtual_adata.X = virt_X

    if verbose:
        print(f"  Expression synthesized for {n_virtual} virtual cells "
              f"(k_sam={k_sam}, median donor pool = "
              f"{int(np.median(donor_counts))})")
        print(f"  Donor matching: {n_local} local, "
              f"{n_expanded} expanded, {n_global} global")

    return virtual_adata, donor_counts, donor_variances


# ===================================================================
# Fast mode  (simple k-NN, same-type averaging)
# ===================================================================

def synthesize_expression_fast(ref_adatas, virtual_adata,
                               cell_type_key, k_sam=3, verbose=True):
    """
    Fast expression synthesis: average of nearest same-type cells.

    This is essentially the same as SpatialZ's ``fast`` mode but
    provided here for consistency.
    """
    import anndata
    combined = anndata.concat(ref_adatas, label='_ref_batch',
                              keys=[f'ref_{i}' for i in range(len(ref_adatas))])
    comb_X     = _dense(combined.X).astype(np.float32)
    comb_types = np.asarray(combined.obs[cell_type_key].values)
    comb_pos   = np.asarray(combined.obsm['spatial'])

    nn = NearestNeighbors(n_neighbors=min(10, len(combined))).fit(comb_pos)
    distances, indices = nn.kneighbors(virtual_adata.obsm['spatial'])

    virt_types = np.asarray(virtual_adata.obs[cell_type_key].values)
    n_virtual  = len(virtual_adata)
    n_genes    = comb_X.shape[1]
    virt_X     = np.zeros((n_virtual, n_genes), dtype=np.float32)

    for i in range(n_virtual):
        qtype = virt_types[i]
        same = [idx for idx in indices[i] if comb_types[idx] == qtype]

        if len(same) > 0:
            sel = same[:min(len(same), k_sam)]
        else:
            all_same = np.where(comb_types == qtype)[0]
            if len(all_same) > 0:
                d = np.linalg.norm(comb_pos[all_same]
                                   - virtual_adata.obsm['spatial'][i],
                                   axis=1)
                sel = [all_same[np.argmin(d)]]
            else:
                sel = list(indices[i][:k_sam])

        expr = comb_X[sel]
        virt_X[i] = expr.mean(axis=0)

    virtual_adata.X = virt_X

    if verbose:
        print(f"  [fast] Expression synthesized for {n_virtual} cells")
    return virtual_adata
