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
                                  alpha, k_sam=3, Beta=100,
                                  verbose=True):
    """
    Niche-coherent expression synthesis (Stage 3 — default mode).

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
    k_sam               : how many donors to draw per virtual cell
    Beta                : temperature for niche-similarity weighting
    verbose             : print progress

    Returns
    -------
    virtual_adata  : (X is filled in)
    donor_counts   : (V,) int — number of matched donor candidates per cell
    """
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

    # Spatial KD-tree for fallback and spatial proximity weighting
    kdtree = cKDTree(comb_pos)

    # Auto-compute spatial bandwidth for xy-proximity weighting:
    # use median nearest-neighbor distance × 3 as the Gaussian sigma
    _nn_dists, _ = kdtree.query(comb_pos, k=2)
    sigma_spatial = float(np.median(_nn_dists[:, 1])) * 5.0
    sigma_spatial = max(sigma_spatial, 1.0)

    # Pre-compute cosine similarities  (combined × virtual)
    if niche_desc_virtual is not None and comb_niche_desc is not None:
        cos_sims = cos_sim(comb_niche_desc, niche_desc_virtual)  # (R, V)
    else:
        cos_sims = None

    virt_X = np.zeros((n_virtual, n_genes), dtype=np.float32)
    donor_counts = np.zeros(n_virtual, dtype=int)

    rng = np.random.RandomState(42)

    for i in range(n_virtual):
        ct = virt_types[i]
        ni = niche_labels_virtual[i] if niche_labels_virtual is not None else -1

        # --- joint mask: same type AND same niche ---
        type_mask  = comb_types == ct
        if ni >= 0 and comb_niche_labels is not None:
            niche_mask = comb_niche_labels == ni
            joint_mask = type_mask & niche_mask
            # Relax to type-only if too few joint matches
            if joint_mask.sum() < 3:
                joint_mask = type_mask
        else:
            joint_mask = type_mask

        cands = np.where(joint_mask)[0]

        if len(cands) == 0:
            # Last resort: nearest neighbours regardless of type
            _, nn = kdtree.query(virt_pos[i], k=min(k_sam, len(combined)))
            nn = np.atleast_1d(nn)
            # Distance-weighted average for fallback too
            fb_dists = np.linalg.norm(comb_pos[nn] - virt_pos[i], axis=1)
            fb_w = np.exp(-0.5 * (fb_dists / sigma_spatial) ** 2)
            fb_w_sum = fb_w.sum()
            if fb_w_sum > 0:
                fb_w /= fb_w_sum
            else:
                fb_w = np.ones(len(nn)) / len(nn)
            virt_X[i] = (comb_X[nn] * fb_w[:, None]).sum(axis=0)
            donor_counts[i] = len(nn)
            continue

        donor_counts[i] = len(cands)

        # --- weights: z-proximity × niche similarity × spatial proximity ---
        w_z = z_weights[cands]

        if cos_sims is not None:
            w_niche = np.exp(Beta * cos_sims[cands, i])
        else:
            w_niche = np.ones(len(cands))

        # Spatial proximity: Gaussian kernel on xy-distance with floor
        xy_dists = np.linalg.norm(comb_pos[cands] - virt_pos[i], axis=1)
        w_spatial = np.exp(-0.5 * (xy_dists / sigma_spatial) ** 2)
        w_spatial = np.maximum(w_spatial, 1e-6)  # floor to avoid zeros

        w = w_z * w_niche * w_spatial
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(len(cands)) / len(cands)

        # --- whole-cell weighted average of sampled donors ---
        k_use = min(k_sam, len(cands))
        if len(cands) <= k_use:
            virt_X[i] = (comb_X[cands] * w[:, None]).sum(axis=0)
        else:
            # Select top-k by weight for deterministic, spatially-focused donors
            top_k = np.argpartition(w, -k_use)[-k_use:]
            sel = cands[top_k]
            sel_w = w[top_k]
            sel_w /= sel_w.sum()
            virt_X[i] = (comb_X[sel] * sel_w[:, None]).sum(axis=0)

    virtual_adata.X = virt_X

    if verbose:
        print(f"  Expression synthesized for {n_virtual} virtual cells "
              f"(median donor pool = {int(np.median(donor_counts))})")

    return virtual_adata, donor_counts


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
