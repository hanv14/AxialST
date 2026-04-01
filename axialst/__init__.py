"""
AxialST — Niche-Coherent Virtual Slice Generation
==================================================

Top-down generation of virtual tissue slices for 3-D spatial cell atlases.

Public API
----------
Generate_axialst
    Synthesize a single virtual slice between two real sections.
Generate_multiple_axialst
    Generate several virtual slices between one pair of sections.
Generate_multiple_slices_axialst
    Process a list of consecutive sections, generating virtual slices
    for every adjacent pair and (optionally) saving to disk.

Quick start
-----------
>>> from axialst import Generate_multiple_slices_axialst
>>> adatas = Generate_multiple_slices_axialst(
...     adata_list, num_sim_list, adatas_id_list,
...     save_path='./output', cell_type_key='cell_class')
"""

from __future__ import annotations

import os
import time
import warnings
import contextlib
import io

import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

# Internal modules
from .utils import (get_unified_type_list, safe_to_dense,
                    auto_sigma, auto_patch_size,
                    compute_niche_descriptors_simple)
from .tissue_programs import (extract_tissue_program,
                              interpolate_tissue_programs,
                              run_mender)
from .patch_generation import generate_virtual_cells
from .expression import (synthesize_expression_default,
                          synthesize_expression_fast)
from .uncertainty import (donor_pool_uncertainty,
                           effective_donor_pool_uncertainty,
                           niche_ambiguity,
                           cross_section_disagreement,
                           z_distance_uncertainty,
                           donor_variance_uncertainty,
                           compute_confidence)


# Re-export so  ``from axialst import *``  works
__all__ = [
    'Generate_axialst',
    'Generate_multiple_axialst',
    'Generate_multiple_slices_axialst',
]


# ===================================================================
#  Helper: run MENDER safely (returns None on failure)
# ===================================================================

def _try_mender(adata_list, cell_type_key, batch_labels,
                scale=6, radius=15, micro_env_key='mender',
                verbose=True):
    """Run MENDER; on import / runtime failure, return None."""
    try:
        descs = run_mender(adata_list, cell_type_key, batch_labels,
                           scale=scale, radius=radius,
                           micro_env_key=micro_env_key)
        if verbose:
            print("  MENDER niche descriptors computed.")
        return descs
    except ImportError:
        if verbose:
            print("  MENDER not installed — using fallback niche descriptors.")
        return None
    except Exception as exc:
        if verbose:
            print(f"  MENDER failed ({exc}) — using fallback niche descriptors.")
        return None


# ===================================================================
#  Core: single virtual slice
# ===================================================================

def Generate_axialst(
    adata1, adata2,
    adata1_id: str = 'above',
    adata2_id: str = 'below',
    alpha: float = 0.5,
    cell_type_key: str = 'cell_type',
    # Stage-1 params
    sigma: float | None = None,
    n_niches: int = 20,
    niche_radius: float = 50.0,
    # Stage-2 params
    patch_size: float | None = None,
    n_cell: int | None = None,
    n_mag: float = 1.0,
    # Stage-3 params
    syn_mode: str = 'default',
    k_sam: int = 10,
    Beta: float = 100.0,
    micro_env_key: str = 'mender',
    smooth_k: int = 6,
    smooth_sigma: float = 1.0,
    smooth_alpha: float = 0.15,
    # Stage-4 params
    compute_uncertainty: bool = True,
    gap_distance: float | None = None,
    reference_gap: float = 1.0,
    # Misc
    add_obs_list: list | None = None,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Generate one virtual slice between two real AnnData sections.

    Parameters mirror SpatialZ's ``Generate_spatialz`` where applicable
    so that the two methods can be swapped with minimal code changes.

    Parameters
    ----------
    adata1, adata2 : AnnData
        Two flanking real sections.  Must contain ``obsm['spatial']``
        (2-D coordinates) and ``obs[cell_type_key]``.
    adata1_id, adata2_id : str
        Unique identifiers appended to obs_names to avoid collisions.
    alpha : float in (0, 1)
        Interpolation parameter.  0 → adata2, 1 → adata1.
    cell_type_key : str
        Column in ``obs`` containing cell-type labels.
    sigma : float or None
        KDE bandwidth.  None → auto-detect from data.
    n_niches : int
        Number of niche archetypes per section.
    niche_radius : float
        Radius for fallback (non-MENDER) niche descriptors.
    patch_size : float or None
        Side-length of Voronoi patches.  None → auto-detect.
    n_cell : int or None
        Target number of virtual cells.  None → interpolated from
        adata1 / adata2 cell counts × *n_mag*.
    n_mag : float
        Cell-density magnification factor.
    syn_mode : {'default', 'fast'}
        Expression synthesis mode.
    k_sam : int
        Number of donor cells to sample per virtual cell.
    Beta : float
        Temperature for niche-similarity weighting.
    micro_env_key : str
        Key in ``obsm`` for MENDER embeddings.
    smooth_k : int
        Number of spatial neighbours for post-synthesis expression
        smoothing.  Set to 0 to disable.
    smooth_sigma : float
        Bandwidth multiplier for spatial smoothing Gaussian kernel.
    smooth_alpha : float
        Blend factor for spatial smoothing (0 = no smoothing,
        1 = full replacement).  Default 0.15 gives a gentle nudge
        that reduces noise without overwriting spatial structure.
    compute_uncertainty : bool
        Whether to compute Stage-4 confidence scores.
    gap_distance : float or None
        Physical z-distance between the two flanking sections.  When
        provided, the z-axis uncertainty is scaled proportionally so
        that larger gaps yield higher uncertainty.  None = uniform.
    reference_gap : float
        Baseline gap distance used to normalise ``gap_distance``.
    add_obs_list : list of str or None
        Additional ``obs`` columns to transfer from real to virtual cells.
    seed : int
        Random seed.
    verbose : bool
        Print progress messages.

    Returns
    -------
    AnnData
        Virtual slice with ``obsm['spatial']``, ``obs[cell_type_key]``,
        filled ``X``, and (optionally) ``obs['confidence']`` and
        individual uncertainty columns.
    """

    def _ptime(msg, t0):
        if verbose:
            print(f"  {msg}: {time.time() - t0:.1f}s")

    # --- input validation -------------------------------------------
    if 'spatial' not in adata1.obsm or 'spatial' not in adata2.obsm:
        raise ValueError("Both adata objects must have obsm['spatial'].")
    if cell_type_key not in adata1.obs or cell_type_key not in adata2.obs:
        raise ValueError(f"Both adata objects must have obs['{cell_type_key}'].")
    if syn_mode not in ('default', 'fast'):
        raise ValueError("syn_mode must be 'default' or 'fast'.")

    # Work on copies so we don't mutate the caller's data
    adata1 = adata1.copy()
    adata2 = adata2.copy()
    adata1.obs_names = [f"{n}_{adata1_id}" for n in adata1.obs_names]
    adata2.obs_names = [f"{n}_{adata2_id}" for n in adata2.obs_names]

    # --- auto-detect hyper-parameters --------------------------------
    type_list = get_unified_type_list(adata1, adata2,
                                      cell_type_key=cell_type_key)
    if sigma is None:
        sigma = auto_sigma(np.vstack([adata1.obsm['spatial'],
                                       adata2.obsm['spatial']]))
        if verbose:
            print(f"  Auto sigma = {sigma:.1f}")

    if patch_size is None:
        patch_size = auto_patch_size(
            np.vstack([adata1.obsm['spatial'], adata2.obsm['spatial']]))
        if verbose:
            print(f"  Auto patch_size = {patch_size:.1f}")

    # =================================================================
    # Stage 1 — Tissue programme extraction & interpolation
    # =================================================================
    if verbose:
        print("Stage 1: Extracting tissue programs …")
    t0 = time.time()

    # Try MENDER for niche descriptors
    mender_descs = None
    if syn_mode == 'default':
        mender_descs = _try_mender(
            [adata1, adata2], cell_type_key,
            ['sec_k', 'sec_k1'],
            micro_env_key=micro_env_key, verbose=verbose)

    tp1 = extract_tissue_program(
        adata1, cell_type_key, type_list, sigma, n_niches,
        niche_radius=niche_radius,
        niche_descriptors=mender_descs[0] if mender_descs else None,
        verbose=verbose)

    tp2 = extract_tissue_program(
        adata2, cell_type_key, type_list, sigma, n_niches,
        niche_radius=niche_radius,
        niche_descriptors=mender_descs[1] if mender_descs else None,
        verbose=verbose)

    interp_tp = interpolate_tissue_programs(tp1, tp2, alpha, verbose=verbose)
    _ptime("Stage 1 total", t0)

    # =================================================================
    # Stage 2 — Niche-coherent patch generation
    # =================================================================
    if verbose:
        print("Stage 2: Generating virtual cells (patches) …")
    t0 = time.time()

    (positions, cell_types_arr, niche_assignments,
     patch_ids, niche_confidences) = generate_virtual_cells(
        interp_tp, patch_size, n_cell=n_cell,
        n_mag=n_mag, seed=seed, verbose=verbose)

    n_virtual = len(positions)
    if n_virtual == 0:
        warnings.warn("No virtual cells generated — check data overlap.")
        return AnnData(X=np.empty((0, adata1.n_vars), dtype=np.float32))

    _ptime("Stage 2 total", t0)

    # Build virtual AnnData skeleton
    var_data = pd.DataFrame(index=adata1.var_names)
    adata3 = AnnData(
        X=np.zeros((n_virtual, adata1.n_vars), dtype=np.float32),
        var=var_data)
    adata3.obsm['spatial'] = positions.astype(np.float32)
    adata3.obs[cell_type_key] = cell_types_arr
    adata3.obs['niche_archetype'] = niche_assignments
    adata3.obs['patch_id'] = patch_ids

    # =================================================================
    # Stage 3 — Expression materialisation
    # =================================================================
    if verbose:
        print("Stage 3: Synthesizing gene expression …")
    t0 = time.time()

    if syn_mode == 'default':
        # Prepare niche descriptors for virtual cells
        # Attempt MENDER on (real + virtual); fallback to composition-based
        niche_desc_virtual = None
        niche_desc_refs = None

        mender3 = _try_mender(
            [adata1, adata2, adata3], cell_type_key,
            ['sec_k', 'sec_k1', 'virtual'],
            micro_env_key=micro_env_key, verbose=verbose)

        if mender3 is not None:
            niche_desc_refs = [mender3[0], mender3[1]]
            niche_desc_virtual = mender3[2]
        else:
            # Fallback: use composition-based descriptors
            niche_desc_refs = [
                compute_niche_descriptors_simple(
                    np.asarray(ad.obsm['spatial']),
                    np.asarray(ad.obs[cell_type_key].values),
                    type_list, radius=niche_radius)
                for ad in [adata1, adata2]]
            niche_desc_virtual = compute_niche_descriptors_simple(
                positions, cell_types_arr, type_list, radius=niche_radius)

        niche_labels_refs = [tp1.niche_labels, tp2.niche_labels]

        adata3, donor_counts, donor_variances = synthesize_expression_default(
            [adata1, adata2], adata3, cell_type_key,
            niche_labels_refs, niche_assignments,
            niche_desc_refs, niche_desc_virtual,
            alpha, k_sam=k_sam, Beta=Beta,
            smooth_k=smooth_k, smooth_sigma=smooth_sigma,
            smooth_alpha=smooth_alpha, verbose=verbose)

    else:  # fast
        adata3 = synthesize_expression_fast(
            [adata1, adata2], adata3, cell_type_key,
            k_sam=k_sam, verbose=verbose)
        donor_counts = np.full(n_virtual, k_sam)
        donor_variances = np.zeros(n_virtual, dtype=np.float32)

    _ptime("Stage 3 total", t0)

    # =================================================================
    # Transfer additional obs annotations
    # =================================================================
    if add_obs_list:
        if verbose:
            print("Transferring additional annotations …")
        from sklearn.neighbors import NearestNeighbors
        combined_pos = np.vstack([adata1.obsm['spatial'],
                                   adata2.obsm['spatial']])
        nn = NearestNeighbors(n_neighbors=1).fit(combined_pos)
        _, nn_idx = nn.kneighbors(positions)
        nn_idx = nn_idx.ravel()

        n1 = len(adata1)
        for obs_key in add_obs_list:
            vals_combined = np.concatenate([
                adata1.obs[obs_key].values,
                adata2.obs[obs_key].values])
            adata3.obs[obs_key] = vals_combined[nn_idx]

    # =================================================================
    # Stage 4 — Uncertainty quantification
    # =================================================================
    if compute_uncertainty:
        if verbose:
            print("Stage 4: Computing uncertainty scores …")
        t0 = time.time()

        # Enhanced donor pool uncertainty (combines count + variance)
        u1 = effective_donor_pool_uncertainty(donor_counts, donor_variances)
        u2 = niche_ambiguity(niche_confidences)
        u3 = cross_section_disagreement(tp1, tp2, positions)
        u4 = z_distance_uncertainty(alpha, gap_distance=gap_distance,
                                     reference_gap=reference_gap)
        u5 = donor_variance_uncertainty(donor_variances)
        conf = compute_confidence(u1, u2, u3, u4, u5=u5)

        adata3.obs['confidence']    = conf
        adata3.obs['u_donor_pool']  = u1
        adata3.obs['u_niche_ambig'] = u2
        adata3.obs['u_cross_sect']  = u3
        adata3.obs['u_z_dist']      = float(u4)
        adata3.obs['u_donor_var']   = u5

        _ptime("Stage 4 total", t0)

    if verbose:
        print(f"Done — virtual slice: {adata3.n_obs} cells, "
              f"{adata3.n_vars} genes.")
    return adata3


# ===================================================================
#  Convenience: multiple slices between one pair
# ===================================================================

def Generate_multiple_axialst(
    adata1, adata2,
    num_sim: int,
    adata1_id: str = 'above',
    adata2_id: str = 'below',
    include_raw: bool = True,
    verbose: bool = True,
    **kwargs,
):
    """
    Generate *num_sim* virtual slices between *adata1* and *adata2*
    by varying alpha from near-1 to near-0.

    Keyword arguments are forwarded to ``Generate_axialst``.

    Returns
    -------
    AnnData  — concatenation of all virtual (and optionally real) slices.
    """
    parts = []
    num_steps = num_sim + 1

    if include_raw:
        a1 = adata1.copy()
        a1.obs['slice_id']  = adata1_id
        a1.obs['data_type'] = 'real'
        parts.append(a1)

    for i in tqdm(range(1, num_steps), desc="AxialST simulations"):
        alpha = 1 - i / num_steps
        sim = Generate_axialst(
            adata1, adata2,
            adata1_id=adata1_id, adata2_id=adata2_id,
            alpha=alpha, verbose=verbose, **kwargs)

        sid = f"{adata1_id}-{adata2_id}-{i}"
        sim.obs['slice_id']  = sid
        sim.obs['data_type'] = 'synthetic'
        parts.append(sim)
        if verbose:
            print(f"  Completed {sid}\n")

    if include_raw:
        a2 = adata2.copy()
        a2.obs['slice_id']  = adata2_id
        a2.obs['data_type'] = 'real'
        parts.append(a2)

    import anndata
    result = anndata.concat(parts, label='slice_id',
                            keys=[p.obs['slice_id'].iloc[0] for p in parts])
    return result


# ===================================================================
#  Convenience: full pipeline over a list of sections
# ===================================================================

def Generate_multiple_slices_axialst(
    adata_list: list,
    num_sim_list: list,
    adatas_id_list: list,
    save_path: str,
    z_positions: list | None = None,
    include_raw: bool = True,
    verbose: bool = True,
    **kwargs,
):
    """
    Process a list of consecutive real sections, generating virtual
    slices for every adjacent pair.

    Parameters
    ----------
    adata_list    : [AnnData, …]   — K consecutive real sections.
    num_sim_list  : [int, …]       — length K−1; virtual slices per gap.
    adatas_id_list: [str, …]       — length K; unique id per section.
    save_path     : str            — directory for .h5ad output files.
    z_positions   : [float, …] or None — length K; physical z-coordinate
                    of each section.  When provided, ``gap_distance`` and
                    ``num_sim`` (if 'auto') are computed from actual
                    inter-section distances, so that thicker gaps get
                    proportionally more virtual slices and higher
                    uncertainty.  None = uniform spacing assumed.
    include_raw   : bool           — include original sections in output.
    **kwargs      : forwarded to ``Generate_axialst``.

    Returns
    -------
    AnnData — concatenation of all real + synthetic slices.
    """
    os.makedirs(save_path, exist_ok=True)
    all_parts = []

    # Compute per-gap distances and reference (median) gap
    if z_positions is not None:
        z_pos = np.asarray(z_positions, dtype=np.float64)
        gap_distances = np.diff(z_pos)           # (K-1,)
        reference_gap = float(np.median(gap_distances))
        if reference_gap <= 0:
            reference_gap = 1.0
        if verbose:
            print(f"  z_positions provided: gaps = {gap_distances}, "
                  f"reference_gap = {reference_gap:.2f}")
    else:
        gap_distances = None
        reference_gap = 1.0

    # Optionally include raw data
    if include_raw:
        for adata, sid in zip(adata_list, adatas_id_list):
            ad = adata.copy()
            ad.obs['slice_id']  = sid
            ad.obs['data_type'] = 'real'
            all_parts.append(ad)
            ad.write(os.path.join(save_path, f"{sid}_raw.h5ad"))

    # Generate virtual slices for each pair
    for i in tqdm(range(len(adata_list) - 1),
                  desc="Generating slices for pairs"):
        n_sim    = num_sim_list[i]
        ad1      = adata_list[i]
        ad2      = adata_list[i + 1]
        id1      = adatas_id_list[i]
        id2      = adatas_id_list[i + 1]

        # Pass gap information through to uncertainty
        pair_kwargs = dict(kwargs)
        if gap_distances is not None:
            pair_kwargs['gap_distance'] = float(gap_distances[i])
            pair_kwargs['reference_gap'] = reference_gap

        for j in range(1, n_sim + 1):
            alpha = 1 - j / (n_sim + 1)
            sim = Generate_axialst(
                ad1, ad2,
                adata1_id=id1, adata2_id=id2,
                alpha=alpha, verbose=verbose, **pair_kwargs)

            sid = f"{id1}-{id2}-{j}"
            sim.obs['slice_id']  = sid
            sim.obs['data_type'] = 'synthetic'
            all_parts.append(sim)
            sim.write(os.path.join(save_path, f"{sid}.h5ad"))
            if verbose:
                print(f"  Saved {sid}.h5ad\n")

    # Concatenate everything
    import anndata
    result = anndata.concat(all_parts, label='slice_id',
                            keys=[p.obs['slice_id'].iloc[0] for p in all_parts])
    return result
