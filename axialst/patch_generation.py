"""
AxialST — Stage 2: Niche-Coherent Patch Generation.

Turn the interpolated tissue-program blueprint into concrete virtual cells.
The slice is divided into spatial patches; each patch is assigned a niche
archetype and a cell-type distribution.  Cells are then materialised inside
every patch using template-based placement learned from real sections.
"""

import numpy as np
from scipy.spatial import cKDTree


# ===================================================================
# Patch grid
# ===================================================================

def compute_tissue_bbox(tp1, tp2, pad_fraction=0.02):
    """Bounding box that covers both real sections, with small padding."""
    all_pos = np.vstack([tp1.positions, tp2.positions])
    lo = all_pos.min(axis=0)
    hi = all_pos.max(axis=0)
    span = hi - lo
    pad = span * pad_fraction
    return (lo[0] - pad[0], hi[0] + pad[0],
            lo[1] - pad[1], hi[1] + pad[1])


def create_patch_grid(bbox, patch_size):
    """Regular grid of patch centres inside *bbox*."""
    x_min, x_max, y_min, y_max = bbox
    xs = np.arange(x_min + patch_size / 2, x_max, patch_size)
    ys = np.arange(y_min + patch_size / 2, y_max, patch_size)
    if len(xs) == 0:
        xs = np.array([(x_min + x_max) / 2])
    if len(ys) == 0:
        ys = np.array([(y_min + y_max) / 2])
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


# ===================================================================
# Spatial templates from real sections
# ===================================================================

def build_niche_templates(tp1, tp2, interp_tp, patch_size, max_exemplars=50):
    """
    For every (interpolated) niche archetype, collect relative cell
    positions from matching real patches in both flanking sections.

    Returns
    -------
    templates : dict  niche_idx → {type_idx: [(dx, dy), …]}
    """
    rng = np.random.RandomState(0)
    half = patch_size / 2.0
    templates = {}

    for k, (ni1, ni2) in enumerate(interp_tp.niche_matching):
        rel = {}    # type_idx → list of (dx, dy)

        if ni1 < tp1.n_niches:
            _collect_relative_positions(tp1, ni1, rel, half,
                                        max_exemplars, rng)
        if ni2 < tp2.n_niches:
            _collect_relative_positions(tp2, ni2, rel, half,
                                        max_exemplars, rng)
        templates[k] = rel

    return templates


def _collect_relative_positions(tp, niche_idx, rel_dict, half_patch,
                                max_exemplars, rng):
    """
    Gather relative (dx, dy) offsets for each cell type from real patches
    of a given niche archetype.
    """
    type_to_idx = {t: i for i, t in enumerate(tp.type_list)}
    niche_cells = np.where(tp.niche_labels == niche_idx)[0]
    if len(niche_cells) == 0:
        return

    tree = cKDTree(tp.positions)

    # Use a subsample of niche cells as pseudo-patch centres
    n_sample = min(len(niche_cells), max_exemplars)
    centres = rng.choice(niche_cells, size=n_sample, replace=False)

    for ci in centres:
        cx, cy = tp.positions[ci]
        nbrs = tree.query_ball_point(tp.positions[ci], r=half_patch)
        for j in nbrs:
            dx = tp.positions[j, 0] - cx
            dy = tp.positions[j, 1] - cy
            t_idx = type_to_idx.get(tp.cell_types[j])
            if t_idx is not None:
                rel_dict.setdefault(t_idx, []).append((dx, dy))


# ===================================================================
# Cell materialisation inside patches
# ===================================================================

def place_cells_in_patch(centre, n_cells, type_indices, niche_id,
                         templates, patch_size, rng):
    """
    Position *n_cells* inside a patch centred at *centre* using a
    template-based strategy.

    Falls back to uniform-random placement when no template exists.
    """
    half = patch_size / 2.0
    positions = np.empty((n_cells, 2), dtype=np.float64)
    template = templates.get(niche_id, {})
    jitter_std = patch_size * 0.05

    for i in range(n_cells):
        offsets = template.get(type_indices[i])
        if offsets and len(offsets) > 0:
            idx = rng.randint(0, len(offsets))
            dx, dy = offsets[idx]
            noise = rng.normal(0, jitter_std, size=2)
            positions[i] = centre + np.array([dx, dy]) + noise
        else:
            positions[i] = centre + rng.uniform(-half, half, size=2)

    return positions


# ===================================================================
# Main entry: generate all virtual cells
# ===================================================================

def generate_virtual_cells(interp_tp, patch_size, n_cell=None,
                           n_mag=1.0, seed=42, verbose=True):
    """
    Stage 2 entry point.

    Parameters
    ----------
    interp_tp   : InterpolatedTissueProgram
    patch_size   : side length of each square patch (in coordinate units)
    n_cell       : target total number of virtual cells (None = auto)
    n_mag        : magnification factor for cell density
    seed         : random seed
    verbose      : print progress

    Returns
    -------
    positions          : (V, 2) float
    cell_types         : (V,)   str labels
    niche_assignments  : (V,)   int   — niche archetype per cell
    patch_ids          : (V,)   int   — which patch each cell belongs to
    niche_confidences  : (V,)   float — cosine-similarity confidence
    """
    rng = np.random.RandomState(seed)
    tp1, tp2 = interp_tp.tp1, interp_tp.tp2
    alpha = interp_tp.alpha
    type_list = interp_tp.type_list

    # ---- patch grid --------------------------------------------------
    bbox = compute_tissue_bbox(tp1, tp2)
    centres = create_patch_grid(bbox, patch_size)

    # ---- evaluate tissue programme at patch centres ------------------
    compositions = interp_tp.eval_composition(centres)    # (P, T)
    densities    = interp_tp.eval_density(centres)        # (P,)

    # Filter empty patches (far outside tissue)
    thresh = densities.max() * 0.01 if densities.max() > 0 else 0
    active = densities > thresh
    centres      = centres[active]
    compositions = compositions[active]
    densities    = densities[active]

    if verbose:
        print(f"  {len(centres)} active patches (of "
              f"{active.size} grid cells, patch_size={patch_size:.0f})")

    if len(centres) == 0:
        return (np.empty((0, 2)), np.array([]), np.array([]),
                np.array([]), np.array([]))

    # ---- niche assignment per patch ----------------------------------
    niche_assign, niche_conf = interp_tp.assign_niche(compositions)

    # ---- cell counts per patch ---------------------------------------
    # Target total
    if n_cell is not None:
        expected_total = float(n_cell)
    else:
        expected_total = (alpha * len(tp1.positions)
                          + (1 - alpha) * len(tp2.positions)) * n_mag

    patch_area = patch_size ** 2
    raw_counts = densities * patch_area        # proportional counts
    raw_sum = raw_counts.sum()
    if raw_sum > 0:
        raw_counts = raw_counts * expected_total / raw_sum

    cell_counts = rng.poisson(np.clip(raw_counts, 0.1, None))
    cell_counts = np.maximum(cell_counts, 0).astype(int)

    if verbose:
        print(f"  Target {expected_total:.0f} cells → "
              f"Poisson-sampled {cell_counts.sum()} cells")

    # ---- spatial templates -------------------------------------------
    templates = build_niche_templates(tp1, tp2, interp_tp, patch_size)

    # ---- materialise cells per patch ---------------------------------
    all_pos, all_types = [], []
    all_niche, all_patch, all_conf = [], [], []

    type_to_idx = {t: i for i, t in enumerate(type_list)}

    for p_idx in range(len(centres)):
        nc = cell_counts[p_idx]
        if nc == 0:
            continue

        comp = compositions[p_idx]
        comp = np.clip(comp, 0, None)
        comp_sum = comp.sum()
        if comp_sum > 0:
            comp /= comp_sum
        else:
            comp = np.ones(len(type_list)) / len(type_list)

        # Sample cell types from multinomial
        type_idx = rng.choice(len(type_list), size=nc, p=comp)
        types    = [type_list[ti] for ti in type_idx]

        # Place cells
        pos = place_cells_in_patch(
            centres[p_idx], nc, type_idx, niche_assign[p_idx],
            templates, patch_size, rng)

        all_pos.append(pos)
        all_types.extend(types)
        all_niche.extend([niche_assign[p_idx]] * nc)
        all_patch.extend([p_idx] * nc)
        all_conf.extend([niche_conf[p_idx]] * nc)

    if len(all_pos) == 0:
        return (np.empty((0, 2)), np.array([]), np.array([]),
                np.array([]), np.array([]))

    return (np.vstack(all_pos),
            np.array(all_types),
            np.array(all_niche, dtype=int),
            np.array(all_patch, dtype=int),
            np.array(all_conf, dtype=np.float64))
