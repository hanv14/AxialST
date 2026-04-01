"""
AxialST — Stage 4: Calibrated Uncertainty Quantification.

Five biologically interpretable uncertainty sources are computed per
virtual cell and combined into a single calibrated confidence score.

Sources
-------
1. Donor pool size       — few matched real cells → higher uncertainty
2. Niche ambiguity       — cell sits at a niche boundary
3. Cross-section disagreement — flanking sections disagree about this region
4. Z-axis distance       — virtual slice far from any real section
                           (gap-aware: scales with actual physical distance)
5. Donor expression variance — high variance among sampled donors → unreliable

The combined confidence is  σ(−(w·u + b))  where weights w and bias b
can be calibrated on held-out data (logistic regression).  Default
weights are empirically tuned for good discrimination.
"""

import numpy as np


# ===================================================================
# Individual uncertainty sources
# ===================================================================

def donor_pool_uncertainty(donor_counts, tau=5.0):
    """
    Source 1.  u₁ = exp(−|donors| / τ)

    Fewer matched donor cells → higher uncertainty.

    Uses effective sample size when weights are available: the
    effective count accounts for weight concentration (entropy).
    """
    return np.exp(-np.asarray(donor_counts, dtype=np.float64) / tau)


def effective_donor_pool_uncertainty(donor_counts, donor_variances,
                                     tau=5.0, var_scale=2.0):
    """
    Enhanced Source 1: combines donor pool size with donor expression
    variance to produce a more discriminative uncertainty.

    Effective uncertainty = pool_size_uncertainty × (1 + var_scale × variance)

    This penalises cells whose donors, even if numerous, disagree
    strongly in expression (high variance = less reliable average).
    """
    counts = np.asarray(donor_counts, dtype=np.float64)
    variances = np.asarray(donor_variances, dtype=np.float64)

    u_pool = np.exp(-counts / tau)

    # Normalise variance to [0, 1] range for stable combination
    v_max = variances.max() if variances.max() > 0 else 1.0
    v_norm = variances / v_max

    u_effective = u_pool * (1.0 + var_scale * v_norm)
    # Clip to [0, 1]
    return np.clip(u_effective, 0.0, 1.0)


def niche_ambiguity(niche_confidences):
    """
    Source 2.  u₂ = 1 − max_n P(niche = n)

    Low confidence in niche assignment → the cell is at a niche boundary.
    """
    return 1.0 - np.asarray(niche_confidences, dtype=np.float64)


def cross_section_disagreement(tp1, tp2, virtual_positions):
    """
    Source 3.  u₃ = JS( P(type | section k) ‖ P(type | section k+1) )

    High Jensen–Shannon divergence → the tissue is changing rapidly and
    interpolation is less reliable.
    """
    comp1 = tp1.eval_composition(virtual_positions)
    comp2 = tp2.eval_composition(virtual_positions)

    eps = 1e-12
    comp1 = comp1 + eps
    comp2 = comp2 + eps
    comp1 = comp1 / comp1.sum(axis=1, keepdims=True)
    comp2 = comp2 / comp2.sum(axis=1, keepdims=True)

    m = 0.5 * (comp1 + comp2)

    def _kl_row(p, q):
        return (p * np.log(p / q)).sum(axis=1)

    js = 0.5 * _kl_row(comp1, m) + 0.5 * _kl_row(comp2, m)
    return np.sqrt(np.clip(js, 0, None))          # JS distance (sqrt of div)


def z_distance_uncertainty(alpha, gap_distance=None, reference_gap=1.0):
    """
    Source 4 (scalar, broadcast to all cells).

    u₄ = min(α, 1−α) / 0.5  ×  gap_scale

    When *gap_distance* is provided (actual physical distance between
    sections), the uncertainty is scaled proportionally so that larger
    gaps between slices yield higher uncertainty:

        gap_scale = gap_distance / reference_gap

    Without gap_distance, gap_scale = 1.0 (backwards compatible).

    Maximum (= 1.0 × gap_scale) at the midpoint; minimum (→ 0) near
    a real section.
    """
    base = min(alpha, 1.0 - alpha) / 0.5
    if gap_distance is not None and reference_gap > 0:
        gap_scale = gap_distance / reference_gap
    else:
        gap_scale = 1.0
    return np.clip(base * gap_scale, 0.0, 1.0)


def donor_variance_uncertainty(donor_variances):
    """
    Source 5.  u₅ = normalised donor expression variance.

    High variance among the selected donor cells indicates that the
    weighted average is uncertain — the donors disagree about what
    expression this cell should have.

    Returns (V,) array in [0, 1].
    """
    v = np.asarray(donor_variances, dtype=np.float64)
    v_max = v.max() if v.max() > 0 else 1.0
    return v / v_max


# ===================================================================
# Combined calibrated confidence
# ===================================================================

def compute_confidence(u1, u2, u3, u4, u5=None,
                       weights=None, bias=0.0):
    """
    Combine uncertainty sources into a calibrated confidence.

    conf(i) = σ( −(w₁u₁ + w₂u₂ + w₃u₃ + w₄u₄ + w₅u₅ + b) )

    Parameters
    ----------
    u1 … u4 : (V,) arrays or scalars
    u5       : (V,) array or None — donor variance uncertainty
    weights  : length-4 or length-5 list/array (default: tuned weights)
    bias     : scalar

    Returns
    -------
    confidence : (V,) array in (0, 1)  — higher is more trustworthy
    """
    if weights is None:
        if u5 is not None:
            # 5-source weights: pool, niche, cross-sect, z-dist, donor-var
            weights = [1.0, 0.8, 1.2, 0.6, 0.9]
            bias = -0.3
        else:
            weights = [1.0, 0.8, 1.2, 0.5]
            bias = -0.5

    u1 = np.atleast_1d(np.asarray(u1, dtype=np.float64))
    u2 = np.atleast_1d(np.asarray(u2, dtype=np.float64))
    u3 = np.atleast_1d(np.asarray(u3, dtype=np.float64))

    # u4 may be scalar
    u4_arr = np.full_like(u1, float(u4))

    logit = -(weights[0] * u1
              + weights[1] * u2
              + weights[2] * u3
              + weights[3] * u4_arr
              + bias)

    if u5 is not None and len(weights) >= 5:
        u5 = np.atleast_1d(np.asarray(u5, dtype=np.float64))
        if len(u5) != len(u1):
            u5 = np.full_like(u1, float(u5.mean()))
        logit -= weights[4] * u5

    return _sigmoid(logit)


def _sigmoid(x):
    # Numerically stable sigmoid
    pos = x >= 0
    z = np.empty_like(x)
    z[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x   = np.exp(x[~pos])
    z[~pos] = exp_x / (1.0 + exp_x)
    return z


# ===================================================================
# Uncertainty calibration helper (optional, leave-one-section-out)
# ===================================================================

def calibrate_weights(held_out_results):
    """
    Fit logistic-regression weights on held-out validation data.

    Parameters
    ----------
    held_out_results : list of dicts, each with keys
        'u1', 'u2', 'u3', 'u4'  — uncertainty arrays for the virtual cells
        'u5'                     — (optional) donor variance uncertainty
        'accuracy'               — binary: 1 if cell-type prediction correct

    Returns
    -------
    weights : (4,) or (5,) array
    bias    : float
    """
    from sklearn.linear_model import LogisticRegression

    Xs, ys = [], []
    has_u5 = 'u5' in held_out_results[0]
    for r in held_out_results:
        n = len(r['u1'])
        cols = [r['u1'], r['u2'], r['u3'], np.full(n, r['u4'])]
        if has_u5:
            cols.append(r['u5'])
        X_block = np.column_stack(cols)
        Xs.append(X_block)
        ys.append(r['accuracy'])

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_all, y_all)

    return clf.coef_[0], clf.intercept_[0]
