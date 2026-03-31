"""
AxialST — Stage 4: Calibrated Uncertainty Quantification.

Four biologically interpretable uncertainty sources are computed per
virtual cell and combined into a single calibrated confidence score.

Sources
-------
1. Donor pool size       — few matched real cells → higher uncertainty
2. Niche ambiguity       — cell sits at a niche boundary
3. Cross-section disagreement — flanking sections disagree about this region
4. Z-axis distance       — virtual slice far from any real section

The combined confidence is  σ(−(w·u + b))  where weights w and bias b
can be calibrated on held-out data (logistic regression).  Default
weights treat all sources equally.
"""

import numpy as np


# ===================================================================
# Individual uncertainty sources
# ===================================================================

def donor_pool_uncertainty(donor_counts, tau=10.0):
    """
    Source 1.  u₁ = exp(−|donors| / τ)

    Fewer matched donor cells → higher uncertainty.
    """
    return np.exp(-np.asarray(donor_counts, dtype=np.float64) / tau)


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


def z_distance_uncertainty(alpha):
    """
    Source 4 (scalar, broadcast to all cells).

    u₄ = min(α, 1−α) / 0.5

    Maximum (= 1.0) at the midpoint; minimum (→ 0) near a real section.
    """
    return min(alpha, 1.0 - alpha) / 0.5


# ===================================================================
# Combined calibrated confidence
# ===================================================================

def compute_confidence(u1, u2, u3, u4,
                       weights=None, bias=0.0):
    """
    Combine four uncertainty sources into a calibrated confidence.

    conf(i) = σ( −(w₁ u₁ + w₂ u₂ + w₃ u₃ + w₄ u₄ + b) )

    Parameters
    ----------
    u1 … u4 : (V,) arrays or scalars
    weights  : length-4 list/array  (default: equal weights = 1)
    bias     : scalar

    Returns
    -------
    confidence : (V,) array in (0, 1)  — higher is more trustworthy
    """
    if weights is None:
        # Default weights chosen so that typical uncertainty levels
        # produce a confidence spread centred near 0.5 – 0.7.
        # These should be calibrated with ``calibrate_weights`` for
        # production use.
        weights = [0.5, 0.5, 0.5, 0.3]
        bias = -1.0            # shift baseline upward

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
        'accuracy'               — binary: 1 if cell-type prediction correct

    Returns
    -------
    weights : (4,) array
    bias    : float
    """
    from sklearn.linear_model import LogisticRegression

    Xs, ys = [], []
    for r in held_out_results:
        n = len(r['u1'])
        X_block = np.column_stack([r['u1'], r['u2'], r['u3'],
                                    np.full(n, r['u4'])])
        Xs.append(X_block)
        ys.append(r['accuracy'])

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_all, y_all)

    return clf.coef_[0], clf.intercept_[0]
