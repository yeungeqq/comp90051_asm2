"""training: calibration for the Amazon robustness experiment."""

import numpy as np
from scipy.special import logsumexp


def original_linear_quantile_in_log_space(stable_anomaly_scores, q):
    """Preserve original-score linear-quantile interpolation in exact arithmetic.

    stable_anomaly_scores are a=-log(S), ordered in ascending anomaly direction.
    If h=(n-1)q=i+w, the original linear quantile of rho-S is rho-S_threshold,
    where S_threshold=(1-w)exp(-a_i)+w*exp(-a_(i+1)). Its stable counterpart is
    -logsumexp(log(1-w)-a_i, log(w)-a_(i+1)). The unknown rho cancels.

    This is not np.quantile(a,q), which generally changes threshold decisions.
    It preserves the underlying real-valued interpolation, not ties created by
    an already rounded legacy decision_function. Keep the original strict >
    comparator. A constant normalization shift in a also shifts this threshold.
    """
    scores = np.asarray(stable_anomaly_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration requires a nonempty finite score vector.")
    if not np.isscalar(q) or not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be a finite scalar in [0,1].")
    ordered = np.sort(scores)
    position = (len(ordered) - 1) * float(q)
    low = int(np.floor(position))
    high = int(np.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(-logsumexp([
        np.log1p(-weight) - ordered[low],
        np.log(weight) - ordered[high],
    ]))
