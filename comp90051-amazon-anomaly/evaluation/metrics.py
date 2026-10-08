"""evaluation: metrics for the Amazon robustness experiment."""

import numpy as np
from config import GROUPS, METRIC_TARGET_TAGS


def metric_tag_set(value):
    """Support the CSV's space-separated tags and an already-parsed tag set."""
    if isinstance(value, (set, frozenset, list, tuple)):
        return set(value)
    return set(str(value).split())


def metric_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def average_precision(y_true, scores):
    """Threshold-based AP from scratch, handling tied scores together.

    AP = sum over DISTINCT score thresholds of precision * change in recall.
    Breaking score ties one observation at a time changes the answer arbitrarily.
    This is not trapezoidal area under a precision-recall curve.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s) or y.ndim != 1 or s.ndim != 1:
        raise ValueError("Labels and scores must be aligned 1D arrays.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("AP requires binary labels and finite scores.")
    n_positive = int(y.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ranked_y, ranked_s = (y[order], s[order])
    ends = np.r_[np.flatnonzero(ranked_s[:-1] != ranked_s[1:]), len(y) - 1]
    tp = np.cumsum(ranked_y)[ends].astype(float)
    precision = tp / (ends + 1)
    recall = tp / n_positive
    return float(np.sum(precision * np.diff(np.r_[0.0, recall])))


def scratch_binary_metrics(groups, scores, flags, tags=None):
    """P is positive; R and N are proxy negatives defined by dataset tags.

    N's rate is best described as a 'comparator alert rate': absent human tags do
    not prove that an image is ecologically pristine.  We retain FPR_N as a short
    table column name, with this qualification in the notebook and report.
    """
    g = np.asarray(groups)
    s = np.asarray(scores, dtype=float)
    f = np.asarray(flags, dtype=bool)
    if not len(g) == len(s) == len(f) or not np.isin(g, GROUPS).all():
        raise ValueError("Metrics require aligned arrays with R/P/N groups.")
    positive, negative = (g == "P", g != "P")
    tp = int(np.sum(f & positive))
    fp = int(np.sum(f & negative))
    result = {
        "n": len(g),
        "n_P": int(positive.sum()),
        "n_R": int((g == "R").sum()),
        "n_N": int((g == "N").sum()),
        "tp": tp,
        "fp": fp,
        "prevalence": metric_ratio(int(positive.sum()), len(g)),
        "AP": average_precision(positive.astype(int), s),
        "precision": metric_ratio(tp, int(f.sum())),
        "recall_P": metric_ratio(tp, int(positive.sum())),
        "FPR_R": metric_ratio(int(np.sum(f & (g == "R"))), int((g == "R").sum())),
        "FPR_N": metric_ratio(int(np.sum(f & (g == "N"))), int((g == "N").sum())),
    }
    if tags is not None:
        if len(tags) != len(g):
            raise ValueError("Tags must align with metric rows.")
        tag_sets = [metric_tag_set(t) for t in tags]
        for tag in METRIC_TARGET_TAGS:
            mask = positive & np.array([tag in ts for ts in tag_sets])
            result[f"n_{tag}"] = int(mask.sum())
            result[f"recall_{tag}"] = metric_ratio(
                int(np.sum(f & mask)), int(mask.sum())
            )
    return result
