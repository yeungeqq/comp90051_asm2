"""Check AP ties, proxy-label metrics, and stable OCSVM threshold interpolation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from evaluation.aggregation import summarize_condition_results
from evaluation.metrics import average_precision, scratch_binary_metrics
from training.calibration import original_linear_quantile_in_log_space


def test_ap_ties_match_sklearn_and_are_order_independent():
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([3, 3, 2, 2, 1, 1])
    expected = average_precision_score(labels, scores)
    assert average_precision(labels, scores) == pytest.approx(expected)
    assert average_precision(labels[::-1], scores[::-1]) == pytest.approx(expected)
    assert np.isnan(average_precision([0, 0], [1, 2]))
    with pytest.raises(ValueError):
        average_precision([1, 0], [np.nan, 1])


def test_metrics_use_correct_group_denominators():
    metrics = scratch_binary_metrics(
        ["P", "P", "R", "R", "N"], [5, 1, 4, 0, 3], [True, False, True, False, True],
        ["clear road", "clear slash_burn", "clear primary", "clear primary", "clear primary water"])
    assert metrics["recall_P"] == 0.5
    assert metrics["FPR_R"] == 0.5
    assert metrics["FPR_N"] == 1
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall_road"] == 1
    assert metrics["recall_slash_burn"] == 0
    assert np.isnan(metrics["recall_selective_logging"])


@pytest.mark.parametrize("q", [0, 0.25, 0.95, 1])
def test_ocsvm_quantile_matches_original_score_alerts(q):
    kernel_sum = np.array([0.02, 0.1, 0.3, 0.8, 1.2])
    stable = -np.log(kernel_sum)
    threshold = original_linear_quantile_in_log_space(stable, q)
    original = 2 - kernel_sum
    np.testing.assert_array_equal(stable > threshold, original > np.quantile(original, q))
    assert original_linear_quantile_in_log_space(stable + 1000, q) == pytest.approx(threshold + 1000)


def test_robustness_deltas_pair_folds():
    frame = pd.DataFrame([
        {"outer_fold": fold, "model": "PCA", "features": "A_RGB", "condition": condition,
         "AP": value, "recall_P": 0.5, "FPR_N": 0.1, "FPR_R": 0.05, "precision": 0.8}
        for fold, clean, blurred in [(0, 0.9, 0.7), (1, 0.6, 0.5)]
        for condition, value in [("clean", clean), ("blur", blurred)]
    ])
    _, pairs, summary = summarize_condition_results(frame.sample(frac=1, random_state=1))
    np.testing.assert_allclose(pairs.sort_values("outer_fold").AP_delta_from_clean, [-0.2, -0.1])
    assert summary.iloc[0].AP_delta_mean == pytest.approx(-0.15)
    assert summary.iloc[0].n_paired_folds == 2
