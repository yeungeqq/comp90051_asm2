"""Verify grouped nested folds isolate fitting, calibration, and holdout."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from data.labels import classify_tags
from data.splits import build_nested_split_plan, make_grouped_folds


def synthetic_manifest():
    rows = []
    for group in "RPN":
        for cluster in range(30):
            for duplicate in range(2 if cluster == 0 else 1):
                rows.append({"image_id": f"{group}_{cluster}_{duplicate}", "group": group,
                             "duplicate_group": f"{group}_{cluster}", "tags": "clear primary"})
    # Include a duplicate cluster spanning content groups.
    rows.extend({"image_id": f"mixed_{group}", "group": group,
                 "duplicate_group": "mixed", "tags": "clear primary"} for group in "RPN")
    return pd.DataFrame(rows)


def test_nested_split_isolation_and_determinism():
    manifest = synthetic_manifest()
    plans, rows, _ = build_nested_split_plan(manifest, 3, 3, 51, 0.2)
    _, repeat, _ = build_nested_split_plan(manifest, 3, 3, 51, 0.2)
    pd.testing.assert_frame_equal(rows, repeat)
    all_held = []
    for held, fit, calibration, inner in plans:
        all_held.extend(held)
        clusters = [set(manifest.iloc[ix].duplicate_group) for ix in (held, fit, calibration)]
        assert all(a.isdisjoint(b) for a, b in combinations(clusters, 2))
        assert manifest.iloc[fit].group.eq("R").all()
        assert manifest.iloc[calibration].group.eq("R").all()
        assert set(manifest.iloc[held].group) == {"R", "P", "N"}
        for inner_fit, validation in inner:
            assert manifest.iloc[inner_fit].group.eq("R").all()
            for a, b in combinations([set(manifest.iloc[ix].duplicate_group)
                                      for ix in (inner_fit, validation, calibration, held)], 2):
                assert a.isdisjoint(b)
    assert sorted(all_held) == list(range(len(manifest)))
    assert rows.groupby(["outer_fold", "duplicate_group"]).role.nunique().max() <= 2


def test_duplicate_clusters_stay_together():
    manifest = synthetic_manifest()
    folds = make_grouped_folds(manifest, 3, 51)
    assert manifest.assign(fold=folds).groupby("duplicate_group").fold.nunique().eq(1).all()
    with pytest.raises(ValueError, match="duplicate groups"):
        make_grouped_folds(manifest.iloc[:2], 3, 51)


@pytest.mark.parametrize("tags,group", [
    ("clear primary", "R"),
    ("clear primary water", "N"),
    ("clear primary road water", "P"),
    ("cloudy primary road", "excluded"),
    ("clear cloudy primary", "excluded"),
    ("clear primary unknown", "excluded"),
])
def test_label_policy(tags, group):
    assert classify_tags(tags.split())[0] == group
