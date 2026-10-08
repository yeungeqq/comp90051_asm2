"""Cohort creation is deterministic and keeps duplicate pixels in one cohort."""

import pandas as pd
import pytest

from data.cohorts import select_cohorts, validate_cohorts


def candidate_source():
    rows = []
    for group in ('R', 'P', 'N'):
        for number in range(12):
            rows.append({
                'image_id': f'{group}_{number:02d}', 'group': group,
                'tags': 'clear primary', 'tiff_path': f'{group}_{number:02d}.tif',
                'tiff_path_count': 1,
            })
    return pd.DataFrame(rows)


def fake_pixel_hash(path):
    # These two IDs have identical pixels but are labelled as different rows.
    digest = 'shared-pixels' if path in {'R_00.tif', 'P_00.tif'} else path
    return digest, None


def test_cohort_selection_is_reproducible_and_disjoint():
    quotas = {'R': 3, 'P': 3, 'N': 3}
    first = select_cohorts(candidate_source(), fake_pixel_hash, seed=51, group_counts=quotas)
    second = select_cohorts(candidate_source(), fake_pixel_hash, seed=51, group_counts=quotas)
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)
        assert left.groupby('group').size().to_dict() == quotas
    development, benchmark = first
    assert set(development.image_id).isdisjoint(benchmark.image_id)
    assert set(development.pixel_sha256).isdisjoint(benchmark.pixel_sha256)


def test_cohort_validator_rejects_cross_cohort_pixel_overlap():
    quotas = {'R': 1, 'P': 1, 'N': 1}
    development = pd.DataFrame({
        'image_id': ['r', 'p', 'n'], 'tags': ['x'] * 3,
        'group': ['R', 'P', 'N'], 'pixel_sha256': ['a', 'b', 'c'],
        'duplicate_group': ['a', 'b', 'c'],
    })
    benchmark = development.copy()
    benchmark['image_id'] = ['r2', 'p2', 'n2']
    with pytest.raises(ValueError, match='overlap: pixel_sha256'):
        validate_cohorts(development, benchmark, quotas)