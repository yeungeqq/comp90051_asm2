"""Check band semantics, deterministic corruptions, and cache alignment."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from config import CONDITIONS
from data.loader import read_rgbn
from models.feature_transformer import feature_token_indices
from preprocessing.degradations import apply_corruption
from preprocessing.features import extract_condition_features, feature_columns, features_from_rgbn


@pytest.mark.parametrize("band_first", [False, True])
def test_tiff_band_order(tmp_path, band_first):
    raw = np.broadcast_to(np.array([100, 200, 300, 400], dtype=np.uint16), (100, 104, 4)).copy()
    path = tmp_path / "chip.tif"
    tifffile.imwrite(path, np.moveaxis(raw, -1, 0) if band_first else raw)
    rgbn = read_rgbn(path)
    np.testing.assert_array_equal(rgbn[0, 0], [300, 200, 100, 400])


def test_feature_values_and_nested_columns():
    image = np.broadcast_to(np.array([100, 200, 300, 400], dtype=np.float32), (16, 20, 4)).copy()
    values, qc = features_from_rgbn(image)
    columns = feature_columns()
    assert [len(columns[name]) for name in columns] == [63, 84, 108]
    assert list(values) == columns["C_RGB_NIR_TEXTURE"]
    assert columns["A_RGB"] == columns["B_RGB_NIR"][:63]
    assert columns["B_RGB_NIR"] == columns["C_RGB_NIR_TEXTURE"][:84]
    assert values["red_mean"] == pytest.approx(100 / 65535)
    assert values["nir_grid_15_mean"] == pytest.approx(400 / 65535)
    assert values["red_std"] == pytest.approx(0, abs=1e-8)
    assert values["glcm_contrast_0"] == 0
    assert values["glcm_homogeneity_0"] == 1
    assert qc["texture_percentile_span_zero"]
    for names in columns.values():
        spatial, global_ = feature_token_indices(names)
        indices = [i for cell in spatial for i in cell] + global_
        assert sorted(indices) == list(range(len(names)))


@pytest.mark.parametrize("condition", CONDITIONS)
def test_corruptions_repeat_without_mutating_source(condition):
    original = np.random.default_rng(8).integers(100, 10000, size=(16, 20, 4)).astype(np.float32)
    untouched = original.copy()
    first, clips = apply_corruption(original, condition, "train_8")
    second, _ = apply_corruption(original, condition, "train_8")
    np.testing.assert_array_equal(original, untouched)
    np.testing.assert_array_equal(first, second)
    assert first.shape == original.shape and first.dtype == np.float32
    assert np.isfinite(first).all() and first.min() >= 0 and first.max() <= 65535
    assert clips.shape == (4,)


def test_cache_reuses_numbers_under_reordered_manifest(tmp_path):
    rows = []
    for number in range(2):
        path = tmp_path / f"train_{number}.tif"
        image = np.random.default_rng(number).integers(100, 5000, size=(100, 100, 4), dtype=np.uint16)
        tifffile.imwrite(path, image)
        rows.append({"image_id": f"train_{number}", "tiff_path": str(path)})
    manifest = pd.DataFrame(rows)
    arguments = {"feature_cache_dir": tmp_path / "cache", "conditions": ("clean", "noise_low")}
    first, audit = extract_condition_features(manifest, **arguments)
    second, reused = extract_condition_features(manifest.iloc[::-1], **arguments)
    assert audit["conditions_computed"] == 4
    assert reused["condition_cache_hits"] == 4
    for condition in arguments["conditions"]:
        for name in feature_columns():
            np.testing.assert_array_equal(first[condition][name][::-1], second[condition][name])


def test_bad_pixels_fail():
    with pytest.raises(ValueError):
        features_from_rgbn(np.full((8, 8, 4), np.nan))


def test_authenticated_feature_package_verifies_source_and_bytes(tmp_path, monkeypatch):
    import config
    from data.loader import file_sha256
    from preprocessing import features

    manifest = pd.DataFrame({"image_id": ["a", "b"], "tags": ["clear primary", "clear road"],
                             "group": ["R", "P"], "pixel_sha256": ["aa", "bb"],
                             "duplicate_group": ["aa", "bb"]})
    manifest.to_csv(tmp_path / "benchmark_manifest.csv", index=False)
    arrays = {f"{condition}__{name}": np.zeros((2, len(columns)), dtype=np.float32)
              for condition in CONDITIONS for name, columns in feature_columns().items()}
    feature_path = tmp_path / "cached_features.npz"
    np.savez(feature_path, **arrays)
    package = {
        "feature_columns": feature_columns(),
        "source_feature_ast_hashes": {"features_from_rgbn": features.visible_definition_ast("features_from_rgbn")},
        "files": {"cached_features.npz": {"sha256": file_sha256(feature_path)}},
        "arrays": {f"formal/{key}": {"shape": list(value.shape), "dtype": value.dtype.str,
                                      "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                   for key, value in arrays.items()},
        "feature_library_version_limit": "synthetic test package",
    }
    package_path = tmp_path / "cached_feature_manifest.json"
    package_path.write_text(json.dumps(package))
    # Test-only authentication of this synthetic two-image fixture.
    monkeypatch.setattr(config, "INPUT_ROOT", tmp_path)
    monkeypatch.setattr(features, "FEATURE_PACKAGE_SHA256", file_sha256(package_path))
    loaded, audit = features.load_authenticated_features(manifest)
    assert audit["all_arrays_verified"]
    assert loaded["clean"]["A_RGB"].shape == (2, 63)
    with pytest.raises(ValueError, match="different ordered cohort"):
        features.load_authenticated_features(manifest.iloc[::-1])
    feature_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="bytes are corrupted"):
        features.load_authenticated_features(manifest)
