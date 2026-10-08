"""Verify selected-model execution, evidence counts, and plots on synthetic data."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import config
from data.splits import build_nested_split_plan
from evaluation import robustness
from main import describe_experiment, main
from training import protocol, tuning
from visualization import plots


def test_selection_and_dry_run(capsys):
    assert config.select_models() == config.MODEL_NAMES
    assert config.select_models("DeepSVDD") == ("DeepSVDD",)
    assert config.select_models(["OCSVM", "PCA"]) == ("PCA", "OCSVM")
    for invalid in ([], ["PCA", "PCA"], ["unknown"]):
        with pytest.raises(ValueError):
            config.select_models(invalid)
    main(["--models", "DeepSVDD", "--dry-run"])
    settings = json.loads(capsys.readouterr().out)
    assert settings["models"] == ["DeepSVDD"]
    assert settings["inner_fits"] == 270 and settings["outer_fits"] == 30
    assert settings["neural_step_budgets"] == {"DeepSVDD": 7680}
    assert describe_experiment()["inner_fits"] == 1620
    assert describe_experiment()["outer_fits"] == 180
    with pytest.raises(SystemExit):
        main(["--models", "unknown", "--dry-run"])


def test_expected_counts_follow_model_scope():
    assert robustness.expected_result_counts(2400, 10, 3) == {
        "inner_fits": 1620, "candidate_means": 540, "selections": 180,
        "outer_fits": 180, "metrics": 1620, "predictions": 388800,
    }
    assert robustness.expected_result_counts(2400, 10, 3, ["LOF"]) == {
        "inner_fits": 270, "candidate_means": 90, "selections": 30,
        "outer_fits": 30, "metrics": 270, "predictions": 64800,
    }


def test_unused_model_support_checks_are_skipped(monkeypatch):
    def unexpected_rank(*args):
        raise AssertionError("PCA support must not be checked for an OCSVM-only run")

    monkeypatch.setattr(tuning, "evaluation_training_rank", unexpected_rank)
    plans = [(np.array([0]), np.array([1, 2, 3]), np.array([4]),
              [(np.array([1, 2, 3]), np.array([0]))])]
    # These intentionally unsupported grids belong only to unselected models.
    clean = {name: np.zeros((5, 2)) for name in config.FEATURE_REPRESENTATIONS}
    tuning.evaluation_check_formal_pca_support(plans, config.FROZEN_GRIDS, clean, models=["OCSVM"])


def test_single_model_evaluation_and_plots(tmp_path, monkeypatch):
    groups = ["R"] * 30 + ["P"] * 12 + ["N"] * 12
    manifest = pd.DataFrame({
        "image_id": [f"chip_{i}" for i in range(len(groups))], "group": groups,
        "tags": ["clear primary" if group == "R" else "clear road" if group == "P"
                 else "clear primary water" for group in groups],
        "duplicate_group": [f"pixel_{i}" for i in range(len(groups))],
        "pixel_sha256": [f"pixel_{i}" for i in range(len(groups))],
    })
    rng = np.random.default_rng(51)
    clean = {name: rng.normal(size=(len(manifest), width)).astype(np.float32)
             for name, width in zip(config.FEATURE_REPRESENTATIONS, (63, 84, 108))}
    features = {condition: {name: values.copy() for name, values in clean.items()}
                for condition in config.CONDITIONS}
    _, splits, _ = build_nested_split_plan(manifest, 2, 2, 51, 0.2)
    splits["role"] = splits.role.replace({"development": "outer_training",
                                           "development_holdout": "outer_holdout"})
    digest = hashlib.sha256(splits.to_csv(index=False).encode()).hexdigest()
    grids = {name: {**candidates, "PCA": [0, 1, 2]}
             for name, candidates in config.FROZEN_GRIDS.items()}
    for module in (robustness, protocol):
        monkeypatch.setattr(module, "OUTER_FOLDS", 2)
        monkeypatch.setattr(module, "INNER_FOLDS", 2)
        monkeypatch.setattr(module, "FROZEN_GRIDS", grids)
    monkeypatch.setattr(tuning, "FROZEN_GRIDS", grids)
    monkeypatch.setattr(robustness, "EXPECTED_SPLIT_SHA256", digest)
    # A synthetic PCA-only run uses CPU; replace hardware validation for this fixture.
    monkeypatch.setattr(protocol.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(robustness, "runtime_identity", lambda: {"test": "synthetic CPU fixture"})
    monkeypatch.setattr(config, "INPUT_ROOT", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    manifest.to_csv(tmp_path / "development_manifest.csv", index=False)

    result = robustness.run_complete_comparison(manifest, features, tmp_path / "results", models=["PCA"])
    assert set(result["metrics"].model) == {"PCA"}
    assert set(result["inner"].model) == {"PCA"}
    assert result["completion"]["counts"] == robustness.expected_result_counts(len(manifest), 2, 2, ["PCA"])
    assert result["completion"]["selected_models"] == ["PCA"]
    assert result["completion"]["execution_complete"]
    assert not result["completion"]["full_comparison_complete"]
    assert not result["completion"]["unique_middle_requirement_met"]
    assert result["completion"]["core_arms"] == 0
    assert result["completion"]["all_arms"] == 3
    identity = json.loads((next((tmp_path / "results").glob("*/experiment_identity.json"))).read_text())
    assert identity["settings"]["models"] == ["PCA"]
    monkeypatch.setattr(plots, "display", lambda value: None)
    plots.show_result_figures(result, tmp_path / "figures")
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5
