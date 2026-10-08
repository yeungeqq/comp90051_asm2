"""Small CPU runs verify every detector and saved-unit resume without Kaggle data."""

import numpy as np
import pandas as pd
import pytest

from models import create_detector
from training.protocol import active_science_identity
from training.trainers import run_inner_unit


@pytest.mark.parametrize("name,parameter,steps", [
    ("PCA", 0, None), ("PCA", 2, None), ("OCSVM", 16, None), ("LOF", 2, None),
    ("MLPAutoencoder", 4, 2), ("DeepSVDD", 0.001, 2), ("TransformerAE", 8, 2),
])
def test_detector_fit_score_contract(name, parameter, steps):
    rng = np.random.default_rng(51)
    training = rng.normal(size=(20, 63)).astype(np.float32)
    unseen = rng.normal(size=(5, 63)).astype(np.float32)
    detector = create_detector(name, parameter, 51, steps, 8, "cpu")
    with pytest.raises(RuntimeError):
        detector.score_samples(unseen)
    detector.fit(training)
    mean = detector.scaler.mean_.copy()
    scores = detector.score_samples(unseen)
    assert scores.shape == (5,) and np.isfinite(scores).all()
    np.testing.assert_array_equal(mean, detector.scaler.mean_)
    assert detector.score_samples(np.empty((0, 63))).shape == (0,)
    assert detector.optimizer_steps_ == (steps or 0)
    if steps:
        assert detector.training_checkpoints_[-1]["optimizer_steps"] == steps


def test_scientific_registry_resolves_modular_definitions():
    first = active_science_identity()
    assert len(first) == 64 and first == active_science_identity()


def test_inner_unit_resumes_and_rejects_changed_evidence(tmp_path):
    manifest = pd.DataFrame({"image_id": [f"train_{i}" for i in range(24)],
                             "group": ["R"] * 20 + ["P", "P", "N", "R"],
                             "tags": ["clear primary"] * 24})
    features = {"A_RGB": np.random.default_rng(51).normal(size=(24, 63)).astype(np.float32)}
    arguments = (manifest, features, np.arange(20), np.arange(20, 24),
                 0, 0, "A_RGB", "PCA", 0, "synthetic-test", tmp_path)
    first, reused = run_inner_unit(*arguments)
    assert not reused
    second, reused = run_inner_unit(*arguments)
    assert reused and first == second
    path = tmp_path / "inner_units" / first["unit_key"] / "validation_scores.csv"
    path.write_text("damaged")
    with pytest.raises(ValueError, match="completed result file changed"):
        run_inner_unit(*arguments)
