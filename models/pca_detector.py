"""models: pca_detector for the Amazon robustness experiment."""

import numpy as np
import time
from config import PCA_ZERO_POLICY
from models.base import ReferenceAnomalyDetector
from preprocessing.scaling import StandardScaler
from sklearn.decomposition import PCA


class PCAAnomalyDetector(ReferenceAnomalyDetector):
    """
    Linear reconstruction using a reduced reference-feature subspace.

    Steps
    -----
    1. Fit scaling using the supplied clean reference training rows.
    2. Learn the model described in the named blocks below.
    3. Reuse the fitted scaler and model when scoring new rows.

    Parameters
    ----------
    parameter : float
        Number of retained principal components; below reference rank.
    seed : int
        Random seed used for this fit.
    optimizer_steps : int or None
        Exact neural update count; non-neural methods do not use it.
    batch_size : int
        Maximum neural training batch size.
    device : str
        'auto' chooses CUDA when available; sklearn methods use CPU.

    Returns
    -------
    score_samples(X) : array, shape [n_images]
        One higher-is-anomalous score for each [F]-dimensional feature vector.
    """
    model_name = 'PCA'

    def _fit_model(self, x):

        # ------------------------------------------------------------
        # Fit the reference-only scaler
        # ------------------------------------------------------------
        # Never learn scaling from calibration, disturbed or outer-test rows.
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        if not np.any(np.var(z, axis=0) > 0):
            raise ValueError('All training features are constant.')
        self.device = 'cpu'
        self.effective_parameter_ = int(self.parameter)

        # ------------------------------------------------------------
        # Learn the full reference principal-component basis
        # ------------------------------------------------------------
        self.pca_ = PCA(svd_solver='full').fit(z)

        # ------------------------------------------------------------
        # Check the feasible retained rank
        # ------------------------------------------------------------
        # Reject impossible ranks instead of silently changing a tuning candidate.
        self.training_rank_ = int(np.linalg.matrix_rank(z - self.pca_.mean_))
        k = int(self.parameter)
        if k >= self.training_rank_ or k >= x.shape[1]:
            raise ValueError(f'PCA k={k} must be below training rank={self.training_rank_} and feature count.')
        self.components_ = self.pca_.components_[:k]

    def _score_model(self, x):
        z = self.scaler.transform(x)
        centered = z - self.pca_.mean_

        # ------------------------------------------------------------
        # Reconstruct in the retained subspace
        # ------------------------------------------------------------
        reconstruction = (centered @ self.components_.T) @ self.components_
        return np.mean((centered - reconstruction) ** 2, axis=1)


class PCAZeroDetector:
    """Reference-only scaling followed by reconstruction at its training mean."""
    model_name = 'PCA'

    def __init__(self, parameter, seed, optimizer_steps, batch_size, device='auto'):
        if isinstance(parameter, (bool, np.bool_)) or not np.isscalar(parameter) or not np.isfinite(parameter) or parameter != 0:
            raise ValueError('This explicit PCA adapter accepts only zero components.')
        if not np.isfinite(seed) or int(seed) != seed or not 0 <= seed < 2 ** 32:
            raise ValueError('seed must be an integer from 0 to 2**32 - 1.')
        if not np.isfinite(batch_size) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError('Batch count must be a positive integer.')
        if optimizer_steps is not None:
            raise ValueError('PCA zero has no neural optimizer budget.')
        self.name, self.parameter = 'PCA', 0.0
        self.seed, self.batch_size = int(seed), int(batch_size)
        self.requested_device, self.device = device, 'cpu'
        self.max_steps = None
        self.parameter_name_, self.model_role_ = 'retained_components', 'baseline'
        self.pca_zero_policy_ = PCA_ZERO_POLICY
        self._fitted = False

    def fit(self, X_reference):
        self._fitted = False
        x = np.asarray(X_reference, dtype=float)
        if x.ndim != 2 or len(x) < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError('Fit requires >=3 finite reference rows and >=2 features.')
        started = time.perf_counter()
        self.n_features_in_ = x.shape[1]
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        if not np.isfinite(z).all() or not np.any(np.var(z, axis=0) > 0):
            raise ValueError('Standardized training features must be finite and nonconstant.')
        # Matches the centering used by the original PCA estimator after scaling.
        self.standardized_training_mean_ = z.mean(axis=0)
        self.training_rank_ = int(np.linalg.matrix_rank(z - self.standardized_training_mean_))
        if self.training_rank_ <= 0:
            raise ValueError('PCA zero requires a nonzero reference rank.')
        self.effective_parameter_ = 0
        self.reconstruction_components_ = 0
        self.components_ = np.empty((0, x.shape[1]), dtype=float)
        self.loss_history_, self.step_loss_history_, self.training_checkpoints_ = [], [], []
        self.optimizer_steps_ = 0
        self.scaler_, self.loss_history = self.scaler, self.loss_history_
        self.fit_seconds = time.perf_counter() - started
        self._fitted = True
        return self

    def score_samples(self, X):
        if not self._fitted:
            raise RuntimeError('Fit on reference-training rows first.')
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('Scoring rows must be finite and match fitted dimensions.')
        if not len(x):
            return np.empty(0, dtype=float)
        centered = self.scaler.transform(x) - self.standardized_training_mean_
        scores = np.mean(centered ** 2, axis=1)
        if scores.shape != (len(x),) or not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite or unaligned PCA-zero scores.')
        return scores
