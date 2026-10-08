"""models: lof for the Amazon robustness experiment."""

import numpy as np
from models.base import ReferenceAnomalyDetector
from preprocessing.scaling import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


class LocalOutlierDetector(ReferenceAnomalyDetector):
    """
    Novelty-mode local density comparison with reference neighbours.

    Steps
    -----
    1. Fit scaling using the supplied clean reference training rows.
    2. Learn the model described in the named blocks below.
    3. Reuse the fitted scaler and model when scoring new rows.

    Parameters
    ----------
    parameter : float
        Number of reference neighbours; strictly below the training row count.
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
    model_name = 'LOF'

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
        neighbors = int(self.parameter)
        if neighbors >= len(x):
            raise ValueError('LOF n_neighbors must be below the training-reference count; no automatic capping.')

        # ------------------------------------------------------------
        # Fit the reference model
        # ------------------------------------------------------------
        # Its internal sklearn offset does not define our external alert threshold.
        self.estimator_ = LocalOutlierFactor(
            n_neighbors=neighbors, novelty=True, contamination='auto',
            metric='minkowski', p=2, n_jobs=2,
        ).fit(z)
        self.effective_parameter_ = int(self.estimator_.n_neighbors_)
        if self.effective_parameter_ != neighbors:
            raise RuntimeError('LOF silently changed the requested neighbor count.')

    def _score_model(self, x):
        z = self.scaler.transform(x)
        return -self.estimator_.score_samples(z)
