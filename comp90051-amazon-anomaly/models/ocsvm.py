"""models: ocsvm for the Amazon robustness experiment."""

import numpy as np
from models.base import ReferenceAnomalyDetector
from preprocessing.scaling import StandardScaler
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.svm import OneClassSVM


def reference_median_squared_distance(z, seed, maximum_rows=512):
    """
    Deterministic bandwidth scale from standardized TRAINING references only.

        A bounded row sample limits pairwise work. The same seed/sample serves every
        multiplier candidate within a fold. Gram distances avoid a large n×n×d tensor.
        Positive distances exclude duplicate feature vectors; all-zero data fail.
    """
    z = np.asarray(z, dtype=float)
    if len(z) > maximum_rows:
        rng = np.random.default_rng(int(seed) + 91337)
        indices = np.sort(rng.choice(len(z), size=maximum_rows, replace=False))
        sample = z[indices]
    else:
        sample = z
    squared_norm = np.sum(sample * sample, axis=1)
    distances = np.maximum(squared_norm[:, None] + squared_norm[None, :] - 2 * sample @ sample.T, 0)
    upper = distances[np.triu_indices(len(sample), 1)]
    positive = upper[upper > np.finfo(float).eps]
    if not len(positive) or not np.isfinite(positive).all():
        raise ValueError('Reference distances cannot define a finite nonzero OCSVM bandwidth.')
    return (float(np.median(positive)), int(len(sample)))


class OneClassSVMDetector(ReferenceAnomalyDetector):
    """
    An RBF boundary around standardized reference feature vectors.

    Steps
    -----
    1. Fit scaling using the supplied clean reference training rows.
    2. Learn the model described in the named blocks below.
    3. Reuse the fitted scaler and model when scoring new rows.

    Parameters
    ----------
    parameter : float
        Multiplier divided by the median positive reference squared distance.
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
    model_name = 'OCSVM'

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
        median, count = reference_median_squared_distance(z, self.seed)
        self.gamma_reference_median_sqdist_, self.gamma_reference_sample_count_ = median, count
        self.effective_parameter_ = self.parameter / median

        # ------------------------------------------------------------
        # Fit the reference model
        # ------------------------------------------------------------
        # Its internal sklearn offset does not define our external alert threshold.
        self.estimator_ = OneClassSVM(
            kernel='rbf', nu=0.05,
            gamma=self.effective_parameter_, cache_size=1024,
        ).fit(z)

    def _score_model(self, x):
        z = self.scaler.transform(x)
        return stable_negative_log_kernel_sum(self.estimator_, z)


def stable_negative_log_kernel_sum(fitted_ocsvm, z):
    """Return -log(sum_i alpha_i exp(-gamma * ||z-support_i||^2)).

    Inputs must already use the fitted model's feature scaling. Positive OCSVM
    dual weights are used exactly as fitted, without renormalizing their mass.
    Zero coefficients are omitted; negative coefficients are rejected. For the
    original anomaly score a=rho-S, this is the strictly increasing transform
    -log(rho-a) in exact arithmetic. It must be computed from supports/weights,
    never from rounded decision scores. Chunking bounds temporary storage.

    Arbitrarily extreme finite inputs can overflow squared distances; these are
    rejected rather than silently returning an infinite diagnostic score.
    """
    if getattr(fitted_ocsvm, "kernel", None) != "rbf":
        raise ValueError("Only fitted RBF OneClassSVM models are supported.")
    if getattr(fitted_ocsvm, "_impl", None) != "one_class":
        raise ValueError("The estimator must implement one-class SVM.")
    if getattr(fitted_ocsvm, "_sparse", False):
        raise ValueError("This candidate supports dense fitted models only.")
    try:
        supports = np.asarray(fitted_ocsvm.support_vectors_, dtype=np.float64)
        dual = np.asarray(fitted_ocsvm.dual_coef_, dtype=np.float64)
        gamma = float(fitted_ocsvm._gamma)
    except AttributeError as exc:
        raise ValueError("The estimator must already be fitted.") from exc
    query = np.asarray(z, dtype=np.float64)
    if supports.ndim != 2 or dual.shape != (1, len(supports)):
        raise ValueError("Support vectors and one-class dual weights are misaligned.")
    if query.ndim != 2 or query.shape[1] != supports.shape[1]:
        raise ValueError("Query features must match the fitted model.")
    if not (np.isfinite(query).all() and np.isfinite(supports).all()
            and np.isfinite(dual).all()):
        raise ValueError("Inputs, supports, and coefficients must be finite.")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("RBF gamma must be finite and positive.")
    alpha = dual[0]
    if np.any(alpha < 0) or not np.any(alpha > 0):
        raise ValueError("A positive kernel sum requires nonnegative, nonzero weights.")
    positive = alpha > 0
    supports = supports[positive]
    log_alpha = np.log(alpha[positive])
    result = np.empty(len(query), dtype=np.float64)
    for query_start in range(0, len(query), 256):
        block = query[query_start:query_start + 256]
        block_log_sum = np.full(len(block), -np.inf)
        for support_start in range(0, len(supports), 1024):
            distance = cdist(block, supports[support_start:support_start + 1024],
                             metric="sqeuclidean")
            with np.errstate(over="ignore", invalid="ignore"):
                exponent = gamma * distance
            if not np.isfinite(exponent).all():
                raise ValueError("Scaled squared distances overflow float64.")
            log_terms = log_alpha[support_start:support_start + 1024][None, :] - exponent
            partial = logsumexp(log_terms, axis=1)
            block_log_sum = np.logaddexp(block_log_sum, partial)
        result[query_start:query_start + len(block)] = -block_log_sum
    if not np.isfinite(result).all():
        raise ValueError("The stable score exceeded the supported finite range.")
    return result
