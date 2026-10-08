"""models: base for the Amazon robustness experiment."""

import numpy as np
import time
import torch
from config import CANDIDATE_MODELS, INTEGER_PARAMETERS, MODEL_ROLES, PARAMETER_NAMES


class ReferenceAnomalyDetector:
    """
    Common validation and diagnostics for reference-only anomaly detection.

    Steps
    -----
    1. Validate the model parameter and declared update budget.
    2. Receive clean reference training rows selected by the external runner.
    3. Fit the selected algorithm and its scaler on those rows only.
    4. Return one score per held-out row, with larger scores more anomalous.

    Parameters
    ----------
    parameter : float
        Model-specific tuning value described in each algorithm section.
    seed : int
        Random seed for this particular fit.
    optimizer_steps : int or None
        Exact update count for neural models; unused by non-neural models.
    batch_size : int
        Maximum neural training batch size.
    device : str
        'auto', 'cpu', or a supported CUDA device.

    Notes
    -----
    fit receives [n_reference, F]; score_samples receives [n_images, F].
    Labels, folds, threshold calibration and corruptions stay outside this class.
    The caller must supply clean R rows; numbers alone do not identify labels.
    """

    model_name = None

    def __init__(self, parameter, seed, optimizer_steps, batch_size, device='auto'):
        name = self.model_name
        if name not in CANDIDATE_MODELS or not np.isfinite(parameter) or parameter <= 0:
            raise ValueError('Unknown candidate or invalid parameter.')
        if name in INTEGER_PARAMETERS and int(parameter) != parameter:
            raise ValueError('This model requires an integer-valued parameter.')
        if name == 'LOF' and parameter < 2:
            raise ValueError('LOF requires at least two neighbors in this protocol.')
        if not np.isfinite(seed) or int(seed) != seed or not 0 <= seed < 2 ** 32:
            raise ValueError('seed must be an integer from 0 to 2**32 - 1.')
        if not np.isfinite(batch_size) or batch_size < 1 or int(batch_size) != batch_size:
            raise ValueError('Batch count must be a positive integer.')
        if optimizer_steps is not None and (not np.isfinite(optimizer_steps) or int(optimizer_steps) != optimizer_steps or optimizer_steps < 1):
            raise ValueError('optimizer_steps must be a positive integer.')
        if name in ('DeepSVDD', 'MLPAutoencoder', 'TransformerAE') and optimizer_steps is None:
            raise ValueError('Neural models require an exact optimizer_steps budget.')
        self.name, self.parameter = name, float(parameter)
        self.seed, self.batch_size = int(seed), int(batch_size)
        self.requested_device = device
        self.max_steps = None if optimizer_steps is None else int(optimizer_steps)
        self.parameter_name_, self.model_role_ = PARAMETER_NAMES[name], MODEL_ROLES[name]
        self._fitted = False

    def fit(self, X_reference):
        """
        Fit on the supplied training-reference rows only.

        Parameters
        ----------
        X_reference : array, shape [n_reference, F]
            Finite clean R feature vectors; at least three rows and two features.

        Returns
        -------
        self : fitted detector
            Stores the scaler, effective parameter and training diagnostics.
        """
        self._fitted = False
        x = np.asarray(X_reference, dtype=float)
        if x.ndim != 2 or len(x) < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError('Fit requires >=3 finite reference rows and >=2 features.')
        started = time.perf_counter()
        self.n_features_in_ = x.shape[1]
        self.loss_history_ = []
        self.optimizer_steps_, self.step_loss_history_ = 0, []
        self.training_checkpoints_ = []
        self.effective_max_samples_ = None
        self.gamma_reference_median_sqdist_ = None
        self.gamma_reference_sample_count_ = None
        self._fit_model(x)
        self.scaler_ = self.scaler
        self.loss_history = self.loss_history_
        if self.name in ('DeepSVDD', 'MLPAutoencoder', 'TransformerAE') and self.optimizer_steps_ != self.max_steps:
            raise RuntimeError('Neural detector did not execute its exact update budget.')
        self.fit_seconds = time.perf_counter() - started
        self._fitted = True
        return self

    def score_samples(self, X):
        """
        Score unseen rows without fitting or changing the scaler.

        Parameters
        ----------
        X : array, shape [n_images, F]
            Features in the same column order used for training.

        Returns
        -------
        scores : array, shape [n_images]
            Finite anomaly scores; higher means more anomalous.
        """
        if not self._fitted:
            raise RuntimeError('Fit on reference-training rows first.')
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('Scoring rows must be finite and match fitted dimensions.')
        if not len(x):
            return np.empty(0, dtype=float)

        # ------------------------------------------------------------
        # Validate the score vector
        # ------------------------------------------------------------
        scores = np.asarray(self._score_model(x), dtype=float)
        if scores.shape != (len(x),) or not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite or unaligned candidate scores.')
        return scores


def record_training_checkpoint(detector, training_data, weight_decay, force=False):
    """Record a fixed-data objective without changing weights, gradients or RNG.

    The data term is a row-weighted mean over ALL clean reference training rows.
    Adam's coupled weight decay corresponds to (weight_decay / 2) * sum(p**2).
    This is a training diagnostic: it uses no validation labels or held-out rows.
    A final checkpoint is recorded even when the budget is not a multiple of 60.
    """
    step = int(detector.optimizer_steps_)
    if not force and step % 60 != 0 and step != detector.max_steps:
        return
    if detector.training_checkpoints_ and detector.training_checkpoints_[-1]['step'] == step:
        return
    if (training_data.ndim != 2 or len(training_data) < 1
            or not torch.isfinite(training_data).all()):
        raise ValueError('Checkpoint data must be finite reference feature rows.')
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Checkpoint weight decay must be finite and nonnegative.')
    if detector.name not in ('MLPAutoencoder', 'DeepSVDD', 'TransformerAE'):
        raise ValueError('Training checkpoints are defined only for neural detectors.')

    # Temporarily switch off training behavior, then restore the original mode.
    was_training = detector.net_.training
    detector.net_.eval()
    try:
        with torch.no_grad():
            total, count = 0.0, 0
            for start in range(0, len(training_data), detector.batch_size):
                batch = training_data[start:start + detector.batch_size].to(detector.device)
                output = detector.net_(batch)
                if detector.name == 'DeepSVDD':
                    row_losses = ((output - detector.center_) ** 2).sum(dim=1)
                else:
                    row_losses = ((output - batch) ** 2).mean(dim=1)
                if row_losses.shape != (len(batch),) or not torch.isfinite(row_losses).all():
                    raise RuntimeError('Nonfinite or misaligned full-reference checkpoint loss.')
                total += float(row_losses.double().sum().cpu())
                count += len(batch)
            data_loss = total / count
            squared_norm = sum(
                float(parameter.detach().double().square().sum().cpu())
                for parameter in detector.net_.parameters()
            )
            regularization_loss = 0.5 * float(weight_decay) * squared_norm
            objective = data_loss + regularization_loss
            if not np.isfinite([data_loss, regularization_loss, objective]).all():
                raise RuntimeError('Nonfinite full-reference training objective.')
    finally:
        detector.net_.train(was_training)
    detector.training_checkpoints_.append({
        'step': step,
        'optimizer_steps': step,
        'data_loss': float(data_loss),
        'regularization_loss': float(regularization_loss),
        'objective': float(objective),
        'n_reference': int(count),
    })
