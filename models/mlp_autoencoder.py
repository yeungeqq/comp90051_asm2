"""models: mlp_autoencoder for the Amazon robustness experiment."""

import numpy as np
import torch
from models.base import ReferenceAnomalyDetector, record_training_checkpoint
from preprocessing.scaling import StandardScaler


class MLPAutoencoderDetector(ReferenceAnomalyDetector):
    """
    Nonlinear reconstruction through a smaller feature representation.

    Steps
    -----
    1. Fit scaling using the supplied clean reference training rows.
    2. Learn the model described in the named blocks below.
    3. Reuse the fitted scaler and model when scoring new rows.

    Parameters
    ----------
    parameter : float
        Width of the compressed feature representation.
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
    model_name = 'MLPAutoencoder'

    def _fit_model(self, x):

        # ------------------------------------------------------------
        # Fit the reference-only scaler
        # ------------------------------------------------------------
        # Never learn scaling from calibration, disturbed or outer-test rows.
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        if not np.isfinite(z).all():
            raise ValueError('Standardized reference features must be finite.')
        if not np.any(np.var(z, axis=0) > 0):
            raise ValueError('All training features are constant; detector is undefined.')
        self._fit_autoencoder(z)
        self.effective_parameter_ = int(self.parameter)

    def _score_model(self, x):
        z = self.scaler.transform(x)
        outputs = []

        # ------------------------------------------------------------
        # Compute without gradient storage
        # ------------------------------------------------------------
        with torch.no_grad():
            for start in range(0, len(z), self.batch_size):
                batch = torch.as_tensor(
                    z[start:start + self.batch_size], dtype=torch.float32,
                    device=self.device,
                )
                outputs.append(((self.net_(batch) - batch) ** 2).mean(1).cpu().numpy())
        return np.asarray(np.concatenate(outputs), dtype=float).reshape(-1)

    def _fit_autoencoder(self, z):
        """
        Learn an undercomplete reconstruction of standardized feature vectors.
        """
        import torch
        from torch import nn
        width = int(self.parameter)
        if width >= z.shape[1]:
            raise ValueError('Autoencoder bottleneck must be smaller than the input feature count.')
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        requested = self.requested_device
        self.device = str(torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested))
        if self.device.startswith('cuda') and (not torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested but is unavailable; use device='auto' or 'cpu'.")
        hidden_width = 64

        # ------------------------------------------------------------
        # Build the neural representation
        # ------------------------------------------------------------
        # Layer sizes and activations are fixed; only the declared parameter is tuned.
        self.net_ = nn.Sequential(
            nn.Linear(z.shape[1], hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, width),
            nn.ReLU(),
            nn.Linear(width, hidden_width),
            nn.ReLU(),
            # Linear output can reconstruct negative standardized features.
            nn.Linear(hidden_width, z.shape[1]),
        ).to(self.device)
        self.n_parameters_ = sum((p.numel() for p in self.net_.parameters()))

        # ------------------------------------------------------------
        # Create the Adam optimizer
        # ------------------------------------------------------------
        self.fixed_weight_decay_ = 1e-05
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=0.001, weight_decay=self.fixed_weight_decay_)
        data = torch.as_tensor(z, dtype=torch.float32)
        if not torch.isfinite(data).all():
            raise ValueError('Reference features exceed float32 capacity.')
        generator = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        self.optimizer_steps_, self.step_loss_history_ = (0, [])

        # ------------------------------------------------------------
        # Plan enough complete passes for the exact update budget
        # ------------------------------------------------------------
        # The loop can stop partway through its last pass; updates, not epochs, control work.
        epochs = int(np.ceil(self.max_steps / np.ceil(len(data) / self.batch_size)))

        # ------------------------------------------------------------
        # Execute the declared training budget
        # ------------------------------------------------------------
        for _ in range(epochs):
            indices = torch.randperm(len(data), generator=generator)
            total_loss, processed = (0.0, 0)
            for start in range(0, len(indices), self.batch_size):
                if self.optimizer_steps_ >= self.max_steps:
                    break
                batch = data[indices[start:start + self.batch_size]].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = ((self.net_(batch) - batch) ** 2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('MLP autoencoder loss became non-finite.')
                loss.backward()
                optimizer.step()

                # ------------------------------------------------------------
                # Record one completed optimizer update
                # ------------------------------------------------------------
                self.optimizer_steps_ += 1
                self.step_loss_history_.append(float(loss.detach().cpu()))
                record_training_checkpoint(self, data, self.fixed_weight_decay_)
                total_loss += float(loss.detach().cpu()) * len(batch)
                processed += len(batch)
            self.loss_history_.append(total_loss / processed)
        self.net_.eval()
