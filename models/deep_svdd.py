"""models: deep_svdd for the Amazon robustness experiment."""

import numpy as np
import torch
from models.base import ReferenceAnomalyDetector, record_training_checkpoint
from preprocessing.scaling import StandardScaler


class DeepSVDDDetector(ReferenceAnomalyDetector):
    """
    Learn a representation close to a fixed reference-only centre.

    Steps
    -----
    1. Fit scaling using the supplied clean reference training rows.
    2. Learn the model described in the named blocks below.
    3. Reuse the fitted scaler and model when scoring new rows.

    Parameters
    ----------
    parameter : float
        Adam learning rate; weight decay stays fixed at 1e-4.
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
    model_name = 'DeepSVDD'

    def _fit_model(self, x):
        if not 0 < self.parameter <= 0.1:
            raise ValueError('Deep SVDD learning rate must lie in (0, 0.1].')

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
        self.fixed_weight_decay_ = 1e-4
        self._fit_svdd(z)
        self.device = str(self.device)
        self.effective_parameter_ = self.parameter
        self.n_parameters_ = int(sum(p.numel() for p in self.net_.parameters()))

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
                outputs.append(((self.net_(batch) - self.center_) ** 2).sum(1).cpu().numpy())
        return np.asarray(np.concatenate(outputs), dtype=float)

    def _fit_svdd(self, z):
        import torch
        from torch import nn
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        requested = self.requested_device
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested)

        # ------------------------------------------------------------
        # Build the neural representation
        # ------------------------------------------------------------
        # Layer sizes and activations are fixed; only the declared parameter is tuned.
        self.net_ = nn.Sequential(
            # Bias-free layers help avoid a trivial constant representation.
            nn.Linear(z.shape[1], 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
        ).to(self.device)
        data = torch.as_tensor(z, dtype=torch.float32)
        if not torch.isfinite(data).all():
            raise ValueError('Reference features exceed float32 capacity.')
        self.net_.eval()

        # ------------------------------------------------------------
        # Compute without gradient storage
        # ------------------------------------------------------------
        with torch.no_grad():
            sums = torch.zeros(16, device=self.device)
            for start in range(0, len(data), self.batch_size):
                sums += self.net_(data[start:start + self.batch_size].to(self.device)).sum(0)

            # ------------------------------------------------------------
            # Fix the centre using initial reference representations
            # ------------------------------------------------------------
            # No validation or held-out image contributes to this centre.
            center = sums / len(data)
            eps = 0.1
            center[(center.abs() < eps) & (center < 0)] = -eps
            center[(center.abs() < eps) & (center >= 0)] = eps
            self.center_ = center.detach()

        # ------------------------------------------------------------
        # Create the Adam optimizer
        # ------------------------------------------------------------
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.parameter, weight_decay=self.fixed_weight_decay_)
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history_ = []
        self.net_.train()
        self.optimizer_steps_, self.step_loss_history_ = (0, [])
        max_steps = self.max_steps

        # ------------------------------------------------------------
        # Plan enough complete passes for the exact update budget
        # ------------------------------------------------------------
        # The loop can stop partway through its last pass; updates, not epochs, control work.
        epochs = int(np.ceil(max_steps / np.ceil(len(data) / self.batch_size)))

        # ------------------------------------------------------------
        # Execute the declared training budget
        # ------------------------------------------------------------
        for _ in range(epochs):
            indices = torch.randperm(len(data), generator=generator)
            total_loss, processed = (0.0, 0)
            for start in range(0, len(indices), self.batch_size):
                if self.optimizer_steps_ >= max_steps:
                    break
                batch = data[indices[start:start + self.batch_size]].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = ((self.net_(batch) - self.center_) ** 2).sum(dim=1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Deep SVDD loss became non-finite.')
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
