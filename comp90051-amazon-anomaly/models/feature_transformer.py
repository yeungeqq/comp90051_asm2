"""models: feature_transformer for the Amazon robustness experiment."""

import numpy as np
import random
import torch
from models.base import ReferenceAnomalyDetector, record_training_checkpoint
from preprocessing.features import feature_columns
from preprocessing.scaling import StandardScaler
from torch import nn


def feature_token_indices(columns):
    """
    Map every source feature exactly once into 16 spatial + 1 global tokens.
    """
    grids = []
    for cell in range(16):
        names = [
            f'{band}_grid_{cell:02d}_mean'
            for band in ('red', 'green', 'blue', 'nir')
            if f'{band}_grid_{cell:02d}_mean' in columns
        ]
        edge = f'edge_grid_{cell:02d}_density'
        if edge in columns:
            names.append(edge)
        grids.append([columns.index(name) for name in names])
    if len({len(group) for group in grids}) != 1:
        raise ValueError('Spatial tokens have inconsistent channel counts.')

    # ------------------------------------------------------------
    # Group spatial engineered features
    # ------------------------------------------------------------
    # Each input row has F features; grouping produces [B, 16, spatial_width].
    spatial = {i for group in grids for i in group}
    global_indices = [i for i in range(len(columns)) if i not in spatial]
    flat = [i for group in grids for i in group] + global_indices
    if len(flat) != len(columns) or len(set(flat)) != len(columns):
        raise ValueError('Token layout must cover every feature exactly once.')
    return (grids, global_indices)


class FeatureTokenReconstructionNetwork(nn.Module):
    """
    Reconstruct engineered statistical features with attention.

    Steps
    -----
    1. Group features into 16 spatial tokens and one global token.
    2. Project both token types to the same embedding width.
    3. Add learned positional embeddings and apply one attention layer.
    4. Decode the contextual global token into the full feature vector.

    Parameters
    ----------
    columns : list of str
        Ordered names of the 63, 84 or 108 engineered features.
    width : int
        Token embedding width; a multiple of four, at least eight.

    Returns
    -------
    forward(x) : tensor, shape [batch_size, n_features]
        Reconstructed features. Internal tokens have shape [batch_size, 17, width].

    Notes
    -----
    These tokens hold feature summaries, not pixel patches. No random masking is
    performed here. Image corruptions are defined outside the detector.
    """

    def __init__(self, columns, width):
        super().__init__()
        if width < 8 or width % 4:
            raise ValueError('Transformer width must be a multiple of four >= 8.')
        grids, globals_ = feature_token_indices(columns)
        self.grid_indices = grids
        self.global_indices = globals_
        self.grid_projection = nn.Linear(len(grids[0]), width)
        self.global_projection = nn.Linear(len(globals_), width)
        self.position = nn.Parameter(torch.zeros(1, 17, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.decoder = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, len(columns)),
        )

    def forward(self, x):
        """
        Reconstruct one batch of standardized features.

        Parameters
        ----------
        x : tensor, shape [B, F]
            B images represented by F standardized engineered features.

        Returns
        -------
        reconstruction : tensor, shape [B, F]
            The decoder uses the attention-updated global token, at index zero.
        """

        # ------------------------------------------------------------
        # Group spatial engineered features
        # ------------------------------------------------------------
        # Each input row has F features; grouping produces [B, 16, spatial_width].
        spatial = torch.stack([x[:, indices] for indices in self.grid_indices], dim=1)
        global_token = self.global_projection(x[:, self.global_indices]).unsqueeze(1)
        tokens = torch.cat([global_token, self.grid_projection(spatial)], dim=1)

        # ------------------------------------------------------------
        # Share information across all 17 tokens
        # ------------------------------------------------------------
        # The global token can attend to each spatial summary.
        encoded = self.encoder(tokens + self.position)
        return self.decoder(encoded[:, 0])


class FeatureTransformerAutoencoderDetector(ReferenceAnomalyDetector):
    """
    Attention-based reconstruction of engineered statistical features.

    Steps
    -----
    1. Fit scaling using the supplied clean reference training rows.
    2. Learn the model described in the named blocks below.
    3. Reuse the fitted scaler and model when scoring new rows.

    Parameters
    ----------
    parameter : float
        Embedding width for each of 17 engineered-feature tokens.
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
    model_name = 'TransformerAE'

    def _fit_model(self, x):
        columns = next((items for items in feature_columns().values() if len(items) == x.shape[1]), None)
        if columns is None:
            raise ValueError('Transformer feature dimensions do not match A/B/C.')
        self.columns, self.width = list(columns), int(self.parameter)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.requested_device == 'auto' else self.requested_device
        if np.max(np.abs(x)) > np.finfo(np.float32).max:
            raise ValueError('Reference features exceed float32 capacity.')
        x = np.asarray(x, dtype=np.float32)
        if len(x) < 3 or x.shape[1] != len(self.columns):
            raise ValueError('Training rows must be R-only and match the declared feature set.')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # ------------------------------------------------------------
        # Fit the reference-only scaler
        # ------------------------------------------------------------
        # Never learn scaling from calibration, disturbed or outer-test rows.
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError('Standardized reference features must be finite.')
        if not np.any(np.var(z, axis=0) > 0):
            raise ValueError('All training features are constant; detector is undefined.')

        # ------------------------------------------------------------
        # Build the neural representation
        # ------------------------------------------------------------
        # Layer sizes and activations are fixed; only the declared parameter is tuned.
        self.net_ = FeatureTokenReconstructionNetwork(self.columns, self.width).to(self.device)

        # ------------------------------------------------------------
        # Create the Adam optimizer
        # ------------------------------------------------------------
        self.fixed_weight_decay_ = 0.0
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=0.001, weight_decay=self.fixed_weight_decay_)
        data = torch.from_numpy(z)
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        self.loss_history_ = []
        self.optimizer_steps_, self.step_loss_history_ = (0, [])
        self.net_.train()

        # ------------------------------------------------------------
        # Plan enough complete passes for the exact update budget
        # ------------------------------------------------------------
        # The loop can stop partway through its last pass; updates, not epochs, control work.
        epochs = int(np.ceil(self.max_steps / len(loader)))

        # ------------------------------------------------------------
        # Execute the declared training budget
        # ------------------------------------------------------------
        for _ in range(epochs):
            losses = []
            for batch, in loader:
                if self.optimizer_steps_ >= self.max_steps:
                    break
                batch = batch.to(self.device)
                optimiser.zero_grad(set_to_none=True)
                loss = torch.mean((self.net_(batch) - batch) ** 2)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite Transformer reconstruction loss.')
                loss.backward()
                optimiser.step()

                # ------------------------------------------------------------
                # Record one completed optimizer update
                # ------------------------------------------------------------
                self.optimizer_steps_ += 1
                losses.append(float(loss.detach().cpu()))
                self.step_loss_history_.append(float(loss.detach().cpu()))
                record_training_checkpoint(self, data, self.fixed_weight_decay_)
            self.loss_history_.append(float(np.mean(losses)))

        self.net_.eval()
        self.effective_parameter_ = int(self.parameter)
        self.n_parameters_ = sum(p.numel() for p in self.net_.parameters())

    def _score_model(self, x):
        z = self.scaler.transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
        self.net_.eval()
        parts = []

        # ------------------------------------------------------------
        # Compute without gradient storage
        # ------------------------------------------------------------
        with torch.no_grad():
            for start in range(0, len(z), 512):
                batch = torch.from_numpy(z[start:start + 512]).to(self.device)
                parts.append(torch.mean((self.net_(batch) - batch) ** 2, dim=1).cpu().numpy())
        scores = np.concatenate(parts) if parts else np.empty(0)
        if len(scores) != len(z) or not np.isfinite(scores).all():
            raise RuntimeError('Invalid Transformer anomaly scores.')
        return scores
