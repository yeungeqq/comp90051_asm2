"""preprocessing: scaling for the Amazon robustness experiment."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def evaluation_training_rank(x):
    """Match the detector's training-only standardization and rank calculation."""
    z = StandardScaler().fit_transform(np.asarray(x, dtype=float))
    return int(np.linalg.matrix_rank(z - z.mean(axis=0)))
