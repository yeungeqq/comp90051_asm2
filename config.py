"""Experiment protocol, feature settings, and Kaggle paths. Imports never start training.

Edit scientific settings before importing the program in a fresh Python process.
Generated cohorts and their split identities are recorded with each result bundle.
"""

import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path

# Execution, data paths, and frozen experiment settings.
RUN_EXPERIMENT = True
SETTINGS_FROZEN = True
INPUT_ROOT = Path('/kaggle/input')
LABELS_PATH = None
OUTPUT_ROOT = Path('/kaggle/working/comp90051-amazon-anomaly/outputs')
SEED = 51
OUTER_FOLDS = 10
INNER_FOLDS = 3
CALIBRATION_FRACTION = 0.20
THRESHOLD_QUANTILE = 0.95
BATCH_SIZE = 128
DEVICE = 'cuda'
USE_FIXED_FEATURE_CACHE = False
MODEL_NAMES = ('PCA', 'OCSVM', 'LOF', 'MLPAutoencoder', 'DeepSVDD', 'TransformerAE')
CORE_MODELS = ('LOF', 'OCSVM', 'DeepSVDD')
FEATURE_REPRESENTATIONS = ('A_RGB', 'B_RGB_NIR', 'C_RGB_NIR_TEXTURE')
STEP_BUDGETS = {'MLPAutoencoder': 3840, 'DeepSVDD': 7680, 'TransformerAE': 3840}
TRAINING_PLATEAU_WINDOW = 60
TRAINING_PLATEAU_COMPARISONS = 3
TRAINING_PLATEAU_RELATIVE_CHANGE = 0.01
CANDIDATE_MODELS = ALGORITHM_NAMES = MODEL_NAMES
PRIMARY_ALGORITHMS = CORE_MODELS
NEURAL_ALGORITHMS = tuple(STEP_BUDGETS)
MODEL_ROLES = {name: 'baseline' if name == 'PCA' else 'candidate' for name in MODEL_NAMES}
PARAMETER_NAMES = {'PCA': 'retained_components', 'OCSVM': 'median_distance_gamma_multiplier',
                  'LOF': 'n_neighbors', 'MLPAutoencoder': 'bottleneck_width',
                  'DeepSVDD': 'learning_rate', 'TransformerAE': 'transformer_width'}
INTEGER_PARAMETERS = {'PCA', 'LOF', 'MLPAutoencoder', 'TransformerAE'}
EVALUATION_METRICS = ('AP', 'recall_P', 'FPR_N', 'FPR_R', 'precision')
METRIC_TARGET_TAGS = ('road', 'selective_logging', 'slash_burn')
GROUPS = ("R", "P", "N")
TARGET_TAGS = frozenset({"road", "selective_logging", "slash_burn"})
NATURAL_TAGS = frozenset({"water", "blooming", "blow_down"})
HUMAN_TAGS = frozenset({"agriculture", "cultivation", "habitation", "road",
    "selective_logging", "slash_burn", "conventional_mine", "artisinal_mine"})
KNOWN_TAGS = frozenset({"agriculture", "artisinal_mine", "bare_ground", "blooming",
    "blow_down", "clear", "cloudy", "conventional_mine", "cultivation",
    "habitation", "haze", "partly_cloudy", "primary", "road", "selective_logging",
    "slash_burn", "water"})
WEATHER_TAGS = frozenset({"clear", "cloudy", "partly_cloudy", "haze"})
COHORT_GROUP_COUNTS = {'N': 238, 'P': 705, 'R': 1457}
EXPECTED_SPLIT_SHA256 = None  # Generated cohort splits are identified at runtime.
FEATURE_PACKAGE_SHA256 = 'cf600855b6304686d843fd2a5b942ea87ebc4ff129e844d3025fefcbe50f7ab5'
CONDITIONS = ("clean", "occlusion_05", "occlusion_10", "occlusion_20", "resolution", "blur", "noise_low", "noise_high", "noise_stress")
PCA_ZERO_POLICY = 'explicit_mean_only_standardized_training_reconstruction'
FROZEN_GRIDS = {'A_RGB': {'PCA': [0, 1, 2],
           'OCSVM': [64.0, 128.0, 256.0],
           'LOF': [2, 5, 10],
           'MLPAutoencoder': [2, 4, 16],
           'DeepSVDD': [0.0001, 0.0003, 0.0005477225575051661],
           'TransformerAE': [8, 32, 128]},
 'B_RGB_NIR': {'PCA': [16, 45, 64],
               'OCSVM': [16, 32, 64],
               'LOF': [80, 320, 512],
               'MLPAutoencoder': [2, 4, 16],
               'DeepSVDD': [0.0001, 0.00031622776601683794, 0.001],
               'TransformerAE': [32, 64, 128]},
 'C_RGB_NIR_TEXTURE': {'PCA': [1, 4, 16],
                       'OCSVM': [8.0, 32.0, 128.0],
                       'LOF': [80, 320, 512],
                       'MLPAutoencoder': [2, 4, 16],
                       'DeepSVDD': [1.1111111111111112e-06, 1e-05, 0.0001],
                       'TransformerAE': [8, 16, 32]}}


def select_models(models=None):
    """Validate a requested subset and return it in the declared model order."""
    if models is None:
        return MODEL_NAMES
    requested = (models,) if isinstance(models, str) else tuple(models)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('Choose at least one model, with no duplicates.')
    unknown = set(requested) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f'Unknown models: {sorted(unknown)}. Choose from {MODEL_NAMES}.')
    return tuple(name for name in MODEL_NAMES if name in requested)


@dataclass(frozen=True)
class FeatureConfig:
    """Fixed settings for reading images and calculating numerical features.

    dn_scale is the uint16 encoding maximum, not an image-specific maximum.
    Dividing by it changes units; it is not a physical reflectance calibration.
    texture_epsilon handles an image whose percentile range is effectively zero.
    No setting is estimated using the evaluation labels or predictions.
    """

    source_band_order: str = "BGRN"  # Official TIFF order, NOT RGBN.
    dn_scale: float = 65535.0  # Fixed uint16 encoding divisor, not dataset min/max.
    minimum_side: int = 100  # Assignment's minimum original image dimension.
    grid_size: int = 4  # 4 by 4 gives 16 local means or edge-density features.
    texture_epsilon: float = 1e-8
    edge_threshold: float = 102.375 / 65535.0  # Fixed Sobel cutoff in raw DN units.
    glcm_levels: int = 32  # Fixed quantisation, independent of training/test data.

    def __post_init__(self):
        if sorted(self.source_band_order) != sorted("RGBN"):
            raise ValueError("source_band_order must contain R, G, B, N exactly once.")
        if self.dn_scale <= 0 or self.texture_epsilon <= 0:
            raise ValueError("dn_scale and texture_epsilon must be positive.")
        if self.grid_size != 4:
            raise ValueError("The planned 63/84/108-feature design requires grid_size=4.")
        if self.minimum_side < 4 or self.glcm_levels < 2:
            raise ValueError("Image side must be >= 4 and GLCM levels >= 2.")


@dataclass(frozen=True)
class CorruptionConfig:
    """Fixed synthetic stress settings in raw digital-number (DN) units."""
    occlusion_fill: float = 0.0
    resolution_scale: float = 0.5
    blur_sigma: float = 1.5
    noise_low_fraction: float = 0.0001
    noise_high_fraction: float = 0.001
    noise_stress_fraction: float = 0.005
    dn_scale: float = 65535.0

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Corruption settings must be finite.")
        if not 0 < self.resolution_scale < 1 or self.blur_sigma <= 0:
            raise ValueError("Resolution scale must be in (0, 1) and blur sigma positive.")
        if self.dn_scale <= 0 or not 0 <= self.occlusion_fill <= self.dn_scale:
            raise ValueError("DN scale and mask fill are incompatible.")
        if not 0 < self.noise_low_fraction < self.noise_high_fraction < self.noise_stress_fraction:
            raise ValueError("Noise levels must satisfy 0 < low < high < stress.")


@dataclass(frozen=True)
class TextureConfig:
    """Frozen within-image percentile limits for the active GLCM definition."""
    lower_percentile: float = 2.0
    upper_percentile: float = 98.0

    def __post_init__(self):
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Texture limits must satisfy 0 <= lower < upper <= 100.")
