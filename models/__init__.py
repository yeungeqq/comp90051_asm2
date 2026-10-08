"""models: __init__ for the Amazon robustness experiment."""

from models.deep_svdd import DeepSVDDDetector
from models.feature_transformer import FeatureTransformerAutoencoderDetector
from models.lof import LocalOutlierDetector
from models.mlp_autoencoder import MLPAutoencoderDetector
from models.ocsvm import OneClassSVMDetector
from models.pca_detector import PCAAnomalyDetector, PCAZeroDetector


DETECTOR_CLASSES = {'PCA': PCAAnomalyDetector, 'OCSVM': OneClassSVMDetector,
    'LOF': LocalOutlierDetector, 'MLPAutoencoder': MLPAutoencoderDetector,
    'DeepSVDD': DeepSVDDDetector, 'TransformerAE': FeatureTransformerAutoencoderDetector}


def create_detector(name, parameter, seed, optimizer_steps, batch_size, device='auto'):
    if name not in DETECTOR_CLASSES:
        raise ValueError('Unknown model: ' + str(name))
    if name == 'PCA' and parameter == 0:
        return PCAZeroDetector(parameter, seed, optimizer_steps, batch_size, device)
    return DETECTOR_CLASSES[name](parameter, seed, optimizer_steps, batch_size, device)
