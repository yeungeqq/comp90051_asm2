"""training: protocol for the Amazon robustness experiment."""

import hashlib
import importlib.metadata
import inspect
import json
import numpy as np
import os
import platform
import subprocess
import torch
import types
from config import (
    BATCH_SIZE,
    CALIBRATION_FRACTION,
    CONDITIONS,
    CorruptionConfig,
    DEVICE,
    FEATURE_REPRESENTATIONS,
    FROZEN_GRIDS,
    FeatureConfig,
    INNER_FOLDS,
    INTEGER_PARAMETERS,
    MODEL_NAMES,
    OUTER_FOLDS,
    SEED,
    SETTINGS_FROZEN,
    STEP_BUDGETS,
    THRESHOLD_QUANTILE,
    TextureConfig,
)
from data.loader import file_sha256
from dataclasses import asdict, dataclass
from pathlib import Path
from preprocessing.features import feature_columns
from threadpoolctl import threadpool_info
from config import select_models


DEFINITION_MODULES = {'FeatureConfig': 'config',
 'read_rgbn': 'data.loader',
 'rgb_for_display': 'data.loader',
 '_index_paths': 'data.loader',
 'discover_inputs': 'data.loader',
 'classify_tags': 'data.labels',
 'build_manifest': 'data.labels',
 'label_summary': 'data.labels',
 'locate_single_input': 'data.loader',
 'file_sha256': 'data.loader',
 'verify_raw_benchmark': 'data.loader',
 'group_label_counts': 'data.splits',
 'make_grouped_folds': 'data.splits',
 'reserve_reference_calibration': 'data.splits',
 'build_nested_split_plan': 'data.splits',
 'evaluation_validate_inputs': 'evaluation.robustness',
 'CorruptionConfig': 'config',
 'image_random_generator': 'preprocessing.degradations',
 'apply_corruption': 'preprocessing.degradations',
 'corrupt_rgbn': 'preprocessing.degradations',
 '_grid_regions': 'preprocessing.features',
 '_band_features': 'preprocessing.features',
 'feature_columns': 'preprocessing.features',
 'TextureConfig': 'config',
 'quantize_luminance': 'preprocessing.features',
 'features_from_rgbn': 'preprocessing.features',
 'feature_protocol': 'preprocessing.features',
 'ConditionFeatureCache': 'preprocessing.features',
 'extract_condition_features': 'preprocessing.features',
 'plot_feature_examples': 'visualization.plots',
 'stable_ast_dump': 'preprocessing.features',
 'visible_definition_ast': 'preprocessing.features',
 'load_authenticated_features': 'preprocessing.features',
 'metric_tag_set': 'evaluation.metrics',
 'metric_ratio': 'evaluation.metrics',
 'average_precision': 'evaluation.metrics',
 'scratch_binary_metrics': 'evaluation.metrics',
 'summarize_condition_results': 'evaluation.aggregation',
 'evaluation_paired_feature_tables': 'evaluation.aggregation',
 'ReferenceAnomalyDetector': 'models.base',
 'record_training_checkpoint': 'models.base',
 'PCAAnomalyDetector': 'models.pca_detector',
 'PCAZeroDetector': 'models.pca_detector',
 'reference_median_squared_distance': 'models.ocsvm',
 'OneClassSVMDetector': 'models.ocsvm',
 'stable_negative_log_kernel_sum': 'models.ocsvm',
 'original_linear_quantile_in_log_space': 'training.calibration',
 'LocalOutlierDetector': 'models.lof',
 'MLPAutoencoderDetector': 'models.mlp_autoencoder',
 'DeepSVDDDetector': 'models.deep_svdd',
 'feature_token_indices': 'models.feature_transformer',
 'FeatureTokenReconstructionNetwork': 'models.feature_transformer',
 'FeatureTransformerAutoencoderDetector': 'models.feature_transformer',
 'create_detector': 'models',
 'json_value': 'training.protocol',
 'canonical_hash': 'training.protocol',
 'code_signature': 'training.protocol',
 'signature_value': 'training.protocol',
 'function_signature': 'training.protocol',
 'active_science_identity': 'training.protocol',
 'runtime_identity': 'training.protocol',
 'validate_frozen_settings': 'training.protocol',
 'evaluation_training_rank': 'preprocessing.scaling',
 'evaluation_check_formal_pca_support': 'training.tuning',
 'training_plateau': 'training.trainers',
 'write_json': 'training.trainers',
 'restore_unit': 'training.trainers',
 'finish_unit': 'training.trainers',
 'unit_identity': 'training.trainers',
 'save_training_evidence': 'training.trainers',
 'run_inner_unit': 'training.trainers',
 'tune_all_arms': 'training.tuning',
 'run_outer_unit': 'training.trainers',
 'evaluation_modal_audit': 'evaluation.aggregation',
 'run_complete_comparison': 'evaluation.robustness'}


SCIENTIFIC_DEFINITIONS = ('FeatureConfig',
 'read_rgbn',
 'rgb_for_display',
 '_index_paths',
 'discover_inputs',
 'classify_tags',
 'build_manifest',
 'label_summary',
 'locate_single_input',
 'file_sha256',
 'verify_raw_benchmark',
 'group_label_counts',
 'make_grouped_folds',
 'reserve_reference_calibration',
 'build_nested_split_plan',
 'evaluation_validate_inputs',
 'CorruptionConfig',
 'image_random_generator',
 'apply_corruption',
 'corrupt_rgbn',
 '_grid_regions',
 '_band_features',
 'feature_columns',
 'TextureConfig',
 'quantize_luminance',
 'features_from_rgbn',
 'feature_protocol',
 'ConditionFeatureCache',
 'extract_condition_features',
 'plot_feature_examples',
 'stable_ast_dump',
 'visible_definition_ast',
 'load_authenticated_features',
 'metric_tag_set',
 'metric_ratio',
 'average_precision',
 'scratch_binary_metrics',
 'summarize_condition_results',
 'evaluation_paired_feature_tables',
 'ReferenceAnomalyDetector',
 'record_training_checkpoint',
 'PCAAnomalyDetector',
 'PCAZeroDetector',
 'reference_median_squared_distance',
 'OneClassSVMDetector',
 'stable_negative_log_kernel_sum',
 'original_linear_quantile_in_log_space',
 'LocalOutlierDetector',
 'MLPAutoencoderDetector',
 'DeepSVDDDetector',
 'feature_token_indices',
 'FeatureTokenReconstructionNetwork',
 'FeatureTransformerAutoencoderDetector',
 'create_detector',
 'json_value',
 'canonical_hash',
 'code_signature',
 'signature_value',
 'function_signature',
 'active_science_identity',
 'runtime_identity',
 'validate_frozen_settings',
 'evaluation_training_rank',
 'evaluation_check_formal_pca_support',
 'training_plateau',
 'write_json',
 'restore_unit',
 'finish_unit',
 'unit_identity',
 'save_training_evidence',
 'run_inner_unit',
 'tune_all_arms',
 'run_outer_unit',
 'evaluation_modal_audit',
 'run_complete_comparison')


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    default=json_value, allow_nan=False).encode()).hexdigest()


def code_signature(code):
    """Describe executable bytecode without notebook cell paths or line numbers."""
    constants = []
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            constants.append(code_signature(item))
        elif isinstance(item, (str, int, float, bool, type(None))):
            constants.append(item)
        else:
            constants.append(repr(item))
    return {'bytecode': code.co_code.hex(), 'constants': constants,
            'names': code.co_names, 'variables': code.co_varnames,
            'freevars': code.co_freevars, 'cellvars': code.co_cellvars,
            'argcount': code.co_argcount, 'posonly': code.co_posonlyargcount,
            'kwonly': code.co_kwonlyargcount, 'flags': code.co_flags,
            'exceptiontable': getattr(code, 'co_exceptiontable', b'').hex()}


def signature_value(value):
    """Represent defaults and scientific constants without memory-address reprs."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [signature_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted([signature_value(item) for item in value], key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(value, dict):
        return {str(key): signature_value(item) for key, item in value.items()}
    if hasattr(value, '__dataclass_fields__') and not inspect.isclass(value):
        return {'configuration_class': type(value).__qualname__, 'fields': signature_value(asdict(value))}
    if isinstance(value, types.CodeType):
        return code_signature(value)
    if inspect.isclass(value):
        return {'class': value.__module__ + '.' + value.__qualname__}
    if inspect.isfunction(value):
        return {'function_code': code_signature(value.__code__),
                'defaults': signature_value(value.__defaults__)}
    raise TypeError('Unrecorded scientific default or closure value: ' + type(value).__name__)


def function_signature(function):
    scientific_globals = {}
    for key in function.__code__.co_names:
        value = function.__globals__.get(key)
        # Modules and callables are recorded through package versions and the
        # explicit definition list. Literal constants must be captured too.
        if isinstance(value, (str, int, float, bool, tuple, list, dict, set, frozenset)):
            if key not in ('SCIENTIFIC_DEFINITIONS', 'IMPLEMENTATION_ID'):
                scientific_globals[key] = signature_value(value)
    return {'code': code_signature(function.__code__),
            'defaults': signature_value(function.__defaults__),
            'keyword_defaults': signature_value(function.__kwdefaults__),
            'closure': [signature_value(c.cell_contents) for c in (function.__closure__ or ())],
            'globals': scientific_globals}


def active_science_identity():
    """An edited function must not silently reuse results from its old definition."""
    definitions = {}
    for name in SCIENTIFIC_DEFINITIONS:
        value = resolve_scientific_definition(name)
        if inspect.isfunction(value):
            definitions[name] = function_signature(value)
        elif inspect.isclass(value):
            members = {}
            for key, member in vars(value).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member):
                    members[key] = function_signature(member)
                elif not key.startswith('__') and isinstance(member, (str, int, float, bool, tuple, list, dict, set, frozenset, type(None))):
                    members[key] = signature_value(member)
            definitions[name] = members
    return canonical_hash({'definitions': definitions, 'module_sources': module_source_hashes()})


def runtime_identity():
    """Library and hardware details are part of a result's identity."""
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit-image', 'tifffile', 'torch')}
    driver = None
    if torch.cuda.is_available():
        driver = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version',
                                          '--format=csv,noheader'], text=True).strip().splitlines()
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'versions': versions, 'device': DEVICE, 'torch_cuda': torch.version.cuda,
            'cuda_driver': driver, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_version': torch.backends.cudnn.version(),
            'threadpools': threadpool_info(),
            'thread_environment': {name: os.environ.get(name) for name in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


def validate_frozen_settings(models=None):
    """Validate distinct effective candidates; never move a winner to the middle."""
    models = select_models(models)
    if not SETTINGS_FROZEN:
        raise ValueError('Parameter candidates have not been frozen in the setup section.')
    if DEVICE != 'cuda' or not torch.cuda.is_available():
        raise ValueError('The declared full experiment requires CUDA. There is no silent CPU fallback.')
    if set(FROZEN_GRIDS) != set(FEATURE_REPRESENTATIONS):
        raise ValueError('All three feature groups are required.')
    for feature, grids in FROZEN_GRIDS.items():
        if set(grids) != set(MODEL_NAMES):
            raise ValueError('Each feature group must contain the same six methods.')
        for model, values in grids.items():
            if len(values) != 3 or values != sorted(set(values)) or not np.isfinite(values).all():
                raise ValueError('Each arm needs three distinct ordered finite candidates.')
            if min(values) < 0 or (model != 'PCA' and min(values) == 0):
                raise ValueError('Only mean-only PCA may use zero.')
            if model in INTEGER_PARAMETERS and any(int(v) != v for v in values):
                raise ValueError('This method needs integer candidates: ' + model)
    return {'models': list(models), 'grids': FROZEN_GRIDS, 'budgets': STEP_BUDGETS, 'seed': SEED,
            'outer_folds': OUTER_FOLDS, 'inner_folds': INNER_FOLDS,
            'calibration_fraction': CALIBRATION_FRACTION, 'threshold_quantile': THRESHOLD_QUANTILE,
            'batch_size': BATCH_SIZE, 'cohort_exposed': True,
            'selection_rule': 'highest_equal_mean_inner_AP_first_ascending_exact_tie',
            'threshold_rule': 'clean_R_linear_95_percentile_strict_greater',
            'features': feature_columns(), 'conditions': list(CONDITIONS),
            'feature_config': asdict(FeatureConfig()), 'texture_config': asdict(TextureConfig()),
            'corruption_config': asdict(CorruptionConfig())}


def resolve_scientific_definition(name):
    """Resolve the notebook's explicit scientific definition across source modules."""
    return getattr(importlib.import_module(DEFINITION_MODULES[name]), name)


def module_source_hashes():
    """Include all Python source so moved/edited helpers invalidate saved units."""
    project = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(project)): file_sha256(path)
            for path in sorted(project.rglob('*.py'))
            if not any(part in {'.venv', '__pycache__', 'outputs', 'tests'} for part in path.relative_to(project).parts)}
