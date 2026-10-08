"""training: trainers for the Amazon robustness experiment."""

import json
import numpy as np
import pandas as pd
from config import (
    BATCH_SIZE,
    CONDITIONS,
    DEVICE,
    SEED,
    STEP_BUDGETS,
    THRESHOLD_QUANTILE,
    TRAINING_PLATEAU_COMPARISONS,
    TRAINING_PLATEAU_RELATIVE_CHANGE,
    TRAINING_PLATEAU_WINDOW,
)
from data.loader import file_sha256
from evaluation.metrics import average_precision, scratch_binary_metrics
from models import create_detector
from pathlib import Path
from training.calibration import original_linear_quantile_in_log_space
from training.protocol import canonical_hash, json_value


def training_plateau(losses, budget):
    """Three consecutive <=1% changes between four 60-step checkpoints.

    `losses` contains checkpoint dictionaries, not shuffled minibatch losses.
    Each checkpoint evaluates the same entire training R set without updates.
    Both decreases and increases must be small. Missing checkpoints fail closed.
    """
    required_steps = [int(budget) - TRAINING_PLATEAU_WINDOW * offset
                      for offset in range(TRAINING_PLATEAU_COMPARISONS, -1, -1)]
    if required_steps[0] < TRAINING_PLATEAU_WINDOW:
        return False, []
    try:
        steps = [int(row["optimizer_steps"]) for row in losses]
        if len(set(steps)) != len(steps) or steps != sorted(steps):
            return False, []
        history = {int(row["optimizer_steps"]): float(row["objective"])
                   for row in losses if int(row["optimizer_steps"]) <= int(budget)}
        objectives = np.asarray([history[step] for step in required_steps])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, []
    if not np.isfinite(objectives).all():
        return False, []
    changes = np.abs(np.diff(objectives)) / np.maximum(np.abs(objectives[:-1]), 1e-12)
    return bool(np.all(changes <= TRAINING_PLATEAU_RELATIVE_CHANGE)), changes.tolist()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False))
    temporary.replace(path)


def restore_unit(directory, identity):
    """No receipt means unfinished work. A corrupt completed receipt is an error."""
    receipt_path = directory / 'receipt.json'
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if canonical_hash(receipt['identity']) != canonical_hash(identity):
        raise ValueError('A result directory contains a different scientific unit.')
    for name, digest in receipt['files'].items():
        if file_sha256(directory / name) != digest:
            raise ValueError('A completed result file changed: ' + name)
    return receipt


def finish_unit(directory, identity, summary):
    """Write the receipt last, after all numerical evidence is safely on disk."""
    files = {p.name: file_sha256(p) for p in directory.iterdir()
             if p.is_file() and p.name not in ('receipt.json', 'receipt.json.tmp')}
    receipt = {'identity': identity, 'summary': summary, 'files': files}
    write_json(directory / 'receipt.json', receipt)
    return receipt


def unit_identity(common, stage, outer, feature, model, parameter, fit, validation,
                  manifest, seed, inner=None, calibration=None):
    """Ordered row identities prevent accidentally swapping image results."""
    return {'experiment': common, 'stage': stage, 'outer_fold': outer,
            'inner_fold': inner, 'features': feature, 'model': model,
            'parameter': parameter, 'seed': seed, 'steps': STEP_BUDGETS.get(model),
            'fit_ids': manifest.iloc[fit].image_id.tolist(),
            'validation_ids': manifest.iloc[validation].image_id.tolist(),
            'calibration_ids': [] if calibration is None else manifest.iloc[calibration].image_id.tolist()}


def save_training_evidence(detector, directory):
    """Keep actual loss history, fixed-R checkpoints, and requested/effective values."""
    diagnostics = {
        'parameter': detector.parameter, 'effective_parameter': detector.effective_parameter_,
        'seed': detector.seed, 'requested_steps': detector.max_steps,
        'optimizer_steps': detector.optimizer_steps_, 'batch_size': detector.batch_size,
        'device': detector.device, 'fit_seconds': detector.fit_seconds,
        'loss_history': detector.loss_history_, 'step_loss_history': detector.step_loss_history_,
        'training_checkpoints': detector.training_checkpoints_,
        'gamma_reference_median_sqdist': getattr(detector, 'gamma_reference_median_sqdist_', None),
        'gamma_reference_sample_count': getattr(detector, 'gamma_reference_sample_count_', None),
        'training_budget_is_resource_cap': detector.name in STEP_BUDGETS,
    }
    if detector.name in STEP_BUDGETS:
        plateau, changes = training_plateau(detector.training_checkpoints_, detector.max_steps)
        diagnostics['plateau_at_cap'] = bool(plateau)
        diagnostics['last_checkpoint_relative_changes'] = changes
    else:
        diagnostics['plateau_at_cap'] = None
    write_json(directory / 'training.json', diagnostics)
    return diagnostics


def run_inner_unit(manifest, clean, fit, validation, outer, inner, feature, model,
                   parameter, common, output):
    seed = SEED + outer * 10000 + inner * 100
    identity = unit_identity(common, 'inner', outer, feature, model, parameter,
                             fit, validation, manifest, seed, inner)
    directory = output / 'inner_units' / canonical_hash(identity)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = restore_unit(directory, identity)
    if receipt is not None:
        return receipt['summary'], True
    detector = create_detector(model, parameter, seed, STEP_BUDGETS.get(model), BATCH_SIZE, DEVICE)
    detector.fit(clean[feature][fit])
    scores = detector.score_samples(clean[feature][validation])
    rows = manifest.iloc[validation][['image_id', 'group', 'tags']].reset_index(drop=True)
    rows['score'] = scores
    rows.to_csv(directory / 'validation_scores.csv', index=False, float_format='%.17g')
    training = save_training_evidence(detector, directory)
    summary = {'outer_fold': outer, 'inner_fold': inner, 'features': feature, 'model': model,
               'parameter': parameter, 'AP': average_precision(rows.group.eq('P').to_numpy(int), scores),
               'effective_parameter': detector.effective_parameter_, 'n_fit_R': len(fit),
               'n_validation': len(validation), 'optimizer_steps': detector.optimizer_steps_,
               'unit_key': directory.name, 'score_unique_count': int(np.unique(scores).size)}
    if not np.isfinite(summary['AP']):
        raise ValueError('Inner AP is undefined; no replacement value is inserted.')
    finish_unit(directory, identity, summary)
    return summary, False


def run_outer_unit(manifest, features, plan, outer, feature, model, parameter, common, output):
    held, fit, calibration, _ = plan
    seed = SEED + outer * 10000 + 900
    identity = unit_identity(common, 'outer', outer, feature, model, parameter,
                            fit, held, manifest, seed, calibration=calibration)
    directory = output / 'outer_units' / canonical_hash(identity)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = restore_unit(directory, identity)
    if receipt is not None:
        return (pd.read_csv(directory / 'metrics.csv', float_precision='round_trip'),
                pd.read_csv(directory / 'predictions.csv', float_precision='round_trip',
                            dtype={'image_id': str, 'group': str, 'tags': str}), receipt['summary'], True)
    detector = create_detector(model, parameter, seed, STEP_BUDGETS.get(model), BATCH_SIZE, DEVICE)
    detector.fit(features['clean'][feature][fit])
    calibration_scores = detector.score_samples(features['clean'][feature][calibration])
    if model == 'OCSVM':
        threshold = original_linear_quantile_in_log_space(calibration_scores, THRESHOLD_QUANTILE)
    else:
        threshold = float(np.quantile(calibration_scores, THRESHOLD_QUANTILE, method='linear'))
    calibration_rows = manifest.iloc[calibration][['image_id', 'group', 'tags']].reset_index(drop=True)
    calibration_rows['score'] = calibration_scores
    calibration_rows.to_csv(directory / 'calibration_scores.csv', index=False, float_format='%.17g')
    save_training_evidence(detector, directory)
    predictions, metrics = [], []
    for condition in CONDITIONS:
        scores = detector.score_samples(features[condition][feature][held])
        flags = scores > threshold
        rows = manifest.iloc[held][['image_id', 'group', 'tags']].reset_index(drop=True)
        rows = rows.assign(outer_fold=outer, features=feature, model=model,
                           condition=condition, score=scores, alert=flags, threshold=threshold)
        predictions.append(rows)
        metrics.append({'outer_fold': outer, 'features': feature, 'model': model,
                        'condition': condition, 'threshold': threshold,
                        'score_unique_count': int(np.unique(scores).size),
                        **scratch_binary_metrics(rows.group, scores, flags, rows.tags)})
    predictions = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(metrics)
    predictions.to_csv(directory / 'predictions.csv', index=False, float_format='%.17g')
    metrics.to_csv(directory / 'metrics.csv', index=False, float_format='%.17g')
    summary = {'outer_fold': outer, 'features': feature, 'model': model,
               'parameter': parameter, 'effective_parameter': detector.effective_parameter_,
               'threshold': threshold, 'n_fit_R': len(fit), 'n_calibration_R': len(calibration),
               'n_held_out': len(held), 'optimizer_steps': detector.optimizer_steps_,
               'fit_seconds': detector.fit_seconds, 'unit_key': directory.name}
    finish_unit(directory, identity, summary)
    return metrics, predictions, summary, False
