"""evaluation: robustness for the Amazon robustness experiment."""

import config
import hashlib
import numpy as np
import pandas as pd
import shutil
import zipfile
from pathlib import Path
from config import (
    CALIBRATION_FRACTION,
    CONDITIONS,
    EXPECTED_SPLIT_SHA256,
    FEATURE_REPRESENTATIONS,
    FROZEN_GRIDS,
    GROUPS,
    INNER_FOLDS,
    MODEL_NAMES,
    OUTER_FOLDS,
    SEED,
    select_models,
)
from data.loader import file_sha256, locate_single_input
from data.splits import build_nested_split_plan
from evaluation.aggregation import (
    evaluation_modal_audit,
    evaluation_paired_feature_tables,
    summarize_condition_results,
)
from training.protocol import (
    active_science_identity,
    canonical_hash,
    runtime_identity,
    validate_frozen_settings,
)
from training.trainers import run_outer_unit, write_json
from training.tuning import evaluation_check_formal_pca_support, tune_all_arms


def evaluation_validate_inputs(manifest, conditions_features, feature_names):
    required = {"image_id", "duplicate_group", "group", "tags"}
    if not required.issubset(manifest.columns):
        raise ValueError(
            f"Manifest lacks columns: {sorted(required - set(manifest.columns))}"
        )
    m = manifest.reset_index(drop=True).copy()
    if m[list(required)].isna().any().any():
        raise ValueError(
            "Manifest IDs, groups, duplicate groups and tags cannot be missing."
        )
    m["image_id"] = m.image_id.astype(str)
    m["duplicate_group"] = m.duplicate_group.astype(str)
    if m.image_id.duplicated().any() or not m.group.isin(GROUPS).all():
        raise ValueError("Image IDs must be unique and groups must be R, P or N.")
    if "clean" not in conditions_features:
        raise ValueError(
            "A 'clean' condition is required for training and threshold calibration."
        )
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Specify at least one unique feature representation.")
    arrays = {}
    conditions = ["clean"] + [c for c in conditions_features if c != "clean"]
    for condition in conditions:
        if not isinstance(condition, str) or not condition:
            raise ValueError("Conditions require nonempty string names.")
        arrays[condition] = {}
        for name in feature_names:
            if name not in conditions_features[condition]:
                raise ValueError(f"Missing {name} features for condition {condition}.")
            x = np.asarray(conditions_features[condition][name], dtype=np.float32)
            if x.ndim != 2 or x.shape[0] != len(m) or x.shape[1] < 1:
                raise ValueError(
                    f"{condition}/{name} is not a manifest-aligned 2D feature matrix."
                )
            if not np.isfinite(x).all():
                raise ValueError(
                    f"{condition}/{name} contains NaN or infinite features."
                )
            if condition != "clean" and x.shape != arrays["clean"][name].shape:
                raise ValueError(
                    "Every condition must retain the same feature dimensions."
                )
            arrays[condition][name] = np.ascontiguousarray(x)
    return (m, arrays)


def expected_result_counts(n_images, n_outer, n_inner, models=None):
    """Expected evidence sizes for the selected models and fixed feature design."""
    models = select_models(models)
    arms = len(models) * len(FEATURE_REPRESENTATIONS)
    outer_fits = n_outer * arms
    return {'inner_fits': outer_fits * n_inner * 3, 'candidate_means': outer_fits * 3,
            'selections': outer_fits, 'outer_fits': outer_fits,
            'metrics': outer_fits * len(CONDITIONS),
            'predictions': n_images * arms * len(CONDITIONS)}


def run_complete_comparison(manifest, features, output, models=None,
                            development_manifest_path=None):
    models = select_models(models)
    settings = validate_frozen_settings(models)
    manifest, features = evaluation_validate_inputs(manifest, features, FEATURE_REPRESENTATIONS)
    plans, split_rows, supports = build_nested_split_plan(manifest, OUTER_FOLDS, INNER_FOLDS,
                                                        SEED, CALIBRATION_FRACTION)
    # The role names below are presentation labels only. The held-out indices stay identical.
    split_rows['role'] = split_rows.role.replace({'development': 'outer_training',
                                                  'development_holdout': 'outer_holdout'})
    split_bytes = split_rows.to_csv(index=False).encode()
    split_digest = hashlib.sha256(split_bytes).hexdigest()
    if EXPECTED_SPLIT_SHA256 and split_digest != EXPECTED_SPLIT_SHA256:
        raise ValueError('Regenerated fold rows differ from the pinned benchmark split.')
    output.mkdir(parents=True, exist_ok=True)
    split_rows.to_csv(output / 'split_manifest.csv', index=False)
    supports.to_csv(output / 'split_supports.csv', index=False)
    evaluation_check_formal_pca_support(plans, FROZEN_GRIDS, features['clean'], models=models)
    expected = expected_result_counts(len(manifest), OUTER_FOLDS, INNER_FOLDS, models)
    feature_hashes = {condition: {name: hashlib.sha256(matrix.tobytes()).hexdigest()
        for name, matrix in arms.items()} for condition, arms in features.items()}
    development_path = (Path(development_manifest_path) if development_manifest_path is not None
                        else locate_single_input('development_manifest.csv'))
    common = {'settings': settings, 'source': active_science_identity(), 'runtime': runtime_identity(),
              'manifest': canonical_hash(manifest[['image_id','group','tags','pixel_sha256','duplicate_group']].to_dict('records')),
              'splits': split_digest, 'feature_values': feature_hashes,
              'development_manifest_sha256': file_sha256(development_path)}
    common_key = canonical_hash(common)
    output = output / common_key
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'experiment_identity.json', common)
    # Keep the complete audit trail inside the final bundle, not only its parent.
    split_rows.to_csv(output / 'split_manifest.csv', index=False)
    supports.to_csv(output / 'split_supports.csv', index=False)
    manifest.to_csv(output / 'benchmark_manifest.csv', index=False)
    shutil.copyfile(development_path, output / 'development_manifest.csv')
    cohort_metadata = development_path.parent / 'cohort_generation.json'
    if cohort_metadata.is_file():
        shutil.copyfile(cohort_metadata, output / 'cohort_generation.json')
    for name in ('raw_image_checks.csv', 'feature_audit.json'):
        evidence_path = config.OUTPUT_ROOT / name
        if evidence_path.is_file():
            shutil.copyfile(evidence_path, output / name)
    # An interrupted execution resumes only exact completed units under this identity.
    inner, candidate_means, selections = tune_all_arms(manifest, features['clean'], plans, common_key, output,
                                                     models=models)
    metrics, predictions, fits = [], [], []
    resumed = 0
    for outer, plan in enumerate(plans):
        for feature in FEATURE_REPRESENTATIONS:
            for model in models:
                chosen = selections[(selections.outer_fold == outer) &
                                    (selections.features == feature) & (selections.model == model)]
                if len(chosen) != 1:
                    raise ValueError('Exactly one saved choice is required for each outer arm.')
                parameter = float(chosen.iloc[0].selected_parameter)
                m, p, f, reused = run_outer_unit(manifest, features, plan, outer,
                                                feature, model, parameter, common_key, output)
                metrics.append(m); predictions.append(p); fits.append(f)
                resumed += int(reused)
                print(f'Outer fits {len(fits)}/{expected["outer_fits"]}; resumed={resumed}; '
                      f'outer={outer + 1} {feature} {model}', flush=True)
    metric_table = pd.concat(metrics, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    fit_table = pd.DataFrame(fits)
    modal = evaluation_modal_audit(selections, FROZEN_GRIDS, OUTER_FOLDS, models=models)
    tables = {'metrics': metric_table, 'predictions': prediction_table, 'outer_fits': fit_table,
              'middle_modes': modal}
    summary, corruption_pairs, corruption_summary = summarize_condition_results(metric_table)
    feature_pairs, feature_summary = evaluation_paired_feature_tables(metric_table)
    tables.update(summary=summary, corruption_pairs=corruption_pairs,
                  corruption_summary=corruption_summary, feature_pairs=feature_pairs,
                  feature_summary=feature_summary)
    for name, frame in tables.items():
        frame.to_csv(output / (name + '.csv'), index=False, float_format='%.17g')
    actual = {'inner_fits': len(inner), 'candidate_means': len(candidate_means),
              'selections': len(selections), 'outer_fits': len(fit_table),
              'metrics': len(metric_table), 'predictions': len(prediction_table)}
    if actual != expected:
        raise ValueError('The run is incomplete: ' + str(actual))
    if metric_table.duplicated(['outer_fold','features','model','condition']).any():
        raise ValueError('A model/feature/fold/condition metric row appears twice.')
    if prediction_table.duplicated(['outer_fold','features','model','condition','image_id']).any():
        raise ValueError('A held-out image prediction appears twice.')
    full_comparison = models == MODEL_NAMES
    completion = {'execution_complete': True, 'counts': actual,
                  'selected_models': list(models), 'full_comparison_complete': full_comparison,
                  'core_middle_pass': int(modal.loc[modal.core_model, 'modal_middle_pass'].sum()),
                  'core_arms': int(modal.core_model.sum()),
                  'all_middle_pass': int(modal.modal_middle_pass.sum()), 'all_arms': len(modal),
                  'selected_middle_requirement_met': bool(modal.modal_middle_pass.all()),
                  'unique_middle_requirement_met': bool(full_comparison and modal.modal_middle_pass.all()),
                  'cohort_exposed': True, 'budget_caps_do_not_prove_convergence': True,
                  'coursework_manual_requirements_verified': False,
                  'output_directory': str(output)}
    write_json(output / 'completion.json', completion)
    inventory = {str(p.relative_to(output)): file_sha256(p) for p in output.rglob('*') if p.is_file()
                 and p.name not in ('integrity_inventory.json', 'result_evidence.zip')}
    write_json(output / 'integrity_inventory.json', inventory)
    with zipfile.ZipFile(output / 'result_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'result_evidence.zip':
                archive.write(path, path.relative_to(output))
    return {**tables, 'inner': inner, 'selections': selections, 'completion': completion}
