"""training: tuning for the Amazon robustness experiment."""

import numpy as np
import pandas as pd
from config import FEATURE_REPRESENTATIONS, FROZEN_GRIDS, select_models
from preprocessing.scaling import evaluation_training_rank
from training.trainers import run_inner_unit


def evaluation_check_formal_pca_support(plans, grids, clean, models=None):
    """Reject unsupported frozen ranks; do not change a grid using formal data.

    Each rank is calculated on a model's own TRAINING-R rows. An unsupported
    configuration aborts this protocol before outer scoring rather than silently
    capping k. No formal validation/holdout feature values enter that rank check.
    """
    models = select_models(models)
    for feature in FEATURE_REPRESENTATIONS:
        if 'PCA' in models:
            largest_k = max(grids[feature]["PCA"])
            smallest_rank = min(
                evaluation_training_rank(clean[feature][fit])
                for _, final_fit, _, inner in plans
                for fit in [final_fit, *[item[0] for item in inner]]
            )
            if largest_k >= smallest_rank:
                raise ValueError(
                    f"Frozen PCA k={largest_k} for {feature} reaches a formal TRAINING rank={smallest_rank}. No candidate was capped or changed; redesign the declared protocol explicitly."
                )
        fit_sizes = [
            len(fit)
            for _, final_fit, _, inner in plans
            for fit in [final_fit, *[item[0] for item in inner]]
        ]
        if 'LOF' in models and max(grids[feature]["LOF"]) >= min(fit_sizes):
            raise ValueError(
                "Frozen LOF neighbors reach the formal TRAINING-reference count; no automatic capping."
            )
        if 'MLPAutoencoder' in models and max(grids[feature]["MLPAutoencoder"]) >= clean[feature].shape[1]:
            raise ValueError(
                "Frozen MLP bottleneck reaches the input feature count; no automatic capping."
            )


def tune_all_arms(manifest, clean, plans, common, output, models=None):
    """The held-out outer scores are never read by this selection function."""
    models = select_models(models)
    records, means, selections = [], [], []
    total = sum(len(inner) for _, _, _, inner in plans) * len(FEATURE_REPRESENTATIONS) * len(models) * 3
    resumed = 0
    for outer, (_, _, _, inner_plans) in enumerate(plans):
        for feature in FEATURE_REPRESENTATIONS:
            for model in models:
                candidates = FROZEN_GRIDS[feature][model]
                candidate_means = []
                for parameter in candidates:
                    scores = []
                    for inner, (fit, validation) in enumerate(inner_plans):
                        row, reused = run_inner_unit(manifest, clean, fit, validation,
                            outer, inner, feature, model, parameter, common, output)
                        records.append(row)
                        resumed += int(reused)
                        scores.append(row['AP'])
                        print(f'Inner fits {len(records)}/{total}; resumed={resumed}; '
                              f'outer={outer + 1} {feature} {model}', flush=True)
                    mean_ap = float(np.mean(scores))
                    candidate_means.append(mean_ap)
                    means.append({'outer_fold': outer, 'features': feature, 'model': model,
                                  'parameter': parameter, 'mean_inner_AP': mean_ap})
                # np.argmax selects the first candidate only when means are EXACTLY tied.
                # Since candidates are ascending, this keeps the declared tie rule.
                winner = int(np.argmax(candidate_means))
                selections.append({'outer_fold': outer, 'features': feature, 'model': model,
                    'selected_parameter': candidates[winner], 'inner_mean_AP': candidate_means[winner],
                    'tie_for_best': int(np.sum(np.asarray(candidate_means) == candidate_means[winner])) > 1})
                pd.DataFrame(records).to_csv(output / 'inner_AP.csv', index=False, float_format='%.17g')
                pd.DataFrame(means).to_csv(output / 'candidate_means.csv', index=False, float_format='%.17g')
                pd.DataFrame(selections).to_csv(output / 'parameter_selections.csv', index=False, float_format='%.17g')
    return pd.DataFrame(records), pd.DataFrame(means), pd.DataFrame(selections)
