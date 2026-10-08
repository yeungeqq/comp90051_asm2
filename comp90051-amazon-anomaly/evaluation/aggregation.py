"""evaluation: aggregation for the Amazon robustness experiment."""

import json
import pandas as pd
from config import EVALUATION_METRICS, FEATURE_REPRESENTATIONS, PRIMARY_ALGORITHMS, select_models


def summarize_condition_results(metrics):
    """Fold SD is descriptive variability, not a confidence interval."""
    keys = ["model", "features", "condition"]
    summary = metrics.groupby(keys, sort=False)[list(EVALUATION_METRICS)].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    clean = metrics.loc[
        metrics.condition == "clean", ["outer_fold", "model", "features", "AP"]
    ]
    clean = clean.rename(columns={"AP": "clean_AP"})
    paired = metrics.merge(
        clean, on=["outer_fold", "model", "features"], validate="many_to_one"
    )
    paired = paired.loc[
        paired.condition != "clean", ["outer_fold", *keys, "AP", "clean_AP"]
    ].copy()
    paired["AP_delta_from_clean"] = paired.AP - paired.clean_AP
    paired_summary = paired.groupby(keys, sort=False).AP_delta_from_clean.agg(
        ["mean", "std", "count"]
    )
    paired_summary = paired_summary.rename(
        columns={
            "mean": "AP_delta_mean",
            "std": "AP_delta_std",
            "count": "n_paired_folds",
        }
    ).reset_index()
    return (summary, paired, paired_summary)


def evaluation_paired_feature_tables(metrics):
    keys = ["outer_fold", "model", "condition"]
    pairs = []
    # Match the very same held-out folds; capacities/parameters were tuned per arm.
    for before, after, label in [
        ("A_RGB", "B_RGB_NIR", "NIR: B84 minus A63"),
        ("B_RGB_NIR", "C_RGB_NIR_TEXTURE", "Texture: C108 minus B84"),
    ]:
        table = metrics[metrics.features.eq(before)][
            keys + list(EVALUATION_METRICS)
        ].merge(
            metrics[metrics.features.eq(after)][keys + list(EVALUATION_METRICS)],
            on=keys,
            suffixes=("_before", "_after"),
            validate="one_to_one",
        )
        table["comparison"] = label
        table["features"] = after
        table["baseline_features"] = before
        for metric in EVALUATION_METRICS:
            table[metric + "_delta"] = (
                table[metric + "_after"] - table[metric + "_before"]
            )
        pairs.append(
            table[
                keys
                + ["comparison", "features", "baseline_features"]
                + [m + "_delta" for m in EVALUATION_METRICS]
            ]
        )
    paired = pd.concat(pairs, ignore_index=True)
    summary = paired.groupby(["comparison", "model", "condition"])[
        [m + "_delta" for m in EVALUATION_METRICS]
    ].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return paired, summary.reset_index()


def evaluation_modal_audit(selections, grids, n_outer, models=None):
    """Literal unique-modal-middle outcome, including endpoint and tie failures."""
    models = select_models(models)
    rows = []
    for feature in FEATURE_REPRESENTATIONS:
        for model in models:
            candidates = grids[feature][model]
            chosen = selections[
                (selections.features == feature) & (selections.model == model)
            ]
            counts = [
                int(chosen.selected_parameter.eq(value).sum()) for value in candidates
            ]
            highest = max(counts)
            modes = [
                value for value, count in zip(candidates, counts) if count == highest
            ]
            passed = (
                len(chosen) == n_outer
                and len(modes) == 1
                and (modes[0] == candidates[1])
            )
            rows.append(
                {
                    "features": feature,
                    "model": model,
                    "grid_low": candidates[0],
                    "grid_middle": candidates[1],
                    "grid_high": candidates[2],
                    "selected_low_count": counts[0],
                    "selected_middle_count": counts[1],
                    "selected_high_count": counts[2],
                    "modal_values": json.dumps(modes),
                    "middle_majority": counts[1] > n_outer / 2,
                    "core_model": model in PRIMARY_ALGORITHMS,
                    "unique_mode": len(modes) == 1,
                    "modal_middle_pass": bool(passed),
                    "outer_selections": len(chosen),
                }
            )
    return pd.DataFrame(rows)
