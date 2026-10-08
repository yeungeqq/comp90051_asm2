"""visualization: plots for the Amazon robustness experiment."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zipfile
from config import (
    CONDITIONS,
    CorruptionConfig,
    FEATURE_REPRESENTATIONS,
    FeatureConfig,
    TextureConfig,
)
from data.loader import file_sha256, read_rgbn
from pathlib import Path
from preprocessing.degradations import corrupt_rgbn
from preprocessing.features import quantize_luminance
from training.trainers import write_json


def display(value):
    """Show a table in notebooks and print it in a plain Python process."""
    try:
        from IPython.display import display as ipython_display
    except ImportError:
        print(value)
    else:
        ipython_display(value)


def plot_feature_examples(manifest, image_ids=None, conditions=("clean", "occlusion_20", "noise_stress"),
                          feature_config=FeatureConfig(), texture_config=TextureConfig(),
                          corruption_config=CorruptionConfig(), seed=51, output_path=None):
    """Plot RGB and GLCM quantisation on the SAME examples under each condition.

    Display limits come from the clean example only and are reused across its
    conditions; these visual limits are never passed into model features.
    """
    import matplotlib.pyplot as plt
    if image_ids is None:
        image_ids = sorted(manifest["image_id"].astype(str).tolist())[:2]
    image_ids, conditions = list(image_ids), tuple(conditions)
    if not image_ids or not conditions or set(conditions) - set(CONDITIONS):
        raise ValueError("Select at least one image and known condition.")
    indexed = manifest.set_index("image_id", drop=False)
    figure, axes = plt.subplots(2 * len(image_ids), len(conditions), squeeze=False,
                               figsize=(3.2 * len(conditions), 5 * len(image_ids)))
    for row_number, image_id in enumerate(image_ids):
        original = read_rgbn(indexed.loc[image_id, "tiff_path"], feature_config)
        low = np.percentile(original[..., :3], 2, axis=(0, 1))
        high = np.percentile(original[..., :3], 98, axis=(0, 1))
        for column, condition in enumerate(conditions):
            transformed = corrupt_rgbn(original, condition, image_id, seed, corruption_config)
            display = np.clip((transformed[..., :3] - low) / np.maximum(high - low, 1), 0, 1)
            gray = np.dot(transformed[..., :3] / feature_config.dn_scale, [.2126, .7152, .0722])
            quantized, qc = quantize_luminance(gray, feature_config, texture_config)
            axes[2 * row_number, column].imshow(display)
            axes[2 * row_number, column].set_title(f"{image_id}: {condition}\nRGB; clean display limits")
            axes[2 * row_number + 1, column].imshow(quantized, cmap="gray", vmin=0, vmax=feature_config.glcm_levels - 1)
            axes[2 * row_number + 1, column].set_title(f"GLCM input: {qc['glcm_occupied_bins']} occupied bins")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle("Texture: within-image 2nd/98th percentiles; display enhancement is separate")
    figure.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=140, bbox_inches="tight")
    return figure


def show_result_figures(result, output):
    """Each error bar is fold SD. It is not a confidence interval."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = result['summary']
    models = tuple(summary.model.drop_duplicates())
    for feature in FEATURE_REPRESENTATIONS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        for ax, metric in zip(axes.flat, ('AP','recall_P','FPR_R','FPR_N')):
            for model in models:
                values = summary[(summary.features == feature) & (summary.model == model)].set_index('condition').loc[list(CONDITIONS)]
                ax.errorbar(range(len(CONDITIONS)), values[metric + '_mean'], yerr=values[metric + '_std'],
                            marker='o', capsize=2, label=model)
            ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=35, ha='right')
            ax.set_ylabel(metric + ' (mean ± fold SD)')
            ax.legend(fontsize=7, ncol=2)
        fig.suptitle(feature + ': the same images across all conditions')
        fig.tight_layout(); fig.savefig(output / (feature + '_robustness.png'), dpi=180); plt.close(fig)
    display(result['middle_modes'][['features','model','grid_low','grid_middle','grid_high',
                                   'selected_low_count','selected_middle_count','selected_high_count',
                                   'modal_middle_pass']])
    display(result['feature_summary'])
    # A paired difference compares the same fold, method and image condition.
    # Positive AP difference means the extra feature group helped on average.
    paired = result['feature_summary']
    for number, comparison in enumerate(paired.comparison.unique(), start=1):
        fig, ax = plt.subplots(figsize=(12, 5))
        for model in models:
            values = paired[(paired.comparison == comparison) & (paired.model == model)].set_index('condition').loc[list(CONDITIONS)]
            ax.errorbar(range(len(CONDITIONS)), values.AP_delta_mean, yerr=values.AP_delta_std,
                        marker='o', capsize=2, label=model)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=35, ha='right')
        ax.set_ylabel('Paired AP difference (mean ± fold SD)')
        ax.set_title(comparison); ax.legend(ncol=3, fontsize=8)
        fig.tight_layout(); fig.savefig(output / f'paired_features_{number}.png', dpi=180); plt.close(fig)

    # These are illustrations from the first fold, selected by image ID rather
    # than by a favourable score. Raw scores are not ranked across fitted folds.
    examples = []
    first_fold = result['predictions'].loc[result['predictions'].outer_fold == 0]
    for _, part in first_fold.groupby(['features','model','condition'], sort=False):
        categories = {
            'detected_disturbance': part.group.eq('P') & part.alert.eq(True),
            'missed_disturbance': part.group.eq('P') & part.alert.eq(False),
            'reference_or_natural_false_alert': part.group.ne('P') & part.alert.eq(True),
        }
        for label, mask in categories.items():
            examples.append(part.loc[mask].sort_values('image_id').head(3).assign(example_type=label))
    example_table = pd.concat(examples, ignore_index=True)
    example_table.to_csv(output / 'detection_examples.csv', index=False, float_format='%.17g')
    display(example_table.head(18))


def finalize_result_archive(output):
    """Include the figures generated after numerical evaluation in the same bundle."""
    inventory = {str(p.relative_to(output)): file_sha256(p) for p in output.rglob('*') if p.is_file()
                 and p.name not in ('integrity_inventory.json', 'result_evidence.zip')}
    write_json(output / 'integrity_inventory.json', inventory)
    with zipfile.ZipFile(output / 'result_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'result_evidence.zip':
                archive.write(path, path.relative_to(output))
