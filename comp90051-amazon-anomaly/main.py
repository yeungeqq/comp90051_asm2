"""Run the fixed Amazon anomaly experiment from Python or a Kaggle notebook."""

import argparse
import config
import json
from pathlib import Path


def describe_experiment(models=None):
    """Describe the workload without scanning images, fitting, or requiring CUDA."""
    models = config.select_models(models)
    outer_fits = config.OUTER_FOLDS * len(config.FEATURE_REPRESENTATIONS) * len(models)
    return {
        "input_root": str(config.INPUT_ROOT),
        "labels_path": str(config.LABELS_PATH) if config.LABELS_PATH else None,
        "output_root": str(config.OUTPUT_ROOT),
        "required_inputs": ["train_v2.csv (or train.csv)", "uncompressed four-band TIFFs"],
        "generated_inputs": ["development_manifest.csv", "benchmark_manifest.csv"],
        "feature_route": "authenticated cache" if config.USE_FIXED_FEATURE_CACHE else "raw TIFF extraction",
        "cache_inputs": ["cached_feature_manifest.json", "cached_features.npz"]
                        if config.USE_FIXED_FEATURE_CACHE else [],
        "device": config.DEVICE,
        "models": list(models),
        "feature_dimensions": {"A_RGB": 63, "B_RGB_NIR": 84, "C_RGB_NIR_TEXTURE": 108},
        "conditions": list(config.CONDITIONS),
        "outer_folds": config.OUTER_FOLDS,
        "inner_folds": config.INNER_FOLDS,
        "inner_fits": outer_fits * config.INNER_FOLDS * 3,
        "outer_fits": outer_fits,
        "neural_step_budgets": {name: config.STEP_BUDGETS[name] for name in models
                                if name in config.STEP_BUDGETS},
        "cohort_exposed": True,
    }


def initialize_runtime():
    """Apply the notebook's deterministic execution settings only on explicit run."""
    import random

    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def run(input_root=None, output_root=None, labels_path=None,
        use_fixed_feature_cache=None, make_plots=True, models=None):
    """Verify inputs, tune, evaluate, export, and return the result tables.

    Paths and the feature-cache route can change without changing the frozen
    experiment. Scientific settings belong in config.py before a fresh process.
    models selects one or more detector names; None runs all six.
    Experiment execution requires CUDA.
    """
    models = config.select_models(models)
    if input_root is not None:
        config.INPUT_ROOT = Path(input_root)
    if output_root is not None:
        config.OUTPUT_ROOT = Path(output_root)
    if labels_path is not None:
        config.LABELS_PATH = Path(labels_path)
    if use_fixed_feature_cache is not None:
        config.USE_FIXED_FEATURE_CACHE = bool(use_fixed_feature_cache)

    initialize_runtime()
    from data.cohorts import create_or_load_cohorts
    from data.loader import verify_raw_benchmark
    from evaluation.robustness import run_complete_comparison
    from preprocessing import features
    from training.protocol import active_science_identity, validate_frozen_settings
    from training.trainers import write_json
    from visualization.plots import finalize_result_archive, plot_feature_examples, show_result_figures

    validate_frozen_settings(models)
    output = config.OUTPUT_ROOT
    for folder in (output, output / "results", output / "figures", output / "checkpoints"):
        folder.mkdir(parents=True, exist_ok=True)
    _, benchmark, development_path, _ = create_or_load_cohorts(
        config.INPUT_ROOT, output / "cohorts", config.LABELS_PATH, seed=config.SEED)
    benchmark = verify_raw_benchmark(benchmark, output)
    print(benchmark.groupby("group").size().rename("images"))

    features.IMPLEMENTATION_ID = active_science_identity()
    if config.USE_FIXED_FEATURE_CACHE:
        matrices, audit = features.load_authenticated_features(benchmark)
    else:
        matrices, audit = features.extract_condition_features(
            benchmark, feature_cache_dir=output / "checkpoints" / "feature_cache", seed=config.SEED)
    write_json(output / "feature_audit.json", audit)
    result = run_complete_comparison(
        benchmark, matrices, output / "results", models=models,
        development_manifest_path=development_path)
    result_directory = Path(result["completion"]["output_directory"])
    if make_plots:
        import shutil

        import matplotlib.pyplot as plt

        figure_directory = output / "figures" / result_directory.name
        figure_directory.mkdir(parents=True, exist_ok=True)
        figure = plot_feature_examples(
            benchmark, image_ids=[benchmark.iloc[0].image_id], conditions=config.CONDITIONS,
            output_path=figure_directory / "feature_examples.png")
        plt.close(figure)
        show_result_figures(result, figure_directory)
        shutil.copytree(figure_directory, result_directory / "figures", dirs_exist_ok=True)
    finalize_result_archive(result_directory)
    print(json.dumps(result["completion"], indent=2))
    print(f"Evidence archive: {result_directory / 'result_evidence.zip'}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=config.INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--labels-path", type=Path, default=config.LABELS_PATH)
    parser.add_argument("--models", nargs="+", choices=config.MODEL_NAMES,
                        help="Run only these models; omitted runs all six.")
    parser.add_argument("--recompute-features", action="store_true",
                        help="Extract from verified TIFFs instead of requiring the fixed NPZ cache.")
    parser.add_argument("--no-plots", action="store_true", help="Export numerical evidence without figures.")
    parser.add_argument("--dry-run", action="store_true", help="Print settings; no data scan or training.")
    args = parser.parse_args(argv)
    models = config.select_models(args.models)
    config.INPUT_ROOT, config.OUTPUT_ROOT = args.input_root, args.output_root
    config.LABELS_PATH = args.labels_path
    if args.recompute_features:
        config.USE_FIXED_FEATURE_CACHE = False
    if args.dry_run or not config.RUN_EXPERIMENT:
        print(json.dumps(describe_experiment(models), indent=2))
        return None
    return run(make_plots=not args.no_plots, models=models)


if __name__ == "__main__":
    main()
