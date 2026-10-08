# COMP90051 — Amazon Satellite Image Anomaly Detection

This project evaluates how image degradation affects human-disturbance detection in four-band Amazon satellite imagery. Six anomaly detectors learn from clean reference-forest images and score held-out images under nine image conditions. The experiment compares RGB, RGB with near-infrared (NIR), and RGB with NIR and texture features to measure detection performance and robustness.

The program runs on Kaggle through `kaggle_runner.ipynb` or the `main.py` command-line entry point. It exports predictions, evaluation tables, figures, training diagnostics, and a verified result archive.

## Requirements

- Python 3.11 or newer.
- A Kaggle Python session with a CUDA GPU enabled for the full experiment.
- The project source and required data attached as Kaggle datasets.
- Python packages listed in `requirements.txt`.

By default, the program trains all six models, performing 1,620 inner fits and 180 outer refits. To train only one model or a subset, follow [Model selection](#model-selection). Neural training uses fixed update budgets: 3,840 for the MLP autoencoder, 7,680 for Deep SVDD, and 3,840 for the feature Transformer autoencoder. Individual detectors and automated tests can run on CPU.

## Run on Kaggle

1. Upload `project.zip` as a Kaggle dataset with an available title, such as **amazon-anomaly-program-v2**. The title can vary; the launcher discovers the source automatically. The ZIP contains the project files directly at its root.
2. Attach the source dataset and the data listed below to a Kaggle Python notebook.
3. Enable a CUDA GPU accelerator in the notebook settings.
4. Import `kaggle_runner.ipynb` into Kaggle and execute its cells in order.

The attached source dataset should have this layout:

```text
<source-dataset>/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── kaggle_runner.ipynb
├── pytest.ini
├── data/
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── visualization/
├── outputs/
└── tests/
```

No additional `comp90051-amazon-anomaly/` directory is required inside the dataset. The launcher discovers the directory containing `main.py`, `config.py`, and `training/protocol.py` anywhere under `/kaggle/input`, and prefers a dataset directory named `program` when there are multiple candidates. This also accommodates longer dataset mount paths.

The source is copied to `/kaggle/working/comp90051-amazon-anomaly` before execution. Attached datasets are read-only; generated results and extraction checkpoints go into this writable working directory. The launcher supports extracted source folders and unextracted `project.zip` archives with either flat or wrapped contents. Set `SOURCE_PROJECT` explicitly if source discovery is ambiguous.

The launcher provides these settings:


| Setting | Purpose |
| --- | --- |
| `SOURCE_PROJECT` | Optional explicit source directory or ZIP; `None` discovers the attached project |
| `MODELS` | Models to train, for example `["DeepSVDD"]`; `None` runs all six |
| `RECOMPUTE_FEATURES` | Set to `True` to extract features from TIFFs instead of loading the authenticated feature package |
| `MAKE_PLOTS` | Enable or disable figure generation |
| `INPUT_ROOT` | Directory containing all attached datasets; defaults to `/kaggle/input` |
| `OUTPUT_ROOT` | Writable directory for results, figures, and extraction checkpoints |
| `LABELS_PATH` | Explicit label CSV path when multiple label files are attached |

Kaggle environments may already provide the required runtime packages. If dependencies are missing, install them before starting the experiment:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-r", str(PROJECT / "requirements.txt")
], check=True)
```

Run this cell after the launcher has defined `PROJECT`. Package downloads require internet access; offline sessions can use attached wheels. Keep the same package versions when resuming a run, because runtime versions are part of the experiment identity.

### Run from a custom Kaggle notebook

Locate the attached source directory, then copy it to writable storage:

```python
from pathlib import Path
import shutil
import subprocess
import sys

candidates = sorted({path.parent for path in Path("/kaggle/input").rglob("main.py")
                     if (path.parent / "config.py").is_file()
                     and (path.parent / "training/protocol.py").is_file()})
preferred = [path for path in candidates if path.name == "program"]
choices = preferred if len(preferred) == 1 else candidates
if len(choices) != 1:
    raise ValueError(f"Expected one attached source project; found {candidates}.")
SOURCE = choices[0]
PROJECT = Path("/kaggle/working/comp90051-amazon-anomaly")
shutil.copytree(SOURCE, PROJECT, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "__MACOSX",
                                             ".DS_Store", "._*", "*.zip"))

# Print the configuration without scanning data or training models.
subprocess.run([sys.executable, str(PROJECT / "main.py"), "--dry-run"], check=True)
```

Start the full experiment:

```python
subprocess.run([sys.executable, "-u", str(PROJECT / "main.py")], check=True)
```

To extract features directly from the verified TIFFs:

```python
subprocess.run([
    sys.executable, "-u", str(PROJECT / "main.py"), "--recompute-features"
], check=True)
```

## Model selection

### Select models in the Kaggle launcher

In the **Inspect the run configuration** code cell of `kaggle_runner.ipynb`, change `MODELS` to the model names you want to run:

```python
MODELS = ["DeepSVDD"]          # Train only Deep SVDD.
# MODELS = ["PCA", "OCSVM"]    # Train two models.
# MODELS = None               # Train all six (default).
```

Rerun the entire configuration cell after changing `MODELS`. This rebuilds `command` and prints a dry run. Confirm that the printed `models`, `inner_fits`, and `outer_fits` match your selection. The dry run does not train models.

Then execute the **Run the experiment** cell:

```python
subprocess.run(command, check=True)
```

Changing `MODELS` without rerunning the configuration cell leaves the existing command unchanged. `check=True` raises an exception if the program exits with an error; it does not choose the models.

The accepted model names are case-sensitive:

| Name | Detector |
| --- | --- |
| `PCA` | PCA reconstruction |
| `OCSVM` | RBF One-Class SVM |
| `LOF` | Novelty Local Outlier Factor |
| `MLPAutoencoder` | MLP autoencoder |
| `DeepSVDD` | Deep Support Vector Data Description |
| `TransformerAE` | Feature Transformer autoencoder |

### Select models through a command or the Python API

With an existing launcher command, select and preview one model directly:

```python
subprocess.run(command + ["--models", "DeepSVDD", "--dry-run"], check=True)
subprocess.run(command + ["--models", "DeepSVDD"], check=True)
```

From the project directory, the equivalent CLI commands are:

```bash
python main.py --models DeepSVDD --dry-run
python main.py --models DeepSVDD
python main.py --models PCA OCSVM
python main.py                          # All six models.
```

For interactive Python use after adding the project to `sys.path`:

```python
from main import run

result = run(models=["DeepSVDD"])
```

### Workload and result scope

Selecting a model still runs its three parameter candidates, three feature sets, ten outer folds, and three inner folds. Evaluation scores all nine image conditions. It does not reduce the fold count or training budget.

| Selected models | Inner fits | Outer refits | Metric rows | Predictions |
| --- | --- | --- | --- | --- |
| One | 270 | 30 | 270 | 64,800 |
| Two | 540 | 60 | 540 | 129,600 |
| All six | 1,620 | 180 | 1,620 | 388,800 |

`RECOMPUTE_FEATURES = False` loads cached image features; the selected models still train. Selected-model runs require the same data and CUDA configuration as the full experiment.

Model selection is recorded in the experiment identity, and plots and completion checks cover only the selected models. In `completion.json`, `execution_complete` confirms completion of that selected scope; `full_comparison_complete` is true only for a run containing all six models. Separate one-model runs are not automatically combined into a full comparison. Changing the selected model set creates a different experiment identity, so completed units from another selection are not automatically reused.

## Data inputs

Attach the source package and image collection separately. Cohort manifests are generated from the labelled TIFF source at runtime.


| Input                          | Requirement                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Four-band training TIFFs       | Uncompressed`.tif` or `.tiff` files stored in BGRN order; one unambiguous path per selected image                  |
| `train_v2.csv` or `train.csv`  | Labels with`image_name` and `tags` columns; the full labelled source must contain at least 10,000 images           |
| Cohort manifests               | Generated automatically under `outputs/cohorts/` on the first run; not required as Kaggle inputs                  |
| Authenticated feature cache    | Not used for generated cohorts; raw TIFF features are recomputed                                                   |

The first run deterministically selects two disjoint cohorts from eligible labelled TIFFs using `SEED` (51 by default). Each cohort contains 1,457 reference images (R), 705 disturbance images (P), and 238 natural-variation controls (N). The generated CSVs include `image_id`, `tags`, `group`, `pixel_sha256`, and `duplicate_group`; selected source paths and other label-audit fields are retained as well. Pixel-identical images are not selected into both cohorts. The manifests and `cohort_generation.json` are saved under `outputs/cohorts/` and reused on subsequent runs. Remove that directory to regenerate both cohorts.

The content groups follow a fixed label policy:

- **R — reference forest:** exactly `clear` and `primary` tags.
- **P — human disturbance:** clear images containing `road`, `selective_logging`, or `slash_burn`; disturbance takes priority over natural-variation tags.
- **N — natural variation:** clear primary forest with `water`, `blooming`, or `blow_down`, and no human-disturbance tags.

The program verifies cohort schema, class counts, disjoint image IDs and pixel hashes, source-label agreement, and benchmark pixel hashes before evaluation. The generated cohort is a new reproducible sample; it does not match the unavailable fixed manifests or reproduce results associated with their hashes. TIFF values are read in RGBN order, and each image side must contain at least 100 pixels. JPEG-only inputs cannot support the NIR comparison. Extract raw-image archives before execution.

Raw feature extraction is the default and is required for these generated cohorts. Feature extraction and model fitting can take substantial time; the transactional feature cache under `outputs/checkpoints/feature_cache` supports resuming interruptions. Feature values may differ under different image contents or library versions; those identities are included in result evidence.

## Experiment protocol

### Models and features

The six detectors are PCA reconstruction, RBF One-Class SVM, novelty Local Outlier Factor, MLP autoencoder, Deep SVDD, and feature Transformer autoencoder. PCA includes an explicit zero-component, mean-only reconstruction candidate.


| Representation      | Dimensions | Features                                                                |
| ------------------- | ---------- | ----------------------------------------------------------------------- |
| `A_RGB`             | 63         | Global band statistics and 4×4 local means for RGB                     |
| `B_RGB_NIR`         | 84         | RGB features plus the corresponding NIR statistics and local means      |
| `C_RGB_NIR_TEXTURE` | 108        | RGB+NIR features plus 16 local edge densities and eight GLCM statistics |

Band and edge features use fixed digital-number scaling. GLCM texture uses each image's own 2nd and 98th luminance percentiles. The Transformer operates on statistical feature tokens.

### Splits, tuning, and calibration

The evaluation uses ten outer folds with three inner folds per outer training partition. Duplicate groups remain together throughout splitting. Separate reference-containing groups reserve calibration data using a 20% calibration fraction.

Each detector and its scaler fit only clean reference rows from the relevant training partition. Three frozen parameter candidates are compared using equally weighted mean inner average precision (AP). An exact tie selects the first candidate in ascending order. The selected detector is refitted on the outer training-reference rows.

The alert threshold is calibrated using the 95th percentile of separate clean reference scores. Alerts use the strict comparison `score > threshold`. OCSVM uses a stable log-kernel anomaly score with threshold interpolation that preserves the underlying linear-score quantile rule.

### Robustness and metrics

Each fitted detector and threshold remain fixed while scoring the same outer held-out images under nine conditions:

- Clean images.
- Square occlusion covering approximately 5%, 10%, or 20% of the image.
- Half-resolution downsampling followed by restoration to the input size.
- Gaussian blur with a 1.5-pixel standard deviation.
- Gaussian noise at low, high, and stress levels: 0.0001, 0.001, and 0.005 times 65,535 digital numbers.

Spatial transformations align all four bands. Corruption randomness depends on image ID, operation family, and seed, making repeated runs independent of image ordering.

Evaluation reports AP, disturbance recall (`recall_P`), precision, reference alert rate (`FPR_R`), and natural-control alert rate (`FPR_N`). Paired comparisons measure AP changes from clean images and the incremental effects of NIR and texture on the same folds. Parameter-selection tables report lower, middle, and upper candidate counts, including ties and failed modal-middle checks.

## Configuration and command-line options

`config.py` contains paths, random seed, cohort group counts, frozen parameter grids, feature settings, degradation settings, and training budgets. Edit scientific settings before importing modules, then start a fresh process. The generated split hash is recorded in the experiment identity and is reproducible for the same source, seed, and code.

```bash
python main.py --help
python main.py --dry-run
python main.py --models DeepSVDD
python main.py --models PCA OCSVM --dry-run
python main.py --input-root /kaggle/input --output-root /kaggle/working/my-results
python main.py --labels-path /kaggle/input/images/train_v2.csv
python main.py --recompute-features --no-plots
```

Set `RUN_EXPERIMENT = False` in `config.py` to make the command-line entry point print settings without running the experiment. A dry run prints configuration only; input and GPU validation occur when the experiment starts.

For interactive use after copying the project:

```python
sys.path.insert(0, str(PROJECT))
from main import run

result = run(output_root="/kaggle/working/my-results", make_plots=True)
display(result["summary"])
display(result["completion"])
```

`run()` also accepts `input_root`, `labels_path`, `use_fixed_feature_cache`, and `models` arguments.

## Project structure


| Files                                                                            | Responsibility                                                                |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `main.py`, `config.py`                                                           | Entry point, reproducibility setup, settings, and paths                       |
| `data/loader.py`, `data/labels.py`, `data/splits.py`                             | TIFF inventory and validation, label policy, and grouped nested splits        |
| `preprocessing/features.py`                                                      | Feature extraction, SQLite extraction cache, and authenticated NPZ loading    |
| `preprocessing/scaling.py`, `preprocessing/degradations.py`                      | Shared scaler, training-rank checks, and deterministic degradations           |
| `models/`                                                                        | Detector interface, model factory, and six detector implementations           |
| `training/tuning.py`, `training/calibration.py`, `training/trainers.py`          | Parameter selection, threshold calculation, fitting, and training diagnostics |
| `training/protocol.py`                                                           | Scientific source registry, runtime identity, and protocol validation         |
| `evaluation/metrics.py`, `evaluation/robustness.py`, `evaluation/aggregation.py` | Metrics, experiment orchestration, and paired summaries                       |
| `visualization/plots.py`                                                         | Figures, illustrative prediction tables, and archive finalization             |
| `tests/`                                                                         | Feature, split, metric, model, cache, and resume checks                       |
| `kaggle_runner.ipynb`                                                            | Kaggle setup, execution, and result inspection                                |

## Outputs and reproducibility

The default output directory is `/kaggle/working/comp90051-amazon-anomaly/outputs/`:

```text
outputs/
├── raw_image_checks.csv
├── benchmark_manifest.csv
├── feature_audit.json
├── results/<experiment-identity>/
│   ├── experiment_identity.json
│   ├── split_manifest.csv, split_supports.csv
│   ├── development_manifest.csv, benchmark_manifest.csv
│   ├── inner_AP.csv, candidate_means.csv, parameter_selections.csv
│   ├── metrics.csv, predictions.csv, outer_fits.csv, middle_modes.csv
│   ├── summary.csv, corruption_pairs.csv, corruption_summary.csv
│   ├── feature_pairs.csv, feature_summary.csv
│   ├── inner_units/<unit-identity>/
│   ├── outer_units/<unit-identity>/
│   ├── completion.json, integrity_inventory.json
│   ├── figures/
│   └── result_evidence.zip
├── figures/<experiment-identity>/
│   ├── feature_examples.png
│   ├── *_robustness.png
│   ├── paired_features_*.png
│   └── detection_examples.csv
└── checkpoints/feature_cache/condition_features.sqlite
```

Figures are generated when plotting is enabled. The SQLite checkpoint is created when extracting features from raw images.

Each completed inner or outer unit stores its numerical evidence, training diagnostics, and a receipt containing file hashes. Reuse requires matching code, settings, runtime, ordered image IDs, splits, feature values, and verified receipt hashes. Interrupted feature extraction resumes from committed image/condition entries.

Rerun with the same writable output directory to resume an interrupted experiment. To resume in a new Kaggle session, restore the saved outputs and matching execution environment first. Completed units reuse saved scores and diagnostics; model weights are not serialized.

A complete six-model run produces 1,620 metric rows and 388,800 prediction rows. Selected-model runs validate counts for their selected scope. `completion.json` records selected models, execution completeness, full-comparison completeness, and the modal-middle requirement separately. `integrity_inventory.json` records evidence-file hashes, and `result_evidence.zip` packages the numerical outputs, audit records, and generated figures for download.

## Troubleshooting a failed Kaggle run

The launcher's experiment cell streams standard output and errors into the notebook and saves a copy to `OUTPUT_ROOT / "run.log"`. If the program fails, the displayed exception includes its final output lines. Inspect the underlying traceback to identify the failing input, dependency, or validation check.

`CalledProcessError: ... returned non-zero exit status 1` from a direct `subprocess.run(..., check=True)` call only reports the child process's exit status. The cause appears earlier in the program's output. To inspect a saved launcher log:

```python
print((OUTPUT_ROOT / "run.log").read_text()[-12000:])
```

The log is replaced on each experiment launch. A dry run checks configuration only; it does not verify attached data, dependencies needed for training, or CUDA availability. If the printed command has no `--models` argument, it requests all six models. Set `MODELS` and rerun the configuration cell to rebuild the command for a selected-model run.

## Automated verification

From the project directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --dry-run
```

Tests use synthetic images and feature matrices, including short CPU training runs for all six detectors. They check band ordering, feature dimensions, deterministic degradations, duplicate isolation, AP ties, threshold interpolation, cache authentication, and saved-unit integrity. Full experiment completion is validated separately by the result counts and evidence files.

## Interpretation

The benchmark is an exposed repeated benchmark because it has influenced adaptive parameter review. Results therefore describe performance on this declared benchmark rather than an untouched test population. Disturbance tags are proxy labels and do not establish illegal activity or early deforestation. Natural-control alert rates reflect the label policy. Fold standard deviations describe variability across folds, and fixed training budgets do not establish convergence.
