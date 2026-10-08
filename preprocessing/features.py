"""preprocessing: features for the Amazon robustness experiment."""

import ast
import hashlib
import inspect
import json
import numpy as np
import pandas as pd
import scipy
import skimage
import sqlite3
import textwrap
import tifffile
import time
from config import (
    CONDITIONS,
    CorruptionConfig,
    FEATURE_PACKAGE_SHA256,
    FEATURE_REPRESENTATIONS,
    FeatureConfig,
    TextureConfig,
)
from data.loader import file_sha256, locate_single_input, read_rgbn
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from preprocessing.degradations import corrupt_rgbn
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import graycomatrix, graycoprops


IMPLEMENTATION_ID = 'not-initialized-by-runner'


def _grid_regions(array, grid_size=4):
    """Yield all 16 nonempty cells; array_split also supports non-256 dimensions."""
    for row in np.array_split(array, grid_size, axis=0):
        yield from np.array_split(row, grid_size, axis=1)


def _band_features(array, prefix):
    values = {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array, ddof=0)),
        f"{prefix}_p10": float(np.percentile(array, 10)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
    }
    values.update({f"{prefix}_grid_{i:02d}_mean": float(np.mean(cell))
                   for i, cell in enumerate(_grid_regions(array))})
    return values


def feature_columns():
    """Names in stable order; feature set A is nested in B, which is nested in C."""
    def band_names(prefix):
        return ([f"{prefix}_{stat}" for stat in ["mean", "std", "p10", "median", "p90"]]
                + [f"{prefix}_grid_{i:02d}_mean" for i in range(16)])
    a = sum((band_names(b) for b in ["red", "green", "blue"]), [])
    b = a + band_names("nir")
    c = b + [f"edge_grid_{i:02d}_density" for i in range(16)]
    c += [f"glcm_{prop}_{angle}" for prop in ["contrast", "homogeneity"]
          for angle in ["0", "45", "90", "135"]]
    assert (len(a), len(b), len(c)) == (63, 84, 108)
    return {"A_RGB": a, "B_RGB_NIR": b, "C_RGB_NIR_TEXTURE": c}


def quantize_luminance(gray, feature_config=FeatureConfig(), texture_config=TextureConfig()):
    """Return integer GLCM input (H,W) and quality measurements for this image.

    gray is luminance in DN/65535 units. We use this image's percentile range,
    not statistics pooled from other images. GLCM expects integer bin numbers.
    The same calculation is applied independently to every corrupted condition.
    """
    low, high = np.percentile(gray, [texture_config.lower_percentile, texture_config.upper_percentile])
    span = float(high - low)
    if span <= feature_config.texture_epsilon:
        unit = np.zeros_like(gray)
    else:
        unit = np.clip((gray - low) / span, 0, 1)
    clipped = float(np.mean((gray < low) | (gray > high)))
    levels = feature_config.glcm_levels
    dtype = np.uint8 if levels <= 256 else np.uint16
    quantized = np.floor(unit * (levels - 1)).astype(dtype)
    return quantized, {
        "texture_percentile_low_dn": float(low * feature_config.dn_scale),
        "texture_percentile_high_dn": float(high * feature_config.dn_scale),
        "texture_quantizer_clipped_fraction": clipped,
        "texture_percentile_span_zero": bool(span <= feature_config.texture_epsilon),
        "glcm_occupied_bins": int(np.unique(quantized).size),
    }


def features_from_rgbn(rgbn, feature_config=FeatureConfig(), texture_config=TextureConfig()):
    """Return (108 ordered feature values, quality dict) from one RGBN image.

    Input: finite nonnegative array shaped (height, width, 4), already RGBN.
    Output: an insertion-ordered dictionary matching feature_columns()['C_RGB_NIR_TEXTURE'].
    We compute all 108 numbers once and later slice A/B/C without recomputing them.
    Band statistics and edges use fixed DN units; only GLCM uses image percentiles.
    """
    array = np.asarray(rgbn, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 4 or min(array.shape[:2]) < 4:
        raise ValueError("Expected H x W x 4 RGBN with sides >=4.")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Feature values require finite nonnegative DN.")
    if feature_config.glcm_levels > 256:
        raise ValueError("GLCM levels above 256 are unsupported for this bounded-memory feature design.")
    array = array / feature_config.dn_scale
    values = {}
    for band, prefix in enumerate(("red", "green", "blue", "nir")):
        values.update(_band_features(array[..., band], prefix))
    gray = np.dot(array[..., :3], [0.2126, 0.7152, 0.0722])
    gradient = np.hypot(sobel(gray, axis=0, mode="reflect") / 8.0,
                        sobel(gray, axis=1, mode="reflect") / 8.0)
    edges = gradient > feature_config.edge_threshold
    values.update({f"edge_grid_{index:02d}_density": float(region.mean())
                   for index, region in enumerate(_grid_regions(edges))})
    quantized, quality = quantize_luminance(gray, feature_config, texture_config)
    glcm = graycomatrix(quantized, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                       levels=feature_config.glcm_levels, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity"):
        for angle, value in zip(("0", "45", "90", "135"), graycoprops(glcm, prop)[0]):
            values[f"glcm_{prop}_{angle}"] = float(value)
    ordered = {name: values[name] for name in feature_columns()["C_RGB_NIR_TEXTURE"]}
    if not np.isfinite(list(ordered.values())).all():
        raise ValueError("Nonfinite features; no zero-filling is performed.")
    quality.update({
                    "texture_clipped_fraction": float(np.mean((gray < 0) | (gray > 1))),
                    "edge_density_mean": float(edges.mean()),
                    "original_height": int(array.shape[0]), "original_width": int(array.shape[1])})
    return ordered, quality


def feature_protocol(feature_config, texture_config, corruption_config, seed):
    """Fingerprint settings and the visible notebook implementation, without files.

    IMPLEMENTATION_ID is set by main.run from the modular scientific source.
    No historical source file or embedded archive is read to calculate this fingerprint.
    Library versions are recorded because numerical routines can change.
    """
    protocol = {
        "implementation_id": str(IMPLEMENTATION_ID),
        "feature_config": asdict(feature_config), "texture_config": asdict(texture_config),
        "corruption_config": asdict(corruption_config), "seed": int(seed),
        "columns": feature_columns()["C_RGB_NIR_TEXTURE"],
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "skimage": skimage.__version__, "tifffile": tifffile.__version__},
    }
    key = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return key, protocol


class ConditionFeatureCache:
    """Transactional, per-image/condition cache; completed entries survive retry.

    Files are identified by image ID, path, size and nanosecond mtime, suitable
    for immutable Kaggle mounts. Pixels deliberately modified while preserving
    metadata require manual cache deletion. Each completed condition is committed
    immediately; interrupted extraction can reuse it on the next call.
    """

    def __init__(self, cache_dir, protocol_key):
        self.protocol_key = protocol_key
        self.connection = None
        self.path = None
        if cache_dir is not None:
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / "condition_features.sqlite"
            self.connection = sqlite3.connect(self.path, timeout=60)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS conditions (protocol_key TEXT, source_key TEXT, "
                "condition TEXT, image_id TEXT, features BLOB, digest TEXT, qc TEXT, "
                "compute_seconds REAL, PRIMARY KEY(protocol_key,source_key,condition))")
            self.connection.commit()

    def get(self, source_key):
        if self.connection is None:
            return {}
        found = {}
        records = self.connection.execute(
            "SELECT condition,features,digest,qc,compute_seconds FROM conditions "
            "WHERE protocol_key=? AND source_key=?", (self.protocol_key, source_key))
        for condition, payload, digest, qc, seconds in records:
            try:
                array = np.frombuffer(payload, dtype="<f4").copy()
                quality = json.loads(qc)
                if (array.shape != (108,) or not np.isfinite(array).all()
                        or sha256(payload).hexdigest() != digest or not isinstance(quality, dict)):
                    continue
                found[condition] = (array, quality, float(seconds))
            except (ValueError, TypeError):
                continue  # A damaged entry is recomputed, never treated as valid data.
        return found

    def put(self, source_key, condition, image_id, array, qc, seconds):
        if self.connection is None:
            return
        payload = np.asarray(array, dtype="<f4").tobytes()
        self.connection.execute(
            "INSERT OR REPLACE INTO conditions VALUES (?,?,?,?,?,?,?,?)",
            (self.protocol_key, source_key, condition, image_id, payload,
             sha256(payload).hexdigest(), json.dumps(qc, allow_nan=False), float(seconds)))
        self.connection.commit()

    def close(self):
        if self.connection is not None:
            self.connection.close()


def extract_condition_features(manifest, feature_cache_dir=None, seed=51,
                               conditions=CONDITIONS, feature_names=None, progress_every=100,
                               feature_config=FeatureConfig(), texture_config=TextureConfig(),
                               corruption_config=CorruptionConfig()):
    """Return (condition -> A/B/C float32 arrays, serializable audit).

    Image IDs follow the supplied manifest exactly. All 108 values are computed
    once for each image/condition and cached together, even when only A or B is
    requested. Changing row order or requesting another feature subset can reuse
    the cache safely. Failed images abort rather than silently changing labels.
    Labels, group assignments and model predictions are not inputs to features.
    """
    started = time.perf_counter()
    names = feature_columns()
    feature_names = tuple(names) if feature_names is None else tuple(feature_names)
    conditions = tuple(conditions)
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError("Choose unique, known conditions.")
    if not feature_names or len(set(feature_names)) != len(feature_names) or set(feature_names) - set(names):
        raise ValueError("Choose unique A/B/C feature names.")
    if len(manifest) == 0 or not {"image_id", "tiff_path"}.issubset(manifest.columns):
        raise ValueError("Pass a nonempty image_id/tiff_path manifest.")
    ids = manifest["image_id"].astype(str).tolist()
    if len(set(ids)) != len(ids) or any(not image_id.strip() for image_id in ids):
        raise ValueError("Image IDs must be unique and nonblank.")
    if "usable" in manifest and not manifest["usable"].eq(True).all():
        raise ValueError("Only audited usable rows may be extracted.")
    if feature_config.dn_scale != corruption_config.dn_scale or int(progress_every) < 1:
        raise ValueError("DN scales must match and progress_every must be positive.")
    protocol_key, protocol = feature_protocol(feature_config, texture_config, corruption_config, seed)
    cache = ConditionFeatureCache(feature_cache_dir, protocol_key)
    all_values = {condition: np.empty((len(manifest), 108), dtype=np.float32) for condition in conditions}
    quality_rows, source_keys = [], []
    hits = 0
    current_compute = {condition: 0.0 for condition in conditions}
    cached_compute = {condition: 0.0 for condition in conditions}
    read_seconds = 0.0
    print(f"Condition features: {len(manifest):,} images × {len(conditions)} conditions; "
          "texture=within-image 2nd/98th percentiles; one 108-dimensional extraction per condition.", flush=True)
    try:
        for row_number, (_, row) in enumerate(manifest.iterrows()):
            image_id = ids[row_number]
            path = Path(row["tiff_path"])
            stat = path.stat()
            source_key = sha256(json.dumps([image_id, str(path.resolve()), stat.st_size, stat.st_mtime_ns]).encode()).hexdigest()
            source_keys.append(source_key)
            entries = cache.get(source_key)
            original = None
            for condition in conditions:
                if condition in entries:
                    array, quality, cost = entries[condition]
                    cached_compute[condition] += cost
                    hits += 1
                else:
                    if original is None:
                        before = time.perf_counter()
                        original = read_rgbn(path, feature_config)
                        read_seconds += time.perf_counter() - before
                    before = time.perf_counter()
                    transformed = corrupt_rgbn(original, condition, image_id, seed, corruption_config)
                    values, quality = features_from_rgbn(transformed, feature_config, texture_config)
                    array = np.asarray(list(values.values()), dtype=np.float32)
                    quality["changed_pixel_fraction"] = float(np.mean(np.any(transformed != original, axis=-1)))
                    cost = time.perf_counter() - before
                    current_compute[condition] += cost
                    cache.put(source_key, condition, image_id, array, quality, cost)
                all_values[condition][row_number] = array
                quality_rows.append({"image_id": image_id, "condition": condition, **quality})
            done = row_number + 1
            if done % int(progress_every) == 0 or done == len(manifest):
                print(f"Condition features {done:,}/{len(manifest):,}; condition-cache hits={hits:,}; "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    finally:
        cache.close()  # Completed conditions remain committed even after failure.
    columns = names["C_RGB_NIR_TEXTURE"]
    indices = {name: [columns.index(column) for column in names[name]] for name in feature_names}
    result = {condition: {name: matrix[:, indices[name]].copy() for name in feature_names}
              for condition, matrix in all_values.items()}
    feature_qc = []
    for condition, by_set in result.items():
        for name, matrix in by_set.items():
            standard_deviations = np.std(matrix.astype(np.float64), axis=0)
            constant = standard_deviations <= 1e-12
            feature_qc.append({"condition": condition, "feature_set": name, "rows": len(matrix),
                "feature_count": matrix.shape[1], "all_finite": bool(np.isfinite(matrix).all()),
                "constant_feature_count": int(constant.sum()),
                "constant_feature_names": [column for column, flag in zip(names[name], constant) if flag]})
    audit = {
        "protocol_key": protocol_key, "protocol": protocol, "image_ids": ids,
        "source_fingerprint": sha256(json.dumps(source_keys).encode()).hexdigest(),
        "conditions": list(conditions), "feature_dimensions": {name: len(names[name]) for name in feature_names},
        "cache_file": str(cache.path) if cache.path else None, "condition_cache_hits": hits,
        "conditions_computed": len(ids) * len(conditions) - hits,
        "timings": {"elapsed_seconds": time.perf_counter() - started, "image_read_seconds": read_seconds,
                    "condition_compute_seconds_this_call": current_compute,
                    "condition_compute_seconds_reused": cached_compute},
        "quality_rows": quality_rows, "feature_qc": feature_qc,
        "interpretation": "GLCM uses within-image 2nd/98th percentile scaling; band and edge features retain fixed DN units. Black masking can change percentile bounds. Corruptions are fixed synthetic stress tests. Constant-feature QC is descriptive and never removes columns.",
    }
    return result, audit


def stable_ast_dump(node):
    """Keep the pinned AST format stable when Python adds empty type parameters."""
    if isinstance(node, ast.AST):
        fields = []
        for name, value in ast.iter_fields(node):
            # Python 3.12 added this unused field to functions and classes.
            # None of the feature functions has a generic type parameter.
            if name == 'type_params' and value == []:
                continue
            if value is None and getattr(type(node), name, ...) is None:
                continue
            fields.append(name + '=' + stable_ast_dump(value))
        return type(node).__name__ + '(' + ', '.join(fields) + ')'
    if isinstance(node, list):
        return '[' + ', '.join(stable_ast_dump(value) for value in node) + ']'
    return repr(node)


def visible_definition_ast(name):
    """Read a visible function or class for source comparison, without executing text."""
    from training.protocol import resolve_scientific_definition

    value = resolve_scientific_definition(name)
    source = inspect.getsource(value)
    tree = ast.parse(textwrap.dedent(source))
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    # Documentation and comments may improve without changing the calculation.
    for part in ast.walk(node):
        if isinstance(part, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and part.body:
            first = part.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                part.body = part.body[1:]
    return hashlib.sha256(stable_ast_dump(node).encode()).hexdigest()


def load_authenticated_features(manifest):
    """Read the exact fixed arrays only if source, order, columns and bytes match."""
    path = locate_single_input('cached_feature_manifest.json')
    if file_sha256(path) != FEATURE_PACKAGE_SHA256:
        raise ValueError('The feature-package manifest differs from its reviewed hash.')
    package = json.loads(path.read_text())
    if package['feature_columns'] != feature_columns():
        raise ValueError('Feature names or ordering differ from the saved arrays.')
    for name, expected in package['source_feature_ast_hashes'].items():
        if visible_definition_ast(name) != expected:
            raise ValueError('Feature code changed: ' + name + '. Recompute features from raw images.')
    fixed = pd.read_csv(path.parent / 'benchmark_manifest.csv', dtype=str)
    columns = ['image_id', 'tags', 'group', 'pixel_sha256', 'duplicate_group']
    if manifest[columns].astype(str).to_dict('records') != fixed[columns].to_dict('records'):
        raise ValueError('The feature cache belongs to a different ordered cohort.')
    feature_path = path.parent / 'cached_features.npz'
    if file_sha256(feature_path) != package['files']['cached_features.npz']['sha256']:
        raise ValueError('Cached feature file bytes are corrupted or changed.')
    arrays = {}
    with np.load(feature_path, allow_pickle=False) as archive:
        expected_keys = {condition + '__' + name for condition in CONDITIONS for name in FEATURE_REPRESENTATIONS}
        if set(archive.files) != expected_keys:
            raise ValueError('The cache must contain exactly all 27 condition/feature arrays.')
        for condition in CONDITIONS:
            arrays[condition] = {}
            for name in FEATURE_REPRESENTATIONS:
                key = condition + '__' + name
                values = np.ascontiguousarray(archive[key])
                expected = package['arrays']['formal/' + key]
                if (list(values.shape) != expected['shape'] or values.dtype.str != expected['dtype']
                        or hashlib.sha256(values.tobytes()).hexdigest() != expected['sha256']
                        or not np.isfinite(values).all()):
                    raise ValueError('Invalid cached feature array: ' + key)
                arrays[condition][name] = values
    # This route verifies saved values. It does not pretend to regenerate every
    # pixel-to-feature calculation using the current library versions.
    audit = {'route': 'authenticated_fixed_feature_values', 'package_sha256': FEATURE_PACKAGE_SHA256,
             'all_arrays_verified': True, 'independent_raw_feature_regeneration': False,
             'feature_library_version_limit': package['feature_library_version_limit'],
             'feature_qc': [{'condition': condition, 'features': name,
                            'rows': len(values), 'columns': values.shape[1], 'all_finite': True}
                           for condition, arms in arrays.items() for name, values in arms.items()]}
    return arrays, audit
