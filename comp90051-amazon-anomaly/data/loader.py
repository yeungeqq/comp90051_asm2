"""data: loader for the Amazon robustness experiment."""

import config
import hashlib
import numpy as np
import os
import pandas as pd
import tifffile
import time
from collections import Counter, defaultdict
from config import FeatureConfig
from data.labels import build_manifest
from pathlib import Path


def read_rgbn(path, config=FeatureConfig()):
    """Return raw numeric H x W x 4 in RGBN order, without display enhancement."""
    raw = tifffile.imread(path)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D four-band TIFF, received shape {raw.shape}.")
    if raw.shape[-1] == 4:
        pass  # Common tifffile layout: height, width, band.
    elif raw.shape[0] == 4:
        raw = np.moveaxis(raw, 0, -1)  # Support band-first files explicitly.
    else:
        raise ValueError(f"Expected exactly four bands, received shape {raw.shape}.")
    if min(raw.shape[:2]) < config.minimum_side:
        raise ValueError(f"Original image is smaller than {config.minimum_side} pixels per side.")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError("TIFF contains nonnumeric or nonfinite pixels.")
    if (raw < 0).any():
        raise ValueError("Negative DN values require source/NoData review.")
    if not np.any(raw > 0):
        raise ValueError("TIFF is entirely zero; inspect source/NoData handling.")
    indices = [config.source_band_order.index(band) for band in "RGBN"]
    return np.ascontiguousarray(raw[..., indices])


def rgb_for_display(rgbn):
    """Per-image contrast stretch for DISPLAY ONLY; never use it for features."""
    rgb = np.asarray(rgbn[..., :3], dtype=float)
    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-8), 0, 1)


def _index_paths(paths):
    """Retain ALL paths for each stem so ambiguous duplicate files are visible."""
    index = defaultdict(list)
    for path in sorted(paths):
        index[Path(path).stem].append(str(path))
    return dict(index)


def discover_inputs(root="/kaggle/input", labels_path=None):
    """Return filename indexes, the selected label CSV, counts and action messages.

    Directory symlinks are not traversed, preventing cycles and unexpected scans
    outside an input mount. Symlinks to files are included, as in the original
    inventory; these uncommon entries can require an individual target lookup.
    Permission errors propagate: an incomplete scan is not a valid inventory.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    started = time.perf_counter()
    print(f"Scanning attached input directory entries: {root}", flush=True)
    tiffs, jpegs, csvs, archives = [], [], [], []
    file_count = 0
    directories = [str(root)]
    while directories:
        # A DirEntry normally reuses the directory listing's type information.
        # Do not replace this with Path(entry.path).is_file(): that adds a
        # separate metadata request for each of tens of thousands of images.
        directory = directories.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    directories.append(entry.path)
                    continue
                if not entry.is_file():
                    continue  # Ignore sockets, broken links, and other non-files.
                file_count += 1
                name = entry.name.lower()
                suffix = os.path.splitext(name)[1]
                if suffix in {".tif", ".tiff"}:
                    tiffs.append(Path(entry.path))
                if suffix in {".jpg", ".jpeg"}:
                    jpegs.append(Path(entry.path))
                if name in {"train_v2.csv", "train.csv"}:
                    csvs.append(Path(entry.path))
                if name.endswith((".zip", ".7z", ".tar", ".tar.gz", ".tgz")):
                    archives.append(Path(entry.path))

    # The file-format choice is explicit and reproducible. The
    # chosen CSV header is the only data content read during this inventory.
    messages = []
    selected = None
    if labels_path is not None:
        selected = Path(labels_path)
        if not selected.is_file():
            raise FileNotFoundError(f"Configured labels CSV does not exist: {selected}")
    else:
        preferred = [path for path in csvs if path.name.lower() == "train_v2.csv"] or csvs
        if len(preferred) == 1:
            selected = preferred[0]
        elif len(preferred) > 1:
            messages.append("Multiple label files found. Set LABELS_PATH explicitly.")
        else:
            messages.append("Attach the labelled training CSV (prefer train_v2.csv).")
    if not tiffs:
        messages.append(
            "No uncompressed TIFFs found. Attach the four-band training TIFFs; "
            "JPG-only data cannot run the NIR comparison. Archives are listed below; "
            "extract them deliberately after checking storage, or attach an "
            "uncompressed copy. This notebook does not auto-extract huge archives."
        )
    if selected is not None:
        columns = set(pd.read_csv(selected, nrows=0).columns)
        if not {"image_name", "tags"}.issubset(columns):
            messages.append("Selected CSV is missing image_name/tags. Check the source.")
    result = {
        "root": str(root), "labels_csv": str(selected) if selected else None,
        "label_candidates": [str(path) for path in sorted(csvs)],
        "tiff_index": _index_paths(tiffs), "jpg_index": _index_paths(jpegs),
        "archives": [str(path) for path in sorted(archives)],
        "counts": {"files": file_count, "tiff": len(tiffs), "jpg": len(jpegs),
                   "candidate_csv": len(csvs), "archives": len(archives)},
        "messages": messages,
    }
    print(f"Input inventory complete: {file_count:,} files, {len(tiffs):,} TIFFs, "
          f"{len(jpegs):,} JPGs; elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return result


def locate_single_input(filename):
    """Find one explicitly named metadata input; ambiguity is an error."""
    matches = list(config.INPUT_ROOT.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f'Attach exactly one {filename}; found {len(matches)}.')
    return matches[0]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(path):
    """Hash canonical RGBN pixel values so equivalent TIFF encodings group together."""
    image = read_rgbn(path)
    canonical = np.ascontiguousarray(image, dtype='<f4')
    return hashlib.sha256(str(canonical.shape).encode() + canonical.tobytes()).hexdigest(), image


def verify_raw_benchmark(benchmark, output):
    """Connect IDs to TIFF files, then check labels and the actual pixel hashes."""
    inventory = discover_inputs(config.INPUT_ROOT, config.LABELS_PATH)
    if not inventory['labels_csv'] or not inventory['tiff_index']:
        raise ValueError('Attach the labelled four-band TIFF collection and label CSV.')
    source = build_manifest(inventory['labels_csv'], inventory['tiff_index'])
    if len(source) < 10000:
        raise ValueError('The original labelled dataset must contain at least 10,000 images.')
    source = source.set_index('image_id', drop=False)
    if not set(benchmark.image_id) <= set(source.index):
        raise ValueError('Some fixed benchmark images are missing from the source.')
    selected = source.loc[benchmark.image_id].reset_index(drop=True)
    for field in ('image_id', 'group', 'tags'):
        if selected[field].tolist() != benchmark[field].tolist():
            raise ValueError('The source disagrees with the fixed manifest: ' + field)
    if not selected.tiff_path_count.eq(1).all():
        raise ValueError('Every benchmark image must have exactly one TIFF path.')

    quality = []
    for number, row in selected.iterrows():
        digest, image = pixel_sha256(row.tiff_path)
        if digest != benchmark.iloc[number].pixel_sha256:
            raise ValueError('Raw pixels changed for ' + row.image_id)
        quality.append({'image_id': row.image_id, 'pixel_sha256': digest,
                        'height': image.shape[0], 'width': image.shape[1],
                        'pixel_min': float(image.min()), 'pixel_max': float(image.max())})
        if (number + 1) % 200 == 0:
            print(f'Raw image checks: {number + 1}/2400', flush=True)
    selected['pixel_sha256'] = benchmark.pixel_sha256.to_numpy()
    selected['duplicate_group'] = benchmark.duplicate_group.to_numpy()
    selected['usable'] = True
    selected['cohort'] = 'exposed_benchmark'
    pd.DataFrame(quality).to_csv(output / 'raw_image_checks.csv', index=False)
    selected.to_csv(output / 'benchmark_manifest.csv', index=False)
    return selected
