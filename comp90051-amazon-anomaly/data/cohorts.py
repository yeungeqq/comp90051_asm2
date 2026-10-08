"""Deterministic, duplicate-safe cohort manifests for the Amazon TIFF dataset."""

import json
import numpy as np
import pandas as pd
from config import COHORT_GROUP_COUNTS, SEED
from data.labels import build_manifest
from data.loader import discover_inputs, file_sha256, pixel_sha256
from pathlib import Path


MANIFEST_COLUMNS = ('image_id', 'tags', 'group', 'pixel_sha256', 'duplicate_group')


def validate_cohorts(development, benchmark, group_counts=COHORT_GROUP_COUNTS):
    """Validate schema, class quotas, uniqueness, and cross-cohort separation."""
    expected = dict(group_counts)
    for role, frame in (('development', development), ('benchmark', benchmark)):
        missing = set(MANIFEST_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f'{role} manifest lacks columns: {sorted(missing)}')
        if frame[list(MANIFEST_COLUMNS)].isna().any().any():
            raise ValueError(f'{role} manifest contains missing cohort fields.')
        if frame.image_id.astype(str).duplicated().any():
            raise ValueError(f'{role} manifest contains duplicate image IDs.')
        if frame.groupby('group').size().to_dict() != expected:
            raise ValueError(f'{role} manifest does not match configured group counts: {expected}.')
        if not frame.pixel_sha256.astype(str).eq(frame.duplicate_group.astype(str)).all():
            raise ValueError(f'{role} manifest has inconsistent duplicate-group hashes.')
        if frame.pixel_sha256.astype(str).duplicated().any():
            raise ValueError(f'{role} manifest contains repeated pixel hashes.')
    for column in ('image_id', 'pixel_sha256', 'duplicate_group'):
        if set(development[column].astype(str)) & set(benchmark[column].astype(str)):
            raise ValueError(f'Development and benchmark overlap: {column}.')
    return development, benchmark


def select_cohorts(source, hash_image, seed=SEED, group_counts=COHORT_GROUP_COUNTS):
    """Select disjoint rows deterministically, skipping repeated pixel hashes.

    ``hash_image`` returns a digest for one TIFF path. Only candidates needed to
    fill the two cohorts are decoded, rather than hashing the entire 21 GB source.
    """
    candidates = source.reset_index(drop=True).copy()
    candidates = candidates[
        candidates.group.isin(group_counts)
        & candidates.tiff_path_count.astype(int).eq(1)
        & candidates.tiff_path.astype(str).ne('')
    ].sort_values('image_id', kind='stable').reset_index(drop=True)
    for group, count in group_counts.items():
        available = int(candidates.group.eq(group).sum())
        if available < count * 2:
            raise ValueError(f'Not enough usable {group} TIFFs for both cohorts: {available}.')

    rng = np.random.default_rng(seed)
    used_ids, used_pixels = set(), set()
    cohorts = {}
    hashes_checked = 0
    for role in ('development', 'benchmark'):
        selected = []
        for group in ('R', 'P', 'N'):
            group_rows = candidates[candidates.group.eq(group)].reset_index(drop=True)
            order = rng.permutation(len(group_rows))
            group_selected = 0
            for position in order:
                if group_selected == group_counts[group]:
                    break
                row = group_rows.iloc[int(position)]
                image_id = str(row.image_id)
                if image_id in used_ids:
                    continue
                try:
                    hashed = hash_image(str(row.tiff_path))
                except (OSError, ValueError) as exc:
                    print(f'Skipping unusable TIFF {row.image_id}: {exc}', flush=True)
                    continue
                digest = hashed[0] if isinstance(hashed, tuple) else hashed
                hashes_checked += 1
                if digest in used_pixels:
                    continue
                record = row.to_dict()
                record['pixel_sha256'] = digest
                record['duplicate_group'] = digest
                record['cohort'] = role
                selected.append(record)
                used_ids.add(image_id)
                used_pixels.add(digest)
                group_selected += 1
                if hashes_checked % 200 == 0:
                    print(f'Cohort source hashes checked: {hashes_checked}', flush=True)
            if group_selected != group_counts[group]:
                raise ValueError(
                    f'Could not fill {role} cohort quota for group {group}: '
                    f'{group_selected}/{group_counts[group]} unique usable images.'
                )
        cohorts[role] = pd.DataFrame(selected).reset_index(drop=True)
        print(f'Created {role} cohort: {len(cohorts[role])} images', flush=True)
    return validate_cohorts(cohorts['development'], cohorts['benchmark'], group_counts)


def create_or_load_cohorts(input_root, output_dir, labels_path=None, seed=SEED,
                           group_counts=COHORT_GROUP_COUNTS):
    """Load cached generated manifests or create them from labels and four-band TIFFs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {role: output_dir / f'{role}_manifest.csv'
             for role in ('development', 'benchmark')}
    inventory = discover_inputs(input_root, labels_path)
    if not inventory['labels_csv'] or not inventory['tiff_index']:
        raise ValueError('Attach train_v2.csv and the extracted four-band train-tif-v2 TIFFs.')
    labels_hash = file_sha256(inventory['labels_csv'])
    metadata_path = output_dir / 'cohort_generation.json'
    present = all(path.is_file() for path in paths.values()) and metadata_path.is_file()
    if present:
        metadata = json.loads(metadata_path.read_text())
        reusable = (
            metadata.get('seed') == int(seed)
            and metadata.get('group_counts') == dict(group_counts)
            and metadata.get('labels_sha256') == labels_hash
            and all(metadata.get(f'{role}_manifest_sha256') == file_sha256(path)
                    for role, path in paths.items())
        )
        if reusable:
            tables = {role: pd.read_csv(path, dtype=str) for role, path in paths.items()}
            development, benchmark = validate_cohorts(
                tables['development'], tables['benchmark'], group_counts)
            return development, benchmark, paths['development'], paths['benchmark']
        print('Generated cohort cache does not match source/settings; regenerating.', flush=True)
    for path in (*paths.values(), metadata_path):
        path.unlink(missing_ok=True)

    source = build_manifest(inventory['labels_csv'], inventory['tiff_index'])
    if len(source) < 10000:
        raise ValueError('The labelled source must contain at least 10,000 images.')
    development, benchmark = select_cohorts(source, pixel_sha256, seed, group_counts)
    temporary_paths = {role: path.with_suffix(path.suffix + '.tmp')
                       for role, path in paths.items()}
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + '.tmp')
    for role, frame in (('development', development), ('benchmark', benchmark)):
        frame.to_csv(temporary_paths[role], index=False)
    metadata = {
        'seed': int(seed), 'group_counts': dict(group_counts),
        'labels_path': str(inventory['labels_csv']),
        'labels_sha256': labels_hash,
        'development_manifest_sha256': file_sha256(temporary_paths['development']),
        'benchmark_manifest_sha256': file_sha256(temporary_paths['benchmark']),
    }
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + '\n')
    for role in paths:
        temporary_paths[role].replace(paths[role])
    temporary_metadata.replace(metadata_path)
    print(f'Generated cohort manifests in {output_dir}', flush=True)
    return development, benchmark, paths['development'], paths['benchmark']