"""data: labels for the Amazon robustness experiment."""

import pandas as pd
from collections import Counter, defaultdict
from config import HUMAN_TAGS, KNOWN_TAGS, NATURAL_TAGS, TARGET_TAGS, WEATHER_TAGS


def classify_tags(tags):
    """Return mutually exclusive group and reason using the predeclared policy."""
    tags = frozenset(tags)
    if not tags:
        return "excluded", "missing_tags"
    if tags - KNOWN_TAGS:
        return "excluded", "unknown_tags_review_required"
    if len(tags & WEATHER_TAGS) != 1:
        return "excluded", "invalid_weather_tag_count"
    if "clear" not in tags:
        return "excluded", "not_clear"
    # P takes priority: a road-plus-water chip is never a negative control.
    if tags & TARGET_TAGS:
        return "P", "target_human_disturbance"
    if tags == {"clear", "primary"}:
        return "R", "reference_forest"
    if "primary" in tags and tags & NATURAL_TAGS and not tags & HUMAN_TAGS:
        return "N", "water_or_natural_variation_control"
    return "excluded", "outside_predeclared_main_groups"


def build_manifest(labels_csv, tiff_index, jpg_index=None):
    """Make one deterministic row per labelled image; never manufacture negatives.

    Multiple CSV rows for an ID are rejected even when their labels agree.
    Multiple TIFF paths for an ID need explicit source selection; they are not
    silently resolved using filesystem order. Index images from the chosen source
    directory instead if you have attached both old and new dataset copies.
    """
    labels = pd.read_csv(labels_csv, dtype={"image_name": "string", "tags": "string"})
    if not {"image_name", "tags"}.issubset(labels.columns):
        raise ValueError("CSV must contain image_name and tags columns.")
    if labels["image_name"].isna().any():
        raise ValueError("Missing image names in the label CSV; inspect before proceeding.")
    labels["image_name"] = labels["image_name"].str.strip()
    if labels["image_name"].eq("").any() or labels["image_name"].duplicated().any():
        raise ValueError("Blank or duplicate image IDs found. Resolve the label source first.")
    jpg_index = jpg_index or {}
    rows = []
    for _, source in labels.sort_values("image_name").iterrows():
        image_id = str(source["image_name"])
        tags = frozenset(str(source["tags"]).split()) if pd.notna(source["tags"]) else frozenset()
        group, reason = classify_tags(tags)
        tiff_paths = tiff_index.get(image_id, [])
        jpg_paths = jpg_index.get(image_id, [])
        if isinstance(tiff_paths, str):
            tiff_paths = [tiff_paths]
        if isinstance(jpg_paths, str):
            jpg_paths = [jpg_paths]
        row = {
            "image_id": image_id, "tags": " ".join(sorted(tags)), "group": group,
            "group_reason": reason, "eligible_label": group in {"R", "P", "N"},
            "tiff_path": tiff_paths[0] if len(tiff_paths) == 1 else "",
            "tiff_path_count": len(tiff_paths),
            "jpg_path": jpg_paths[0] if len(jpg_paths) == 1 else "",
            "unknown_tags": " ".join(sorted(tags - KNOWN_TAGS)),
        }
        row.update({f"tag_{tag}": tag in tags for tag in sorted(KNOWN_TAGS)})
        # This stricter P subset is descriptive, not evidence of early-stage damage.
        row["forest_context_target"] = (
            group == "P" and "primary" in tags and not tags &
            {"agriculture", "cultivation", "habitation", "conventional_mine", "artisinal_mine"}
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("The labels file has zero rows.")
    return result.set_index("image_id", drop=False)


def label_summary(manifest):
    """Return full-source tag counts and before/after unique-image group counts."""
    tags = Counter(tag for text in manifest["tags"] for tag in text.split())
    groups = manifest.groupby("group", dropna=False).size().rename("label_eligible_count").to_frame()
    if "usable" in manifest:
        groups["usable_unique_count"] = manifest[manifest["usable"]].groupby("group").size()
        groups["usable_unique_count"] = groups["usable_unique_count"].fillna(0).astype(int)
    return groups, pd.Series(tags, name="source_tag_count").sort_values(ascending=False)
