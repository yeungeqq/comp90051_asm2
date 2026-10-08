"""data: splits for the Amazon robustness experiment."""

import numpy as np
import pandas as pd
from config import GROUPS
from types import SimpleNamespace


def group_label_counts(manifest):
    """One row per duplicate cluster; a cluster may contain multiple label groups."""
    return pd.crosstab(manifest["duplicate_group"], manifest["group"]).reindex(
        columns=list(GROUPS), fill_value=0
    )


def make_grouped_folds(manifest, n_folds, seed):
    """Greedy group-aware, approximately stratified folds WITHOUT sklearn.

    A duplicate group always travels together.  The objective balances R/P/N
    counts relative to each group's desired per-fold count.  This cannot create
    missing geographic metadata or guarantee geographic independence.
    """
    table = group_label_counts(manifest)
    if len(table) < n_folds:
        raise ValueError(f"Only {len(table)} duplicate groups for {n_folds} folds.")
    support = (table > 0).sum()
    if (support < n_folds).any():
        raise ValueError(
            f"Insufficient distinct duplicate groups for {n_folds} folds: {support.to_dict()}. Audit sample definitions/support first."
        )
    counts = table.to_numpy(dtype=float)
    ids = table.index.to_numpy()
    target = counts.sum(axis=0) / n_folds
    rng = np.random.default_rng(seed)
    priority = np.sum(counts / target, axis=1)
    order = np.lexsort((rng.random(len(ids)), -priority))
    loads = np.zeros((n_folds, len(GROUPS)), dtype=float)
    cluster_counts = np.zeros(n_folds, dtype=int)
    assignment = {}
    for i in order:
        costs = []
        for fold in range(n_folds):
            hypothetical = loads.copy()
            hypothetical[fold] += counts[i]
            cost = np.sum((hypothetical / target) ** 2)
            costs.append(cost)
        best = np.flatnonzero(np.isclose(costs, np.min(costs), rtol=0, atol=1e-12))
        smallest = best[cluster_counts[best] == cluster_counts[best].min()]
        chosen = int(rng.choice(smallest))
        loads[chosen] += counts[i]
        cluster_counts[chosen] += 1
        assignment[ids[i]] = chosen
    folds = manifest["duplicate_group"].map(assignment).to_numpy(dtype=int)
    for fold in range(n_folds):
        present = set(manifest.loc[folds == fold, "group"])
        if present != set(GROUPS):
            raise ValueError(
                f"Fold {fold} lacks a group after grouped allocation ({present}). Inspect duplicate clusters or revise the prespecified design."
            )
    return folds


def reserve_reference_calibration(outer_train, config, seed):
    """Reserve reference-containing clusters; exclude the WHOLE cluster from dev."""
    table = group_label_counts(outer_train)
    candidates = table.index[table["R"] > 0].to_numpy()
    n_cal = max(1, int(np.ceil(config.calibration_fraction * len(candidates))))
    if len(candidates) - n_cal < config.inner_folds:
        raise ValueError("Too few R clusters after reserving threshold calibration.")
    rng = np.random.default_rng(seed)
    pure = [x for x in candidates if table.loc[x, ["P", "N"]].sum() == 0]
    mixed = [x for x in candidates if table.loc[x, ["P", "N"]].sum() != 0]
    rng.shuffle(pure)
    rng.shuffle(mixed)
    chosen = set((pure + mixed)[:n_cal])
    held = outer_train["duplicate_group"].isin(chosen).to_numpy()
    return (np.flatnonzero(~held), np.flatnonzero(held))


def build_nested_split_plan(m, outer_folds, inner_folds, seed, calibration_fraction):
    """Freeze ALL splits before any model runs, grouping duplicates throughout."""
    outer_ids = make_grouped_folds(m, outer_folds, seed)
    config = SimpleNamespace(
        calibration_fraction=calibration_fraction, inner_folds=inner_folds
    )
    plans, rows, supports = ([], [], [])
    for outer in range(outer_folds):
        held = np.flatnonzero(outer_ids == outer)
        available = np.flatnonzero(outer_ids != outer)
        dev_local, cal_local = reserve_reference_calibration(
            m.iloc[available].reset_index(drop=True), config, seed + 100 + outer
        )
        dev, cal_cluster = (available[dev_local], available[cal_local])
        cal = cal_cluster[m.iloc[cal_cluster].group.to_numpy() == "R"]
        development = m.iloc[dev].reset_index(drop=True)
        inner_ids = make_grouped_folds(development, inner_folds, seed + 200 + outer)
        final_fit = dev[development.group.to_numpy() == "R"]
        if len(final_fit) < 3 or len(cal) < 1:
            raise ValueError("Insufficient reference fitting/calibration support.")
        inner_plans = []
        for inner in range(inner_folds):
            fit = dev[(inner_ids != inner) & (development.group.to_numpy() == "R")]
            val = dev[inner_ids == inner]
            if len(fit) < 3:
                raise ValueError("An inner fit requires at least 3 reference images.")
            if not set(m.iloc[fit].duplicate_group).isdisjoint(
                m.iloc[val].duplicate_group
            ):
                raise RuntimeError(
                    "Duplicate leakage between inner fit and validation."
                )
            inner_plans.append((fit, val))
        cluster_sets = [
            set(m.iloc[ix].duplicate_group) for ix in (dev, cal_cluster, held)
        ]
        if any((cluster_sets[i] & cluster_sets[j] for i in range(3) for j in range(i))):
            raise RuntimeError(
                "Duplicate leakage between development, calibration and holdout."
            )
        role = {int(i): "development" for i in dev}
        role.update({int(i): "calibration_cluster_excluded" for i in cal_cluster})
        role.update({int(i): "calibration_reference" for i in cal})
        role.update({int(i): "development_holdout" for i in held})
        inner_by_idx = dict(zip(dev.tolist(), inner_ids.tolist()))
        for i, record in m.iterrows():
            rows.append(
                {
                    "outer_fold": outer,
                    "image_id": record.image_id,
                    "duplicate_group": record.duplicate_group,
                    "group": record.group,
                    "role": role[i],
                    "inner_validation_fold": inner_by_idx.get(i, -1),
                }
            )
        for name, ix in [
            ("fit_reference", final_fit),
            ("calibration_reference", cal),
            ("development_holdout", held),
        ]:
            supports.append(
                {
                    "outer_fold": outer,
                    "role": name,
                    "n": len(ix),
                    **{f"n_{g}": int((m.iloc[ix].group == g).sum()) for g in GROUPS},
                }
            )
        plans.append((held, final_fit, cal, inner_plans))
    return (plans, pd.DataFrame(rows), pd.DataFrame(supports))
