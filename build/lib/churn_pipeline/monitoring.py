from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class DriftReport:
    feature_metrics: pd.DataFrame
    aggregate_flags: Dict[str, bool]


def population_stability_index(
    expected: pd.Series, actual: pd.Series, buckets: int = 10, epsilon: float = 1e-6
) -> float:
    """Compute the population stability index between two numeric distributions."""

    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, buckets + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) < 2:
        return 0.0

    expected_bins = np.clip(np.digitize(expected, breaks, right=False) - 1, 0, len(breaks) - 2)
    actual_bins = np.clip(np.digitize(actual, breaks, right=False) - 1, 0, len(breaks) - 2)

    psi = 0.0
    for b in range(len(breaks) - 1):
        exp_ratio = max((expected_bins == b).sum() / len(expected), epsilon)
        act_ratio = max((actual_bins == b).sum() / len(actual), epsilon)
        psi += (act_ratio - exp_ratio) * np.log(act_ratio / exp_ratio)
    return float(psi)


def kolmogorov_smirnov_statistic(expected: pd.Series, actual: pd.Series) -> float:
    """Compute the Kolmogorov-Smirnov statistic without SciPy."""

    expected = np.sort(expected.dropna())
    actual = np.sort(actual.dropna())
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    all_values = np.concatenate([expected, actual])
    all_values = np.sort(np.unique(all_values))

    exp_idx = np.searchsorted(expected, all_values, side="right")
    act_idx = np.searchsorted(actual, all_values, side="right")
    exp_cdf = exp_idx / expected.size
    act_cdf = act_idx / actual.size
    return float(np.max(np.abs(exp_cdf - act_cdf)))


def compute_data_drift_report(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.1,
) -> DriftReport:
    """Generate a simple drift report comparing baseline vs. current data."""

    if feature_columns is None:
        feature_columns = [
            col for col in baseline.columns if pd.api.types.is_numeric_dtype(baseline[col])
        ]

    records = []
    flags = {"psi": False, "ks": False}

    for col in feature_columns:
        if col not in current.columns:
            continue
        if not pd.api.types.is_numeric_dtype(baseline[col]):
            continue

        psi = population_stability_index(baseline[col], current[col])
        ks = kolmogorov_smirnov_statistic(baseline[col], current[col])
        records.append({"feature": col, "psi": psi, "ks": ks})

        if not np.isnan(psi) and psi > psi_threshold:
            flags["psi"] = True
        if not np.isnan(ks) and ks > ks_threshold:
            flags["ks"] = True

    metrics_df = pd.DataFrame(records).sort_values("psi", ascending=False)
    return DriftReport(feature_metrics=metrics_df, aggregate_flags=flags)


def performance_drift(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    degrade_threshold: float = 0.1,
) -> Dict[str, float]:
    """Compute relative metric deltas to flag concept drift via performance drops."""

    deltas: Dict[str, float] = {}
    for metric, baseline_value in baseline_metrics.items():
        if metric not in current_metrics:
            continue
        current_value = current_metrics[metric]
        if baseline_value == 0:
            deltas[metric] = float("nan")
            continue
        deltas[metric] = (current_value - baseline_value) / abs(baseline_value)

    deltas["needs_retrain"] = any(
        (value < -abs(degrade_threshold)) for key, value in deltas.items() if key != "needs_retrain"
    )
    return deltas
