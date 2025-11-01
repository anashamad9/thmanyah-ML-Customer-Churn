from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from churn_pipeline.monitoring import compute_data_drift_report, performance_drift
from churn_pipeline.utils import load_structured_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data and performance drift checks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/monitoring.yaml"),
        help="Monitoring configuration file.",
    )
    return parser.parse_args()


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for {path}")


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cfg = load_structured_file(args.config)

    baseline_features = _load_table(Path(cfg["baseline_features_path"]))
    current_features = _load_table(Path(cfg["current_features_path"]))

    drift_report = compute_data_drift_report(
        baseline_features,
        current_features,
        psi_threshold=float(cfg.get("psi_threshold", 0.2)),
        ks_threshold=float(cfg.get("ks_threshold", 0.1)),
    )

    print("=== Data Drift Report ===")
    print(drift_report.feature_metrics.to_string(index=False))
    print("Flags:", drift_report.aggregate_flags)

    if "baseline_metrics_path" in cfg and "current_metrics_path" in cfg:
        baseline_metrics = _load_metrics(Path(cfg["baseline_metrics_path"]))
        current_metrics = _load_metrics(Path(cfg["current_metrics_path"]))
        perf_deltas = performance_drift(
            baseline_metrics,
            current_metrics,
            degrade_threshold=float(cfg.get("performance_drop_threshold", 0.1)),
        )
        print("=== Performance Drift ===")
        print(json.dumps(perf_deltas, indent=2))


if __name__ == "__main__":
    main()
