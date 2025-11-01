from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

import pandas as pd

from churn_pipeline.config import FeatureConfig
from churn_pipeline.features import build_feature_matrix


def _load_subset(path: Path, limit: int = 5000) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in islice(handle, limit):
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def test_feature_builder_creates_labels(tmp_path: Path) -> None:
    data_path = Path("customer_churn_mini.json")
    if not data_path.exists():
        raise RuntimeError("Expected dataset customer_churn_mini.json not found.")

    df = _load_subset(data_path)
    df = df[df["userId"].notnull() & (df["userId"] != "") & (df["userId"] != "None")]

    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["registration"] = pd.to_datetime(df["registration"], unit="ms", errors="coerce")
    feature_cfg = FeatureConfig(min_events_per_user=5, min_sessions_per_user=1)

    features = build_feature_matrix(df, feature_cfg)

    assert not features.empty
    assert "churned" in features.columns
    assert features["churned"].isin([0, 1]).all()
