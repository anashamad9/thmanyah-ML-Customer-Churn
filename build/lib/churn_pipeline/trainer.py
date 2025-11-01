from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig
from .data_loader import clean_event_log, load_event_log
from .evaluation import compute_classification_metrics
from .features import build_feature_matrix

try:
    import mlflow
    from mlflow import sklearn as mlflow_sklearn
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    mlflow = None
    mlflow_sklearn = None


def _temporal_split(feature_df: pd.DataFrame, config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    holdout_days = config.split_config.temporal_holdout_days
    cutoff = feature_df["label_ts"].max() - pd.Timedelta(days=holdout_days)
    train_df = feature_df[feature_df["label_ts"] < cutoff]
    test_df = feature_df[feature_df["label_ts"] >= cutoff]

    if len(train_df) < config.split_config.min_train_users or len(test_df) == 0:
        train_df, test_df = _stratified_split(feature_df, config.split_config.test_ratio, config.random_state)
    return train_df, test_df


def _stratified_split(feature_df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        feature_df.index,
        test_size=test_ratio,
        random_state=seed,
        stratify=feature_df["churned"],
    )
    return feature_df.loc[train_idx], feature_df.loc[test_idx]


def _prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, ...]:
    feature_cols = [col for col in train_df.columns if col not in {target_column, "label_ts"}]
    X_train = train_df[feature_cols]
    y_train = train_df[target_column].values
    X_test = test_df[feature_cols]
    y_test = test_df[target_column].values

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    remainder_cols = sorted(set(feature_cols) - set(numeric_cols))

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if remainder_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                remainder_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = GradientBoostingClassifier(random_state=0)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return clf, X_train, y_train, X_test, y_test


def train(config: TrainingConfig) -> Dict[str, float]:
    """Run the end-to-end training pipeline."""

    events = load_event_log(config.data_path)
    events = clean_event_log(events)

    feature_df = build_feature_matrix(events, config.feature_config)
    if feature_df.empty:
        raise ValueError("No user features generated. Check feature configuration thresholds.")

    if config.split_config.method == "temporal":
        train_df, test_df = _temporal_split(feature_df, config)
    else:
        train_df, test_df = _stratified_split(
            feature_df, config.split_config.test_ratio, config.random_state
        )

    pipeline, X_train, y_train, X_test, y_test = _prepare_features(train_df, test_df, config.target_column)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    eval_result = compute_classification_metrics(y_test, y_pred, y_score)

    # Persist artifacts
    config.ensure_dirs()
    model_path = config.artifacts_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    _export_dataframe(feature_df, config.artifacts_dir / "features")
    _export_dataframe(test_df, config.artifacts_dir / "evaluation_set")
    (config.artifacts_dir / "metrics.json").write_text(json.dumps(eval_result.metrics, indent=2), encoding="utf-8")

    if mlflow:
        mlflow.set_tracking_uri(str(config.tracking_uri))
        mlflow.set_experiment(config.experiment_name)
        with mlflow.start_run(run_name=f"{config.model_name}"):
            mlflow.log_params(
                {
                    "model_name": config.model_name,
                    "random_state": config.random_state,
                    "lookback_days": config.feature_config.lookback_days,
                }
            )
            mlflow.log_metrics(eval_result.metrics)
            if mlflow_sklearn:
                mlflow_sklearn.log_model(pipeline, artifact_path="model")

    return eval_result.metrics


def _export_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist dataframe to parquet when possible, otherwise fallback to CSV."""

    parquet_path = path.with_suffix(".parquet")
    csv_path = path.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path)
    except Exception:  # pragma: no cover - optional dependency
        df.to_csv(csv_path)
