from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .utils import load_structured_file


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    lookback_days: int = 30
    min_events_per_user: int = 10
    min_sessions_per_user: int = 1
    include_gender: bool = True
    include_level: bool = True
    include_location: bool = False


@dataclass
class SplitConfig:
    """Configuration controlling how training/validation splits are generated."""

    method: str = "temporal"  # accepted: temporal, stratified
    test_ratio: float = 0.2
    min_train_users: int = 50
    temporal_holdout_days: int = 14


@dataclass
class TrainingConfig:
    """Configuration for model training end-to-end."""

    data_path: Path
    artifacts_dir: Path = Path("artifacts")
    model_registry: Path = Path("models")
    tracking_uri: Optional[str] = "mlruns"
    experiment_name: str = "customer-churn"
    random_state: int = 42
    model_name: str = "gradient_boosting"
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    split_config: SplitConfig = field(default_factory=SplitConfig)
    target_column: str = "churned"

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry.mkdir(parents=True, exist_ok=True)


def load_training_config(path: Path | str) -> TrainingConfig:
    """Load the YAML training configuration file."""

    payload = load_structured_file(Path(path))

    data_path = payload.get("data_path", "customer_churn_mini.json")
    feature_cfg = FeatureConfig(**payload.get("feature_config", {}))
    split_cfg = SplitConfig(**payload.get("split_config", {}))
    cfg = TrainingConfig(
        data_path=Path(data_path),
        artifacts_dir=Path(payload.get("artifacts_dir", "artifacts")),
        model_registry=Path(payload.get("model_registry", "models")),
        tracking_uri=payload.get("tracking_uri", "mlruns"),
        experiment_name=payload.get("experiment_name", "customer-churn"),
        random_state=payload.get("random_state", 42),
        model_name=payload.get("model_name", "gradient_boosting"),
        feature_config=feature_cfg,
        split_config=split_cfg,
        target_column=payload.get("target_column", "churned"),
    )
    cfg.ensure_dirs()
    return cfg
