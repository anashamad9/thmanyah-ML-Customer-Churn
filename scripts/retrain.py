from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from churn_pipeline import load_training_config
from churn_pipeline.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger model retraining on new data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Optional path to a new dataset. Overrides the data_path in the config.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Copy the trained model into the model registry directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    if args.data:
        config.data_path = args.data
    metrics = train(config)
    print(metrics)

    if args.register:
        source = config.artifacts_dir / "model.joblib"
        target = config.model_registry / "latest_model.joblib"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"Registered model at {target}")


if __name__ == "__main__":
    main()
