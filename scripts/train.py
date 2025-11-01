from __future__ import annotations

import argparse
import json
from pathlib import Path

from churn_pipeline import load_training_config
from churn_pipeline.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train customer churn model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the training configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    metrics = train(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
