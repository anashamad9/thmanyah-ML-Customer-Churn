from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn_pipeline.config import FeatureConfig
from churn_pipeline.data_loader import clean_event_log, load_event_log
from churn_pipeline.features import CHURN_EVENT, build_feature_matrix

plt.switch_backend("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory analysis for churn dataset.")
    parser.add_argument("--data", type=Path, default=Path("customer_churn_mini.json"), help="Path to the raw event log.")
    parser.add_argument("--output", type=Path, default=Path("docs/charts"), help="Directory where plots will be stored.")
    parser.add_argument("--limit-users", type=int, default=None, help="Optional cap on number of users to analyse (for speed).")
    parser.add_argument("--feature-lookback", type=int, default=30, help="Days lookback for features.")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def assign_churn_labels(events: pd.DataFrame) -> pd.DataFrame:
    churn_users = set(events.loc[events["page"] == CHURN_EVENT, "userId"])
    events = events.copy()
    events["churned"] = events["userId"].isin(churn_users).astype(int)
    return events


def _top_categories(series: pd.Series, top_n: int = 10) -> pd.Series:
    counts = series.value_counts().head(top_n)
    counts.index = counts.index.astype(str)
    return counts


def plot_churn_rate_by_level(features: pd.DataFrame, output: Path) -> None:
    if "current_level_paid" not in features.columns:
        return
    df = features.copy()
    df["subscription_level"] = np.where(df["current_level_paid"] == 1, "paid", "free")
    churn_rates = df.groupby("subscription_level")["churned"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    churn_rates.plot(kind="bar", color=["#ff7b7b", "#4c72b0"], ax=ax)
    ax.set_ylabel("Churn Rate")
    ax.set_title("Churn Rate by Subscription Level")
    ax.set_ylim(0, 1)
    for patch in ax.patches:
        ax.annotate(f"{patch.get_height():.2f}", (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "churn_rate_by_level.png", dpi=160)
    plt.close(fig)


def plot_event_engagement(events: pd.DataFrame, output: Path) -> None:
    engagement = (
        events.assign(date=events["ts"].dt.date)
        .groupby(["date", "churned"])
        .size()
        .unstack(fill_value=0)
    )
    if engagement.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    engagement.plot(ax=ax)
    ax.set_ylabel("Events per day")
    ax.set_title("Daily Engagement by Churn Outcome")
    ax.legend(["Retained", "Churned"])
    fig.tight_layout()
    fig.savefig(output / "daily_engagement.png", dpi=160)
    plt.close(fig)


def plot_top_pages(events: pd.DataFrame, output: Path) -> None:
    top_pages_churn = _top_categories(events.loc[events["churned"] == 1, "page"])
    top_pages_stay = _top_categories(events.loc[events["churned"] == 0, "page"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    top_pages_churn.plot(kind="barh", ax=axes[0], color="#c44e52")
    axes[0].set_title("Top Pages - Churned Users")
    axes[0].invert_xaxis()
    top_pages_stay.plot(kind="barh", ax=axes[1], color="#4c72b0")
    axes[1].set_title("Top Pages - Retained Users")
    fig.tight_layout()
    fig.savefig(output / "top_pages.png", dpi=160)
    plt.close(fig)


def plot_feature_differences(features: pd.DataFrame, output: Path, top_n: int = 10) -> None:
    numeric_cols: Iterable[str] = [
        col for col in features.select_dtypes(include=[np.number]).columns if col not in {"churned"}
    ]
    if not numeric_cols:
        return

    group_means = features.groupby("churned")[numeric_cols].mean().T
    group_means.columns = ["retained", "churned"]
    group_means["delta"] = group_means["churned"] - group_means["retained"]
    top_diffs = group_means.reindex(group_means["delta"].abs().sort_values(ascending=False).index).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_diffs["delta"].plot(kind="bar", color=np.where(top_diffs["delta"] >= 0, "#c44e52", "#4c72b0"), ax=ax)
    ax.set_ylabel("Mean Difference (Churned - Retained)")
    ax.set_title("Top Feature Differences")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output / "feature_differences.png", dpi=160)
    plt.close(fig)

    # Save table alongside chart for documentation
    top_diffs.round(3).to_csv(output / "feature_differences.csv")


def generate_textual_summary(features: pd.DataFrame, output: Path) -> None:
    summary_path = output / "summary.json"

    churn_rate = features["churned"].mean()
    top_features = pd.read_csv(output / "feature_differences.csv") if (output / "feature_differences.csv").exists() else None

    summary_payload = {
        "n_users": int(features.shape[0]),
        "churn_rate": float(churn_rate),
        "mean_num_events_churned": float(features.loc[features["churned"] == 1, "num_events"].mean()),
        "mean_num_events_retained": float(features.loc[features["churned"] == 0, "num_events"].mean()),
    }

    if top_features is not None:
        summary_payload["top_feature_differences"] = top_features.head(5).to_dict(orient="records")

    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output)

    events = load_event_log(args.data)
    events = clean_event_log(events)
    events = assign_churn_labels(events)

    if args.limit_users:
        allowed_users = set(events["userId"].unique()[: args.limit_users])
        events = events[events["userId"].isin(allowed_users)]

    feature_cfg = FeatureConfig(lookback_days=args.feature_lookback)
    features = build_feature_matrix(events, feature_cfg)

    if features.empty:
        raise ValueError("No features generated. Lower filtering thresholds or provide more data.")

    plot_churn_rate_by_level(features, args.output)
    plot_event_engagement(events, args.output)
    plot_top_pages(events, args.output)
    plot_feature_differences(features, args.output)
    generate_textual_summary(features, args.output)

    print(f"Analysis complete. Charts stored in {args.output}")


if __name__ == "__main__":
    main()
