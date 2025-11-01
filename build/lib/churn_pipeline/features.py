from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig

CHURN_EVENT = "Cancellation Confirmation"


def _session_stats(events: pd.DataFrame) -> Tuple[float, float, float]:
    """Return statistics for session duration in minutes."""

    if events.empty:
        return 0.0, 0.0, 0.0
    grouped = events.groupby("sessionId")["ts"]
    durations = (grouped.max() - grouped.min()).dt.total_seconds() / 60.0
    if durations.empty:
        return 0.0, 0.0, 0.0
    mean_val = float(durations.mean())
    median_val = float(durations.median())
    std_val = float(durations.std(ddof=0)) if len(durations) > 1 else 0.0
    if np.isnan(mean_val):
        mean_val = 0.0
    if np.isnan(median_val):
        median_val = 0.0
    if np.isnan(std_val):
        std_val = 0.0
    return mean_val, median_val, std_val


def _event_rate(events: pd.DataFrame) -> float:
    if events.empty:
        return 0.0
    timespan = (events["ts"].max() - events["ts"].min()).total_seconds() / 3600.0
    if timespan <= 0:
        return float(len(events))
    return float(len(events) / timespan)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def build_user_features(events: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    """Aggregate event level data into user level features."""

    feature_rows: List[dict] = []

    for user_id, user_events in events.groupby("userId"):
        user_events = user_events.sort_values("ts")
        churn_mask = user_events["page"] == CHURN_EVENT
        churned = churn_mask.any()

        cutoff_ts = user_events["ts"].max()
        if churned:
            cutoff_ts = user_events.loc[churn_mask, "ts"].iloc[0]
            user_events = user_events[user_events["ts"] < cutoff_ts]

        if feature_config.lookback_days:
            window_start = cutoff_ts - pd.Timedelta(days=feature_config.lookback_days)
            user_events = user_events[user_events["ts"] >= window_start]

        if len(user_events) < feature_config.min_events_per_user:
            continue

        sessions = user_events["sessionId"].nunique()
        if sessions < feature_config.min_sessions_per_user:
            continue

        session_item_counts = user_events.groupby("sessionId")["itemInSession"].max()
        session_item_std = float(session_item_counts.std(ddof=0)) if len(session_item_counts) > 1 else 0.0
        snapshot = {
            "userId": user_id,
            "label_ts": cutoff_ts,
            "churned": int(churned),
            "num_events": len(user_events),
            "num_sessions": sessions,
            "num_songs": int((user_events["page"] == "NextSong").sum()),
            "num_errors": int((user_events["page"] == "Error").sum()),
            "num_add_friend": int((user_events["page"] == "Add Friend").sum()),
            "num_add_playlist": int((user_events["page"] == "Add to Playlist").sum()),
            "num_roll_advert": int((user_events["page"] == "Roll Advert").sum()),
            "num_thumb_up": int((user_events["page"] == "Thumbs Up").sum()),
            "num_thumb_down": int((user_events["page"] == "Thumbs Down").sum()),
            "avg_items_per_session": float(session_item_counts.mean()),
            "std_items_per_session": session_item_std,
            "total_listening_minutes": float(user_events["length"].fillna(0.0).sum() / 60.0),
            "event_rate_per_hour": _event_rate(user_events),
            "distinct_artists": int(user_events["artist"].nunique()),
            "distinct_songs": int(user_events["song"].nunique()),
        }

        mean_session, median_session, std_session = _session_stats(user_events)
        snapshot["avg_session_minutes"] = mean_session
        snapshot["median_session_minutes"] = median_session
        snapshot["std_session_minutes"] = std_session

        paid_events = (user_events["level"] == "paid").sum()
        snapshot["paid_event_ratio"] = _safe_ratio(paid_events, len(user_events))

        if feature_config.include_gender and "gender" in user_events.columns:
            # One-hot encode gender values M/F/Other
            genders = user_events["gender"].dropna().unique()
            last_gender = genders[-1] if len(genders) else None
            snapshot["gender_M"] = 1 if last_gender == "M" else 0
            snapshot["gender_F"] = 1 if last_gender == "F" else 0

        if feature_config.include_level:
            levels = user_events["level"].dropna()
            last_level = levels.iloc[-1] if len(levels) else None
            snapshot["current_level_paid"] = 1 if last_level == "paid" else 0

        if feature_config.include_location and "location" in user_events.columns:
            snapshot["num_locations"] = int(user_events["location"].nunique())

        if "registration" in user_events.columns:
            reg = user_events["registration"].dropna()
            if not reg.empty:
                reg_time = reg.iloc[0]
                snapshot["account_age_days"] = float((cutoff_ts - reg_time).days)

        timespan = user_events["ts"].max() - user_events["ts"].min()
        snapshot["active_days"] = float(timespan.days if pd.notnull(timespan) else 0)

        feature_rows.append(snapshot)

    feature_df = pd.DataFrame(feature_rows)
    if not feature_df.empty:
        feature_df = feature_df.set_index("userId")
        feature_df = feature_df.sort_values("label_ts")

    return feature_df


def build_feature_matrix(events: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    """Convenience wrapper returning features without configuration mutation."""

    return build_user_features(events, feature_config)
