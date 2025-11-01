from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def load_event_log(path: Path | str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load a JSON lines event log into a pandas DataFrame."""

    df = pd.read_json(Path(path), lines=True)
    if columns is not None:
        df = df[list(columns)]
    return df


def clean_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning of the raw event log."""

    cleaned = df.copy()
    cleaned = cleaned[cleaned["userId"].notnull()]
    cleaned = cleaned[cleaned["userId"] != ""]
    cleaned = cleaned[cleaned["userId"] != "None"]
    cleaned["ts"] = pd.to_datetime(cleaned["ts"], unit="ms")
    cleaned["registration"] = pd.to_datetime(cleaned["registration"], unit="ms", errors="coerce")
    cleaned["sessionId"] = (
        pd.to_numeric(cleaned["sessionId"], errors="coerce").fillna(-1).astype(int)
    )
    cleaned["itemInSession"] = (
        pd.to_numeric(cleaned["itemInSession"], errors="coerce").fillna(0).astype(int)
    )
    cleaned["userId"] = cleaned["userId"].astype(str)
    if "length" in cleaned.columns:
        cleaned["length"] = pd.to_numeric(cleaned["length"], errors="coerce").fillna(0.0)
    return cleaned.reset_index(drop=True)
