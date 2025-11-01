from __future__ import annotations

import os
from functools import lru_cache
import random
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from churn_pipeline.config import load_training_config
from churn_pipeline.data_loader import clean_event_log
from churn_pipeline.features import build_feature_matrix
from .sample_data import SAMPLE_EVENT_PAYLOADS

try:
    import joblib
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("joblib is required to run the API. Install project dependencies first.") from exc


class Event(BaseModel):
    ts: int
    userId: str
    sessionId: int
    page: str
    auth: str | None = None
    method: str | None = None
    status: int | None = None
    level: str | None = None
    itemInSession: int | None = None
    location: str | None = None
    userAgent: str | None = None
    lastName: str | None = None
    firstName: str | None = None
    registration: int | None = None
    gender: str | None = None
    artist: str | None = None
    song: str | None = None
    length: float | None = Field(default=None, description="Song length in seconds")


class PredictionRequest(BaseModel):
    events: List[Event]


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_label: int
    user_id: str


app = FastAPI(title="Customer Churn Predictor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelService:
    def __init__(self) -> None:
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        config_path = Path(os.getenv("CHURN_CONFIG", "configs/training.yaml"))
        self.config = load_training_config(config_path)
        self.artifacts_dir = artifacts_dir
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        model_path = self.artifacts_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run training before starting the API."
            )
        return joblib.load(model_path)

    def predict(self, events: List[Event]) -> PredictionResponse:
        event_dicts = [event.model_dump() for event in events]
        user_ids = {event["userId"] for event in event_dicts}
        if len(user_ids) != 1:
            raise HTTPException(status_code=400, detail="Events must belong to a single userId.")

        df = pd.DataFrame(event_dicts)
        df = clean_event_log(df)
        features = build_feature_matrix(df, self.config.feature_config)
        if features.empty:
            raise HTTPException(status_code=422, detail="Insufficient data to build user features.")

        # Drop supervised columns before inference
        inference_df = features.drop(columns=[self.config.target_column, "label_ts"], errors="ignore")

        proba = self.pipeline.predict_proba(inference_df)[0][1]
        label = int(proba >= 0.5)

        return PredictionResponse(
            churn_probability=float(proba),
            churn_label=label,
            user_id=next(iter(user_ids)),
        )


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    return ModelService()


@app.on_event("startup")
def _startup() -> None:
    get_service()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    service = get_service()
    return service.predict(request.events)


class SampleResponse(BaseModel):
    user_id: str
    events: List[Event]


@app.get("/sample", response_model=SampleResponse)
def sample_user_events() -> SampleResponse:
    """Return a pseudo-random event payload for demo purposes."""

    sample = random.choice(SAMPLE_EVENT_PAYLOADS)
    events = [Event.model_validate(event) for event in sample["events"]]
    return SampleResponse(user_id=sample["userId"], events=events)
