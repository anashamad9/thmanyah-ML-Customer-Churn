from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn import metrics


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> EvaluationResult:
    """Compute a standard classification metric suite."""

    metric_payload: Dict[str, float] = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        try:
            metric_payload["roc_auc"] = float(metrics.roc_auc_score(y_true, y_score))
        except ValueError:
            metric_payload["roc_auc"] = float("nan")
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        metric_payload["pr_auc"] = float(metrics.auc(recall, precision))

    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    return EvaluationResult(metrics=metric_payload, confusion_matrix=conf_mat)
