from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.services.errors import BackendError
from app.services.storage_service import load_lab_history, load_prediction_history, save_evaluation_record


def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        raise BackendError("UPDATE_SKIPPED_INSUFFICIENT_LABELS", "no matched prediction/lab samples")
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if len(y_true) >= 2 and np.sum((y_true - np.mean(y_true)) ** 2) > 0:
        r2 = float(1 - np.sum(err**2) / np.sum((y_true - np.mean(y_true)) ** 2))
    else:
        r2 = float("nan")
    return {"samples": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2}


def matched_active_predictions(target: str, start_time=None, end_time=None):
    predictions = load_prediction_history(target)
    labs = load_lab_history(target)
    if predictions.empty or labs.empty:
        return pd.DataFrame(columns=["prediction_time", "prediction", "sample_time", "value"])

    if start_time is not None:
        predictions = predictions[predictions["prediction_time"] >= pd.to_datetime(start_time)]
        labs = labs[labs["sample_time"] >= pd.to_datetime(start_time)]
    if end_time is not None:
        predictions = predictions[predictions["prediction_time"] < pd.to_datetime(end_time)]
        labs = labs[labs["sample_time"] < pd.to_datetime(end_time)]

    merged = pd.merge(
        predictions,
        labs,
        left_on="prediction_time",
        right_on="sample_time",
        how="inner",
    )
    return merged.sort_values("prediction_time")


def evaluate_active(target: str, cycle: str, start_time=None, end_time=None):
    matched = matched_active_predictions(target, start_time, end_time)
    if matched.empty:
        raise BackendError("UPDATE_SKIPPED_INSUFFICIENT_LABELS", f"{target}: no matched samples")
    metrics = metrics_dict(matched["value"].to_numpy(), matched["prediction"].to_numpy())
    record = {
        "target": target,
        "cycle": cycle,
        "model_version": "active",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "start_time": str(start_time) if start_time is not None else "",
        "end_time": str(end_time) if end_time is not None else "",
        **metrics,
    }
    save_evaluation_record(target, record)
    return record
