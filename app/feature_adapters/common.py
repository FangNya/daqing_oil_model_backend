from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import interpolate

from app.config import BASE_DIR
from app.services.errors import BackendError


def backend_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model_artifacts(model_dir: str | Path) -> dict:
    model_dir = backend_path(model_dir)
    return {
        "model_dir": model_dir,
        "config": load_json(model_dir / "model_config.json"),
        "feature_columns": joblib.load(model_dir / "feature_columns.pkl"),
        "engineered_feature_names": joblib.load(model_dir / "engineered_feature_names.pkl"),
        "feature_scaler": joblib.load(model_dir / "feature_scaler.pkl"),
    }


def validate_operation_data(op_data: pd.DataFrame, columns: list[str]):
    if op_data.empty:
        raise BackendError("MISSING_OPERATION_DATA", "operation_records is empty")
    if "TimeStamp" not in op_data.columns:
        raise BackendError("MISSING_TIMESTAMP", "operation_records require TimeStamp")
    missing = [col for col in columns if col not in op_data.columns]
    if missing:
        raise BackendError(
            "MISSING_FEATURE_COLUMNS",
            f"operation_records missing columns: {', '.join(missing)}",
        )


def normalize_operation_data(op_data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    validate_operation_data(op_data, columns)
    df = op_data.copy()
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    if df["TimeStamp"].isna().any():
        raise BackendError("MISSING_TIMESTAMP", "operation_records contain invalid TimeStamp")
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").drop_duplicates("TimeStamp", keep="last")
    df[columns] = df[columns].ffill().bfill()
    return df.reset_index(drop=True)


def normalize_lab_history(lab_history: pd.DataFrame) -> pd.DataFrame:
    if lab_history is None or lab_history.empty:
        return pd.DataFrame(columns=["sample_time", "dry_point"])
    df = lab_history.rename(columns={"value": "dry_point"}).copy()
    df["sample_time"] = pd.to_datetime(df["sample_time"], errors="coerce")
    df["dry_point"] = pd.to_numeric(df["dry_point"], errors="coerce")
    return df.dropna(subset=["sample_time", "dry_point"]).sort_values("sample_time")


def lowpass_filter_1st_order(series: np.ndarray, dt_minutes: float, tau_minutes: float) -> np.ndarray:
    if len(series) == 0:
        return series.astype(float)
    if not np.isfinite(dt_minutes) or dt_minutes <= 0:
        dt_minutes = 2.0
    alpha = dt_minutes / (tau_minutes + dt_minutes)
    y = np.zeros_like(series, dtype=float)
    y[0] = series[0]
    for idx in range(1, len(series)):
        y[idx] = alpha * series[idx] + (1 - alpha) * y[idx - 1]
    return y


def add_last_prediction(op_data: pd.DataFrame, lab_history: pd.DataFrame) -> pd.DataFrame:
    history = normalize_lab_history(lab_history)
    prepared = op_data.sort_values("TimeStamp").copy()
    if history.empty:
        prepared["last_prediction"] = np.nan
        return prepared
    merged = pd.merge_asof(
        prepared,
        history[["sample_time", "dry_point"]],
        left_on="TimeStamp",
        right_on="sample_time",
        direction="backward",
        allow_exact_matches=False,
    )
    merged["last_prediction"] = merged["dry_point"].ffill()
    return merged.drop(columns=["sample_time", "dry_point"])


def interpolate_or_tail(window: pd.DataFrame, feature_columns: list[str], max_points: int) -> np.ndarray:
    if len(window) >= max_points:
        return window[feature_columns].tail(max_points).to_numpy(dtype=float)
    x_orig = np.linspace(0, 1, len(window))
    x_new = np.linspace(0, 1, max_points)
    seq = []
    for col in feature_columns:
        values = window[col].to_numpy(dtype=float)
        if len(values) < 2:
            seq.append(np.full(max_points, values[0] if len(values) else 0.0))
        else:
            func = interpolate.interp1d(x_orig, values, kind="linear", fill_value="extrapolate")
            seq.append(func(x_new))
    return np.array(seq).T


def build_window_sequence(
    prepared_op: pd.DataFrame,
    prediction_time,
    feature_columns: list[str],
    delay_hours: int,
    window_hours: int,
    max_points: int,
    min_points: int = 20,
) -> np.ndarray:
    prediction_time = pd.to_datetime(prediction_time, errors="coerce")
    if pd.isna(prediction_time):
        raise BackendError("MISSING_TIMESTAMP", "invalid prediction_time")
    end_time = prediction_time - timedelta(hours=delay_hours)
    start_time = end_time - timedelta(hours=window_hours)
    window = prepared_op[
        (prepared_op["TimeStamp"] >= start_time) & (prepared_op["TimeStamp"] <= end_time)
    ][feature_columns]
    if len(window) < min_points:
        raise BackendError(
            "INSUFFICIENT_WINDOW_POINTS",
            f"operation window has {len(window)} points, require at least {min_points}",
        )
    if window.isnull().any().any():
        raise BackendError("FEATURE_BUILD_FAILED", "operation window contains null feature values")
    return interpolate_or_tail(window, feature_columns, max_points)


def extract_enhanced_stats_fft(X: np.ndarray, feature_columns: list[str], n_fft: int = 3) -> tuple[np.ndarray, list[str]]:
    stats = {
        "mean": np.mean(X, axis=1),
        "std": np.std(X, axis=1),
        "min": np.min(X, axis=1),
        "max": np.max(X, axis=1),
        "range": np.max(X, axis=1) - np.min(X, axis=1),
        "median": np.median(X, axis=1),
        "q25": np.percentile(X, 25, axis=1),
        "q75": np.percentile(X, 75, axis=1),
        "first": X[:, 0, :],
        "last": X[:, -1, :],
        "slope": X[:, -1, :] - X[:, 0, :],
    }
    parts = []
    names = []
    for stat_name, values in stats.items():
        parts.append(values)
        names.extend([f"{col}__{stat_name}" for col in feature_columns])

    fft_rows = []
    for sample in X:
        row = []
        for feature_idx in range(sample.shape[1]):
            ts = sample[:, feature_idx]
            if np.std(ts) == 0:
                fft_mag = np.zeros(n_fft)
            else:
                fft_mag = np.abs(np.fft.fft(ts - np.mean(ts)))[:n_fft]
            row.extend(fft_mag)
        fft_rows.append(row)
    parts.append(np.array(fft_rows))
    for col in feature_columns:
        for idx in range(n_fft):
            names.append(f"{col}__fft{idx}")
    return np.concatenate(parts, axis=1), names
