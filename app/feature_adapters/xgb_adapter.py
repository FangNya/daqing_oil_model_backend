from __future__ import annotations

import numpy as np
import pandas as pd

from app.feature_adapters.common import (
    add_last_prediction,
    build_window_sequence,
    extract_enhanced_stats_fft,
    load_model_artifacts,
    lowpass_filter_1st_order,
    normalize_operation_data,
)
from app.services.errors import BackendError


FILTER_WARMUP_HOURS = 7 * 24


class XGBFeatureAdapter:
    def __init__(self, target: str, model_dir: str, tau_minutes: float):
        self.target = target
        self.artifacts = load_model_artifacts(model_dir)
        self.model_dir = self.artifacts["model_dir"]
        self.config = self.artifacts["config"]
        self.raw_feature_columns = list(self.artifacts["feature_columns"])
        self.op_feature_columns = [col for col in self.raw_feature_columns if col != "last_prediction"]
        self.engineered_feature_names = list(self.artifacts["engineered_feature_names"])
        self.feature_scaler = self.artifacts["feature_scaler"]
        self.tau_minutes = tau_minutes
        # Preserve the recursive low-pass state without loading the full operation history.
        self.history_warmup_hours = FILTER_WARMUP_HOURS
        best = self.config.get("best_config", {})
        self.delay_hours = int(best.get("delay_hours", 1))
        self.window_hours = int(best.get("window_hours", 1))
        self.max_points = int(best.get("max_points", 30))

    def prepare_operation_data(self, op_data: pd.DataFrame, lab_history: pd.DataFrame) -> pd.DataFrame:
        prepared = normalize_operation_data(op_data, self.op_feature_columns)
        dt_minutes = prepared["TimeStamp"].diff().dt.total_seconds().median() / 60
        for col in self.op_feature_columns:
            prepared[col] = lowpass_filter_1st_order(
                prepared[col].to_numpy(dtype=float),
                dt_minutes=dt_minutes,
                tau_minutes=self.tau_minutes,
            )
        prepared = add_last_prediction(prepared, lab_history)
        return prepared

    def build_predict_input(self, op_data: pd.DataFrame, lab_history: pd.DataFrame, prediction_time) -> np.ndarray:
        prepared = self.prepare_operation_data(op_data, lab_history)
        seq = build_window_sequence(
            prepared,
            prediction_time,
            self.raw_feature_columns,
            self.delay_hours,
            self.window_hours,
            self.max_points,
            min_points=20,
        )
        engineered, names = extract_enhanced_stats_fft(seq.reshape(1, *seq.shape), self.raw_feature_columns, n_fft=3)
        if names != self.engineered_feature_names:
            raise BackendError("FEATURE_BUILD_FAILED", "engineered feature names do not match active model")
        return self.feature_scaler.transform(engineered)

    def build_training_matrix(self, op_data: pd.DataFrame, lab_history: pd.DataFrame, labels: pd.DataFrame):
        labels = labels.copy()
        labels["sample_time"] = pd.to_datetime(labels["sample_time"], errors="coerce")
        labels["value"] = pd.to_numeric(labels["value"], errors="coerce")
        labels = labels.dropna(subset=["sample_time", "value"]).sort_values("sample_time")
        prepared = self.prepare_operation_data(op_data, lab_history)
        rows = []
        y_values = []
        for _, row in labels.iterrows():
            try:
                seq = build_window_sequence(
                    prepared,
                    row["sample_time"],
                    self.raw_feature_columns,
                    self.delay_hours,
                    self.window_hours,
                    self.max_points,
                    min_points=20,
                )
            except BackendError:
                continue
            rows.append(seq)
            y_values.append(float(row["value"]))
        if not rows:
            return np.empty((0, len(self.engineered_feature_names))), np.array([])
        engineered, names = extract_enhanced_stats_fft(np.array(rows), self.raw_feature_columns, n_fft=3)
        if names != self.engineered_feature_names:
            raise BackendError("FEATURE_BUILD_FAILED", "engineered feature names do not match active model")
        return self.feature_scaler.transform(engineered), np.array(y_values, dtype=float)
