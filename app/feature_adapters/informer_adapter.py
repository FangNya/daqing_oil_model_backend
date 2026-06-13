from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from app.feature_adapters.common import (
    add_last_prediction,
    backend_path,
    build_window_sequence,
    normalize_operation_data,
)
from app.services.errors import BackendError


class InformerFeatureAdapter:
    def __init__(self, config: dict):
        self.config = config
        self.model_dir = backend_path(config["active_model_dir"])
        self.op_feature_columns = list(config["op_feature_columns"])
        self.feature_columns = self.op_feature_columns + ["last_prediction"]
        self.delay_hours = int(config["delay_hours"])
        self.window_hours = int(config["window_hours"])
        self.max_points = int(config["max_points"])
        self.min_points = int(config.get("min_points", 20))
        self.scalers = joblib.load(self.model_dir / "scalers.pkl")

    def prepare_operation_data(self, op_data: pd.DataFrame, lab_history: pd.DataFrame) -> pd.DataFrame:
        prepared = normalize_operation_data(op_data, self.op_feature_columns)
        prepared = add_last_prediction(prepared, lab_history)
        return prepared

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X_scaled = X.copy().astype(float)
        for idx in range(X.shape[2]):
            scaler = self.scalers[idx]
            for sample_idx in range(X.shape[0]):
                X_scaled[sample_idx, :, idx] = scaler.transform(X[sample_idx, :, idx].reshape(-1, 1)).flatten()
        return X_scaled

    def build_predict_input(self, op_data: pd.DataFrame, lab_history: pd.DataFrame, prediction_time) -> np.ndarray:
        prepared = self.prepare_operation_data(op_data, lab_history)
        seq = build_window_sequence(
            prepared,
            prediction_time,
            self.feature_columns,
            self.delay_hours,
            self.window_hours,
            self.max_points,
            self.min_points,
        )
        return self._scale(seq.reshape(1, *seq.shape))

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
                    self.feature_columns,
                    self.delay_hours,
                    self.window_hours,
                    self.max_points,
                    self.min_points,
                )
            except BackendError:
                continue
            rows.append(seq)
            y_values.append(float(row["value"]))
        if not rows:
            return np.empty((0, self.max_points, len(self.feature_columns))), np.array([])
        return self._scale(np.array(rows)), np.array(y_values, dtype=float)
