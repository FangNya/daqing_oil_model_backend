from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb

from app.feature_adapters.common import backend_path
from app.models.base import BasePredictor
from app.services.errors import BackendError


class XGBoostPredictor(BasePredictor):
    def __init__(self, model_dir: str | Path):
        self.model_dir = backend_path(model_dir)
        self.model_path = self.model_dir / "xgboost_model.json"
        if not self.model_path.exists():
            raise BackendError("MODEL_NOT_LOADED", f"missing model file: {self.model_path}", status_code=500)
        self.booster = xgb.Booster()
        self.booster.load_model(str(self.model_path))

    def predict(self, X):
        try:
            y = self.booster.predict(xgb.DMatrix(X))
            return float(np.asarray(y).reshape(-1)[0])
        except Exception as exc:
            raise BackendError("PREDICTION_FAILED", str(exc), status_code=500) from exc
