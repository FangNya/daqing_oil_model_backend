import logging
import math

import pandas as pd
from flask import jsonify, request

from app.services.prediction_service import predict_target
from app.services.storage_service import load_lab_history, load_prediction_history


logger = logging.getLogger(__name__)

SAFE_DEFAULTS = {
    "qiya": 63.6,
    "C5": 0.07,
    "bingxi": 0.4,
    "gandian": 201.2,
}


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _now_text():
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def _latest_valid_prediction(target: str):
    try:
        history = load_prediction_history(target)
    except Exception as exc:
        logger.warning("load prediction fallback failed for %s: %s", target, exc)
        return None
    if history.empty or "prediction" not in history.columns:
        return None
    for value in reversed(history["prediction"].tolist()):
        number = _finite_float(value)
        if number is not None:
            return number
    return None


def _latest_valid_lab_value(target: str):
    try:
        history = load_lab_history(target)
    except Exception as exc:
        logger.warning("load lab fallback failed for %s: %s", target, exc)
        return None
    if history.empty or "value" not in history.columns:
        return None
    for value in reversed(history["value"].tolist()):
        number = _finite_float(value)
        if number is not None:
            return number
    return None


def _safe_fallback_value(target: str):
    value = _latest_valid_prediction(target)
    if value is not None:
        return value, "latest_prediction"
    value = _latest_valid_lab_value(target)
    if value is not None:
        return value, "latest_lab"
    return SAFE_DEFAULTS[target], "static_default"


def _safe_prediction_body(target: str, body=None, reason: str = "invalid_result"):
    body = dict(body) if isinstance(body, dict) else {}
    value = _finite_float(body.get("result"))
    if value is None:
        value = _finite_float(body.get("prediction"))
    if value is None:
        value, source = _safe_fallback_value(target)
        body["fallback"] = True
        body["fallback_reason"] = reason
        body["fallback_source"] = source
    body.setdefault("target", target)
    body.setdefault("prediction_time", body.get("time") or _now_text())
    body["time"] = body.get("time") or body["prediction_time"]
    body["prediction"] = value
    body["result"] = str(value)
    body["status"] = "ok"
    return body


def handle_predict(target: str):
    payload = request.get_json(silent=True) or {}
    try:
        body = predict_target(target, payload)
        safe_body = _safe_prediction_body(target, body)
        if safe_body.get("fallback"):
            logger.warning("prediction response fallback used for %s: %s", target, safe_body.get("fallback_reason"))
        return jsonify(safe_body)
    except Exception as exc:
        logger.exception("prediction failed for %s, returning safe fallback", target)
        return jsonify(_safe_prediction_body(target, reason=type(exc).__name__))
