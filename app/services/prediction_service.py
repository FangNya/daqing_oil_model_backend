from __future__ import annotations

import importlib
import json
from datetime import timezone

import pandas as pd

from app.config import RUNTIME_DIR, get_stale_threshold_days
from app.feature_adapters.common import validate_operation_data
from app.models.informer_model import InformerPredictor
from app.models.xgboost_model import XGBoostPredictor
from app.services.errors import BackendError
from app.services.registry_service import get_target_config, resolve_cycle
from app.services.storage_service import (
    append_lab_records,
    append_operation_records,
    load_lab_history,
    load_operation_history,
    save_prediction_record,
)


_MODEL_CACHE = {}


def _coerce_timestamp(value, code: str = "MISSING_TIMESTAMP"):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise BackendError(code, "operation_records require valid TimeStamp")
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _format_timestamp(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _parse_end_time(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, (int, float)):
        unit = "ms" if value > 100000000000 else "s"
        ts = pd.to_datetime(value, unit=unit, errors="coerce")
    else:
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _records_from_payload(payload: dict, adapter):
    raw_records = payload.get("operation_records") or payload.get("produce") or []
    records = [dict(row) for row in raw_records]
    for row in records:
        if "TimeStamp" in row:
            continue
        for key in ("timestamp", "time", "Time"):
            if row.get(key) is not None:
                row["TimeStamp"] = row[key]
                break

    missing_time = [row.get("TimeStamp") in (None, "") for row in records]
    if records and all(missing_time):
        end_time = _parse_end_time(payload.get("endTime"))
        if end_time is not None:
            return _with_synthetic_timestamps(records, adapter, end_time)
    return records


def _with_synthetic_timestamps(records, adapter, end_time):
    records = [dict(row) for row in records]
    if len(records) == 1:
        times = [end_time]
    else:
        start_time = end_time - pd.Timedelta(hours=adapter.window_hours)
        times = pd.date_range(start=start_time, end=end_time, periods=len(records))
    for row, ts in zip(records, times):
        row["TimeStamp"] = _format_timestamp(ts)
    return records


def _prediction_time(operation_records, lab_records, adapter, payload, stale_count: int):
    if lab_records:
        times = pd.to_datetime([row.get("sample_time") for row in lab_records], errors="coerce")
        if times.isna().any():
            raise BackendError("INVALID_LAB_RECORD", "lab_records contain invalid sample_time")
        return times.max()
    times = pd.to_datetime([row.get("TimeStamp") for row in operation_records], errors="coerce")
    if len(times) == 0 or times.isna().any():
        raise BackendError("MISSING_TIMESTAMP", "operation_records require valid TimeStamp")
    if stale_count:
        end_time = _parse_end_time(payload.get("endTime"))
        if end_time is not None:
            return end_time + pd.Timedelta(hours=adapter.delay_hours)
    return times.max() + pd.Timedelta(hours=adapter.delay_hours)


def _split_stale_records(records, threshold_days: int):
    now = pd.Timestamp.now()
    max_delta = pd.Timedelta(days=threshold_days)
    fresh_records = []
    stale_records = []
    for row in records:
        ts = _coerce_timestamp(row.get("TimeStamp"))
        if abs(now - ts) > max_delta:
            stale_records.append(row)
        else:
            fresh_records.append(row)
    return fresh_records, stale_records


def _time_bounds(records):
    if not records:
        return None, None
    times = [_coerce_timestamp(row.get("TimeStamp")) for row in records]
    return _format_timestamp(min(times)), _format_timestamp(max(times))


def _combine_operation_history(op_history: pd.DataFrame, operation_records):
    request_df = pd.DataFrame(operation_records)
    if request_df.empty:
        return op_history
    combined = pd.concat([op_history, request_df], ignore_index=True)
    if "TimeStamp" not in combined.columns:
        return combined
    combined["TimeStamp"] = pd.to_datetime(combined["TimeStamp"], errors="coerce")
    combined = combined.dropna(subset=["TimeStamp"])
    return combined.drop_duplicates("TimeStamp", keep="last").sort_values("TimeStamp").reset_index(drop=True)


def _shift_request_records_to_window(operation_records, adapter, pred_time):
    request_df = pd.DataFrame(operation_records)
    if request_df.empty:
        return request_df
    request_df["TimeStamp"] = pd.to_datetime(request_df["TimeStamp"], errors="coerce")
    request_df = request_df.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").reset_index(drop=True)
    end_time = pd.Timestamp(pred_time) - pd.Timedelta(hours=adapter.delay_hours)
    if len(request_df) == 1:
        request_df["TimeStamp"] = [end_time]
    else:
        start_time = end_time - pd.Timedelta(hours=adapter.window_hours)
        request_df["TimeStamp"] = pd.date_range(start=start_time, end=end_time, periods=len(request_df))
    return request_df


def _write_cold_start_log(target: str, record: dict):
    try:
        log_dir = RUNTIME_DIR / "logs" / "prediction_cold_start"
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
            "target": target,
            **record,
        }
        with open(log_dir / f"{target}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _load_adapter(config):
    module = importlib.import_module(f"app.feature_adapters.{config['feature_adapter']}")
    return module.create_adapter(config)


def _load_model(config):
    cache_key = (config["engine"], config["active_model_dir"])
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if config["engine"] == "xgboost":
        model = XGBoostPredictor(config["active_model_dir"])
    elif config["engine"] == "informer":
        model = InformerPredictor(config["active_model_dir"])
    else:
        raise BackendError("MODEL_NOT_LOADED", f"unsupported engine: {config['engine']}", status_code=500)
    _MODEL_CACHE[cache_key] = model
    return model


def clear_model_cache():
    _MODEL_CACHE.clear()


def predict_target(target: str, payload: dict):
    payload = payload or {}
    lab_records = payload.get("lab_records") or []
    try:
        cycle = resolve_cycle(target, payload.get("cycle"))
        config = get_target_config(target)
        threshold_days = get_stale_threshold_days()
    except ValueError as exc:
        code = str(exc).split(":", 1)[0]
        raise BackendError(code, str(exc)) from exc

    adapter = _load_adapter(config)
    operation_records = _records_from_payload(payload, adapter)
    if not operation_records:
        raise BackendError("MISSING_OPERATION_DATA", "operation_records or produce is required")

    validate_operation_data(pd.DataFrame(operation_records), adapter.op_feature_columns)
    fresh_records, stale_records = _split_stale_records(operation_records, threshold_days)
    fresh_start, fresh_end = _time_bounds(fresh_records)
    stale_start, stale_end = _time_bounds(stale_records)

    try:
        append_operation_records(target, fresh_records)
        append_lab_records(target, lab_records)
    except ValueError as exc:
        code = str(exc).split(":", 1)[0]
        raise BackendError(code, str(exc)) from exc
    except Exception as exc:
        raise BackendError("DATA_SAVE_FAILED", str(exc), status_code=500) from exc

    op_history = load_operation_history(target)
    lab_history = load_lab_history(target)
    pred_time = _prediction_time(operation_records, lab_records, adapter, payload, len(stale_records))
    prediction_history = _combine_operation_history(op_history, operation_records)

    model = _load_model(config)
    fallback_used = False
    try:
        X = adapter.build_predict_input(prediction_history, lab_history, pred_time)
    except BackendError as exc:
        if exc.code != "INSUFFICIENT_WINDOW_POINTS":
            if stale_records:
                _write_cold_start_log(
                    target,
                    {
                        "status": "failed",
                        "reason": exc.code,
                        "fresh_count": len(fresh_records),
                        "stale_count": len(stale_records),
                        "fresh_time_start": fresh_start,
                        "fresh_time_end": fresh_end,
                        "stale_time_start": stale_start,
                        "stale_time_end": stale_end,
                        "threshold_days": threshold_days,
                        "fallback_used": fallback_used,
                    },
                )
            raise
        fallback_used = True
        shifted_history = _shift_request_records_to_window(operation_records, adapter, pred_time)
        try:
            X = adapter.build_predict_input(shifted_history, lab_history, pred_time)
        except BackendError as fallback_exc:
            if stale_records:
                _write_cold_start_log(
                    target,
                    {
                        "status": "failed",
                        "reason": fallback_exc.code,
                        "fresh_count": len(fresh_records),
                        "stale_count": len(stale_records),
                        "fresh_time_start": fresh_start,
                        "fresh_time_end": fresh_end,
                        "stale_time_start": stale_start,
                        "stale_time_end": stale_end,
                        "threshold_days": threshold_days,
                        "fallback_used": fallback_used,
                    },
                )
            raise
    prediction = model.predict(X)

    save_prediction_record(
        target,
        {
            "prediction_time": pred_time,
            "prediction": prediction,
            "model_version": "active",
            "cycle": cycle,
        },
    )

    prediction_time = _format_timestamp(pred_time)
    if stale_records:
        _write_cold_start_log(
            target,
            {
                "status": "ok",
                "fresh_count": len(fresh_records),
                "stale_count": len(stale_records),
                "fresh_time_start": fresh_start,
                "fresh_time_end": fresh_end,
                "stale_time_start": stale_start,
                "stale_time_end": stale_end,
                "threshold_days": threshold_days,
                "fallback_used": fallback_used,
                "prediction_time": prediction_time,
            },
        )

    return {
        "target": target,
        "prediction_time": prediction_time,
        "prediction": prediction,
        "time": prediction_time,
        "result": str(prediction),
        "model_version": "active",
        "active_cycle": cycle,
        "status": "ok",
    }
