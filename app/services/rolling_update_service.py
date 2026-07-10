from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from app.feature_adapters.common import backend_path
from app.models.informer_model import InformerPredictor
from app.models.xgboost_model import XGBoostPredictor
from app.services.errors import BackendError
from app.services.evaluation_service import evaluate_active, metrics_dict, to_naive_timestamp
from app.services.prediction_service import _load_adapter, clear_model_cache
from app.services.registry_service import get_target_config, load_registry, save_registry
from app.services.storage_service import (
    load_lab_history,
    load_operation_history,
)


MIN_UPDATE_SAMPLES = {
    "bingxi": 1,
    "gandian": 3,
    "qiya": 5,
    "C5": 5,
}

INFORMER_UPDATE_EPOCHS = {
    "bingxi": 50,
    "gandian": 30,
}


def _cycle_days(cycle: str):
    if cycle == "none":
        return None
    if cycle.endswith("d"):
        return int(cycle[:-1])
    raise BackendError("INVALID_CYCLE", f"invalid cycle: {cycle}")


def _now():
    return datetime.now(timezone.utc)


def _version_name(target: str):
    return f"{target}_{_now().strftime('%Y%m%d%H%M%S')}"


def _write_log(target: str, message: str, extra: dict | None = None):
    log_dir = backend_path("runtime/logs/rolling_update")
    log_dir.mkdir(parents=True, exist_ok=True)
    row = {"created_at": _now().isoformat(), "target": target, "message": message, **(extra or {})}
    path = log_dir / f"{target}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _labels_in_period(target: str, start_time, end_time):
    labs = load_lab_history(target)
    if labs.empty:
        return pd.DataFrame(columns=["sample_time", "value"])
    start_time = to_naive_timestamp(start_time)
    end_time = to_naive_timestamp(end_time)
    mask = (labs["sample_time"] >= start_time) & (labs["sample_time"] < end_time)
    return labs.loc[mask, ["sample_time", "value"]].sort_values("sample_time")


def _candidate_base_dir(target: str, version: str):
    return backend_path(f"runtime/models/{target}/candidates/{version}")


def _archive_active(target: str, version: str):
    active_dir = backend_path(f"runtime/models/{target}/active")
    archive_dir = backend_path(f"runtime/models/{target}/archive/{version}")
    if active_dir.exists():
        shutil.copytree(active_dir, archive_dir, dirs_exist_ok=True)
    return archive_dir


def _copy_candidate_to_active(candidate_dir: Path, target: str):
    active_dir = backend_path(f"runtime/models/{target}/active")
    active_dir.mkdir(parents=True, exist_ok=True)
    for item in candidate_dir.iterdir():
        dst = active_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)


def _remove_model_dir(target: str, path: str | Path, reason: str):
    model_root = backend_path(f"runtime/models/{target}").resolve()
    target_path = backend_path(path).resolve()
    if target_path == model_root or model_root not in target_path.parents:
        _write_log(target, "model directory cleanup skipped", {"path": str(path), "reason": reason})
        return
    try:
        if target_path.exists():
            shutil.rmtree(target_path)
            _write_log(target, "model directory removed", {"path": str(path), "reason": reason})
    except Exception as exc:
        _write_log(
            target,
            "model directory cleanup failed",
            {"path": str(path), "reason": reason, "error": str(exc)},
        )


def _make_xgb_candidate(target: str, config: dict, labels: pd.DataFrame):
    adapter = _load_adapter(config)
    op_history = load_operation_history(target)
    lab_history = load_lab_history(target)
    X_train, y_train = adapter.build_training_matrix(op_history, lab_history, labels)
    if len(y_train) < MIN_UPDATE_SAMPLES[target]:
        raise BackendError(
            "UPDATE_SKIPPED_INSUFFICIENT_LABELS",
            f"{target}: {len(y_train)} valid update samples",
        )

    active_dir = backend_path(config["active_model_dir"])
    booster = xgb.Booster()
    booster.load_model(str(active_dir / "xgboost_model.json"))
    best = adapter.config.get("best_config", {})
    params = {
        "objective": best.get("objective", "reg:squarederror"),
        "max_depth": int(best.get("max_depth", 3)),
        "eta": min(float(best.get("learning_rate", 0.005)), 0.005),
        "min_child_weight": float(best.get("min_child_weight", 1.0)),
        "subsample": float(best.get("subsample", 1.0)),
        "colsample_bytree": float(best.get("colsample_bytree", 1.0)),
        "lambda": float(best.get("reg_lambda", 1.0)),
        "alpha": float(best.get("reg_alpha", 0.0)),
        "seed": 42,
    }
    updated = xgb.train(params, xgb.DMatrix(X_train, label=y_train), num_boost_round=10, xgb_model=booster)

    version = _version_name(target)
    candidate_dir = _candidate_base_dir(target, version)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    updated.save_model(candidate_dir / "xgboost_model.json")
    for filename in ["feature_scaler.pkl", "feature_columns.pkl", "engineered_feature_names.pkl", "model_config.json"]:
        shutil.copy2(active_dir / filename, candidate_dir / filename)
    return candidate_dir, version, len(y_train)


def _make_informer_candidate(target: str, config: dict, labels: pd.DataFrame):
    adapter = _load_adapter(config)
    op_history = load_operation_history(target)
    lab_history = load_lab_history(target)
    X_train, y_train = adapter.build_training_matrix(op_history, lab_history, labels)
    if len(y_train) < MIN_UPDATE_SAMPLES[target]:
        raise BackendError(
            "UPDATE_SKIPPED_INSUFFICIENT_LABELS",
            f"{target}: {len(y_train)} valid update samples",
        )

    active_dir = backend_path(config["active_model_dir"])
    predictor = InformerPredictor(active_dir)
    y_scaler = joblib.load(active_dir / "y_scaler.pkl")
    y_scaled = y_scaler.transform(np.asarray(y_train).reshape(-1, 1)).flatten()
    predictor.model.fit(
        X_train,
        y_scaled,
        epochs=INFORMER_UPDATE_EPOCHS.get(target, 30),
        batch_size=5,
        verbose=0,
    )

    version = _version_name(target)
    candidate_dir = _candidate_base_dir(target, version)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    predictor.model.save(candidate_dir / "best_drypoint_model.h5")
    for filename in ["scalers.pkl", "y_scaler.pkl", "model_config.json"]:
        shutil.copy2(active_dir / filename, candidate_dir / filename)
    return candidate_dir, version, len(y_train)


def create_candidate(target: str, cycle_start, cycle_end):
    config = get_target_config(target)
    labels = _labels_in_period(target, cycle_start, cycle_end)
    if len(labels) < MIN_UPDATE_SAMPLES[target]:
        raise BackendError(
            "UPDATE_SKIPPED_INSUFFICIENT_LABELS",
            f"{target}: {len(labels)} labels in cycle",
        )
    if config["engine"] == "xgboost":
        return _make_xgb_candidate(target, config, labels)
    if config["engine"] == "informer":
        return _make_informer_candidate(target, config, labels)
    raise BackendError("ROLLING_UPDATE_FAILED", f"unsupported engine: {config['engine']}")


def evaluate_model_dir(target: str, model_dir: str, labels: pd.DataFrame):
    if labels.empty:
        raise BackendError("UPDATE_SKIPPED_INSUFFICIENT_LABELS", f"{target}: no labels for candidate evaluation")
    config = get_target_config(target)
    candidate_config = dict(config)
    candidate_config["active_model_dir"] = model_dir
    adapter = _load_adapter(candidate_config)
    op_history = load_operation_history(target)
    lab_history = load_lab_history(target)
    X, y_true = adapter.build_training_matrix(op_history, lab_history, labels)
    if len(y_true) < MIN_UPDATE_SAMPLES[target]:
        raise BackendError(
            "UPDATE_SKIPPED_INSUFFICIENT_LABELS",
            f"{target}: {len(y_true)} valid candidate evaluation samples",
        )
    if candidate_config["engine"] == "xgboost":
        predictor = XGBoostPredictor(model_dir)
        y_pred = predictor.booster.predict(xgb.DMatrix(X))
    elif candidate_config["engine"] == "informer":
        predictor = InformerPredictor(model_dir)
        pred_scaled = predictor.model.predict(X, verbose=0).reshape(-1)
        y_pred = predictor.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    else:
        raise BackendError("ROLLING_UPDATE_FAILED", f"unsupported engine: {candidate_config['engine']}")
    return metrics_dict(y_true, y_pred)


def _maybe_promote_existing_candidate(target: str, config: dict, active_metrics: dict, cycle_start, cycle_end):
    candidate_dir = config.get("candidate_model_dir")
    if not candidate_dir:
        return {"status": "no_candidate"}
    labels = _labels_in_period(target, cycle_start, cycle_end)
    candidate_metrics = evaluate_model_dir(target, candidate_dir, labels)
    if candidate_metrics["mae"] < active_metrics["mae"]:
        result = promote_candidate(
            target,
            candidate_dir,
            reason=f"candidate_mae={candidate_metrics['mae']:.6f} < active_mae={active_metrics['mae']:.6f}",
        )
        result["candidate_metrics"] = candidate_metrics
        result["active_metrics"] = active_metrics
        return result
    _write_log(
        target,
        "candidate rejected",
        {
            "candidate_dir": candidate_dir,
            "candidate_mae": candidate_metrics["mae"],
            "active_mae": active_metrics["mae"],
        },
    )
    registry = load_registry()
    registry[target]["candidate_model_dir"] = None
    save_registry(registry)
    _remove_model_dir(target, candidate_dir, "candidate_rejected")
    return {
        "status": "candidate_rejected",
        "candidate_metrics": candidate_metrics,
        "active_metrics": active_metrics,
    }


def maybe_run_target_update(target: str, now=None):
    now = to_naive_timestamp(now or _now())
    config = get_target_config(target)
    cycle = config.get("default_cycle", "none")
    days = _cycle_days(cycle)
    if days is None:
        try:
            evaluate_active(target, cycle)
        except BackendError as exc:
            _write_log(target, "evaluation skipped", {"reason": exc.message})
        return {"target": target, "status": "skipped", "reason": "cycle none"}

    last_update = config.get("last_update_time")
    cycle_start = to_naive_timestamp(last_update) if last_update else now - timedelta(days=days)
    if (now - cycle_start).days < days:
        return {"target": target, "status": "not_due", "cycle": cycle}

    cycle_end = cycle_start + timedelta(days=days)
    registry = load_registry()
    try:
        eval_record = evaluate_active(target, cycle, cycle_start, cycle_end)
        selection_result = _maybe_promote_existing_candidate(target, config, eval_record, cycle_start, cycle_end)
        config = get_target_config(target)
        candidate_dir, version, samples = create_candidate(target, cycle_start, cycle_end)
    except BackendError as exc:
        registry[target]["last_update_time"] = now.isoformat()
        save_registry(registry)
        _write_log(target, "rolling update skipped", {"reason": exc.message, "cycle": cycle})
        return {"target": target, "status": "skipped", "reason": exc.code}

    registry = load_registry()
    registry[target]["candidate_model_dir"] = str(candidate_dir.relative_to(backend_path(".")))
    registry[target]["last_update_time"] = now.isoformat()
    save_registry(registry)
    clear_model_cache()
    _write_log(
        target,
        "candidate generated",
        {
            "version": version,
            "samples": samples,
            "cycle": cycle,
            "active_mae": eval_record.get("mae"),
            "selection_status": selection_result.get("status"),
        },
    )
    return {
        "target": target,
        "status": "candidate_generated",
        "candidate": version,
        "samples": samples,
        "selection": selection_result,
    }


def promote_candidate(target: str, candidate_dir: str, reason: str):
    candidate_path = backend_path(candidate_dir)
    if not candidate_path.exists():
        raise BackendError("MODEL_NOT_LOADED", f"candidate missing: {candidate_dir}", status_code=500)
    version = _version_name(target)
    archive_dir = _archive_active(target, version)
    _copy_candidate_to_active(candidate_path, target)
    registry = load_registry()
    registry[target]["candidate_model_dir"] = None
    registry[target]["last_selection_time"] = _now().isoformat()
    save_registry(registry)
    clear_model_cache()
    _write_log(target, "candidate promoted", {"candidate_dir": candidate_dir, "reason": reason})
    _remove_model_dir(target, candidate_path, "promoted_candidate_copied_to_active")
    _remove_model_dir(target, archive_dir, "old_active_replaced_by_promoted_candidate")
    return {"target": target, "status": "promoted", "candidate_dir": candidate_dir}


def run_daily_check(now=None):
    results = []
    for target in ["bingxi", "C5", "gandian", "qiya"]:
        results.append(maybe_run_target_update(target, now=now))
    return results
