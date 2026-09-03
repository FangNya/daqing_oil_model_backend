import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import RUNTIME_DIR


def _parse_time(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"MISSING_TIMESTAMP: invalid timestamp {value}")
    return ts


def _quarter_path(base_dir: Path, target: str, ts, suffix: str):
    ts = _parse_time(ts)
    quarter = (ts.month - 1) // 3 + 1
    return base_dir / target / f"{ts.year}-Q{quarter}.{suffix}"


def _month_path(base_dir: Path, target: str, ts, suffix: str = "csv"):
    ts = _parse_time(ts)
    return base_dir / target / f"{ts.year}-{ts.month:02d}.{suffix}"


def _read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_excel(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path)


def _normalize_bound(value):
    if value is None:
        return None
    ts = _parse_time(value)
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _partitioned_paths(directory: Path, suffix: str, freq: str, start_time=None, end_time=None):
    paths = []
    for path in directory.glob(f"*.{suffix}"):
        try:
            period = pd.Period(path.stem, freq=freq)
        except ValueError:
            continue
        paths.append((period, path))
    paths.sort(key=lambda item: item[0])

    start_period = start_time.to_period(freq) if start_time is not None else None
    end_period = end_time.to_period(freq) if end_time is not None else None
    selected = [
        (period, path)
        for period, path in paths
        if (start_period is None or period >= start_period)
        and (end_period is None or period <= end_period)
    ]
    return paths, selected


def _load_partitioned_history(
    directory: Path,
    suffix: str,
    freq: str,
    time_column: str,
    empty_columns,
    normalize_frame,
    start_time=None,
    end_time=None,
    include_preceding: bool = False,
):
    start_time = _normalize_bound(start_time)
    end_time = _normalize_bound(end_time)
    all_paths, selected_paths = _partitioned_paths(directory, suffix, freq, start_time, end_time)
    selected_path_set = {path for _, path in selected_paths}
    reader = _read_excel if suffix == "xlsx" else _read_csv

    frames = []
    preceding = []
    for _, path in selected_paths:
        df = normalize_frame(reader(path))
        if df.empty:
            continue
        if include_preceding and start_time is not None:
            before = df[df[time_column] < start_time]
            if not before.empty:
                preceding.append(before.sort_values(time_column).tail(1))
        if start_time is not None:
            df = df[df[time_column] >= start_time]
        if end_time is not None:
            df = df[df[time_column] <= end_time]
        if not df.empty:
            frames.append(df)

    if include_preceding and start_time is not None and not preceding:
        start_period = start_time.to_period(freq)
        for period, path in reversed(all_paths):
            if period >= start_period or path in selected_path_set:
                continue
            df = normalize_frame(reader(path))
            if df.empty:
                continue
            before = df[df[time_column] < start_time]
            if not before.empty:
                preceding.append(before.sort_values(time_column).tail(1))
                break

    if preceding:
        frames.insert(0, pd.concat(preceding, ignore_index=True).sort_values(time_column).tail(1))
    if not frames:
        return pd.DataFrame(columns=empty_columns)
    return pd.concat(frames, ignore_index=True)


def _normalize_operation_frame(df: pd.DataFrame):
    if df.empty or "TimeStamp" not in df.columns:
        return pd.DataFrame(columns=["TimeStamp"])
    df = df.copy()
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    return df.dropna(subset=["TimeStamp"])


def _normalize_lab_frame(df: pd.DataFrame):
    if df.empty or "sample_time" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["sample_time", "value"])
    df = df.copy()
    df["sample_time"] = pd.to_datetime(df["sample_time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["sample_time", "value"])


def _normalize_prediction_frame(df: pd.DataFrame):
    required = ["prediction_time", "prediction"]
    if df.empty or any(column not in df.columns for column in required):
        return pd.DataFrame(columns=required)
    df = df.copy()
    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    return df.dropna(subset=required)


def append_operation_records(target: str, records):
    if not records:
        return []
    df = pd.DataFrame(records)
    if "TimeStamp" not in df.columns:
        raise ValueError("MISSING_TIMESTAMP: operation_records require TimeStamp")
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    if df["TimeStamp"].isna().any():
        raise ValueError("MISSING_TIMESTAMP: invalid TimeStamp in operation_records")

    written = []
    base_dir = RUNTIME_DIR / "data" / "operation"
    for _, group in df.groupby(df["TimeStamp"].dt.to_period("Q")):
        path = _quarter_path(base_dir, target, group["TimeStamp"].min(), "csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_csv(path)
        merged = pd.concat([existing, group], ignore_index=True)
        merged["TimeStamp"] = pd.to_datetime(merged["TimeStamp"], errors="coerce")
        merged = merged.dropna(subset=["TimeStamp"])
        merged = merged.drop_duplicates(subset=["TimeStamp"], keep="last").sort_values("TimeStamp")
        merged.to_csv(path, index=False)
        written.append(str(path.relative_to(RUNTIME_DIR.parent)))
    return written


def append_lab_records(target: str, records):
    if not records:
        return []
    df = pd.DataFrame(records)
    if "sample_time" not in df.columns or "value" not in df.columns:
        raise ValueError("INVALID_LAB_RECORD: lab_records require sample_time and value")
    df["sample_time"] = pd.to_datetime(df["sample_time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["sample_time"].isna().any() or df["value"].isna().any():
        raise ValueError("INVALID_LAB_RECORD: invalid sample_time or value")

    written = []
    base_dir = RUNTIME_DIR / "data" / "lab"
    for _, group in df.groupby(df["sample_time"].dt.to_period("Q")):
        path = _quarter_path(base_dir, target, group["sample_time"].min(), "xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_excel(path)
        merged = pd.concat([existing, group], ignore_index=True)
        merged["sample_time"] = pd.to_datetime(merged["sample_time"], errors="coerce")
        merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
        merged = merged.dropna(subset=["sample_time", "value"])
        merged = merged.drop_duplicates(subset=["sample_time"], keep="last").sort_values("sample_time")
        merged.to_excel(path, index=False)
        written.append(str(path.relative_to(RUNTIME_DIR.parent)))
    return written


def load_lab_history(target: str, start_time=None, end_time=None, include_preceding: bool = False):
    lab_dir = RUNTIME_DIR / "data" / "lab" / target
    df = _load_partitioned_history(
        lab_dir,
        "xlsx",
        "Q",
        "sample_time",
        ["sample_time", "value"],
        _normalize_lab_frame,
        start_time,
        end_time,
        include_preceding,
    )
    return df.drop_duplicates("sample_time", keep="last").sort_values("sample_time")


def load_operation_history(target: str, start_time=None, end_time=None, include_preceding: bool = False):
    operation_dir = RUNTIME_DIR / "data" / "operation" / target
    df = _load_partitioned_history(
        operation_dir,
        "csv",
        "Q",
        "TimeStamp",
        ["TimeStamp"],
        _normalize_operation_frame,
        start_time,
        end_time,
        include_preceding,
    )
    df = df.drop_duplicates("TimeStamp", keep="last").sort_values("TimeStamp")
    return df.reset_index(drop=True)


def save_prediction_record(target: str, record):
    record = dict(record)
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    path = _month_path(RUNTIME_DIR / "predictions", target, record["prediction_time"], "csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_csv(path)
    merged = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    merged.to_csv(path, index=False)
    return str(path.relative_to(RUNTIME_DIR.parent))


def load_prediction_history(target: str, start_time=None, end_time=None):
    prediction_dir = RUNTIME_DIR / "predictions" / target
    df = _load_partitioned_history(
        prediction_dir,
        "csv",
        "M",
        "prediction_time",
        ["prediction_time", "prediction", "model_version", "cycle", "created_at"],
        _normalize_prediction_frame,
        start_time,
        end_time,
    )
    return df.sort_values("prediction_time")


def load_latest_valid_prediction(target: str):
    prediction_dir = RUNTIME_DIR / "predictions" / target
    all_paths, _ = _partitioned_paths(prediction_dir, "csv", "M")
    for _, path in reversed(all_paths):
        df = _normalize_prediction_frame(_read_csv(path))
        if df.empty:
            continue
        for value in reversed(df.sort_values("prediction_time")["prediction"].tolist()):
            number = float(value)
            if math.isfinite(number):
                return number
    return None


def load_latest_valid_lab(target: str):
    lab_dir = RUNTIME_DIR / "data" / "lab" / target
    all_paths, _ = _partitioned_paths(lab_dir, "xlsx", "Q")
    for _, path in reversed(all_paths):
        df = _normalize_lab_frame(_read_excel(path))
        if df.empty:
            continue
        for value in reversed(df.sort_values("sample_time")["value"].tolist()):
            number = float(value)
            if math.isfinite(number):
                return number
    return None


def save_evaluation_record(target: str, record):
    record = dict(record)
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    path = _month_path(RUNTIME_DIR / "evaluations", target, record.get("evaluated_at") or record["created_at"], "csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_csv(path)
    merged = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    merged.to_csv(path, index=False)
    return str(path.relative_to(RUNTIME_DIR.parent))
