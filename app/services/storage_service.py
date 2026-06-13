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


def load_lab_history(target: str):
    lab_dir = RUNTIME_DIR / "data" / "lab" / target
    frames = []
    for path in sorted(lab_dir.glob("*.xlsx")):
        df = pd.read_excel(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["sample_time", "value"])
    df = pd.concat(frames, ignore_index=True)
    df["sample_time"] = pd.to_datetime(df["sample_time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["sample_time", "value"]).drop_duplicates("sample_time", keep="last").sort_values("sample_time")


def load_operation_history(target: str):
    operation_dir = RUNTIME_DIR / "data" / "operation" / target
    frames = []
    for path in sorted(operation_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["TimeStamp"])
    df = pd.concat(frames, ignore_index=True)
    if "TimeStamp" not in df.columns:
        return pd.DataFrame(columns=["TimeStamp"])
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.dropna(subset=["TimeStamp"]).drop_duplicates("TimeStamp", keep="last").sort_values("TimeStamp")
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


def load_prediction_history(target: str):
    prediction_dir = RUNTIME_DIR / "predictions" / target
    frames = []
    for path in sorted(prediction_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["prediction_time", "prediction", "model_version", "cycle", "created_at"])
    df = pd.concat(frames, ignore_index=True)
    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    return df.dropna(subset=["prediction_time", "prediction"]).sort_values("prediction_time")


def save_evaluation_record(target: str, record):
    record = dict(record)
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    path = _month_path(RUNTIME_DIR / "evaluations", target, record.get("evaluated_at") or record["created_at"], "csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_csv(path)
    merged = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    merged.to_csv(path, index=False)
    return str(path.relative_to(RUNTIME_DIR.parent))
