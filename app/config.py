import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = BASE_DIR / "runtime"
REGISTRY_PATH = RUNTIME_DIR / "registry" / "model_registry.json"
ENV_PATH = BASE_DIR / ".env"

VALID_CYCLES = {"none", "10d", "20d", "30d"}


def _load_env_file(path: Path = ENV_PATH):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()

TARGET_SPECS = {
    "bingxi": {
        "display_name": "丙烯",
        "engine": "informer",
        "default_cycle": "10d",
        "active_model_dir": "runtime/models/bingxi/active",
        "feature_adapter": "bingxi_adapter",
        "delay_hours": 1,
        "window_hours": 2,
        "max_points": 60,
        "min_points": 20,
        "op_feature_columns": [
            "ARG2_TIC1345_PV",
            "ARG2_FIC1303_PV",
            "ARG2_FIC1210_PV",
            "ARG2_FIC1307_PV",
            "ARG2_FIC1306_PV",
            "ARG2_FIC1305_PV",
            "ARG2_FIC1304_PV",
            "ARG2_PI1304_PV",
            "ARG2_TI1308_PV",
            "ARG2_TI1310_PV",
            "ARG2_TI1312_PV",
            "ARG2_TI1314_PV",
            "ARG2_FI1314_PV",
            "ARG2_TI1341_PV",
            "ARG2_TI1347_PV",
            "ARG2_TI1304_PV",
            "ARG2_TIC1101_PV",
            "ARG2_TIC1103_PV",
            "ARG2_TI1233C_PV",
            "ARG2_TI1306_PV",
        ],
    },
    "gandian": {
        "display_name": "干点",
        "engine": "informer",
        "default_cycle": "none",
        "active_model_dir": "runtime/models/gandian/active",
        "feature_adapter": "gandian_adapter",
        "delay_hours": 3,
        "window_hours": 1,
        "max_points": 30,
        "min_points": 20,
        "op_feature_columns": [
            "ARG2_TIC1201B_PV",
            "ARG2_PI1204_PV",
            "ARG2_FIC1214_PV",
            "ARG2_FI1160_PV",
            "ARG2_FIC1210_PV",
            "ARG2_FIC1203_PV",
            "ARG2_FI1405_PV",
            "ARG2_FI1314_PV",
            "ARG2_FI1312_PV",
            "ARG2_PI1308_PV",
            "ARG2_TI1304_PV",
        ],
    },
    "qiya": {
        "display_name": "蒸气压",
        "engine": "xgboost",
        "default_cycle": "10d",
        "active_model_dir": "runtime/models/qiya/active",
        "feature_adapter": "qiya_adapter",
        "min_points": 20,
    },
    "C5": {
        "display_name": "C5",
        "engine": "xgboost",
        "default_cycle": "none",
        "active_model_dir": "runtime/models/C5/active",
        "feature_adapter": "C5_adapter",
        "min_points": 20,
    },
}

_CYCLE_ENV_KEYS = {
    "bingxi": "ROLLING_CYCLE_BINGXI",
    "qiya": "ROLLING_CYCLE_QIYA",
    "C5": "ROLLING_CYCLE_C5",
    "gandian": "ROLLING_CYCLE_GANDIAN",
}


def get_rolling_cycle(target: str, default: str) -> str:
    key = _CYCLE_ENV_KEYS.get(target)
    cycle = os.getenv(key, default).strip() if key else default
    if cycle not in VALID_CYCLES:
        raise ValueError(f"INVALID_CYCLE: {cycle}")
    return cycle


def get_stale_threshold_days() -> int:
    raw_value = os.getenv("OPC_STALE_THRESHOLD_DAYS", "7").strip()
    try:
        days = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"INVALID_STALE_THRESHOLD_DAYS: {raw_value}") from exc
    if days < 1:
        raise ValueError(f"INVALID_STALE_THRESHOLD_DAYS: {raw_value}")
    return days
