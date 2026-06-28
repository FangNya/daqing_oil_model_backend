from __future__ import annotations

import os
from dataclasses import dataclass
from app.config import RUNTIME_DIR  # noqa: F401


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"INVALID_SCANNER_ENV: {name}={raw}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"INVALID_SCANNER_ENV: {name}={raw}") from exc


@dataclass(frozen=True)
class ScannerSettings:
    port: str
    baud: int
    boot_interval: float
    boot_timeout: float
    scan_interval: float
    scan_timeout: float
    confirm_count: int
    log_dir: str


def load_settings() -> ScannerSettings:
    return ScannerSettings(
        port=os.getenv("SCANNER_PORT", "COM3").strip(),
        baud=_env_int("SCANNER_BAUD", 9600),
        boot_interval=_env_float("SCANNER_BOOT_INTERVAL", 3.0),
        boot_timeout=_env_float("SCANNER_BOOT_TIMEOUT", 2.0),
        scan_interval=_env_float("SCANNER_SCAN_INTERVAL", 5.0),
        scan_timeout=_env_float("SCANNER_SCAN_TIMEOUT", 2.0),
        confirm_count=_env_int("SCANNER_CONFIRM_COUNT", 3),
        log_dir=os.getenv("SCANNER_LOG_DIR", str(RUNTIME_DIR / "logs" / "scanner")).strip(),
    )
