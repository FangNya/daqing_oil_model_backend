from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"


def _load_env_file(path: Path = ENV_PATH):
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_ENV = _load_env_file()


def _get(name: str, default: str) -> str:
    return _ENV.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"INVALID_SCANNER_ENV: {name}={raw}") from exc


def _get_float(name: str, default: float) -> float:
    raw = _get(name, str(default))
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
    confirm_interval: float
    confirm_count: int
    log_dir: Path


def load_settings() -> ScannerSettings:
    from app.config import RUNTIME_DIR

    return ScannerSettings(
        port=_get("SCANNER_PORT", "COM3"),
        baud=_get_int("SCANNER_BAUD", 9600),
        boot_interval=_get_float("SCANNER_BOOT_INTERVAL", 3.0),
        boot_timeout=_get_float("SCANNER_BOOT_TIMEOUT", 2.0),
        scan_interval=_get_float("SCANNER_SCAN_INTERVAL", 5.0),
        scan_timeout=_get_float("SCANNER_SCAN_TIMEOUT", 2.0),
        confirm_interval=_get_float("SCANNER_CONFIRM_INTERVAL", 5.0),
        confirm_count=_get_int("SCANNER_CONFIRM_COUNT", 3),
        log_dir=Path(_get("SCANNER_LOG_DIR", str(RUNTIME_DIR / "logs" / "scanner"))),
    )
