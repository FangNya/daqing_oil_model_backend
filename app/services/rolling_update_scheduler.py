from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone

from app.config import RUNTIME_DIR
from app.services.rolling_update_service import run_daily_check


logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 24 * 60 * 60
_DEFAULT_INITIAL_DELAY_SECONDS = 60
_FALSE_VALUES = {"0", "false", "no", "off"}

_lock = threading.Lock()
_stop_event = threading.Event()
_thread: threading.Thread | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in _FALSE_VALUES


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value.strip())
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw_value, default)
        return default
    if value < minimum:
        logger.warning("Invalid %s=%r, using default %s", name, raw_value, default)
        return default
    return value


def _write_scheduler_log(message: str, extra: dict | None = None):
    try:
        log_dir = RUNTIME_DIR / "logs" / "rolling_update"
        log_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": "scheduler",
            "message": message,
            **(extra or {}),
        }
        with open(log_dir / "scheduler.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write rolling update scheduler log", exc_info=True)


def _run_loop(interval_seconds: int, initial_delay_seconds: int):
    _write_scheduler_log(
        "scheduler started",
        {
            "interval_seconds": interval_seconds,
            "initial_delay_seconds": initial_delay_seconds,
        },
    )
    if _stop_event.wait(initial_delay_seconds):
        _write_scheduler_log("scheduler stopped before first check")
        return

    while not _stop_event.is_set():
        try:
            results = run_daily_check()
            _write_scheduler_log("daily check completed", {"results": results})
        except Exception as exc:
            logger.exception("Rolling update daily check failed")
            _write_scheduler_log("daily check failed", {"error": repr(exc)})

        if _stop_event.wait(interval_seconds):
            break

    _write_scheduler_log("scheduler stopped")


def start_rolling_update_scheduler() -> bool:
    if not _env_bool("ROLLING_UPDATE_ENABLED", True):
        _write_scheduler_log("scheduler disabled by environment")
        return False

    interval_seconds = _env_int("ROLLING_UPDATE_INTERVAL_SECONDS", _DEFAULT_INTERVAL_SECONDS, minimum=1)
    initial_delay_seconds = _env_int(
        "ROLLING_UPDATE_INITIAL_DELAY_SECONDS",
        _DEFAULT_INITIAL_DELAY_SECONDS,
        minimum=0,
    )

    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _stop_event.clear()
        _thread = threading.Thread(
            target=_run_loop,
            args=(interval_seconds, initial_delay_seconds),
            name="RollingUpdateScheduler",
            daemon=True,
        )
        _thread.start()
        return True


def stop_rolling_update_scheduler(timeout: float = 5.0):
    _stop_event.set()
    with _lock:
        thread = _thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
