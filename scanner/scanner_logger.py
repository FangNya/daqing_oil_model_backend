from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scanner_config import load_settings


def _log_path() -> Path:
    settings = load_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings.log_dir / "scanner.jsonl"


def log_event(event: str, level: str = "info", **payload):
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **payload,
    }
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
