from __future__ import annotations

from app.services.storage_service import append_lab_records
from scanner_parser import scan_payloads


def persist_scan_record(record):
    results = []
    for payload in scan_payloads(record):
        written_paths = append_lab_records(
            payload["target"],
            [
                {
                    "sample_time": payload["sample_time"],
                    "value": payload["value"],
                }
            ],
        )
        results.append(
            {
                "target": payload["target"],
                "sample_time": payload["sample_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "value": payload["value"],
                "written_paths": written_paths,
            }
        )
    return results
