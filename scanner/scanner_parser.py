from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

import pandas as pd


TARGET_SPECS = [
    ("bingxi", "bx_time", "bx_entry", "丙烯"),
    ("gandian", "pr_time", "pr_entry", "终馏点"),
    ("qiya", "tp_time", "tp_entry", "蒸气压"),
    ("C5", "c5_time", "c5_entry", "碳五及以上"),
]


@dataclass(frozen=True)
class ScanRecord:
    raw_text: str
    normalized_text: str
    fields: Dict[str, str]


def normalize_text(text: str) -> str:
    lines = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", "", line)
        lines.append(line)
    return "\n".join(lines)


def parse_scan_text(text: str) -> ScanRecord:
    raw_text = str(text).strip()
    normalized_text = normalize_text(raw_text)
    fields: dict[str, str] = {}

    for line in normalized_text.split("\n"):
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            fields[key] = value

    required_keys = {"bx_entry", "bx_time", "c5_entry", "c5_time", "pr_entry", "pr_time", "tp_entry", "tp_time"}
    missing = sorted(required_keys - set(fields))
    if missing:
        raise ValueError(f"二维码字段不完整，缺少: {', '.join(missing)}")

    return ScanRecord(raw_text=raw_text, normalized_text=normalized_text, fields=fields)


def normalize_lab_value(value: str):
    text = str(value).strip()
    if not text:
        return text
    if text.startswith(("≤", "≥")):
        text = text[1:]
    text = text.replace("%", "").replace("℃", "").replace("kPa", "").strip()
    match = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    if not match:
        return value
    number = float(match.group(0))
    return int(number) if number.is_integer() else number


def sample_time_for_target(record: ScanRecord, target: str):
    for spec_target, time_key, _, _ in TARGET_SPECS:
        if spec_target == target:
            ts = pd.to_datetime(record.fields.get(time_key), errors="coerce")
            if pd.isna(ts):
                raise ValueError(f"二维码中没有可用的采样时间: {time_key}")
            return ts
    raise ValueError(f"未知目标: {target}")


def scan_payloads(record: ScanRecord):
    payloads = []
    for target, time_key, value_key, _ in TARGET_SPECS:
        sample_time = pd.to_datetime(record.fields.get(time_key), errors="coerce")
        if pd.isna(sample_time):
            raise ValueError(f"二维码中没有可用的采样时间: {time_key}")
        payloads.append(
            {
                "target": target,
                "sample_time": sample_time,
                "value": normalize_lab_value(record.fields[value_key]),
            }
        )
    return payloads
