from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import time

import serial

from scanner_config import load_settings
from scanner_lab_writer import persist_scan_record
from scanner_logger import log_event
from scanner_parser import ScanRecord, parse_scan_text


class Scanner:
    START = bytes([0x16, 0x54, 0x0D])
    STOP = bytes([0x16, 0x55, 0x0D])

    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port, baud, timeout=0.2)

    def start(self):
        self.ser.write(self.START)
        self.ser.flush()

    def stop(self):
        self.ser.write(self.STOP)
        self.ser.flush()

    def read(self, timeout: float):
        begin = time.time()
        buffer = bytearray()
        last_recv = None

        while True:
            now = time.time()
            if now - begin > timeout:
                break

            waiting = self.ser.in_waiting
            if waiting:
                buffer.extend(self.ser.read(waiting))
                last_recv = now

            if last_recv and now - last_recv > 0.3:
                break

            time.sleep(0.01)

        if not buffer:
            return None
        return buffer.decode("utf-8", errors="ignore").strip()

    def scan_once(self, timeout: float):
        try:
            self.start()
            return self.read(timeout)
        finally:
            self.stop()

    def close(self):
        self.ser.close()


def wait_first_qr(scanner: Scanner, boot_timeout: float, boot_interval: float) -> ScanRecord:
    log_event("scanner_wait", message="等待首次二维码")

    while True:
        result = scanner.scan_once(boot_timeout)
        if result:
            try:
                record = parse_scan_text(result)
            except ValueError as exc:
                log_event("scanner_parse_failed", level="warning", error=str(exc), raw_text=result)
                continue

            print("首次识别成功：")
            print(record.raw_text)
            log_event("scanner_first_ok", raw_text=record.raw_text, normalized_text=record.normalized_text)
            return record

        print("未识别，继续等待...")
        log_event("scanner_wait_retry", message="未识别，继续等待")
        time.sleep(boot_interval)


def confirm_scan(scanner: Scanner, first_record: ScanRecord, scan_timeout: float, scan_interval: float, confirm_count: int):
    current = first_record.normalized_text
    count = 1

    while count < confirm_count:
        time.sleep(scan_interval)
        result = scanner.scan_once(scan_timeout)
        if not result:
            print("本轮无二维码，确认计数重置")
            log_event("scanner_confirm_reset", reason="empty_scan", raw_text=first_record.raw_text)
            return None

        try:
            record = parse_scan_text(result)
        except ValueError as exc:
            print("解析失败，确认计数重置：", exc)
            log_event("scanner_confirm_reset", reason="parse_failed", error=str(exc), raw_text=result)
            return None

        if record.normalized_text != current:
            print("二维码内容变化，确认计数重置")
            log_event(
                "scanner_confirm_reset",
                reason="content_changed",
                previous=current,
                current=record.normalized_text,
            )
            return None

        count += 1
        print(f"确认通过 {count}/{confirm_count}")
        log_event("scanner_confirm_progress", count=count, confirm_count=confirm_count)

    return first_record


def monitor(scanner: Scanner, settings):
    first_record = wait_first_qr(scanner, settings.boot_timeout, settings.boot_interval)
    last = first_record.normalized_text

    print("\n进入定时扫描模式")
    log_event("scanner_monitor_start", normalized_text=last)

    while True:
        time.sleep(settings.scan_interval)

        result = scanner.scan_once(settings.scan_timeout)
        if not result:
            print("本轮无二维码")
            log_event("scanner_empty_scan")
            continue

        try:
            record = parse_scan_text(result)
        except ValueError as exc:
            print("解析失败：", exc)
            log_event("scanner_parse_failed", level="warning", error=str(exc), raw_text=result)
            continue

        if record.normalized_text == last:
            print("二维码未更新")
            log_event("scanner_duplicate_scan", normalized_text=record.normalized_text)
            continue

        print("\n检测到新二维码，开始连续确认：")
        log_event("scanner_new_qr", raw_text=record.raw_text, normalized_text=record.normalized_text)

        confirmed = confirm_scan(
            scanner,
            record,
            settings.scan_timeout,
            settings.scan_interval,
            settings.confirm_count,
        )
        if not confirmed:
            continue

        last = confirmed.normalized_text

        print("确认通过，准备写入：")
        print(confirmed.raw_text)
        log_event("scanner_confirmed", raw_text=confirmed.raw_text, normalized_text=confirmed.normalized_text)

        try:
            result_info = persist_scan_record(confirmed)
            print("写入成功：", result_info)
            log_event("scanner_write_success", result=result_info)
        except Exception as exc:  # noqa: BLE001
            print("写入失败：", exc)
            log_event("scanner_write_failed", level="error", error=str(exc), raw_text=confirmed.raw_text)


def main():
    settings = load_settings()
    scanner = Scanner(settings.port, settings.baud)

    try:
        monitor(scanner, settings)
    finally:
        scanner.close()


if __name__ == "__main__":
    main()
