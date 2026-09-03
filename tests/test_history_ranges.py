import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from app.services import storage_service


class HistoryRangeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temp_dir.name) / "runtime"
        self.runtime_patch = mock.patch.object(storage_service, "RUNTIME_DIR", self.runtime_dir)
        self.runtime_patch.start()

    def tearDown(self):
        self.runtime_patch.stop()
        self.temp_dir.cleanup()

    def _write_csv(self, relative_path, rows):
        path = self.runtime_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_excel(self, relative_path, rows):
        path = self.runtime_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_excel(path, index=False)
        return path

    def test_operation_range_reads_overlapping_partitions_and_one_preceding_record(self):
        q1 = self._write_csv(
            "data/operation/demo/2025-Q1.csv",
            [{"TimeStamp": "2025-03-31 23:59:00", "x": 1}],
        )
        q2 = self._write_csv(
            "data/operation/demo/2025-Q2.csv",
            [
                {"TimeStamp": "2025-04-15 00:00:00", "x": 2},
                {"TimeStamp": "2025-06-30 23:59:00", "x": 3},
            ],
        )
        q3 = self._write_csv(
            "data/operation/demo/2025-Q3.csv",
            [
                {"TimeStamp": "2025-07-01 00:00:00", "x": 4},
                {"TimeStamp": "2025-07-02 00:00:00", "x": 5},
            ],
        )
        q4 = self._write_csv(
            "data/operation/demo/2025-Q4.csv",
            [{"TimeStamp": "2025-10-01 00:00:00", "x": 6}],
        )
        original_read_csv = storage_service._read_csv
        opened = []

        def tracked_read(path):
            opened.append(path)
            return original_read_csv(path)

        with mock.patch.object(storage_service, "_read_csv", side_effect=tracked_read):
            result = storage_service.load_operation_history(
                "demo",
                start_time="2025-04-01 00:00:00",
                end_time="2025-07-01 00:00:00",
                include_preceding=True,
            )

        self.assertEqual(result["x"].tolist(), [1, 2, 3, 4])
        self.assertEqual(set(opened), {q1, q2, q3})
        self.assertNotIn(q4, opened)

    def test_lab_range_is_inclusive_and_uses_nearest_preceding_record(self):
        self._write_excel(
            "data/lab/demo/2025-Q1.xlsx",
            [{"sample_time": "2025-03-01 00:00:00", "value": 1.0}],
        )
        self._write_excel(
            "data/lab/demo/2025-Q2.xlsx",
            [
                {"sample_time": "2025-04-01 00:00:00", "value": 2.0},
                {"sample_time": "2025-04-10 00:00:00", "value": 3.0},
                {"sample_time": "2025-04-20 00:00:00", "value": 4.0},
            ],
        )

        result = storage_service.load_lab_history(
            "demo",
            start_time="2025-04-10 00:00:00",
            end_time="2025-04-20 00:00:00",
            include_preceding=True,
        )

        self.assertEqual(result["value"].tolist(), [2.0, 3.0, 4.0])

    def test_prediction_range_is_inclusive(self):
        self._write_csv(
            "predictions/demo/2025-01.csv",
            [{"prediction_time": "2025-01-31 23:59:00", "prediction": 1.0}],
        )
        self._write_csv(
            "predictions/demo/2025-02.csv",
            [
                {"prediction_time": "2025-02-01 00:00:00", "prediction": 2.0},
                {"prediction_time": "2025-02-28 23:59:00", "prediction": 3.0},
            ],
        )

        result = storage_service.load_prediction_history(
            "demo",
            start_time="2025-02-01 00:00:00",
            end_time="2025-02-28 23:59:00",
        )

        self.assertEqual(result["prediction"].tolist(), [2.0, 3.0])

    def test_latest_valid_prediction_stops_after_first_partition_with_value(self):
        oldest = self._write_csv(
            "predictions/demo/2025-01.csv",
            [{"prediction_time": "2025-01-01", "prediction": 1.0}],
        )
        previous = self._write_csv(
            "predictions/demo/2025-02.csv",
            [{"prediction_time": "2025-02-01", "prediction": 2.0}],
        )
        latest = self._write_csv(
            "predictions/demo/2025-03.csv",
            [{"prediction_time": "2025-03-01", "prediction": float("inf")}],
        )
        original_read_csv = storage_service._read_csv
        opened = []

        def tracked_read(path):
            opened.append(path)
            return original_read_csv(path)

        with mock.patch.object(storage_service, "_read_csv", side_effect=tracked_read):
            value = storage_service.load_latest_valid_prediction("demo")

        self.assertEqual(value, 2.0)
        self.assertEqual(opened, [latest, previous])
        self.assertNotIn(oldest, opened)

    def test_latest_valid_lab_stops_after_first_partition_with_value(self):
        oldest = self._write_excel(
            "data/lab/demo/2025-Q1.xlsx",
            [{"sample_time": "2025-01-01", "value": 1.0}],
        )
        latest = self._write_excel(
            "data/lab/demo/2025-Q2.xlsx",
            [{"sample_time": "2025-04-01", "value": 2.0}],
        )
        original_read_excel = storage_service._read_excel
        opened = []

        def tracked_read(path):
            opened.append(path)
            return original_read_excel(path)

        with mock.patch.object(storage_service, "_read_excel", side_effect=tracked_read):
            value = storage_service.load_latest_valid_lab("demo")

        self.assertEqual(value, 2.0)
        self.assertEqual(opened, [latest])
        self.assertNotIn(oldest, opened)


if __name__ == "__main__":
    unittest.main()
