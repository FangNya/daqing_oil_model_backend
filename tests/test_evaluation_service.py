import unittest
from unittest.mock import patch

import pandas as pd

from app.services import evaluation_service as service
from app.services.errors import BackendError


class PredictionLabMatchingTests(unittest.TestCase):
    def predictions(self, times, values=None, created_at=None):
        frame = pd.DataFrame({
            "prediction_time": pd.to_datetime(times),
            "prediction": values if values is not None else list(range(len(times))),
        })
        if created_at is not None:
            frame["created_at"] = created_at
        return frame

    def labs(self, times, values=None):
        return pd.DataFrame({
            "sample_time": pd.to_datetime(times),
            "value": values if values is not None else [10.0] * len(times),
        })

    def matched(self, predictions, labs, start=None, end=None):
        with patch.object(service, "load_prediction_history", return_value=predictions), patch.object(
            service, "load_lab_history", return_value=labs
        ):
            return service.matched_active_predictions("qiya", start, end)

    def test_exact_and_nearby_predictions_on_both_sides(self):
        for seconds in [0, -7, 24, -120, 120]:
            with self.subTest(seconds=seconds):
                sample = pd.Timestamp("2026-10-23 08:00:00")
                result = self.matched(
                    self.predictions([sample + pd.Timedelta(seconds=seconds)]), self.labs([sample])
                )
                self.assertEqual(len(result), 1)

    def test_predictions_outside_tolerance_are_excluded(self):
        for seconds in [-121, 121]:
            with self.subTest(seconds=seconds):
                sample = pd.Timestamp("2026-10-23 08:00:00")
                result = self.matched(
                    self.predictions([sample + pd.Timedelta(seconds=seconds)]), self.labs([sample])
                )
                self.assertTrue(result.empty)

    def test_nearest_prediction_wins_and_each_lab_is_counted_once(self):
        result = self.matched(
            self.predictions(["2026-10-23 07:59:30", "2026-10-23 08:00:10", "2026-10-23 08:00:45"], [1, 2, 3]),
            self.labs(["2026-10-23 08:00:00"]),
        )
        self.assertEqual(result["prediction"].tolist(), [2])

    def test_equally_near_predictions_choose_the_earlier_one(self):
        result = self.matched(
            self.predictions(["2026-10-23 08:00:30", "2026-10-23 07:59:30"], [2, 1]),
            self.labs(["2026-10-23 08:00:00"]),
        )
        self.assertEqual(result["prediction"].tolist(), [1])

    def test_repeated_predictions_choose_latest_created_at_not_input_order(self):
        result = self.matched(
            self.predictions(
                ["2026-10-23 07:59:53"] * 3, [2, 1, 9],
                ["2026-10-23T00:00:02+00:00", "2026-10-23T00:00:01+00:00", "invalid"],
            ),
            self.labs(["2026-10-23 08:00:00"]),
        )
        self.assertEqual(result["prediction"].tolist(), [2])
        self.assertNotIn("_match_created_at", result.columns)

    def test_missing_created_at_uses_last_stored_prediction(self):
        result = self.matched(
            self.predictions(["2026-10-23 07:59:53"] * 2, [1, 2]),
            self.labs(["2026-10-23 08:00:00"]),
        )
        self.assertEqual(result["prediction"].tolist(), [2])

    def test_repeated_lab_samples_are_not_double_counted(self):
        result = self.matched(
            self.predictions(["2026-10-23 07:59:53"], [1]),
            self.labs(["2026-10-23 08:00:00"] * 2, [10, 11]),
        )
        self.assertEqual(result["value"].tolist(), [11])

    def test_cycle_bounds_apply_to_labs_and_allow_cross_month_prediction(self):
        start, end = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02")
        calls = []
        predictions = self.predictions(["2026-09-30 23:59:53", "2026-10-02 00:00:30"], [1, 2])
        labs = self.labs(["2026-09-30 23:59:59", "2026-10-01 00:00:00", "2026-10-01 23:59:59", "2026-10-02 00:00:00"])

        def load_predictions(target, start_time=None, end_time=None):
            calls.append((target, start_time, end_time))
            return predictions[(predictions["prediction_time"] >= start_time) & (predictions["prediction_time"] <= end_time)]

        def load_labs(target, start_time=None, end_time=None):
            return labs[(labs["sample_time"] >= start_time) & (labs["sample_time"] <= end_time)]

        with patch.object(service, "load_prediction_history", load_predictions), patch.object(service, "load_lab_history", load_labs):
            result = service.matched_active_predictions("qiya", start, end)
        self.assertEqual(result["sample_time"].tolist(), [start, end - pd.Timedelta(seconds=1)])
        self.assertEqual(calls, [("qiya", start - pd.Timedelta(minutes=2), end + pd.Timedelta(minutes=2))])

    def test_single_file_patch_supports_older_storage_functions(self):
        predictions = self.predictions(["2026-10-23 07:59:53"])
        labs = self.labs(["2026-10-23 08:00:00"])

        def old_predictions(target):
            return predictions

        def old_labs(target):
            return labs

        with patch.object(service, "load_prediction_history", old_predictions), patch.object(service, "load_lab_history", old_labs):
            result = service.matched_active_predictions("qiya", "2026-10-23", "2026-10-24")
        self.assertEqual(len(result), 1)

    def test_empty_predictions_labs_and_period_return_no_samples(self):
        predictions = self.predictions(["2026-10-23 07:59:53"])
        labs = self.labs(["2026-10-23 08:00:00"])
        for pred, lab, start, end in [
            (predictions.iloc[:0], labs, None, None),
            (predictions, labs.iloc[:0], None, None),
            (predictions, labs, "2026-10-24", "2026-10-25"),
        ]:
            with self.subTest(start=start, end=end):
                self.assertTrue(self.matched(pred, lab, start, end).empty)

    def test_evaluation_uses_matched_pairs_for_metrics_and_saves_once(self):
        with patch.object(service, "load_prediction_history", return_value=self.predictions(
            ["2026-10-22 07:59:36", "2026-10-23 07:59:53"], [11, 12]
        )), patch.object(service, "load_lab_history", return_value=self.labs(
            ["2026-10-22 08:00:00", "2026-10-23 08:00:00"], [10, 10]
        )), patch.object(service, "save_evaluation_record") as save:
            result = service.evaluate_active("qiya", "10d", "2026-10-22", "2026-10-24")
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["mae"], 1.5)
        save.assert_called_once_with("qiya", result)

    def test_no_match_keeps_existing_skip_error_and_does_not_save(self):
        with patch.object(service, "load_prediction_history", return_value=self.predictions(
            ["2026-10-23 07:50:00"]
        )), patch.object(service, "load_lab_history", return_value=self.labs(
            ["2026-10-23 08:00:00"]
        )), patch.object(service, "save_evaluation_record") as save:
            with self.assertRaises(BackendError) as raised:
                service.evaluate_active("qiya", "10d")
        self.assertEqual(raised.exception.code, "UPDATE_SKIPPED_INSUFFICIENT_LABELS")
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
