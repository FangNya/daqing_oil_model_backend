import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from app.services import evaluation_service, prediction_service, rolling_update_service


class PredictionHistoryWindowTests(unittest.TestCase):
    def test_prediction_loads_only_model_window(self):
        prediction_time = pd.Timestamp("2026-09-03 12:00:00")
        adapter = SimpleNamespace(
            op_feature_columns=["x"],
            delay_hours=1,
            window_hours=2,
            build_predict_input=mock.Mock(return_value=np.zeros((1, 1))),
        )
        model = SimpleNamespace(predict=mock.Mock(return_value=12.5))
        lab_history = pd.DataFrame(
            [{"sample_time": "2026-09-03 08:59:00", "value": 10.0}]
        )
        operation_history = pd.DataFrame(
            [{"TimeStamp": "2026-09-03 08:59:00", "x": 1.0}]
        )
        payload = {
            "operation_records": [
                {"TimeStamp": prediction_time.strftime("%Y-%m-%d %H:%M:%S"), "x": 2.0}
            ]
        }

        with (
            mock.patch.object(prediction_service, "resolve_cycle", return_value="10d"),
            mock.patch.object(
                prediction_service,
                "get_target_config",
                return_value={"engine": "informer", "active_model_dir": "unused"},
            ),
            mock.patch.object(prediction_service, "get_stale_threshold_days", return_value=3650),
            mock.patch.object(prediction_service, "_load_adapter", return_value=adapter),
            mock.patch.object(prediction_service, "append_lab_records"),
            mock.patch.object(prediction_service, "load_lab_history", return_value=lab_history) as load_lab,
            mock.patch.object(
                prediction_service,
                "load_operation_history",
                return_value=operation_history,
            ) as load_operation,
            mock.patch.object(prediction_service, "_load_model", return_value=model),
            mock.patch.object(prediction_service, "_append_operation_records_or_error"),
            mock.patch.object(prediction_service, "save_prediction_record"),
        ):
            result = prediction_service.predict_target("bingxi", payload)

        expected_start = prediction_time - pd.Timedelta(hours=3)
        load_lab.assert_called_once_with(
            "bingxi",
            start_time=expected_start,
            end_time=prediction_time,
            include_preceding=True,
        )
        load_operation.assert_called_once_with(
            "bingxi",
            start_time=expected_start,
            end_time=prediction_time - pd.Timedelta(hours=1),
            include_preceding=True,
        )
        self.assertEqual(result["prediction"], 12.5)


class RollingHistoryWindowTests(unittest.TestCase):
    def test_training_history_range_comes_from_labels_and_adapter(self):
        labels = pd.DataFrame(
            {
                "sample_time": ["2026-09-01 12:00:00", "2026-09-03 12:00:00"],
                "value": [1.0, 2.0],
            }
        )
        adapter = SimpleNamespace(delay_hours=3, window_hours=2)

        with (
            mock.patch.object(
                rolling_update_service,
                "load_operation_history",
                return_value=pd.DataFrame(),
            ) as load_operation,
            mock.patch.object(
                rolling_update_service,
                "load_lab_history",
                return_value=pd.DataFrame(),
            ) as load_lab,
        ):
            rolling_update_service._load_training_histories("gandian", adapter, labels)

        expected_start = pd.Timestamp("2026-09-01 07:00:00")
        expected_end = pd.Timestamp("2026-09-03 09:00:00")
        load_operation.assert_called_once_with(
            "gandian",
            start_time=expected_start,
            end_time=expected_end,
            include_preceding=True,
        )
        load_lab.assert_called_once_with(
            "gandian",
            start_time=expected_start,
            end_time=expected_end,
            include_preceding=True,
        )

    def test_training_history_range_includes_adapter_warmup(self):
        labels = pd.DataFrame({"sample_time": ["2026-09-03 12:00:00"], "value": [1.0]})
        adapter = SimpleNamespace(delay_hours=1, window_hours=1, history_warmup_hours=168)

        with (
            mock.patch.object(
                rolling_update_service,
                "load_operation_history",
                return_value=pd.DataFrame(),
            ) as load_operation,
            mock.patch.object(
                rolling_update_service,
                "load_lab_history",
                return_value=pd.DataFrame(),
            ),
        ):
            rolling_update_service._load_training_histories("qiya", adapter, labels)

        load_operation.assert_called_once_with(
            "qiya",
            start_time=pd.Timestamp("2026-08-27 10:00:00"),
            end_time=pd.Timestamp("2026-09-03 11:00:00"),
            include_preceding=True,
        )

    def test_none_cycle_evaluates_only_latest_30_days(self):
        now = pd.Timestamp("2026-09-03 12:00:00")

        with (
            mock.patch.object(
                rolling_update_service,
                "get_target_config",
                return_value={"default_cycle": "none"},
            ),
            mock.patch.object(rolling_update_service, "evaluate_active") as evaluate,
        ):
            result = rolling_update_service.maybe_run_target_update("C5", now=now)

        evaluate.assert_called_once_with("C5", "none", now - pd.Timedelta(days=30), now)
        self.assertEqual(result["status"], "skipped")


class EvaluationHistoryWindowTests(unittest.TestCase):
    def test_evaluation_loads_range_and_excludes_end_boundary(self):
        start = pd.Timestamp("2026-08-01 00:00:00")
        end = pd.Timestamp("2026-09-01 00:00:00")
        predictions = pd.DataFrame(
            {
                "prediction_time": [start, end],
                "prediction": [1.0, 2.0],
            }
        )
        labs = pd.DataFrame(
            {
                "sample_time": [start, end],
                "value": [1.5, 2.5],
            }
        )

        with (
            mock.patch.object(
                evaluation_service,
                "load_prediction_history",
                return_value=predictions,
            ) as load_predictions,
            mock.patch.object(evaluation_service, "load_lab_history", return_value=labs) as load_labs,
        ):
            result = evaluation_service.matched_active_predictions("demo", start, end)

        load_predictions.assert_called_once_with(
            "demo",
            start_time=start - evaluation_service.PREDICTION_MATCH_TOLERANCE,
            end_time=end + evaluation_service.PREDICTION_MATCH_TOLERANCE,
        )
        load_labs.assert_called_once_with("demo", start_time=start, end_time=end)
        self.assertEqual(result["prediction_time"].tolist(), [start])


if __name__ == "__main__":
    unittest.main()

