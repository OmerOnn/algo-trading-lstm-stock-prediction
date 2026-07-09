import unittest
from unittest.mock import patch

from train_all_models import train_all_models


class TrainingWorkflowTest(unittest.TestCase):
    def test_trains_all_configured_horizons_without_cli_filter(self):
        config = {
            "prediction_horizons": [1, 5, 21],
            "prediction_horizon": 21,
            "device": "mps",
            "tickers": ["AAPL"],
            "benchmark_ticker": "SPY",
            "start_date": "2018-01-01",
            "end_date": None,
            "macro_tickers": {},
        }

        device_info = type(
            "DeviceInfoStub",
            (),
            {
                "device": type("DeviceStub", (), {"type": "mps", "__str__": lambda self: "mps"})(),
                "device_name": "Apple Silicon GPU",
                "accelerator": "MPS / Apple Metal GPU",
            },
        )()

        with patch("train_all_models.load_config", return_value=config), patch(
            "train_all_models.get_best_device",
            return_value=device_info,
        ), patch(
            "train_all_models.resolve_xgboost_backend",
            return_value=("cpu", "Apple Metal GPU is not supported by XGBoost, so it is running on CPU."),
        ), patch("train_all_models.preload_training_data") as preload_training_data, patch(
            "train_all_models.train_lstm_models_for_horizons"
        ) as train_lstm_models_for_horizons, patch(
            "train_all_models.train_xgboost_models_for_horizons"
        ) as train_xgboost_models_for_horizons:
            horizons = train_all_models()

        self.assertEqual(horizons, [1, 5, 21])
        preload_training_data.assert_called_once()
        train_lstm_models_for_horizons.assert_called_once_with(config, [1, 5, 21])
        train_xgboost_models_for_horizons.assert_called_once_with(config, [1, 5, 21])

    def test_trains_single_horizon_and_runs_comparisons_when_requested(self):
        config = {
            "prediction_horizons": [1, 5, 21],
            "prediction_horizon": 21,
            "device": "cpu",
            "tickers": ["AAPL"],
            "benchmark_ticker": "SPY",
            "start_date": "2018-01-01",
            "end_date": None,
            "macro_tickers": {},
        }

        device_info = type(
            "DeviceInfoStub",
            (),
            {
                "device": type("DeviceStub", (), {"type": "cpu", "__str__": lambda self: "cpu"})(),
                "device_name": "CPU",
                "accelerator": "CPU",
            },
        )()

        with patch("train_all_models.load_config", return_value=config), patch(
            "train_all_models.get_best_device",
            return_value=device_info,
        ), patch(
            "train_all_models.resolve_xgboost_backend",
            return_value=("cuda", "CUDA GPU acceleration is active for XGBoost."),
        ), patch("train_all_models.preload_training_data"), patch(
            "train_all_models.train_lstm_models_for_horizons"
        ), patch(
            "train_all_models.train_xgboost_models_for_horizons"
        ), patch("compare_results.main") as compare_results_main:
            horizons = train_all_models(selected_horizon=5, compare=True)

        self.assertEqual(horizons, [5])
        compare_results_main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
