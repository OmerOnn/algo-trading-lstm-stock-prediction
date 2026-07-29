import unittest

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame, tune_signal_threshold
from src.dataset import StockSequenceDataset
from src.regression import add_model_target, decode_model_output, regression_metrics
from train import chronological_train_validation_test_split


class RegressionRedesignTest(unittest.TestCase):
    def test_volatility_scaled_target_decodes_to_percentage_return(self):
        frame = pd.DataFrame(
            {
                "future_return": [0.02, -0.04],
                "volatility_20d": [0.01, 0.02],
            }
        )
        transformed = add_model_target(
            frame,
            horizon=4,
            target_config={
                "mode": "volatility_scaled",
                "daily_volatility_floor": 0.001,
                "target_clip": 10.0,
            },
        )
        decoded = decode_model_output(
            transformed["model_target"].to_numpy(),
            transformed["target_scale"].to_numpy(),
        )
        np.testing.assert_allclose(decoded, frame["future_return"].to_numpy())

    def test_purged_split_keeps_labels_out_of_later_periods(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        frame = pd.DataFrame(
            {
                "Ticker": "AAA",
                "future_return": np.linspace(-0.1, 0.1, len(dates)),
            },
            index=dates,
        )
        train, validation, test = chronological_train_validation_test_split(
            frame,
            train_ratio=0.60,
            validation_ratio=0.20,
            purge_horizon=5,
        )
        self.assertLessEqual(dates.get_loc(train.index.max()) + 5, dates.get_loc(validation.index.min()) - 1)
        self.assertLessEqual(
            dates.get_loc(validation.index.max()) + 5,
            dates.get_loc(test.index.min()) - 1,
        )

    def test_sequence_training_window_includes_target_date_features(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        frame = pd.DataFrame(
            {
                "Ticker": "AAA",
                "feature": np.arange(5, dtype=float),
                "future_return": 0.01,
                "model_target": 0.5,
                "target_scale": 0.02,
            },
            index=dates,
        )
        dataset = StockSequenceDataset(frame, ["feature"], window_size=3)
        features, _, _, _ = dataset[0]
        np.testing.assert_allclose(features.numpy().ravel(), [0.0, 1.0, 2.0])
        self.assertEqual(dataset.metadata[0]["date"], str(dates[2].date()))

    def test_predictive_score_rewards_a_useful_regression_forecast(self):
        actual = np.asarray([-0.08, -0.02, 0.01, 0.05, 0.09])
        zero = regression_metrics(actual, np.zeros_like(actual))
        useful = regression_metrics(actual, actual * 0.8)
        self.assertGreater(useful["predictive_score"], zero["predictive_score"])
        self.assertGreater(useful["rank_information_coefficient"], 0.9)

    def test_backtest_uses_non_overlapping_horizon_returns(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        metadata = [{"ticker": "AAA", "date": str(date.date())} for date in dates]
        cfg = BacktestConfig()
        frame = build_signal_frame(
            metadata,
            true_return=np.full(10, 0.02),
            predicted_return=np.full(10, 0.02),
            cfg=cfg,
        )
        daily, metrics = backtest_signals(frame, cfg, horizon=3)
        self.assertEqual(len(daily), 4)
        self.assertTrue(metrics["overlapping_predictions_removed"])

    def test_threshold_is_tuned_on_validation_above_cost_floor(self):
        dates = pd.bdate_range("2024-01-01", periods=60)
        metadata = [{"ticker": "AAA", "date": str(date.date())} for date in dates]
        predicted = np.linspace(-0.04, 0.04, 60)
        actual = predicted + 0.002
        cfg = BacktestConfig(transaction_cost_pct=0.001, slippage_pct=0.0005)
        threshold, details = tune_signal_threshold(
            metadata,
            actual,
            predicted,
            cfg,
            horizon=1,
            min_active_trades=5,
        )
        self.assertGreaterEqual(threshold, details["minimum_cost_threshold"])
        self.assertEqual(details["selection_dataset"], "validation")


if __name__ == "__main__":
    unittest.main()
