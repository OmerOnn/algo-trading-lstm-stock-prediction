import unittest

import numpy as np

from src.backtest import BacktestConfig, build_signal_frame, cost_aware_signal_threshold


class BacktestRegressionTest(unittest.TestCase):
    def test_signal_frame_derives_signals_from_predicted_return(self):
        cfg = BacktestConfig(
            transaction_cost_pct=0.001,
            slippage_pct=0.0005,
            signal_threshold_multiplier=1.0,
            min_signal_edge=0.0,
        )
        metadata = [
            {"ticker": "AAA", "date": "2024-01-01"},
            {"ticker": "BBB", "date": "2024-01-02"},
            {"ticker": "CCC", "date": "2024-01-03"},
        ]
        signal_df = build_signal_frame(
            metadata=metadata,
            true_return=np.array([0.01, -0.02, 0.0001]),
            predicted_return=np.array([0.01, -0.02, 0.0001]),
            cfg=cfg,
        )

        threshold = cost_aware_signal_threshold(cfg)
        self.assertAlmostEqual(signal_df["signal_threshold"].iloc[0], threshold)
        self.assertEqual(signal_df["predicted_signal"].tolist(), ["BUY", "SELL", "HOLD"])


if __name__ == "__main__":
    unittest.main()
