"""Tests for the uncertainty, decision and cross-sectional evaluation layers."""

import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame, tune_decision_config
from src.decision import DecisionConfig, decide_batch, edge_z_score, position_sizes
from src.features import get_feature_columns, is_market_wide_feature
from src.model import StockReturnPredictor
from src.regression import (
    add_model_target,
    compose_total_return,
    cross_sectional_metrics,
    estimate_market_drift,
    fit_return_calibration,
    full_metrics,
    target_component_column,
)
from src.uncertainty import (
    block_bootstrap_row_indices,
    describe_confidence,
    enable_mc_dropout,
    fit_interval_calibration,
    interval_metrics,
    mc_dropout_predict,
)
from src.validation import purged_walk_forward_splits


def build_panel(n_dates: int = 200, n_tickers: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    frame = pd.DataFrame(
        [{"Ticker": f"T{i}", "date": date} for i in range(n_tickers) for date in dates]
    ).set_index("date")
    rows = len(frame)
    frame["volatility_20d"] = 0.02
    frame["idiosyncratic_volatility_20d"] = 0.015
    frame["benchmark_future_return"] = rng.normal(0.01, 0.05, rows)
    frame["future_return"] = frame["benchmark_future_return"] + rng.normal(0.0, 0.06, rows)
    frame["return_20d"] = rng.normal(0.0, 0.08, rows)
    frame["excess_return_20d"] = rng.normal(0.0, 0.08, rows)
    return frame


class MarketExcessTargetTest(unittest.TestCase):
    def test_market_excess_target_removes_the_benchmark_leg(self):
        panel = build_panel()
        config = {"mode": "market_excess", "volatility_column": "idiosyncratic_volatility_20d"}
        transformed = add_model_target(panel, horizon=21, target_config=config)

        np.testing.assert_allclose(
            transformed["future_excess_return"].to_numpy(),
            (panel["future_return"] - panel["benchmark_future_return"]).to_numpy(),
        )
        self.assertEqual(target_component_column(config), "future_excess_return")
        # The excess leg must be less volatile than the raw return it came from.
        self.assertLess(
            transformed["future_excess_return"].std(), transformed["future_return"].std()
        )

    def test_market_excess_target_requires_the_benchmark_column(self):
        panel = build_panel().drop(columns=["benchmark_future_return"])
        with self.assertRaises(KeyError):
            add_model_target(panel, horizon=21, target_config={"mode": "market_excess"})

    def test_market_drift_is_estimated_once_per_date(self):
        panel = build_panel(n_dates=120, n_tickers=10)
        drift = estimate_market_drift(panel, {"mode": "market_excess"})
        self.assertEqual(drift["sample_size"], 120)
        self.assertAlmostEqual(
            drift["market_drift"],
            float(panel.groupby(panel.index)["benchmark_future_return"].first().mean()),
            places=10,
        )

    def test_total_return_is_the_component_plus_the_drift(self):
        np.testing.assert_allclose(
            compose_total_return(np.asarray([0.01, -0.02]), 0.008),
            np.asarray([0.018, -0.012]),
        )


class CrossSectionalMetricsTest(unittest.TestCase):
    def test_cross_sectional_ic_ignores_a_common_market_move(self):
        """A forecast that only predicts the market level has no ranking skill."""
        dates = pd.bdate_range("2021-01-01", periods=50).repeat(10)
        rng = np.random.default_rng(3)
        market = np.repeat(rng.normal(0, 0.05, 50), 10)
        truth = market + rng.normal(0, 0.03, len(dates))

        market_only = cross_sectional_metrics(dates, truth, market, horizon=21)
        self.assertLess(abs(market_only["mean_ic"]), 0.05)

        informative = cross_sectional_metrics(dates, truth, truth * 0.5, horizon=21)
        self.assertGreater(informative["mean_ic"], 0.9)

    def test_metrics_report_significance_and_quantile_spread(self):
        dates = pd.bdate_range("2021-01-01", periods=60).repeat(20)
        rng = np.random.default_rng(7)
        truth = rng.normal(0, 0.06, len(dates))
        prediction = 0.4 * truth + rng.normal(0, 0.04, len(dates))
        metrics = full_metrics(dates, truth, prediction, horizon=21)

        self.assertGreater(metrics["cross_sectional_ic"], 0.0)
        self.assertGreater(metrics["cross_sectional_ic_t_statistic"], 1.0)
        self.assertGreater(metrics["cross_sectional_long_short_spread_per_period"], 0.0)
        self.assertEqual(metrics["cross_sectional_evaluated_dates"], 60)


class CalibrationGuardRailTest(unittest.TestCase):
    def test_a_flattening_calibration_is_rejected(self):
        """The old bug: a negative least-squares slope clipped to zero, collapsing
        every prediction to a constant while improving MAE."""
        rng = np.random.default_rng(11)
        truth = rng.normal(0, 0.06, 2000)
        prediction = -0.3 * truth + rng.normal(0, 0.02, 2000)  # anti-correlated
        calibration = fit_return_calibration(truth, prediction)
        self.assertFalse(calibration["enabled"])
        self.assertEqual(calibration["slope"], 1.0)

    def test_a_genuine_magnitude_correction_is_accepted(self):
        rng = np.random.default_rng(13)
        truth = rng.normal(0, 0.06, 4000)
        prediction = 3.0 * truth  # right ranking, wrong scale
        calibration = fit_return_calibration(truth, prediction)
        self.assertTrue(calibration["enabled"])
        self.assertLess(calibration["slope"], 1.0)


class IntervalCalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.n = 4000
        self.scales = np.full(self.n, 0.06)
        self.truth = rng.normal(0, 0.06, self.n)
        self.prediction = 0.2 * self.truth + rng.normal(0, 0.01, self.n)
        self.model_std = np.full(self.n, 0.01)

    def test_validation_coverage_matches_the_nominal_level(self):
        calibration = fit_interval_calibration(
            self.truth, self.prediction, self.model_std, self.scales, confidence_level=0.80
        )
        self.assertAlmostEqual(calibration.validation_coverage, 0.80, places=2)

    def test_interval_widens_with_the_volatility_scale(self):
        calibration = fit_interval_calibration(
            self.truth, self.prediction, self.model_std, self.scales, confidence_level=0.80
        )
        narrow = calibration.sigma(np.asarray([0.01]), np.asarray([0.02]))
        wide = calibration.sigma(np.asarray([0.01]), np.asarray([0.10]))
        self.assertGreater(wide[0], narrow[0])

    def test_interval_metrics_penalise_an_overconfident_band(self):
        honest = interval_metrics(self.truth, self.prediction - 0.1, self.prediction + 0.1, 0.80)
        overconfident = interval_metrics(
            self.truth, self.prediction - 0.005, self.prediction + 0.005, 0.80
        )
        self.assertGreater(honest["coverage_picp"], overconfident["coverage_picp"])
        self.assertLess(honest["winkler_score"], overconfident["winkler_score"])

    def test_confidence_label_reflects_signal_to_noise(self):
        self.assertEqual(describe_confidence(0.05, 0.02)["confidence_label"], "High")
        self.assertEqual(describe_confidence(0.005, 0.09)["confidence_label"], "Low")
        self.assertGreater(describe_confidence(0.05, 0.02)["direction_probability"], 0.9)
        self.assertAlmostEqual(describe_confidence(0.0, 0.09)["direction_probability"], 0.5, places=3)


class MonteCarloDropoutTest(unittest.TestCase):
    def test_dropout_stays_stochastic_while_normalisation_does_not(self):
        model = StockReturnPredictor(input_size=6, hidden_size=8, dropout=0.5, input_dropout=0.5)
        enable_mc_dropout(model)
        self.assertTrue(any(m.training for m in model.modules() if isinstance(m, nn.Dropout)))
        self.assertFalse(model.norm.training)

    def test_repeated_passes_produce_a_positive_spread(self):
        torch.manual_seed(0)
        model = StockReturnPredictor(input_size=6, hidden_size=8, dropout=0.4, input_dropout=0.3)
        x = torch.randn(4, 10, 6)
        mean, std = mc_dropout_predict(model, x, passes=25)
        self.assertEqual(mean.shape, (4,))
        self.assertTrue(np.all(std > 0))
        # The helper must leave the model in eval mode for deterministic use.
        self.assertFalse(model.training)


class BlockBootstrapTest(unittest.TestCase):
    def test_resample_is_non_empty_and_keeps_whole_dates_together(self):
        panel = build_panel(n_dates=60, n_tickers=5)
        rng = np.random.default_rng(23)
        indices = block_bootstrap_row_indices(panel.index, rng, block_size=21)

        # Regression guard: a Timestamp/datetime64 hash mismatch once made this
        # silently return zero rows, which trained every bootstrap member on an
        # empty dataset and produced constant predictions.
        self.assertGreater(len(indices), 0)
        self.assertAlmostEqual(len(indices) / len(panel), 1.0, delta=0.35)

        # Every selected date contributes its full cross-section.
        counts = pd.Series(panel.index[indices]).value_counts()
        self.assertGreater(len(counts), 0)
        self.assertTrue((counts % 5 == 0).all())

    def test_resample_works_for_both_timestamp_representations(self):
        panel = build_panel(n_dates=40, n_tickers=4)
        rng = np.random.default_rng(5)
        from_index = block_bootstrap_row_indices(panel.index, rng, block_size=7)
        from_numpy = block_bootstrap_row_indices(
            panel.index.to_numpy(), np.random.default_rng(5), block_size=7
        )
        self.assertGreater(len(from_index), 0)
        np.testing.assert_array_equal(np.sort(from_index), np.sort(from_numpy))


class DecisionRuleTest(unittest.TestCase):
    def test_risk_adjusted_rule_penalises_a_volatile_forecast(self):
        """Same expected return, different uncertainty, different decision."""
        cfg = DecisionConfig(rule="risk_adjusted", threshold=0.01, min_z_score=0.20)
        signals = decide_batch(
            np.asarray([0.03, 0.03]), np.asarray([0.05, 0.40]), cfg
        )
        self.assertEqual(list(signals), ["BUY", "HOLD"])

    def test_point_rule_ignores_uncertainty(self):
        cfg = DecisionConfig(rule="point", threshold=0.01)
        signals = decide_batch(np.asarray([0.03, 0.03]), np.asarray([0.05, 0.40]), cfg)
        self.assertEqual(list(signals), ["BUY", "BUY"])

    def test_edge_is_measured_beyond_the_cost_hurdle(self):
        # A forecast that exactly equals the hurdle has no edge left.
        self.assertAlmostEqual(float(edge_z_score(np.asarray([0.01]), np.asarray([0.1]), 0.01)[0]), 0.0)
        self.assertGreater(float(edge_z_score(np.asarray([0.05]), np.asarray([0.1]), 0.01)[0]), 0.0)

    def test_a_forecast_below_the_hurdle_is_hold_not_sell(self):
        """Regression guard: a zero edge satisfies both `>= 0` and `<= -0`, and the
        SELL branch used to overwrite BUY whenever the tuned floor was zero."""
        cfg = DecisionConfig(rule="risk_adjusted", threshold=0.0292, min_z_score=0.0)
        signals = decide_batch(
            np.asarray([0.0107, 0.0249, 0.0317, -0.0107, -0.0400]),
            np.asarray([0.108, 0.104, 0.135, 0.108, 0.110]),
            cfg,
        )
        self.assertEqual(list(signals), ["HOLD", "HOLD", "BUY", "HOLD", "SELL"])

    def test_point_rule_with_a_zero_threshold_is_not_ambiguous(self):
        cfg = DecisionConfig(rule="point", threshold=0.0)
        signals = decide_batch(
            np.asarray([0.01, -0.01, 0.0]), np.asarray([0.1, 0.1, 0.1]), cfg
        )
        self.assertEqual(list(signals), ["BUY", "SELL", "HOLD"])

    def test_no_observation_is_ever_both_long_and_short(self):
        rng = np.random.default_rng(41)
        predictions = rng.normal(0, 0.05, 5000)
        sigma = np.full(5000, 0.09)
        for rule in ("risk_adjusted", "point"):
            for floor in (0.0, 0.05, 0.2):
                cfg = DecisionConfig(rule=rule, threshold=0.0, min_z_score=floor)
                signals = decide_batch(predictions, sigma, cfg)
                buys = signals == "BUY"
                sells = signals == "SELL"
                self.assertFalse(np.any(buys & sells), f"{rule}/{floor} overlapped")
                # A signal must never contradict the sign of its own forecast.
                self.assertTrue(np.all(predictions[buys] > 0))
                self.assertTrue(np.all(predictions[sells] < 0))

    def test_shorts_are_suppressed_when_shorting_is_disabled(self):
        cfg = DecisionConfig(rule="point", threshold=0.01, allow_short=False)
        signals = np.asarray(["SELL", "BUY"], dtype=object)
        sizes = position_sizes(signals, np.asarray([-0.05, 0.05]), np.asarray([0.05, 0.05]), cfg)
        self.assertEqual(list(sizes), [0.0, 1.0])

    def test_confidence_sizing_scales_exposure_with_conviction(self):
        cfg = DecisionConfig(rule="risk_adjusted", threshold=0.0, min_z_score=0.2, position_sizing="confidence")
        signals = np.asarray(["BUY", "BUY"], dtype=object)
        sizes = position_sizes(signals, np.asarray([0.01, 0.10]), np.asarray([0.05, 0.05]), cfg)
        self.assertLess(sizes[0], sizes[1])
        self.assertLessEqual(sizes[1], 1.0)


class BacktestIntegrationTest(unittest.TestCase):
    def test_decision_tuning_is_frozen_from_validation_only(self):
        dates = pd.bdate_range("2024-01-01", periods=120)
        metadata = [
            {"ticker": f"T{i}", "date": str(date.date())} for date in dates for i in range(6)
        ]
        rng = np.random.default_rng(29)
        truth = rng.normal(0.005, 0.05, len(metadata))
        prediction = 0.3 * truth + rng.normal(0, 0.01, len(metadata))
        sigma = np.full(len(metadata), 0.05)
        cfg = BacktestConfig(transaction_cost_pct=0.001, slippage_pct=0.0005)

        decision_cfg, report = tune_decision_config(
            metadata, truth, prediction, sigma, cfg, horizon=21,
            base_decision_cfg=DecisionConfig(rule="risk_adjusted"), min_active_trades=5,
        )
        self.assertEqual(report["selection_dataset"], "validation")
        self.assertGreaterEqual(decision_cfg.threshold, report["minimum_cost_threshold"])

    def test_partial_positions_pay_partial_transaction_cost(self):
        metadata = [{"ticker": "AAA", "date": "2024-01-02"}, {"ticker": "BBB", "date": "2024-01-02"}]
        cfg = BacktestConfig(transaction_cost_pct=0.001, slippage_pct=0.0005)
        frame = build_signal_frame(
            metadata, np.asarray([0.02, 0.02]), np.asarray([0.02, 0.02]), cfg,
            threshold=0.003, sigma=np.asarray([0.05, 0.05]),
            decision_cfg=DecisionConfig(rule="risk_adjusted", min_z_score=0.05),
        )
        frame.loc[0, "position"] = 0.5
        frame.loc[1, "position"] = 1.0
        _, metrics = backtest_signals(frame, cfg, horizon=1)
        self.assertAlmostEqual(metrics["average_gross_exposure"], 0.75)


class FeatureSelectionTest(unittest.TestCase):
    def test_market_wide_features_are_identified_and_droppable(self):
        self.assertTrue(is_market_wide_feature("benchmark_return_20d"))
        self.assertTrue(is_market_wide_feature("macro_vix_close"))
        # Stock-specific market-relative features must survive.
        self.assertFalse(is_market_wide_feature("excess_return_20d"))
        self.assertFalse(is_market_wide_feature("market_beta_60d"))
        self.assertFalse(is_market_wide_feature("idiosyncratic_volatility_20d"))

        frame = pd.DataFrame(
            {
                "Ticker": ["A"],
                "future_return": [0.01],
                "benchmark_return_20d": [0.02],
                "macro_vix_close": [18.0],
                "excess_return_20d": [0.03],
                "rsi_14": [55.0],
            }
        )
        self.assertEqual(
            sorted(get_feature_columns(frame, exclude_market_wide=True)),
            ["excess_return_20d", "rsi_14"],
        )
        self.assertEqual(len(get_feature_columns(frame, exclude_market_wide=False)), 4)


class WalkForwardSplitTest(unittest.TestCase):
    def test_folds_are_purged_and_move_forward_in_time(self):
        dates = pd.bdate_range("2015-01-01", periods=1500)
        splits = purged_walk_forward_splits(dates, folds=3, purge_horizon=21)
        self.assertEqual(len(splits), 3)

        previous_test_end = None
        for split in splits:
            self.assertGreater(len(split.train_dates), 0)
            self.assertGreater(len(split.validation_dates), 0)
            self.assertGreater(len(split.test_dates), 0)
            # Purge gaps: no train label may reach into validation, none into test.
            train_end = pd.Timestamp(split.train_dates[-1])
            validation_start = pd.Timestamp(split.validation_dates[0])
            validation_end = pd.Timestamp(split.validation_dates[-1])
            test_start = pd.Timestamp(split.test_dates[0])
            self.assertGreaterEqual((validation_start - train_end).days, 21)
            self.assertGreaterEqual((test_start - validation_end).days, 21)
            if previous_test_end is not None:
                self.assertGreater(test_start, previous_test_end)
            previous_test_end = pd.Timestamp(split.test_dates[-1])


if __name__ == "__main__":
    unittest.main()
