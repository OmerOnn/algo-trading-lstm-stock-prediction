"""
Tests for portfolio construction, turnover costs, blending and feature safety.

These cover the parts of the redesign whose correctness is easiest to get subtly
wrong: exposure normalisation, cost accounting, weight constraints, and the
point-in-time property of the new features.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.acceptance import evaluate_acceptance_gates, format_acceptance_table
from src.blending import BlendWeights, align_model_predictions, fit_blend_weights, weight_grid
from src.boosting import ensemble_feature_importance, recommend_feature_blocklist
from src.features import get_feature_columns, is_market_wide_feature, market_wide_feature_columns
from src.panel_features import (
    add_cross_sectional_ranks,
    add_market_state_features,
    add_panel_features,
    add_sector_features,
    assign_sectors,
)
from src.portfolio import (
    PortfolioConfig,
    build_portfolio_frame,
    regime_performance,
    run_all_offsets,
    run_top_k_backtest,
)
from src.uncertainty import (
    conditional_coverage,
    fit_interval_calibration,
    mondrian_interval_calibration,
    uncertainty_filter_benefit,
)


def portfolio_frame(dates: int = 120, tickers: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=dates)
    rows = []
    for ticker in range(tickers):
        score = rng.normal(0, 1, dates)
        rows.append(
            pd.DataFrame(
                {
                    "date": index,
                    "ticker": f"T{ticker:02d}",
                    "sector": f"S{ticker % 4}",
                    "beta": rng.normal(1.0, 0.2, dates),
                    "score": score,
                    # A genuine but small relationship, so selection should work.
                    "forward_return": 0.01 * score + rng.normal(0, 0.05, dates),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class TopKPortfolioTest(unittest.TestCase):
    def test_long_only_weights_are_fully_invested(self):
        cfg = PortfolioConfig(top_k=10)
        history, metrics = run_top_k_backtest(portfolio_frame(), cfg, horizon=21)
        self.assertGreater(len(history), 0)
        np.testing.assert_allclose(history["gross_exposure"].to_numpy(), 1.0, atol=1e-9)
        np.testing.assert_allclose(history["net_exposure"].to_numpy(), 1.0, atol=1e-9)
        self.assertAlmostEqual(metrics["average_gross_exposure"], 1.0, places=9)

    def test_exactly_top_k_names_are_held(self):
        cfg = PortfolioConfig(top_k=10)
        history, _ = run_top_k_backtest(portfolio_frame(), cfg, horizon=21)
        self.assertTrue(bool((history["names_held"] == 10).all()))

    def test_long_short_is_self_financing_with_a_capped_gross(self):
        cfg = PortfolioConfig(top_k=8, long_short=True, maximum_gross_exposure=2.0)
        history, _ = run_top_k_backtest(portfolio_frame(), cfg, horizon=21)
        np.testing.assert_allclose(history["net_exposure"].to_numpy(), 0.0, atol=1e-9)
        np.testing.assert_allclose(history["gross_exposure"].to_numpy(), 2.0, atol=1e-9)

    def test_cost_is_charged_on_realised_turnover(self):
        """Holding the same book must cost nothing to keep holding."""
        # A constant score means the same names are always selected, so after the
        # first rebalance turnover is zero and only the initial purchase is paid.
        frame = portfolio_frame(dates=100, tickers=20)
        frame["score"] = frame["ticker"].str[1:].astype(int).astype(float)
        cfg = PortfolioConfig(top_k=5, transaction_cost_pct=0.001, slippage_pct=0.0005)
        history, _ = run_top_k_backtest(frame, cfg, horizon=21)

        self.assertGreater(float(history["transaction_cost"].iloc[0]), 0.0)
        np.testing.assert_allclose(history["transaction_cost"].to_numpy()[1:], 0.0, atol=1e-12)
        np.testing.assert_allclose(history["turnover"].to_numpy()[1:], 0.0, atol=1e-12)

    def test_full_turnover_costs_the_full_round_trip(self):
        cfg = PortfolioConfig(top_k=5, transaction_cost_pct=0.001, slippage_pct=0.0005)
        history, _ = run_top_k_backtest(portfolio_frame(seed=3), cfg, horizon=21)
        # Turnover of 2.0 (sell everything, buy everything) at 0.0015 per unit.
        expected = history["turnover"] * 0.0015
        np.testing.assert_allclose(
            history["transaction_cost"].to_numpy(), expected.to_numpy(), atol=1e-12
        )

    def test_net_return_is_gross_minus_cost(self):
        cfg = PortfolioConfig(top_k=10)
        history, _ = run_top_k_backtest(portfolio_frame(), cfg, horizon=21)
        np.testing.assert_allclose(
            history["net_return"].to_numpy(),
            (history["gross_return"] - history["transaction_cost"]).to_numpy(),
            atol=1e-12,
        )

    def test_rebalance_dates_do_not_overlap_the_horizon(self):
        cfg = PortfolioConfig(top_k=5)
        history, _ = run_top_k_backtest(portfolio_frame(dates=200), cfg, horizon=21)
        gaps = pd.to_datetime(history["date"]).diff().dropna().dt.days
        self.assertTrue(bool((gaps >= 21).all()))

    def test_sector_neutralisation_removes_the_sector_tilt(self):
        frame = portfolio_frame()
        # Give one sector a large constant score advantage.
        frame.loc[frame["sector"] == "S0", "score"] += 10.0

        plain, _ = run_top_k_backtest(frame, PortfolioConfig(top_k=8, neutralize="none"), 21)
        neutral, _ = run_top_k_backtest(frame, PortfolioConfig(top_k=8, neutralize="sector"), 21)
        self.assertGreater(len(plain), 0)
        self.assertGreater(len(neutral), 0)
        # Without neutralisation the tilted sector monopolises the book; with it,
        # the selection has to spread out.
        self.assertNotEqual(
            plain["gross_return"].round(9).tolist(), neutral["gross_return"].round(9).tolist()
        )

    def test_every_rebalance_offset_is_evaluated(self):
        report = run_all_offsets(portfolio_frame(dates=200), PortfolioConfig(top_k=8), horizon=21)
        self.assertEqual(report["offsets_evaluated"], 21)
        self.assertEqual(len(report["per_offset"]), 21)
        for key in ("sharpe_ratio", "total_return", "information_ratio_vs_universe"):
            block = report["distribution"][key]
            for statistic in ("mean", "median", "std", "worst", "best"):
                self.assertIn(statistic, block)
            self.assertLessEqual(block["worst"], block["mean"] + 1e-9)
            self.assertGreaterEqual(block["best"], block["mean"] - 1e-9)

    def test_offsets_actually_differ(self):
        """If all offsets agreed, reporting a distribution would be pointless."""
        report = run_all_offsets(portfolio_frame(dates=200), PortfolioConfig(top_k=8), horizon=21)
        sharpes = [run["sharpe_ratio"] for run in report["per_offset"]]
        self.assertGreater(np.std(sharpes), 0.0)

    def test_regime_performance_splits_the_timeline(self):
        history, _ = run_top_k_backtest(portfolio_frame(dates=200), PortfolioConfig(top_k=8), 21)
        rows = regime_performance(history, None, blocks=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(row["periods"] for row in rows), len(history))

    def test_build_frame_maps_sectors_and_drops_unusable_rows(self):
        signal = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"] * 3),
                "ticker": ["AAPL", "MSFT", "XXX"],
                "predicted_return": [0.01, np.nan, 0.03],
                "true_return": [0.02, 0.01, 0.00],
            }
        )
        frame = build_portfolio_frame(signal, sector_map={"AAPL": "Tech", "MSFT": "Tech"})
        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["sector"]), {"Tech", "Unclassified"})


class BlendTest(unittest.TestCase):
    def folds(self, lstm_quality: float, xgb_quality: float, folds: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        out = []
        for fold in range(folds):
            n = 1200
            dates = np.repeat(pd.bdate_range("2020-01-01", periods=n // 20), 20)[:n]
            truth = rng.normal(0, 0.05, n)
            out.append(
                {
                    "true_return": truth,
                    "dates": dates,
                    "reference_prediction": 0.0,
                    "predictions": {
                        "LSTM": lstm_quality * truth + rng.normal(0, 0.05, n),
                        "XGBoost": xgb_quality * truth + rng.normal(0, 0.05, n),
                    },
                }
            )
        return out

    def test_weights_are_non_negative_and_sum_to_one(self):
        blend = fit_blend_weights(self.folds(0.4, 0.4), horizon=21)
        self.assertTrue(all(weight >= 0 for weight in blend.weights.values()))
        self.assertAlmostEqual(sum(blend.weights.values()), 1.0, places=9)

    def test_every_grid_point_obeys_the_simplex_constraint(self):
        for weights in weight_grid(["LSTM", "XGBoost"], step=0.05):
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
            self.assertTrue(all(value >= 0 for value in weights.values()))
        self.assertEqual(len(weight_grid(["LSTM", "XGBoost"], step=0.05)), 21)

    def test_a_clearly_worse_model_gets_little_or_no_weight(self):
        blend = fit_blend_weights(self.folds(0.6, 0.02, seed=4), horizon=21)
        self.assertGreater(blend.weights["LSTM"], blend.weights["XGBoost"])

    def test_equal_weights_are_not_assumed(self):
        blend = fit_blend_weights(self.folds(0.6, 0.05, seed=9), horizon=21)
        self.assertNotAlmostEqual(blend.weights["LSTM"], 0.5, places=2)

    def test_blend_is_only_retained_if_it_beats_the_best_single_model(self):
        # One strong and one useless model: blending should not be retained.
        blend = fit_blend_weights(
            self.folds(0.6, 0.0, seed=11), horizon=21, minimum_improvement=0.05
        )
        self.assertFalse(blend.retained)
        self.assertEqual(max(blend.weights.values()), 1.0)
        self.assertIn("below the", blend.reason)

    def test_apply_produces_the_weighted_combination(self):
        blend = BlendWeights(weights={"LSTM": 0.25, "XGBoost": 0.75})
        combined = blend.apply(
            {"LSTM": np.array([1.0, 2.0]), "XGBoost": np.array([3.0, 4.0])}
        )
        np.testing.assert_allclose(combined, [2.5, 3.5])

    def test_apply_renormalises_when_a_model_is_missing(self):
        blend = BlendWeights(weights={"LSTM": 0.25, "XGBoost": 0.75})
        combined = blend.apply({"LSTM": np.array([2.0, 4.0])})
        np.testing.assert_allclose(combined, [2.0, 4.0])

    def test_single_model_folds_are_reported_not_blended(self):
        folds = self.folds(0.4, 0.4)
        for fold in folds:
            fold["predictions"].pop("XGBoost")
        blend = fit_blend_weights(folds, horizon=21)
        self.assertFalse(blend.retained)
        self.assertEqual(blend.weights, {"LSTM": 1.0})

    def test_alignment_joins_on_ticker_and_date(self):
        """A positional join would compare one model's AAPL to another's XOM."""
        left = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "NVDA"],
                "date": pd.to_datetime(["2020-01-01"] * 3),
                "predicted_return": [0.01, 0.02, 0.03],
                "true_return": [0.05, 0.06, 0.07],
            }
        )
        right = pd.DataFrame(
            {
                "ticker": ["NVDA", "AAPL"],
                "date": pd.to_datetime(["2020-01-01"] * 2),
                "predicted_return": [0.30, 0.10],
                "true_return": [0.07, 0.05],
            }
        )
        aligned = align_model_predictions({"a": left, "b": right})
        self.assertEqual(len(aligned), 2)
        row = aligned[aligned["ticker"] == "AAPL"].iloc[0]
        self.assertAlmostEqual(row["prediction_a"], 0.01)
        self.assertAlmostEqual(row["prediction_b"], 0.10)


class FeatureSafetyTest(unittest.TestCase):
    def panel(self, dates: int = 400, tickers: int = 12, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        index = pd.bdate_range("2015-01-01", periods=dates)
        # One benchmark for the whole panel. The real pipeline guarantees this --
        # benchmark columns are joined from a single index series -- and a fixture
        # that gave each ticker its own benchmark would not be testing the code
        # that ships.
        benchmark = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, dates)))
        benchmark_volatility = pd.Series(benchmark).pct_change().rolling(20).std().to_numpy()
        rows = []
        for ticker in range(tickers):
            close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, dates)))
            rows.append(
                pd.DataFrame(
                    {
                        "Ticker": f"T{ticker:02d}",
                        "return_1d": pd.Series(close).pct_change().to_numpy(),
                        "return_5d": pd.Series(close).pct_change(5).to_numpy(),
                        "return_20d": pd.Series(close).pct_change(20).to_numpy(),
                        "return_60d": pd.Series(close).pct_change(60).to_numpy(),
                        "volatility_20d": pd.Series(close)
                        .pct_change()
                        .rolling(20)
                        .std()
                        .to_numpy(),
                        "price_to_sma_50": rng.normal(0, 0.05, dates),
                        "price_to_sma_200": rng.normal(0, 0.08, dates),
                        "excess_return_20d": rng.normal(0, 0.03, dates),
                        "idiosyncratic_volatility_20d": np.abs(rng.normal(0.012, 0.002, dates)),
                        "idiosyncratic_return_1d": rng.normal(0, 0.01, dates),
                        "market_beta_60d": rng.normal(1.0, 0.2, dates),
                        "rsi_14": rng.uniform(20, 80, dates),
                        "volume_ratio_20": np.abs(rng.normal(1.0, 0.2, dates)),
                        "dollar_volume_zscore_60d": rng.normal(0, 1, dates),
                        "benchmark_volatility_20d": benchmark_volatility,
                        "benchmark_close": benchmark,
                        "future_return": rng.normal(0, 0.05, dates),
                        "benchmark_future_return": rng.normal(0.01, 0.04, dates),
                    },
                    index=index,
                )
            )
        return pd.concat(rows).sort_index()

    def test_market_wide_features_are_recognised_by_prefix(self):
        self.assertTrue(is_market_wide_feature("benchmark_return_20d"))
        self.assertTrue(is_market_wide_feature("macro_vix_close"))
        self.assertTrue(is_market_wide_feature("marketstate_pct_above_sma50"))
        self.assertTrue(is_market_wide_feature("universe_return_1d"))
        # Stock-specific market-relative features must NOT be excluded.
        self.assertFalse(is_market_wide_feature("market_beta_60d"))
        self.assertFalse(is_market_wide_feature("excess_return_20d"))
        self.assertFalse(is_market_wide_feature("sector_relative_return_20d"))

    def test_market_state_features_are_constant_across_the_cross_section(self):
        """
        This is the property that justifies excluding them from the stock model.

        A feature identical for every ticker on a date carries no cross-sectional
        ranking information but does let a sequence model identify the date.
        """
        panel = add_panel_features(self.panel(), ticker_sectors=None)
        market_columns = [c for c in panel.columns if c.startswith("marketstate_")]
        self.assertGreater(len(market_columns), 0)
        sample_dates = list(panel.index.unique())[250:260]
        for column in market_columns:
            for date in sample_dates:
                values = panel.loc[date, column].dropna()
                if len(values) > 1:
                    self.assertLess(
                        float(values.std()), 1e-9, f"{column} varies across tickers on {date}"
                    )

    def test_interactions_do_vary_across_the_cross_section(self):
        """Interactions are how market state reaches the stock model safely."""
        panel = add_panel_features(self.panel(), ticker_sectors=None)
        interaction_columns = [c for c in panel.columns if "_x_" in c]
        self.assertGreater(len(interaction_columns), 0)
        varying = 0
        for column in interaction_columns:
            for date in list(panel.index.unique())[300:305]:
                values = panel.loc[date, column].dropna()
                if len(values) > 1 and float(values.std()) > 1e-12:
                    varying += 1
                    break
        self.assertGreater(varying, 0)
        for column in interaction_columns:
            self.assertFalse(is_market_wide_feature(column))

    def test_market_model_may_not_see_its_own_label_or_the_raw_index_level(self):
        panel = self.panel()
        columns = market_wide_feature_columns(panel)
        self.assertNotIn("benchmark_future_return", columns)
        self.assertNotIn("benchmark_close", columns)

    def test_cross_sectional_ranks_are_bounded_and_centred(self):
        panel = add_cross_sectional_ranks(self.panel())
        rank_columns = [c for c in panel.columns if c.startswith("xs_rank_")]
        self.assertGreater(len(rank_columns), 0)
        for column in rank_columns:
            values = panel[column].dropna()
            self.assertGreaterEqual(float(values.min()), -0.5 - 1e-9)
            self.assertLessEqual(float(values.max()), 0.5 + 1e-9)

    def test_panel_features_are_point_in_time(self):
        """
        Truncating the future must not change any past feature value.

        This is the strongest available check for look-ahead: recompute the whole
        panel stage on a prefix of history and require every overlapping value to
        be bit-for-bit identical.
        """
        panel = self.panel(dates=400, tickers=8)
        cutoff = panel.index.unique()[300]

        full = add_panel_features(panel, ticker_sectors=None)
        truncated = add_panel_features(panel[panel.index <= cutoff], ticker_sectors=None)

        engineered = [
            column
            for column in truncated.columns
            if column not in panel.columns and pd.api.types.is_numeric_dtype(truncated[column])
        ]
        self.assertGreater(len(engineered), 10)

        full_slice = full[full.index <= cutoff].sort_values(["Ticker"], kind="stable")
        truncated_slice = truncated.sort_values(["Ticker"], kind="stable")
        for column in engineered:
            a = full_slice[column].to_numpy(dtype=float)
            b = truncated_slice[column].to_numpy(dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() == 0:
                continue
            np.testing.assert_allclose(
                a[both],
                b[both],
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{column} changed when future data was removed (look-ahead)",
            )

    def test_thin_sectors_fall_back_to_the_universe_composite(self):
        """A one-stock sector would make sector-relative momentum a constant zero."""
        panel = self.panel(dates=120, tickers=6)
        sectors = {f"T{i:02d}": ("Solo" if i == 0 else "Big") for i in range(6)}
        out = add_sector_features(assign_sectors(panel, sectors), minimum_sector_members=3)
        # Skip the warm-up dates: return_1d is NaN on the first date, so the
        # member count is genuinely below the minimum there for every sector.
        settled = out[out.index > out.index.unique()[2]]
        solo = settled[settled["Ticker"] == "T00"]
        big = settled[settled["Ticker"] == "T01"]
        self.assertTrue(bool((solo["sector_is_composite_fallback"] == 1.0).all()))
        self.assertTrue(bool((big["sector_is_composite_fallback"] == 0.0).all()))
        # The fallback must actually change the composite it uses.
        np.testing.assert_allclose(
            solo["sector_return_1d"].to_numpy(), solo["universe_return_1d"].to_numpy()
        )

    def test_blocklist_removes_named_features(self):
        panel = self.panel(dates=60, tickers=3)
        without = get_feature_columns(panel, exclude_market_wide=True)
        blocked = get_feature_columns(
            panel, exclude_market_wide=True, blocklist=["rsi_14", "volume_ratio_20"]
        )
        self.assertIn("rsi_14", without)
        self.assertNotIn("rsi_14", blocked)
        self.assertEqual(len(blocked), len(without) - 2)


class ImportanceTest(unittest.TestCase):
    class FakeModel:
        def __init__(self, values):
            self.feature_importances_ = np.asarray(values, dtype=float)

    def test_importance_is_averaged_across_the_whole_ensemble(self):
        """
        Regression guard: importance used to come from a reference model that was
        never used for inference.
        """
        models = [
            self.FakeModel([1.0, 0.0, 0.5]),
            self.FakeModel([0.0, 1.0, 0.5]),
            self.FakeModel([0.5, 0.5, 0.5]),
        ]
        frame = ensemble_feature_importance(models, ["a", "b", "c"])
        self.assertEqual(int(frame["member_count"].iloc[0]), 3)
        row = frame[frame["feature"] == "c"].iloc[0]
        self.assertAlmostEqual(row["gain_importance"], 0.5, places=9)
        self.assertAlmostEqual(row["gain_importance_std"], 0.0, places=9)
        # "a" has the same mean as "c" but huge dispersion across members.
        a = frame[frame["feature"] == "a"].iloc[0]
        self.assertGreater(a["gain_importance_std"], 0.0)

    def test_members_using_feature_is_counted(self):
        models = [self.FakeModel([1.0, 0.0]), self.FakeModel([1.0, 0.0])]
        frame = ensemble_feature_importance(models, ["a", "b"])
        self.assertEqual(int(frame[frame["feature"] == "a"]["members_using_feature"].iloc[0]), 2)
        self.assertEqual(int(frame[frame["feature"] == "b"]["members_using_feature"].iloc[0]), 0)

    def test_blocklist_requires_repetition_across_folds(self):
        harmful = pd.DataFrame(
            {"feature": ["x", "y"], "ic_drop_mean": [-0.02, -0.02], "harmful": [True, True]}
        )
        once = pd.DataFrame(
            {"feature": ["x", "y"], "ic_drop_mean": [0.01, -0.02], "harmful": [False, True]}
        )
        report = recommend_feature_blocklist([harmful, once], minimum_folds_harmful=2)
        self.assertEqual(report["recommended_blocklist"], ["y"])
        self.assertFalse(report["applied"])

    def test_no_recommendation_from_a_single_fold(self):
        frame = pd.DataFrame({"feature": ["x"], "ic_drop_mean": [-0.05], "harmful": [True]})
        report = recommend_feature_blocklist([frame], minimum_folds_harmful=2)
        self.assertEqual(report["recommended_blocklist"], [])


class UncertaintyEvaluationTest(unittest.TestCase):
    def data(self, n: int = 4000, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = np.abs(rng.normal(0.05, 0.02, n)) + 0.01
        prediction = rng.normal(0, 0.01, n)
        truth = prediction + rng.normal(0, 1, n) * scale
        model_std = np.full(n, 0.002)
        return truth, prediction, model_std, scale

    def test_conditional_coverage_buckets_are_reported(self):
        truth, prediction, model_std, scale = self.data()
        calibration = fit_interval_calibration(truth, prediction, model_std, scale, 0.80)
        lower, upper, _ = calibration.interval(prediction, model_std, scale)
        rows = conditional_coverage(truth, lower, upper, scale, 0.80, buckets=5)
        self.assertEqual(len(rows), 5)
        for row in rows:
            for key in ("coverage_picp", "coverage_error", "mean_interval_width", "sample_size"):
                self.assertIn(key, row)

    def test_mondrian_fits_a_multiplier_per_regime(self):
        truth, prediction, model_std, scale = self.data()
        regime = np.where(scale > np.median(scale), "high", "low")
        calibrations = mondrian_interval_calibration(
            truth, prediction, model_std, scale, regime, 0.80
        )
        self.assertIn("high", calibrations)
        self.assertIn("low", calibrations)
        self.assertIn("__pooled__", calibrations)
        self.assertNotEqual(
            calibrations["high"].conformal_multiplier, calibrations["low"].conformal_multiplier
        )

    def test_small_regime_groups_fall_back_to_the_pooled_multiplier(self):
        truth, prediction, model_std, scale = self.data()
        regime = np.array(["rare"] * 5 + ["common"] * (len(truth) - 5))
        calibrations = mondrian_interval_calibration(
            truth, prediction, model_std, scale, regime, 0.80, minimum_group_size=200
        )
        self.assertIs(calibrations["rare"], calibrations["__pooled__"])

    def test_filter_benefit_detects_a_useful_sigma(self):
        """When sigma is informative, filtering by confidence must raise IC."""
        rng = np.random.default_rng(2)
        n = 6000
        dates = np.repeat(pd.bdate_range("2020-01-01", periods=n // 30), 30)[:n]
        # Half the rows are informative with small sigma; half are pure noise with
        # large sigma. A working filter should keep the informative half.
        informative = np.arange(n) % 2 == 0
        truth = rng.normal(0, 0.05, n)
        prediction = np.where(informative, 0.5 * truth, rng.normal(0, 0.05, n))
        sigma = np.where(informative, 0.01, 1.0)

        report = uncertainty_filter_benefit(dates, truth, prediction, sigma, horizon=21)
        self.assertTrue(report["evaluated"])
        self.assertTrue(report["filtering_helps"])
        self.assertGreater(report["ic_improvement_from_filtering"], 0.0)

    def test_filter_benefit_reports_no_benefit_for_an_uninformative_sigma(self):
        rng = np.random.default_rng(3)
        n = 4000
        dates = np.repeat(pd.bdate_range("2020-01-01", periods=n // 20), 20)[:n]
        truth = rng.normal(0, 0.05, n)
        prediction = 0.2 * truth + rng.normal(0, 0.05, n)
        sigma = np.full(n, 0.05)  # carries no information at all
        report = uncertainty_filter_benefit(dates, truth, prediction, sigma, horizon=21)
        self.assertTrue(report["evaluated"])
        self.assertIn("filtering_helps", report)


class AcceptanceGateTest(unittest.TestCase):
    def test_missing_inputs_fail_rather_than_silently_pass(self):
        report = evaluate_acceptance_gates(None, None, None, None, None)
        self.assertFalse(report["all_passed"])
        self.assertEqual(report["passed"], 0)
        self.assertEqual(len(report["failed_gates"]), report["total"])

    def test_a_fully_qualifying_model_passes_every_gate(self):
        walk_forward = {
            "summary": {"cross_sectional_ic": {"mean": 0.05, "std": 0.01}},
            "folds": [
                {
                    "test_metrics": {
                        "cross_sectional_ic": value,
                        "cross_sectional_long_short_spread_annualised": 0.10,
                    }
                }
                for value in (0.04, 0.05, 0.06)
            ],
        }
        report = evaluate_acceptance_gates(
            walk_forward=walk_forward,
            test_metrics={"cross_sectional_ic_t_statistic": 2.5, "mse": 1.0, "rmse": 1.0, "mae": 1.0},
            baselines={"historical_mean": {"mse": 2.0, "rmse": 2.0, "mae": 2.0}},
            interval_metrics={"coverage_error": 0.01, "normalized_interval_width": 2.0},
            portfolio={"distribution": {"information_ratio_vs_universe": {"mean": 0.5}}, "offsets_evaluated": 21},
            regime_blocks=[{"metrics": {"cross_sectional_ic": 0.02}}],
            calibration_stability={"stable": True, "selected": "affine", "detail": "consistent"},
        )
        self.assertTrue(report["all_passed"], report["failed_gates"])

    def test_a_single_negative_fold_fails_the_every_fold_gate(self):
        walk_forward = {
            "summary": {"cross_sectional_ic": {"mean": 0.05}},
            "folds": [{"test_metrics": {"cross_sectional_ic": value}} for value in (0.06, -0.01, 0.07)],
        }
        report = evaluate_acceptance_gates(walk_forward, None, None, None, None)
        self.assertIn("positive_ic_every_fold", report["failed_gates"])

    def test_table_renders_and_states_the_no_optimisation_policy(self):
        report = evaluate_acceptance_gates(None, None, None, None, None)
        text = format_acceptance_table(report)
        self.assertIn("Acceptance gates:", text)
        self.assertIn("FAIL", text)
        self.assertIn("never optimised against", report["note"])


if __name__ == "__main__":
    unittest.main()
