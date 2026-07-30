"""
Tests for the modelling core: metrics, losses, batching, targets and selection.

Each test pins a behaviour that a plausible refactor could silently break, and
several of them are regression guards for bugs that actually occurred in this
project.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.boosting import round_ladder, round_score, select_rounds
from src.calibration import (
    ReturnCalibration,
    calibration_monotonicity,
    cross_sectional_center,
    decile_calibration_report,
    fit_calibration_candidates,
    select_calibration,
)
from src.dataset import DateGroupedBatchSampler, StockSequenceDataset
from src.losses import CompositeRegressionLoss, LOSS_PRESETS, grouped_correlation, resolve_loss_config
from src.market_model import (
    MarketReturnModel,
    build_market_frame,
    compose_hierarchical_return,
    fit_market_return_model,
)
from src.model import StockReturnPredictor, VariationalSequenceDropout
from src.regression import (
    RESIDUAL_RETURN_COLUMN,
    SELECTION_SCORE_KEY,
    add_model_target,
    add_selection_score,
    clipped_beta,
    evaluate_baselines,
    full_metrics,
    regression_metrics,
    resolve_target_config,
    target_component_column,
)
from src.validation import chronological_train_validation_test_split, purged_walk_forward_splits


def make_panel(dates: int = 400, tickers: int = 12, seed: int = 0) -> pd.DataFrame:
    """A small synthetic panel with the columns the pipeline guarantees."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=dates)
    rows = []
    for ticker in range(tickers):
        signal = rng.normal(0, 1, dates)
        benchmark = rng.normal(0.0004, 0.01, dates)
        rows.append(
            pd.DataFrame(
                {
                    "Ticker": f"T{ticker:02d}",
                    "sector": f"S{ticker % 3}",
                    "return_1d": rng.normal(0, 0.015, dates),
                    "return_20d": signal * 0.02,
                    "return_60d": rng.normal(0, 0.05, dates),
                    "excess_return_20d": signal * 0.015,
                    "volatility_20d": np.abs(rng.normal(0.015, 0.003, dates)) + 0.005,
                    "idiosyncratic_volatility_20d": np.abs(rng.normal(0.012, 0.003, dates)) + 0.005,
                    "market_beta_60d": rng.normal(1.0, 0.25, dates),
                    "benchmark_return_20d": benchmark * 20,
                    "benchmark_volatility_20d": np.abs(rng.normal(0.01, 0.002, dates)) + 0.004,
                    "marketstate_pct_above_sma50": rng.uniform(0.2, 0.8, dates),
                    "future_return": signal * 0.01 + rng.normal(0, 0.05, dates),
                    "benchmark_future_return": benchmark * 21,
                },
                index=index,
            )
        )
    return pd.concat(rows).sort_index()


# ---------------------------------------------------------------------------
# Phase 2: MSE and the selection criterion
# ---------------------------------------------------------------------------


class MseMetricTest(unittest.TestCase):
    def test_mse_is_reported_and_is_the_square_of_rmse(self):
        truth = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
        predicted = np.array([0.02, -0.01, 0.02, -0.03, 0.04])
        metrics = regression_metrics(truth, predicted)
        self.assertIn("mse", metrics)
        self.assertAlmostEqual(metrics["mse"], float(np.mean((predicted - truth) ** 2)), places=12)
        self.assertAlmostEqual(metrics["rmse"], float(np.sqrt(metrics["mse"])), places=12)

    def test_mse_appears_in_every_metric_payload(self):
        panel = make_panel(dates=120, tickers=6)
        metrics = full_metrics(
            panel.index, panel["future_return"].to_numpy(), panel["return_20d"].to_numpy(), horizon=21
        )
        self.assertIn("mse", metrics)

        baselines = evaluate_baselines(panel.iloc[:400], panel.iloc[400:], 21, "future_return")
        for name, payload in baselines.items():
            self.assertIn("mse", payload, f"baseline {name} has no mse")

    def test_skill_is_measured_against_the_supplied_reference(self):
        truth = np.array([0.02, -0.01, 0.04, -0.03])
        reference = 0.0
        perfect = regression_metrics(truth, truth, reference_prediction=reference)
        self.assertAlmostEqual(perfect["mse_skill_vs_historical_mean"], 1.0, places=9)

        useless = regression_metrics(
            truth, np.full_like(truth, reference), reference_prediction=reference
        )
        self.assertAlmostEqual(useless["mse_skill_vs_historical_mean"], 0.0, places=9)


class SelectionScoreTest(unittest.TestCase):
    """
    The checkpoint criterion must reject both degenerate ways to look good.

    Regression guard: selecting on MSE alone made every earlier run peak at epoch
    one, because a constant forecast is a strong squared-error solution on a
    near-unforecastable target.
    """

    def test_constant_forecast_scores_zero(self):
        metrics = {"cross_sectional_ic": 0.0, "mse_skill_vs_historical_mean": 0.0}
        scored = add_selection_score(metrics, magnitude_weight=0.25)
        self.assertAlmostEqual(scored[SELECTION_SCORE_KEY], 0.0, places=9)

    def test_ranking_only_model_is_penalised_for_bad_magnitudes(self):
        ranking_only = add_selection_score(
            {"cross_sectional_ic": 0.05, "mse_skill_vs_historical_mean": -0.40}
        )
        balanced = add_selection_score(
            {"cross_sectional_ic": 0.05, "mse_skill_vs_historical_mean": 0.02}
        )
        self.assertLess(ranking_only[SELECTION_SCORE_KEY], balanced[SELECTION_SCORE_KEY])

    def test_score_requires_both_kinds_of_skill_to_be_high(self):
        both = add_selection_score(
            {"cross_sectional_ic": 0.05, "mse_skill_vs_historical_mean": 0.10}
        )
        magnitude_only = add_selection_score(
            {"cross_sectional_ic": 0.0, "mse_skill_vs_historical_mean": 0.10}
        )
        self.assertGreater(both[SELECTION_SCORE_KEY], magnitude_only[SELECTION_SCORE_KEY])

    def test_non_finite_magnitude_does_not_poison_the_score(self):
        scored = add_selection_score(
            {"cross_sectional_ic": 0.04, "mse_skill_vs_historical_mean": float("nan")}
        )
        self.assertTrue(np.isfinite(scored[SELECTION_SCORE_KEY]))


# ---------------------------------------------------------------------------
# Phase 2/5: the composite loss and date-aware batching
# ---------------------------------------------------------------------------


class CompositeLossTest(unittest.TestCase):
    def test_mse_term_actually_contributes_to_the_gradient(self):
        """MSE must be optimised, not merely reported after training."""
        torch.manual_seed(0)
        predicted = torch.zeros(64, requires_grad=True)
        target = torch.linspace(-1, 1, 64)
        groups = torch.zeros(64, dtype=torch.long)

        mse_only = CompositeRegressionLoss(
            mse_weight=1.0, huber_weight=0.0, cross_sectional_ic_weight=0.0
        )
        mse_only(predicted, target, groups).backward()
        self.assertGreater(float(predicted.grad.abs().sum()), 0.0)

    def test_each_preset_produces_a_distinct_objective(self):
        torch.manual_seed(11)
        predicted = torch.linspace(-1, 1, 200)
        # A noisy, imperfectly ordered target, so the correlation term is neither
        # exactly 1 nor exactly 0 and every preset weights something different.
        target = 0.5 * predicted + torch.randn(200) * 0.4
        groups = torch.zeros(200, dtype=torch.long)

        losses = {
            name: round(
                float(CompositeRegressionLoss.from_config(preset)(predicted, target, groups).item()),
                8,
            )
            for name, preset in LOSS_PRESETS.items()
        }
        self.assertEqual(
            len(set(losses.values())), len(losses), f"presets collapsed to the same value: {losses}"
        )
        # The IC-weighted preset must differ from the magnitude-only pair.
        self.assertNotEqual(losses["mse_huber"], losses["mse_huber_ic"])

    def test_all_zero_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_loss_config(
                {"mse_weight": 0.0, "huber_weight": 0.0, "cross_sectional_ic_weight": 0.0}
            )

    def test_terms_are_reported_separately(self):
        loss = CompositeRegressionLoss.from_config(None)
        components = loss.terms(
            torch.linspace(-1, 1, 30), torch.linspace(-1, 1, 30), torch.zeros(30, dtype=torch.long)
        )
        self.assertEqual(set(components), {"mse", "huber", "cross_sectional_ic"})
        self.assertAlmostEqual(
            float(loss.combine(components).item()),
            float(loss(torch.linspace(-1, 1, 30), torch.linspace(-1, 1, 30), torch.zeros(30, dtype=torch.long)).item()),
            places=9,
        )


class GroupedCorrelationTest(unittest.TestCase):
    def test_perfect_within_date_ordering_gives_one(self):
        predicted = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        target = predicted.clone()
        groups = torch.tensor([0] * 5 + [1] * 5)
        self.assertAlmostEqual(
            float(grouped_correlation(predicted, target, groups, 5).item()), 1.0, places=5
        )

    def test_inverted_ordering_gives_minus_one(self):
        predicted = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = -predicted
        groups = torch.zeros(5, dtype=torch.long)
        self.assertAlmostEqual(
            float(grouped_correlation(predicted, target, groups, 5).item()), -1.0, places=5
        )

    def test_each_date_is_weighted_equally_regardless_of_size(self):
        """
        A big date must not dominate a small one.

        Date 0 has 40 names and is perfectly ordered; date 1 has 5 names and is
        perfectly inverted. Equal weighting per date puts the mean at 0, whereas
        row weighting would put it near +0.78.
        """
        big = torch.arange(40, dtype=torch.float32)
        small = torch.arange(5, dtype=torch.float32)
        predicted = torch.cat([big, small])
        target = torch.cat([big, -small])
        groups = torch.tensor([0] * 40 + [1] * 5)
        value = float(grouped_correlation(predicted, target, groups, 5).item())
        self.assertAlmostEqual(value, 0.0, places=4)

    def test_dates_below_the_minimum_are_skipped(self):
        predicted = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 1.0])
        groups = torch.zeros(2, dtype=torch.long)
        self.assertEqual(float(grouped_correlation(predicted, target, groups, 5).item()), 0.0)

    def test_no_qualifying_group_returns_zero_not_nan(self):
        value = grouped_correlation(
            torch.ones(3), torch.ones(3), torch.zeros(3, dtype=torch.long), 5
        )
        self.assertTrue(np.isfinite(float(value.item())))
        self.assertEqual(float(value.item()), 0.0)


class DateBatchingTest(unittest.TestCase):
    def frame(self, dates: int = 30, tickers: int = 8) -> pd.DataFrame:
        index = pd.bdate_range("2020-01-01", periods=dates)
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "Ticker": f"T{t}",
                        "feature": np.arange(dates, dtype=float),
                        "future_return": 0.01,
                        "model_target": 0.5,
                        "target_scale": 0.02,
                    },
                    index=index,
                )
                for t in range(tickers)
            ]
        ).sort_index()

    def test_dataset_emits_a_date_code(self):
        dataset = StockSequenceDataset(self.frame(), ["feature"], window_size=3)
        self.assertEqual(len(dataset[0]), 5)
        self.assertEqual(len(dataset.date_codes), len(dataset))

    def test_date_codes_are_chronological(self):
        dataset = StockSequenceDataset(self.frame(), ["feature"], window_size=3)
        by_date = {}
        for index, meta in enumerate(dataset.metadata):
            by_date.setdefault(meta["date"], set()).add(int(dataset.date_codes[index]))
        # Every row on a date shares one code, and codes increase with the date.
        for codes in by_date.values():
            self.assertEqual(len(codes), 1)
        ordered = [next(iter(by_date[key])) for key in sorted(by_date)]
        self.assertEqual(ordered, sorted(ordered))

    def test_every_batch_holds_complete_date_cross_sections(self):
        dataset = StockSequenceDataset(self.frame(), ["feature"], window_size=3)
        sampler = DateGroupedBatchSampler(dataset, dates_per_batch=2, shuffle=False, seed=0)
        sizes_by_date = {}
        for index in range(len(dataset)):
            code = int(dataset.date_codes[index])
            sizes_by_date[code] = sizes_by_date.get(code, 0) + 1

        for batch in sampler:
            counts = {}
            for index in batch:
                code = int(dataset.date_codes[index])
                counts[code] = counts.get(code, 0) + 1
            for code, count in counts.items():
                self.assertEqual(
                    count,
                    sizes_by_date[code],
                    "a batch split a date, so its cross-section is incomplete",
                )

    def test_every_row_is_used_exactly_once_per_epoch(self):
        dataset = StockSequenceDataset(self.frame(), ["feature"], window_size=3)
        sampler = DateGroupedBatchSampler(dataset, dates_per_batch=3, shuffle=True, seed=1)
        seen = [index for batch in sampler for index in batch]
        self.assertEqual(sorted(seen), list(range(len(dataset))))

    def test_shuffling_changes_the_batch_order_between_epochs(self):
        dataset = StockSequenceDataset(self.frame(), ["feature"], window_size=3)
        sampler = DateGroupedBatchSampler(dataset, dates_per_batch=2, shuffle=True, seed=7)
        first = [list(batch) for batch in sampler]
        second = [list(batch) for batch in sampler]
        self.assertNotEqual(first, second)

    def test_row_cap_is_respected(self):
        dataset = StockSequenceDataset(self.frame(tickers=20), ["feature"], window_size=3)
        sampler = DateGroupedBatchSampler(
            dataset, dates_per_batch=10, shuffle=False, seed=0, max_rows_per_batch=25
        )
        for batch in sampler:
            # A batch may exceed the cap only because a single date is larger.
            self.assertLessEqual(len(batch), 25 + 20)


class ModelArchitectureTest(unittest.TestCase):
    def test_recurrent_dropout_shares_one_mask_across_timesteps(self):
        torch.manual_seed(0)
        dropout = VariationalSequenceDropout(0.5)
        dropout.train()
        x = torch.ones(4, 10, 6)
        out = dropout(x)
        # A shared mask means every timestep of a given (row, feature) is equal.
        for row in range(4):
            for feature in range(6):
                column = out[row, :, feature]
                self.assertEqual(len(torch.unique(column)), 1)

    def test_recurrent_dropout_is_identity_in_eval(self):
        dropout = VariationalSequenceDropout(0.5)
        dropout.eval()
        x = torch.randn(3, 5, 4)
        torch.testing.assert_close(dropout(x), x)

    def test_auxiliary_heads_are_regression_heads_and_optional(self):
        model = StockReturnPredictor(input_size=7, hidden_size=8, auxiliary_horizons=[5, 63])
        main, auxiliaries = model.forward_with_auxiliaries(torch.randn(4, 6, 7))
        self.assertEqual(main.shape, (4,))
        self.assertEqual(set(auxiliaries), {"h5", "h63"})
        for output in auxiliaries.values():
            self.assertEqual(output.shape, (4,))

        plain = StockReturnPredictor(input_size=7, hidden_size=8)
        self.assertEqual(len(plain.auxiliary_heads), 0)
        self.assertEqual(plain(torch.randn(2, 6, 7)).shape, (2,))

    def test_main_forward_is_unaffected_by_auxiliary_heads(self):
        torch.manual_seed(3)
        model = StockReturnPredictor(input_size=5, hidden_size=6, auxiliary_horizons=[5])
        model.eval()
        x = torch.randn(3, 4, 5)
        main_only = model(x)
        main_with, _ = model.forward_with_auxiliaries(x)
        torch.testing.assert_close(main_only, main_with)


# ---------------------------------------------------------------------------
# Phase 4: target decomposition and total-return reconstruction
# ---------------------------------------------------------------------------


class TargetDecompositionTest(unittest.TestCase):
    def test_beta_neutral_residual_removes_the_beta_weighted_market_move(self):
        panel = make_panel(dates=100, tickers=5)
        cfg = resolve_target_config({"mode": "beta_neutral_residual"})
        out = add_model_target(panel, 21, cfg)

        self.assertEqual(target_component_column(cfg), RESIDUAL_RETURN_COLUMN)
        expected = (
            out["future_return"].to_numpy()
            - out["target_beta"].to_numpy() * out["benchmark_future_return"].to_numpy()
        )
        np.testing.assert_allclose(out[RESIDUAL_RETURN_COLUMN].to_numpy(), expected, atol=1e-12)

    def test_beta_is_clipped_so_one_bad_estimate_cannot_explode_a_target(self):
        panel = make_panel(dates=60, tickers=3)
        panel.loc[panel.index[0], "market_beta_60d"] = 50.0
        panel.loc[panel.index[1], "market_beta_60d"] = -20.0
        beta = clipped_beta(panel, {"mode": "beta_neutral_residual", "beta_clip": [0.0, 3.0]})
        self.assertLessEqual(float(beta.max()), 3.0)
        self.assertGreaterEqual(float(beta.min()), 0.0)

    def test_missing_beta_falls_back_to_one(self):
        panel = make_panel(dates=40, tickers=3).drop(columns=["market_beta_60d"])
        beta = clipped_beta(panel, {"mode": "beta_neutral_residual"})
        np.testing.assert_allclose(beta, 1.0)

    def test_total_return_reconstruction_is_an_exact_identity(self):
        market = np.array([0.01, 0.02, -0.01])
        residual = np.array([0.005, -0.002, 0.003])
        beta = np.array([1.2, 0.8, 1.0])
        parts = compose_hierarchical_return(market, residual, beta=beta)
        np.testing.assert_allclose(
            parts["total"],
            parts["market_component"] + parts["sector_component"] + parts["residual_component"],
        )
        np.testing.assert_allclose(parts["market_component"], beta * market)

    def test_decomposed_target_requires_benchmark_forward_returns(self):
        panel = make_panel(dates=40, tickers=3).drop(columns=["benchmark_future_return"])
        with self.assertRaises(KeyError):
            add_model_target(panel, 21, {"mode": "beta_neutral_residual"})


class MarketModelTest(unittest.TestCase):
    def test_market_frame_has_one_row_per_date(self):
        panel = make_panel(dates=200, tickers=9)
        frame = build_market_frame(panel, ["benchmark_return_20d", "marketstate_pct_above_sma50"])
        self.assertEqual(len(frame), panel.index.nunique())

    def test_zero_shrinkage_reproduces_the_constant_drift_exactly(self):
        panel = make_panel(dates=300, tickers=6)
        model = MarketReturnModel(drift=0.0123, shrinkage=0.0, feature_columns=["x"])
        frame = pd.DataFrame({"x": np.arange(10, dtype=float)})
        np.testing.assert_allclose(model.predict(frame), np.full(10, 0.0123))

    def test_a_useless_market_model_gets_zero_weight(self):
        """Validation without skill must collapse the market leg to the drift."""
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2015-01-01", periods=900)
        frame = pd.DataFrame(
            {
                "Ticker": "AAA",
                "marketstate_pct_above_sma50": rng.normal(0, 1, len(dates)),
                "benchmark_future_return": rng.normal(0.01, 0.04, len(dates)),
            },
            index=dates,
        )
        train, validation = frame.iloc[:700], frame.iloc[700:]
        model = fit_market_return_model(
            train, validation, ["marketstate_pct_above_sma50"], maximum_shrinkage=0.6
        )
        self.assertGreaterEqual(model.shrinkage, 0.0)
        self.assertLessEqual(model.shrinkage, 0.6)

    def test_non_finite_features_do_not_produce_a_nan_forecast(self):
        model = MarketReturnModel(
            drift=0.01,
            shrinkage=0.5,
            feature_columns=["a", "b"],
            coefficients={"a": 1.0, "b": -1.0},
            intercept=0.0,
            feature_mean=[0.0, 0.0],
            feature_std=[1.0, 1.0],
        )
        frame = pd.DataFrame({"a": [np.inf, np.nan, 1.0], "b": [1.0, 2.0, np.nan]})
        predictions = model.predict(frame)
        self.assertTrue(np.all(np.isfinite(predictions)))


# ---------------------------------------------------------------------------
# Phase 6: boosting round selection
# ---------------------------------------------------------------------------


class RoundSelectionTest(unittest.TestCase):
    def curve(self, best_rounds: int, ic_at_best: float = 0.05) -> list[dict]:
        rows = []
        for rounds in (1, 5, 10, 20, 40, 80):
            distance = abs(np.log(rounds) - np.log(best_rounds))
            rows.append(
                {
                    "rounds": rounds,
                    "cross_sectional_ic": ic_at_best - 0.01 * distance,
                    "mse_skill": 0.01,
                }
            )
        return rows

    def test_no_minimum_round_floor_is_applied(self):
        """
        Regression guard for the reported bug.

        The previous code did ``max(50, best_iteration + 1)``, so a selection of
        one round silently became fifty and the metadata described a model that
        was never used.
        """
        selection = select_rounds([self.curve(1)])
        self.assertEqual(selection.rounds, 1)
        self.assertNotEqual(selection.rounds, 50)
        self.assertIn("no minimum-round floor", selection.to_dict()["note"])

    def test_selection_prefers_the_consistent_count(self):
        # Fold A likes 5 rounds, fold B likes 80; the stable middle should win
        # over either extreme once the dispersion penalty is applied.
        selection = select_rounds([self.curve(5), self.curve(80)])
        self.assertIn(selection.rounds, (10, 20, 40))

    def test_only_rungs_present_in_every_fold_are_eligible(self):
        short = [{"rounds": 1, "cross_sectional_ic": 0.9, "mse_skill": 0.0}]
        long = self.curve(20)
        selection = select_rounds([short, long])
        self.assertEqual(selection.rounds, 1)

    def test_ladder_is_bounded_by_what_was_fitted(self):
        self.assertEqual(round_ladder(20, (1, 5, 10, 20, 40, 80)), [1, 5, 10, 20])
        self.assertIn(37, round_ladder(37))

    def test_round_score_requires_both_kinds_of_skill(self):
        ranking_only = round_score({"cross_sectional_ic": 0.05, "mse_skill": -0.5})
        balanced = round_score({"cross_sectional_ic": 0.05, "mse_skill": 0.05})
        self.assertLess(ranking_only, balanced)

    def test_empty_input_returns_a_documented_default(self):
        selection = select_rounds([])
        self.assertGreater(selection.rounds, 0)
        self.assertIn("no fold curves", selection.reason)


# ---------------------------------------------------------------------------
# Phase 8: calibration
# ---------------------------------------------------------------------------


class CalibrationTest(unittest.TestCase):
    def signal_and_outcome(self, n: int = 3000, slope: float = 0.3, seed: int = 0):
        rng = np.random.default_rng(seed)
        dates = np.repeat(pd.bdate_range("2018-01-01", periods=n // 30), 30)[:n]
        predicted = rng.normal(0, 0.05, n)
        truth = slope * predicted + rng.normal(0, 0.05, n)
        return truth, predicted, dates

    def test_a_flattening_calibration_is_rejected(self):
        """
        Regression guard: a negative or near-zero slope collapses the model.

        A least-squares fit on pure noise improves MAE by shrinking every
        prediction towards a constant, which deletes the only thing the model
        produced.
        """
        rng = np.random.default_rng(1)
        n = 4000
        predicted = rng.normal(0, 0.05, n)
        truth = -0.5 * predicted + rng.normal(0, 0.01, n)  # inverted relationship
        dates = np.repeat(pd.bdate_range("2018-01-01", periods=n // 20), 20)[:n]

        candidates = fit_calibration_candidates(truth, predicted, dates)
        for name, calibration in candidates.items():
            if name == "identity":
                continue
            applied = calibration.apply(predicted, dates)
            # Nothing may invert the ordering the model produced.
            if np.std(applied) > 0:
                correlation = np.corrcoef(applied, predicted)[0, 1]
                self.assertGreater(correlation, 0.0, f"{name} inverted the ranking")

    def test_cross_sectional_centering_removes_the_per_date_mean(self):
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        dates = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3)
        centred = cross_sectional_center(values, dates)
        self.assertAlmostEqual(float(centred[:3].mean()), 0.0, places=12)
        self.assertAlmostEqual(float(centred[3:].mean()), 0.0, places=12)
        # Ordering within each date is preserved.
        self.assertTrue(np.all(np.diff(centred[:3]) > 0))

    def test_residual_leg_carries_no_common_intercept(self):
        truth, predicted, dates = self.signal_and_outcome()
        candidates = fit_calibration_candidates(
            truth + 0.05, predicted, dates, allow_intercept=False, cross_sectional_centering=True
        )
        for name, calibration in candidates.items():
            if calibration.method in {"affine", "ridge"}:
                self.assertFalse(
                    calibration.allow_intercept, f"{name} would add a common intercept"
                )
                applied = calibration.apply(predicted, dates)
                # With centring plus no intercept the alpha averages to ~zero.
                self.assertLess(abs(float(applied.mean())), 0.01)

    def test_shrinkage_reduces_magnitude_without_changing_order(self):
        calibration = ReturnCalibration(method="affine", slope=1.0, shrinkage=0.5)
        values = np.array([-0.04, -0.01, 0.02, 0.06])
        applied = calibration.apply(values)
        np.testing.assert_allclose(applied, values * 0.5)
        self.assertTrue(np.all(np.diff(applied) > 0))

    def test_selection_prefers_consistency_across_folds(self):
        truth_a, predicted_a, dates_a = self.signal_and_outcome(seed=1)
        truth_b, predicted_b, dates_b = self.signal_and_outcome(seed=2)
        folds = [
            fit_calibration_candidates(truth_a, predicted_a, dates_a),
            fit_calibration_candidates(truth_b, predicted_b, dates_b),
        ]
        name, report = select_calibration(folds)
        self.assertIn(name, folds[0])
        self.assertEqual(report["rule"], "mean(out-of-fold score) - std(out-of-fold score)")
        self.assertEqual(report["folds"], 2)

    def test_identity_wins_when_nothing_beats_it(self):
        name, _ = select_calibration([fit_calibration_candidates(np.zeros(10), np.zeros(10), None)])
        self.assertEqual(name, "identity")

    def test_decile_report_has_the_required_columns(self):
        truth, predicted, _ = self.signal_and_outcome()
        rows = decile_calibration_report(truth, predicted, deciles=10)
        self.assertEqual(len(rows), 10)
        for row in rows:
            for key in (
                "decile",
                "count",
                "mean_predicted_return",
                "mean_realized_return",
                "realized_standard_error",
                "directional_hit_rate",
            ):
                self.assertIn(key, row)
        self.assertEqual(sum(row["count"] for row in rows), len(truth))

    def test_decile_summary_recovers_a_known_slope(self):
        truth, predicted, _ = self.signal_and_outcome(n=20000, slope=0.3, seed=3)
        summary = calibration_monotonicity(decile_calibration_report(truth, predicted))
        self.assertAlmostEqual(summary["realized_slope_vs_predicted"], 0.3, delta=0.12)
        self.assertGreater(summary["rank_correlation"], 0.9)


# ---------------------------------------------------------------------------
# Purged splits and walk-forward refitting
# ---------------------------------------------------------------------------


class PurgedSplitTest(unittest.TestCase):
    def test_holdout_split_leaves_a_purge_gap(self):
        panel = make_panel(dates=500, tickers=4)
        train, validation, test = chronological_train_validation_test_split(
            panel, 0.7, 0.15, purge_horizon=21
        )
        self.assertLess(train.index.max(), validation.index.min())
        self.assertLess(validation.index.max(), test.index.min())
        gap = len(
            [d for d in panel.index.unique() if train.index.max() < d < validation.index.min()]
        )
        self.assertGreaterEqual(gap, 1)

    def test_walk_forward_windows_move_forward_and_never_overlap(self):
        panel = make_panel(dates=800, tickers=4)
        splits = purged_walk_forward_splits(
            panel.index.unique(), folds=3, purge_horizon=21, expanding=True
        )
        self.assertEqual(len(splits), 3)
        previous_end = None
        for split in splits:
            self.assertGreater(len(split.train_dates), 0)
            self.assertGreater(len(split.test_dates), 0)
            self.assertLess(max(split.train_dates), min(split.validation_dates))
            self.assertLess(max(split.validation_dates), min(split.test_dates))
            if previous_end is not None:
                self.assertGreater(min(split.test_dates), previous_end)
            previous_end = max(split.test_dates)

    def test_expanding_origin_grows_the_training_window(self):
        panel = make_panel(dates=900, tickers=3)
        splits = purged_walk_forward_splits(
            panel.index.unique(), folds=3, purge_horizon=21, expanding=True
        )
        sizes = [len(split.train_dates) for split in splits]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_no_training_date_appears_in_any_test_window(self):
        panel = make_panel(dates=700, tickers=3)
        for split in purged_walk_forward_splits(
            panel.index.unique(), folds=3, purge_horizon=21
        ):
            self.assertEqual(set(split.train_dates) & set(split.test_dates), set())
            self.assertEqual(set(split.validation_dates) & set(split.test_dates), set())


if __name__ == "__main__":
    unittest.main()
