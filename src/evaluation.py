"""
Shared evaluation and reporting for both model families.

Both trainers hand this module the same thing — a table of per-row forecasts —
and it does everything downstream: calibration, hierarchical composition,
interval construction, metrics, baselines, decision-rule tuning, backtesting and
the acceptance checklist.

That is the point. When the LSTM and XGBoost reports are produced by the same
code, a difference between them is a difference between the models. When each
trainer implements its own evaluation, a difference can just as easily be a
difference between two evaluation implementations, and there is no way to tell
which from the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.acceptance import calibration_stability, evaluate_acceptance_gates, format_acceptance_table
from src.backtest import backtest_signals, build_signal_frame, tune_decision_config
from src.calibration import (
    ReturnCalibration,
    calibration_monotonicity,
    decile_calibration_report,
    fit_calibration_candidates,
    select_calibration,
)
from src.decision import DecisionConfig
from src.market_model import MarketReturnModel, compose_hierarchical_return
from src.portfolio import PortfolioConfig, build_portfolio_frame, run_all_offsets, run_top_k_backtest
from src.regression import (
    EXCESS_RETURN_COLUMN,
    RESIDUAL_RETURN_COLUMN,
    TOTAL_RETURN_COLUMN,
    chronological_block_metrics,
    evaluate_baselines,
    full_metrics,
)
from src.training_common import build_backtest_config, build_base_decision_config, strip_arrays
from src.uncertainty import (
    conditional_coverage_report,
    fit_interval_calibration,
    interval_metrics,
    uncertainty_filter_benefit,
)


# The columns every forecast table must provide. Named explicitly so a trainer
# that forgets one fails immediately with a clear message rather than producing a
# silently wrong report.
REQUIRED_COLUMNS = (
    "ticker",
    "date",
    "true_total_return",
    "true_component_return",
    "raw_prediction",
    "model_std",
    "target_scale",
)


@dataclass
class ForecastTable:
    """One split's per-row forecasts, in the modelled component's return space."""

    name: str
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in self.frame.columns]
        if missing:
            raise KeyError(f"ForecastTable '{self.name}' is missing columns: {missing}")
        self.frame = self.frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def dates(self) -> pd.Series:
        return pd.to_datetime(self.frame["date"])

    def column(self, name: str, default: float = 0.0) -> np.ndarray:
        if name not in self.frame.columns:
            return np.full(len(self.frame), float(default), dtype=float)
        values = self.frame[name].astype(float).to_numpy()
        return np.where(np.isfinite(values), values, float(default))

    def metadata(self) -> list[dict]:
        return [
            {"ticker": str(ticker), "date": str(pd.Timestamp(date).date())}
            for ticker, date in zip(self.frame["ticker"], self.frame["date"])
        ]


def build_forecast_table(
    name: str,
    tickers,
    dates,
    true_total_return,
    true_component_return,
    raw_prediction,
    model_std,
    target_scale,
    beta=None,
    sector=None,
    volatility_scale=None,
    market_return_forecast=None,
) -> ForecastTable:
    """Assemble a forecast table from aligned arrays."""
    frame = pd.DataFrame(
        {
            "ticker": np.asarray(tickers, dtype=object),
            "date": pd.to_datetime(pd.Series(list(dates))).to_numpy(),
            "true_total_return": np.asarray(true_total_return, dtype=float),
            "true_component_return": np.asarray(true_component_return, dtype=float),
            "raw_prediction": np.asarray(raw_prediction, dtype=float),
            "model_std": np.asarray(model_std, dtype=float),
            "target_scale": np.asarray(target_scale, dtype=float),
        }
    )
    if beta is not None:
        frame["beta"] = np.asarray(beta, dtype=float)
    if sector is not None:
        frame["sector"] = np.asarray(sector, dtype=object)
    if volatility_scale is not None:
        frame["volatility_scale"] = np.asarray(volatility_scale, dtype=float)
    if market_return_forecast is not None:
        frame["market_return_forecast"] = np.asarray(market_return_forecast, dtype=float)

    # A row with no realised label cannot be scored and must not be silently
    # counted as a correct or incorrect forecast.
    frame = frame.dropna(subset=["true_total_return", "true_component_return", "raw_prediction"])
    return ForecastTable(name=name, frame=frame)


@dataclass
class EvaluationResult:
    """Everything a training run needs to report and to persist."""

    payload: dict = field(default_factory=dict)
    signal_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: ReturnCalibration = field(default_factory=ReturnCalibration)
    interval_calibration: object = None
    decision_config: DecisionConfig | None = None


def evaluate_model(
    config: dict,
    horizon: int,
    model_name: str,
    validation: ForecastTable,
    test: ForecastTable,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    market_model: MarketReturnModel,
    target_config: dict,
    component_column: str,
    walk_forward: dict | None = None,
    calibration_folds: list[dict] | None = None,
    extra: dict | None = None,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Run the complete evaluation for one fitted model.

    ``calibration_folds`` holds out-of-fold predictions from walk-forward, and is
    what the calibration family is selected on. When it is absent the selection
    falls back to the validation split — still never the test split — and the
    payload says so, because a calibration chosen on one window deserves less
    trust than one that held up across folds.
    """
    uncertainty_cfg = config.get("uncertainty", {}) or {}
    confidence_level = float(uncertainty_cfg.get("confidence_level", 0.80))
    minimum_sigma = float(uncertainty_cfg.get("minimum_sigma", 0.005))

    # ------------------------------------------------------------------
    # 1. Return calibration, selected out of fold.
    # ------------------------------------------------------------------
    residual_leg = component_column in {EXCESS_RETURN_COLUMN, RESIDUAL_RETURN_COLUMN}
    calibration_kwargs = {
        # The residual leg is a relative view: no common intercept, and centred
        # cross-sectionally so it sums to roughly zero across the universe.
        "allow_intercept": not residual_leg,
        "cross_sectional_centering": residual_leg,
    }

    if calibration_folds:
        fold_candidates = [
            fit_calibration_candidates(
                fold["true_return"], fold["predicted_return"], fold["dates"], **calibration_kwargs
            )
            for fold in calibration_folds
        ]
        selected_name, calibration_report = select_calibration(fold_candidates)
        calibration_report["source"] = "purged walk-forward out-of-fold predictions"
        validation_candidates = fit_calibration_candidates(
            validation.column("true_component_return"),
            validation.column("raw_prediction"),
            validation.dates,
            **calibration_kwargs,
        )
        calibration = validation_candidates.get(selected_name, ReturnCalibration())
    else:
        validation_candidates = fit_calibration_candidates(
            validation.column("true_component_return"),
            validation.column("raw_prediction"),
            validation.dates,
            **calibration_kwargs,
        )
        selected_name, calibration_report = select_calibration([validation_candidates])
        calibration_report["source"] = (
            "validation split only (no walk-forward folds were available); a "
            "single-window calibration is weaker evidence than a cross-fold one"
        )
        calibration = validation_candidates.get(selected_name, ReturnCalibration())

    # A candidate chosen on the folds can still be rejected when re-fitted on the
    # validation window (the rank-preservation guard applies there too). When that
    # happens the transform is the identity, and the log has to say so rather than
    # claim a calibration is in force that is not.
    calibration_report["fold_selected_candidate"] = selected_name
    calibration_report["applied_method"] = calibration.method
    calibration_report["fold_selection_overridden_on_validation"] = bool(
        selected_name != "identity" and calibration.method == "identity"
    )
    if verbose:
        print(f"\nReturn calibration selected out of fold: {selected_name}")
        if calibration_report["fold_selection_overridden_on_validation"]:
            print(f"  overridden on validation -> identity: {calibration.reason}")
        else:
            print(f"  applied: {calibration.method} | {calibration.reason}")

    validation_component = calibration.apply(
        validation.column("raw_prediction"), validation.dates
    )
    test_component = calibration.apply(test.column("raw_prediction"), test.dates)

    # ------------------------------------------------------------------
    # 2. Prediction intervals, calibrated on validation only.
    # ------------------------------------------------------------------
    interval_calibration = fit_interval_calibration(
        validation.column("true_component_return"),
        validation_component,
        validation.column("model_std"),
        validation.column("target_scale"),
        confidence_level=confidence_level,
        minimum_sigma=minimum_sigma,
    )
    validation_lower, validation_upper, validation_sigma = interval_calibration.interval(
        validation_component, validation.column("model_std"), validation.column("target_scale")
    )
    test_lower, test_upper, test_sigma = interval_calibration.interval(
        test_component, test.column("model_std"), test.column("target_scale")
    )

    if verbose:
        print(
            f"Interval calibration: level {confidence_level:.0%} | "
            f"multiplier {interval_calibration.conformal_multiplier:.3f} | "
            f"validation coverage {interval_calibration.validation_coverage:.3f}"
        )

    # ------------------------------------------------------------------
    # 3. Hierarchical composition into the user-facing total return.
    # ------------------------------------------------------------------
    def compose(table: ForecastTable, component: np.ndarray) -> dict[str, np.ndarray]:
        market_forecast = table.column("market_return_forecast", default=market_model.drift)
        beta = table.column("beta", default=1.0) if residual_leg else np.ones(len(table))
        if component_column == EXCESS_RETURN_COLUMN:
            # A plain market-excess target already assumes beta 1, so applying a
            # per-stock beta here would double-count the market exposure.
            beta = np.ones(len(table))
        return compose_hierarchical_return(market_forecast, component, beta=beta)

    validation_parts = compose(validation, validation_component)
    test_parts = compose(test, test_component)
    validation_total = validation_parts["total"]
    test_total = test_parts["total"]

    train_component_mean = float(train_df[component_column].astype(float).mean())
    train_total_mean = float(train_df[TOTAL_RETURN_COLUMN].astype(float).mean())

    component_metrics = {
        "validation": strip_arrays(
            full_metrics(
                validation.dates,
                validation.column("true_component_return"),
                validation_component,
                horizon,
                reference_prediction=train_component_mean,
            )
        ),
        "test": strip_arrays(
            full_metrics(
                test.dates,
                test.column("true_component_return"),
                test_component,
                horizon,
                reference_prediction=train_component_mean,
            )
        ),
    }
    total_metrics = {
        "validation": strip_arrays(
            full_metrics(
                validation.dates,
                validation.column("true_total_return"),
                validation_total,
                horizon,
                reference_prediction=train_total_mean,
            )
        ),
        "test": strip_arrays(
            full_metrics(
                test.dates,
                test.column("true_total_return"),
                test_total,
                horizon,
                reference_prediction=train_total_mean,
            )
        ),
    }

    # ------------------------------------------------------------------
    # 4. Decision rule (validation only) and the signal backtest.
    # ------------------------------------------------------------------
    backtest_cfg = build_backtest_config(config)
    decision_cfg, decision_tuning = tune_decision_config(
        metadata=validation.metadata(),
        true_return=validation.column("true_total_return"),
        predicted_return=validation_total,
        sigma=validation_sigma,
        cfg=backtest_cfg,
        horizon=horizon,
        base_decision_cfg=build_base_decision_config(config),
        min_active_trades=int(config.get("threshold_min_active_trades", 20)),
    )
    if verbose:
        print(
            f"Validation-selected decision rule: {decision_cfg.rule} | "
            f"threshold {decision_cfg.threshold * 100:.2f}% | min z {decision_cfg.min_z_score:.2f}"
        )

    signal_frame = build_signal_frame(
        metadata=test.metadata(),
        true_return=test.column("true_total_return"),
        predicted_return=test_total,
        cfg=backtest_cfg,
        threshold=decision_cfg.threshold,
        sigma=test_sigma,
        decision_cfg=decision_cfg,
    )
    signal_frame["lower_bound"] = test_lower + test_parts["market_component"]
    signal_frame["upper_bound"] = test_upper + test_parts["market_component"]
    signal_frame["market_component"] = test_parts["market_component"]
    signal_frame["residual_component"] = test_parts["residual_component"]
    signal_frame["sector_component"] = test_parts["sector_component"]

    backtest_frame, backtest_metrics = backtest_signals(signal_frame, backtest_cfg, horizon=horizon)

    point_frame = build_signal_frame(
        metadata=test.metadata(),
        true_return=test.column("true_total_return"),
        predicted_return=test_total,
        cfg=backtest_cfg,
        threshold=decision_cfg.threshold,
        sigma=None,
    )
    _, point_backtest_metrics = backtest_signals(point_frame, backtest_cfg, horizon=horizon)

    # ------------------------------------------------------------------
    # 5. Fully invested portfolio backtest, every rebalance offset.
    # ------------------------------------------------------------------
    portfolio_cfg = config.get("portfolio", {}) or {}
    sector_map = config.get("ticker_sectors")
    portfolio_frame = build_portfolio_frame(
        signal_frame,
        score_column="predicted_return",
        return_column="true_return",
        sector_map=sector_map,
        beta_series=test.column("beta", default=1.0),
    )

    def portfolio_variant(neutralize: str, long_short: bool) -> dict:
        cfg = PortfolioConfig(
            top_k=int(portfolio_cfg.get("top_k", 20)),
            top_quantile=float(portfolio_cfg.get("top_quantile", 0.20)),
            long_short=bool(long_short),
            transaction_cost_pct=float(config["transaction_cost_pct"]),
            slippage_pct=float(config["slippage_pct"]),
            initial_cash=float(config["initial_cash"]),
            neutralize=neutralize,
        )
        return run_all_offsets(portfolio_frame, cfg, horizon)

    long_only_cfg = PortfolioConfig(
        top_k=int(portfolio_cfg.get("top_k", 20)),
        top_quantile=float(portfolio_cfg.get("top_quantile", 0.20)),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        initial_cash=float(config["initial_cash"]),
    )
    portfolio_history, _ = run_top_k_backtest(portfolio_frame, long_only_cfg, horizon, offset=0)
    portfolio_report = {
        "top_k_long_only": portfolio_variant("none", False),
        "top_k_sector_neutral": portfolio_variant("sector", False),
        "top_k_beta_neutral": portfolio_variant("beta", False),
    }
    if bool(config.get("allow_short", False)):
        portfolio_report["long_short"] = portfolio_variant("none", True)

    if verbose:
        distribution = portfolio_report["top_k_long_only"].get("distribution", {})
        if distribution:
            sharpe = distribution["sharpe_ratio"]
            information = distribution["information_ratio_vs_universe"]
            print(
                f"\nFully invested top-{long_only_cfg.top_k} portfolio across "
                f"{portfolio_report['top_k_long_only']['offsets_evaluated']} rebalance offsets:"
            )
            print(
                f"  Sharpe mean {sharpe['mean']:+.3f} median {sharpe['median']:+.3f} "
                f"worst {sharpe['worst']:+.3f} best {sharpe['best']:+.3f}"
            )
            print(
                f"  Information ratio vs universe: mean {information['mean']:+.3f} "
                f"worst {information['worst']:+.3f}"
            )

    # ------------------------------------------------------------------
    # 6. Uncertainty diagnostics.
    # ------------------------------------------------------------------
    volatility_scale = test.column("volatility_scale", default=0.0)
    if not np.any(volatility_scale):
        volatility_scale = test.column("target_scale", default=1.0)

    uncertainty_report = {
        "confidence_level": confidence_level,
        "calibration": interval_calibration.to_dict(),
        "validation_interval_metrics": interval_metrics(
            validation.column("true_component_return"),
            validation_lower,
            validation_upper,
            confidence_level,
        ),
        "test_interval_metrics": interval_metrics(
            test.column("true_component_return"), test_lower, test_upper, confidence_level
        ),
        "conditional_coverage": conditional_coverage_report(
            test.column("true_component_return"),
            test_lower,
            test_upper,
            test_component,
            volatility_scale,
            confidence_level=confidence_level,
        ),
        "filter_benefit": uncertainty_filter_benefit(
            test.dates,
            test.column("true_component_return"),
            test_component,
            test_sigma,
            horizon=horizon,
        ),
        "mean_epistemic_std": float(np.mean(test.column("model_std"))),
        "mean_total_sigma": float(np.mean(test_sigma)),
        "epistemic_share_of_variance": float(
            np.mean(np.square(test.column("model_std")))
            / max(float(np.mean(np.square(test_sigma))), 1e-12)
        ),
    }

    if verbose:
        test_interval = uncertainty_report["test_interval_metrics"]
        conditional = uncertainty_report["conditional_coverage"]
        print(
            f"\nTest interval coverage {test_interval['coverage_picp']:.3f} "
            f"(nominal {confidence_level:.2f}) | width "
            f"{test_interval['mean_interval_width_mpiw'] * 100:.2f}%"
        )
        print(
            f"  worst conditional coverage error: volatility "
            f"{conditional['worst_absolute_coverage_error_by_volatility']:.3f} | "
            f"magnitude {conditional['worst_absolute_coverage_error_by_magnitude']:.3f}"
        )
        benefit = uncertainty_report["filter_benefit"]
        if benefit.get("evaluated"):
            print(
                f"  filtering by confidence changes IC by "
                f"{benefit['ic_improvement_from_filtering']:+.4f} "
                f"({'useful' if benefit['filtering_helps'] else 'no benefit'})"
            )

    # ------------------------------------------------------------------
    # 7. Baselines, decile calibration, regime blocks, acceptance gates.
    # ------------------------------------------------------------------
    baselines_total = evaluate_baselines(
        train_df, test_df, horizon, TOTAL_RETURN_COLUMN, feature_columns
    )
    baselines_component = evaluate_baselines(
        train_df, test_df, horizon, component_column, feature_columns
    )

    decile_rows = decile_calibration_report(test.column("true_total_return"), test_total)
    decile_component = decile_calibration_report(
        test.column("true_component_return"), test_component
    )
    regime_blocks = chronological_block_metrics(
        test.metadata(),
        test.column("true_component_return"),
        test_component,
        blocks=int(config.get("evaluation_regime_blocks", 4)),
        horizon=horizon,
    )

    fold_records = (walk_forward or {}).get("folds") or [{}]
    stability = calibration_stability(
        [
            {"calibration": calibration.method, "decision_rule": decision_cfg.rule}
            for _ in fold_records
        ]
    )
    acceptance = evaluate_acceptance_gates(
        walk_forward=walk_forward,
        test_metrics=component_metrics["test"],
        baselines=baselines_component,
        interval_metrics=uncertainty_report["test_interval_metrics"],
        portfolio=portfolio_report["top_k_long_only"],
        regime_blocks=regime_blocks,
        calibration_stability=stability,
        thresholds=config.get("acceptance_thresholds"),
    )
    if verbose:
        print()
        print(format_acceptance_table(acceptance))

    payload = {
        "model_name": model_name,
        "prediction_horizon": horizon,
        "validation_size": int(len(validation)),
        "test_size": int(len(test)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_configuration": target_config,
        "modelled_component": component_column,
        "market_model": market_model.to_dict(),
        "hierarchical_decomposition": {
            "formula": "beta * expected_market_return + expected_sector_return + expected_stock_residual",
            "mean_market_component": float(np.mean(test_parts["market_component"])),
            "mean_sector_component": float(np.mean(test_parts["sector_component"])),
            "mean_residual_component": float(np.mean(test_parts["residual_component"])),
            "mean_total": float(np.mean(test_total)),
            "mean_beta_applied": float(np.mean(test_parts["beta"])),
            "sector_component_implemented": False,
            "sector_component_note": (
                "sector-relative information enters through the feature set "
                "(sector composites, sector beta, sector-relative momentum) rather "
                "than as a separately forecast leg; the leg is reported as zero so "
                "the composition stays an identity"
            ),
        },
        "return_calibration": calibration.to_dict(),
        "return_calibration_selection": calibration_report,
        "component_metrics": component_metrics,
        "total_return_metrics": total_metrics,
        "validation_metrics": total_metrics["validation"],
        "test_metrics": total_metrics["test"],
        "regression_baselines": baselines_total,
        "regression_baselines_component": baselines_component,
        "decile_calibration_total_return": decile_rows,
        "decile_calibration_component": decile_component,
        "decile_calibration_summary": calibration_monotonicity(decile_rows),
        "uncertainty": uncertainty_report,
        "decision_rule": decision_cfg.to_dict(),
        "decision_rule_tuning": decision_tuning,
        "backtest_metrics": backtest_metrics,
        "backtest_metrics_point_rule_ablation": point_backtest_metrics,
        "portfolio_backtest": portfolio_report,
        "test_regime_blocks": regime_blocks,
        "walk_forward": walk_forward,
        "acceptance_gates": acceptance,
        "split_method": "purged_chronological_holdout",
        "purge_trading_days": horizon if bool(config.get("purge_overlapping_labels", True)) else 0,
        "holdout_policy": (
            "the test period is treated as a development holdout: it is reported "
            "but never used to select features, hyperparameters, losses, "
            "calibration, blend weights or decision thresholds"
        ),
    }
    if extra:
        payload.update(extra)

    return EvaluationResult(
        payload=payload,
        signal_frame=signal_frame,
        backtest_frame=backtest_frame,
        portfolio_frame=portfolio_history,
        calibration=calibration,
        interval_calibration=interval_calibration,
        decision_config=decision_cfg,
    )


def print_decile_table(rows: list[dict], title: str = "Decile calibration") -> None:
    """Print the predicted-versus-realised table that reveals magnitude skill."""
    if not rows:
        return
    print(f"\n{title}:")
    print(
        f"  {'decile':>6} {'count':>7} {'predicted':>11} {'realized':>11} "
        f"{'std err':>9} {'hit rate':>9}"
    )
    print("  " + "-" * 58)
    for row in rows:
        print(
            f"  {row['decile']:>6} {row['count']:>7} "
            f"{row['mean_predicted_return'] * 100:>10.2f}% "
            f"{row['mean_realized_return'] * 100:>10.2f}% "
            f"{row['realized_standard_error'] * 100:>8.2f}% "
            f"{row['directional_hit_rate']:>8.1%}"
        )
