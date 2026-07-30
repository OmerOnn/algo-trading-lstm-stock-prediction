"""
Inference: predict the expected future return of one or more tickers.

Every prediction carries an uncertainty range. The point forecast alone is not
actionable for equity returns, where the irreducible noise is an order of
magnitude larger than any attainable edge; the interval is what tells the user
whether the number means anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.backtest import BacktestConfig, cost_aware_signal_threshold
from src.data_download import download_earnings_data, download_macro_data, download_price_data
from src.decision import DecisionConfig, decide, direction_probability
from src.device import get_best_device
from src.features import (
    add_benchmark_features,
    add_macro_features,
    add_earnings_features,
    add_regime_normalized_features,
    add_technical_indicators,
)
from src.model import StockReturnPredictor
from src.panel_features import add_panel_features
from src.calibration import ReturnCalibration
from src.regression import apply_return_calibration, resolve_target_config, target_scale
from src.uncertainty import IntervalCalibration, describe_confidence, mc_dropout_predict


ROOT = Path(__file__).resolve().parent


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_available_horizons(config: dict) -> list[int]:
    model_base = Path(config["model_output_path"])
    scaler_base = Path(config["scaler_output_path"])
    metadata_base = Path(config["metadata_output_path"])
    discovered: list[int] = []
    for model_path in (ROOT / model_base.parent).glob(f"{model_base.stem}_h*{model_base.suffix}"):
        raw_horizon = model_path.stem.rsplit("_h", 1)[-1]
        if not raw_horizon.isdigit():
            continue
        horizon = int(raw_horizon)
        scaler_path = ROOT / scaler_base.with_name(f"{scaler_base.stem}_h{horizon}{scaler_base.suffix}")
        metadata_path = ROOT / metadata_base.with_name(
            f"{metadata_base.stem}_h{horizon}{metadata_base.suffix}"
        )
        if scaler_path.exists() and metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    metadata = json.load(file)
                if int(metadata.get("artifact_schema_version", 1)) >= 3:
                    discovered.append(horizon)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    if discovered:
        return sorted(set(discovered))
    if "prediction_horizons" in config and config["prediction_horizons"]:
        return [int(h) for h in config["prediction_horizons"]]
    return [int(config.get("prediction_horizon", 21))]


def artifact_path(base_path: str | Path, horizon: int) -> Path:
    path = Path(base_path)
    return ROOT / path.with_name(f"{path.stem}_h{horizon}{path.suffix}")


def lstm_model_exists(config: dict, horizon: int) -> bool:
    """
    True only for artifacts this code can actually load.

    The schema check matters: checkpoints from before the architecture change
    exist on disk for several horizons but would fail to load into the current
    model, so treating "file present" as "model available" would surface a
    crash to the user instead of a clean "not trained yet" message.
    """
    try:
        _, _, metadata_path = resolve_artifact_paths(config, int(horizon))
    except FileNotFoundError:
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return int(json.load(file).get("artifact_schema_version", 1)) >= 3
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def available_models_for_horizon(config: dict, horizon: int) -> dict[str, bool]:
    """Which trained model families exist for a horizon, without raising."""
    return {
        "LSTM": lstm_model_exists(config, int(horizon)),
        "XGBoost": xgboost_model_exists(int(horizon)),
    }


def resolve_artifact_paths(config: dict, horizon: int) -> tuple[Path, Path, Path]:
    model_path = artifact_path(config["model_output_path"], horizon)
    scaler_path = artifact_path(config["scaler_output_path"], horizon)
    metadata_path = artifact_path(config["metadata_output_path"], horizon)

    if model_path.exists() and scaler_path.exists() and metadata_path.exists():
        return model_path, scaler_path, metadata_path

    default_horizon = int(config.get("default_prediction_horizon", config.get("prediction_horizon", 21)))
    if horizon == default_horizon:
        fallback = (
            ROOT / config["model_output_path"],
            ROOT / config["scaler_output_path"],
            ROOT / config["metadata_output_path"],
        )
        if all(path.exists() for path in fallback):
            return fallback

    raise FileNotFoundError(
        f"No trained model artifacts were found for horizon={horizon}. "
        f"Run: python3 train_lstm.py --horizon {horizon}"
    )


def _per_ticker_features(ticker: str, config: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Indicators, benchmark context, macro and trailing z-scores for one ticker."""
    price_df = download_price_data(ticker, config["start_date"], config["end_date"])
    benchmark_df = download_price_data(config["benchmark_ticker"], config["start_date"], config["end_date"])
    macro_df = pd.DataFrame()
    if config.get("macro_tickers"):
        macro_df = download_macro_data(config["macro_tickers"], config["start_date"], config["end_date"])

    df = add_technical_indicators(price_df)
    df = add_benchmark_features(df, benchmark_df)
    df = add_macro_features(df, macro_df)
    earnings_feature_names = {
        "is_earnings_day",
        "is_near_earnings",
        "days_since_earnings",
        "days_to_earnings",
        "eps_estimate",
        "reported_eps",
        "eps_surprise_pct",
    }
    if bool(config.get("use_earnings_features", False)) or earnings_feature_names.intersection(feature_columns):
        df = add_earnings_features(df, download_earnings_data(ticker))
    df = add_regime_normalized_features(df, window=int(config.get("regime_normalization_window", 252)))
    return df.replace([np.inf, -np.inf], np.nan)


# The universe panel is expensive to build and identical for every ticker in a
# request, so it is cached per (universe, date range) for the life of the process.
_UNIVERSE_PANEL_CACHE: dict[tuple, pd.DataFrame] = {}


def build_universe_panel(config: dict, feature_columns: list[str]) -> pd.DataFrame:
    """
    Build the whole universe's feature panel.

    Inference cannot be done one ticker at a time any more. Cross-sectional
    ranks, breadth, dispersion, average correlation and the sector composites are
    all defined *relative to the other stocks on the same date*, so producing
    them for one ticker in isolation is not possible — the cross-section is the
    feature. The panel is therefore built for the configured universe and the
    requested ticker's row is selected from it.
    """
    universe = tuple(config["tickers"])
    key = (universe, str(config["start_date"]), str(config.get("end_date")))
    if key in _UNIVERSE_PANEL_CACHE:
        return _UNIVERSE_PANEL_CACHE[key]

    frames = []
    for ticker in universe:
        try:
            frames.append(_per_ticker_features(ticker, config, feature_columns))
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not break inference
            print(f"Warning: could not build features for {ticker}: {exc}")
    if not frames:
        raise ValueError("No universe data could be built, so cross-sectional features are unavailable.")

    panel = pd.concat(frames).sort_index()
    panel = add_panel_features(
        panel,
        ticker_sectors=config.get("ticker_sectors"),
        minimum_sector_members=int(config.get("minimum_sector_members", 3)),
        regime_normalization_window=int(config.get("regime_normalization_window", 252)),
    )
    _UNIVERSE_PANEL_CACHE[key] = panel
    return panel


def build_latest_features(ticker: str, config: dict, feature_columns: list[str], horizon: int) -> pd.DataFrame:
    """Feature rows for one ticker, taken from the full universe panel."""
    panel = build_universe_panel(config, feature_columns)
    df = panel[panel["Ticker"].astype(str).str.upper() == str(ticker).upper()].copy()
    if df.empty:
        raise ValueError(
            f"{ticker} is not in the configured universe, so its cross-sectional "
            "features cannot be built. Add it to `tickers` in configs/config.yaml."
        )

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Feature engineering did not produce {len(missing)} required columns for {ticker}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return df.dropna(subset=feature_columns).copy()


def latest_row_beta(df: pd.DataFrame) -> float | None:
    """
    The stock's rolling market beta on the most recent row.

    The market leg of the decomposition is ``beta * expected_market_return``, so
    serving has to apply the same beta the target was built with. Returns None
    when the column is absent or non-finite, and the caller then falls back to a
    beta of 1.
    """
    if "market_beta_60d" not in df.columns or df.empty:
        return None
    value = float(df["market_beta_60d"].iloc[-1])
    return value if np.isfinite(value) else None


def build_backtest_config(config: dict) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        allow_short=bool(config["allow_short"]),
        signal_threshold_multiplier=float(config.get("signal_threshold_multiplier", 1.0)),
        min_signal_edge=float(config.get("min_signal_edge", 0.0)),
    )


def build_decision_config(config: dict, metadata: dict) -> DecisionConfig:
    """Rebuild the validation-frozen decision rule stored with the model."""
    stored = metadata.get("decision_rule") or {}
    fallback_threshold = cost_aware_signal_threshold(build_backtest_config(config))
    return DecisionConfig(
        rule=str(stored.get("rule", config.get("decision", {}).get("rule", "risk_adjusted"))),
        threshold=float(stored.get("threshold", fallback_threshold)),
        min_z_score=float(stored.get("min_z_score", 0.15)),
        allow_short=bool(stored.get("allow_short", config.get("allow_short", False))),
        min_direction_probability=float(stored.get("min_direction_probability", 0.0)),
        position_sizing=str(stored.get("position_sizing", "binary")),
    )


def assemble_prediction(
    ticker: str,
    latest_date: str,
    horizon: int,
    component_return: float,
    component_model_std: float,
    latest_scale: float,
    metadata: dict,
    config: dict,
    model_name: str,
    latest_beta: float | None = None,
) -> dict:
    """
    Turn a raw model output into the full user-facing prediction record.

    Shared by both model families so the LSTM and XGBoost outputs are directly
    comparable: same calibration order, same interval construction, same signal
    rule and the same confidence vocabulary.
    """
    calibration_payload = metadata.get("return_calibration") or {}
    if "method" in calibration_payload:
        # Schema 4: the full ReturnCalibration (affine / ridge / isotonic, with
        # shrinkage and optional cross-sectional centring). No dates are passed:
        # a single-ticker request has no cross-section, so the calibration falls
        # back to its stored centring offset instead of centring a value against
        # itself, which would return exactly zero.
        component_return = float(
            ReturnCalibration.from_dict(calibration_payload).apply(
                np.asarray([component_return])
            )[0]
        )
    else:
        # Schema 3 and earlier stored a plain affine {slope, intercept, enabled}.
        component_return = float(
            apply_return_calibration(np.asarray([component_return]), calibration_payload)[0]
        )

    interval_calibration = IntervalCalibration.from_dict(metadata.get("interval_calibration"))
    lower_component, upper_component, sigma = interval_calibration.interval(
        np.asarray([component_return]),
        np.asarray([component_model_std]),
        np.asarray([latest_scale]),
    )

    # The market leg. Schema 4 stores the whole stage-1 model; when its
    # fold-selected shrinkage is zero this is exactly the historical drift, which
    # is what older artifacts stored directly.
    market_payload = metadata.get("market_model") or metadata.get("market_drift") or {}
    market_drift = float(market_payload.get("drift", market_payload.get("market_drift", 0.0)))

    # Beta-weighted, matching the training-time decomposition. Falls back to 1.0
    # for artifacts trained on a plain market-excess target.
    beta = float(latest_beta if latest_beta is not None else 1.0)
    if str(metadata.get("modelled_component", "")) != "future_residual_return":
        beta = 1.0
    market_component = beta * market_drift

    expected_return = component_return + market_component
    lower = float(lower_component[0]) + market_component
    upper = float(upper_component[0]) + market_component
    sigma_value = float(sigma[0])

    decision_cfg = build_decision_config(config, metadata)
    signal = decide(expected_return, sigma_value, decision_cfg)
    confidence = describe_confidence(expected_return, sigma_value)

    return {
        "model": model_name,
        "ticker": ticker,
        "latest_data_date": latest_date,
        "prediction_horizon_trading_days": int(horizon),
        "signal": signal,
        "expected_return": expected_return,
        "expected_return_pct": expected_return * 100.0,
        "predicted_return": expected_return,
        "market_drift": market_drift,
        "market_drift_pct": market_drift * 100.0,
        # The hierarchical decomposition, reported component by component so the
        # user can see whether a forecast is a market call or a stock call.
        "market_component": market_component,
        "market_component_pct": market_component * 100.0,
        "sector_component": 0.0,
        "sector_component_pct": 0.0,
        "stock_specific_component": component_return,
        "stock_specific_component_pct": component_return * 100.0,
        "beta_applied": beta,
        "model_excess_return": component_return,
        "model_excess_return_pct": component_return * 100.0,
        "confidence_level": float(interval_calibration.confidence_level),
        "lower_bound": lower,
        "lower_bound_pct": lower * 100.0,
        "upper_bound": upper,
        "upper_bound_pct": upper * 100.0,
        "interval_width_pct": (upper - lower) * 100.0,
        "forecast_sigma": sigma_value,
        "forecast_sigma_pct": sigma_value * 100.0,
        "model_uncertainty_pct": component_model_std * 100.0,
        "confidence_label": confidence["confidence_label"],
        "confidence_explanation": confidence["confidence_explanation"],
        "signal_to_noise": confidence["signal_to_noise"],
        "direction_probability": float(
            direction_probability(np.asarray([expected_return]), np.asarray([sigma_value]))[0]
        ),
        "decision_rule": decision_cfg.rule,
        "signal_threshold": decision_cfg.threshold,
        "signal_threshold_pct": decision_cfg.threshold * 100.0,
        "signal_strength": abs(expected_return) / max(decision_cfg.threshold, 1e-9),
    }


def load_model_and_metadata(config: dict, horizon: int | None = None):
    """Load the ensemble, the scaler and every calibration needed at inference."""
    selected_horizon = int(
        horizon or config.get("default_prediction_horizon", config.get("prediction_horizon", 21))
    )
    model_path, scaler_path, metadata_path = resolve_artifact_paths(config, selected_horizon)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]
    scaler = joblib.load(scaler_path)
    device_info = get_best_device(config.get("device", "auto"))
    device = device_info.device

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dicts = (
        checkpoint["ensemble_state_dicts"]
        if isinstance(checkpoint, dict) and "ensemble_state_dicts" in checkpoint
        else [checkpoint]
    )

    models = []
    for state_dict in state_dicts:
        model = StockReturnPredictor(
            input_size=len(feature_columns),
            hidden_size=int(metadata["hidden_size"]),
            num_layers=int(metadata["num_layers"]),
            dropout=float(metadata["dropout"]),
            model_type=str(metadata.get("model_type", config.get("model_type", "lstm"))),
            input_dropout=float(metadata.get("input_dropout", 0.10)),
            # Both must come from the metadata, not the current config. Auxiliary
            # heads add parameters to the state dict, so a model trained with them
            # cannot be loaded into an architecture built without them.
            recurrent_dropout=float(metadata.get("recurrent_dropout", 0.0)),
            auxiliary_horizons=list(metadata.get("auxiliary_horizons", []) or []),
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    metadata["runtime_device"] = str(device)
    metadata["runtime_accelerator"] = device_info.accelerator
    metadata["runtime_device_name"] = device_info.device_name
    metadata["artifact_model_path"] = str(model_path)
    metadata["artifact_scaler_path"] = str(scaler_path)
    metadata["artifact_metadata_path"] = str(metadata_path)
    return models, metadata, scaler, feature_columns, device


def predict_ticker_with_artifacts(
    ticker: str,
    config: dict,
    model,
    metadata: dict,
    scaler,
    feature_columns: list[str],
    device: torch.device,
    horizon: int | None = None,
) -> dict:
    """
    Produce the full prediction record for one ticker.

    ``model`` accepts either a single module or the ensemble list returned by
    :func:`load_model_and_metadata`.
    """
    models = model if isinstance(model, (list, tuple)) else [model]
    selected_horizon = int(
        horizon or metadata.get("prediction_horizon", config.get("prediction_horizon", 21))
    )

    df = build_latest_features(ticker, config, feature_columns, selected_horizon)
    window_size = int(metadata["window_size"])
    if len(df) < window_size:
        raise ValueError(
            f"Only {len(df)} usable rows after feature engineering; {window_size} are required."
        )

    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.transform(df_scaled[feature_columns])
    latest_window = df_scaled[feature_columns].tail(window_size).to_numpy(dtype=np.float32)
    x = torch.tensor(latest_window, dtype=torch.float32).unsqueeze(0)

    # Monte Carlo dropout across every ensemble member. Combined with the law of
    # total variance this gives the same epistemic estimate used at training time.
    passes = int(metadata.get("mc_dropout_passes", 30))
    member_means, member_variances = [], []
    for member in models:
        mean, std = mc_dropout_predict(member, x, passes=max(passes, 20), device=device)
        member_means.append(float(mean[0]))
        member_variances.append(float(std[0]) ** 2)

    means = np.asarray(member_means, dtype=float)
    scaled_mean = float(means.mean())
    within = float(np.mean(member_variances))
    between = float(means.var(ddof=1)) if len(means) > 1 else 0.0
    scaled_model_std = float(np.sqrt(within + between))

    target_configuration = resolve_target_config(
        metadata.get("target_configuration", {"mode": "raw_return"})
    )
    latest_scale = float(target_scale(df.tail(1), selected_horizon, target_configuration)[0])

    latest_beta = latest_row_beta(df)
    return assemble_prediction(
        ticker=ticker,
        latest_date=str(df.index[-1].date()),
        horizon=selected_horizon,
        component_return=scaled_mean * latest_scale,
        component_model_std=scaled_model_std * latest_scale,
        latest_scale=latest_scale,
        metadata=metadata,
        config=config,
        model_name="LSTM",
        latest_beta=latest_beta,
    ) | {"ensemble_size": len(models)}


# ---------------------------------------------------------------------------
# XGBoost inference
# ---------------------------------------------------------------------------


def xgboost_artifact_paths(horizon: int) -> tuple[Path, Path]:
    return (
        ROOT / "models" / f"xgboost_regressor_h{horizon}.joblib",
        ROOT / "models" / f"xgboost_metadata_h{horizon}.json",
    )


def xgboost_model_exists(horizon: int) -> bool:
    model_path, metadata_path = xgboost_artifact_paths(int(horizon))
    if not (model_path.exists() and metadata_path.exists()):
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return int(json.load(file).get("artifact_schema_version", 1)) >= 3
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def load_xgboost_model_and_metadata(config: dict, horizon: int | None = None):
    """Load the bootstrap ensemble and its calibrations for one horizon."""
    selected_horizon = int(
        horizon or config.get("default_prediction_horizon", config.get("prediction_horizon", 21))
    )
    model_path, metadata_path = xgboost_artifact_paths(selected_horizon)
    if not (model_path.exists() and metadata_path.exists()):
        raise FileNotFoundError(
            f"No trained XGBoost artifacts were found for horizon={selected_horizon}. "
            f"Run: python3 train_xgboost.py --horizon {selected_horizon}"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    ensemble = joblib.load(model_path)
    metadata["artifact_model_path"] = str(model_path)
    metadata["artifact_metadata_path"] = str(metadata_path)
    return ensemble, metadata, metadata["feature_columns"]


def predict_xgboost_ticker_with_artifacts(
    ticker: str,
    config: dict,
    ensemble,
    metadata: dict,
    feature_columns: list[str],
    horizon: int | None = None,
) -> dict:
    """
    Prediction record for one ticker from the XGBoost bootstrap ensemble.

    The trees consume raw (unscaled) features, so no feature scaler is applied
    here -- unlike the LSTM path. Epistemic uncertainty is the spread across
    bootstrap members.
    """
    selected_horizon = int(
        horizon or metadata.get("prediction_horizon", config.get("prediction_horizon", 21))
    )
    df = build_latest_features(ticker, config, feature_columns, selected_horizon)
    if df.empty:
        raise ValueError("No usable rows remain after feature engineering.")

    latest_row = df[feature_columns].tail(1)
    mean_scaled, std_scaled = ensemble.predict(latest_row)

    target_configuration = resolve_target_config(
        metadata.get("target_configuration", {"mode": "raw_return"})
    )
    latest_scale = float(target_scale(df.tail(1), selected_horizon, target_configuration)[0])

    latest_beta = latest_row_beta(df)
    return assemble_prediction(
        ticker=ticker,
        latest_date=str(df.index[-1].date()),
        horizon=selected_horizon,
        component_return=float(np.asarray(mean_scaled)[0]) * latest_scale,
        component_model_std=float(np.asarray(std_scaled)[0]) * latest_scale,
        latest_scale=latest_scale,
        metadata=metadata,
        config=config,
        model_name="XGBoost",
        latest_beta=latest_beta,
    ) | {"ensemble_size": len(ensemble)}


def predict_ticker(ticker: str, config: dict, horizon: int | None = None) -> dict:
    models, metadata, scaler, feature_columns, device = load_model_and_metadata(config, horizon=horizon)
    return predict_ticker_with_artifacts(
        ticker, config, models, metadata, scaler, feature_columns, device, horizon
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future return for one or more stock tickers.")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker, for example AAPL")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon in trading days")
    parser.add_argument("--output", type=str, default="reports/latest_predictions.csv", help="CSV output path")
    return parser.parse_args()


def main() -> None:
    config = load_config()
    args = parse_args()
    selected_horizon = int(
        args.horizon or config.get("default_prediction_horizon", config.get("prediction_horizon", 21))
    )

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        tickers = [args.ticker.upper().strip()]
    else:
        tickers = [input("Enter stock ticker symbol, for example AAPL: ").upper().strip()]

    if not tickers or not tickers[0]:
        raise ValueError("At least one ticker is required.")

    models, metadata, scaler, feature_columns, device = load_model_and_metadata(
        config, horizon=selected_horizon
    )
    results = []
    for ticker in tickers:
        result = predict_ticker_with_artifacts(
            ticker, config, models, metadata, scaler, feature_columns, device, horizon=selected_horizon
        )
        results.append(result)

        print(f"\n{result['ticker']} - {result['prediction_horizon_trading_days']} trading days ahead")
        print("-" * 56)
        print(f"Latest market date      : {result['latest_data_date']}")
        print(f"Expected movement       : {result['expected_return_pct']:+.2f}%")
        print(
            f"Estimated range ({result['confidence_level']:.0%})  : "
            f"{result['lower_bound_pct']:+.2f}% to {result['upper_bound_pct']:+.2f}%"
        )
        print(
            f"Confidence              : {result['confidence_label']} "
            f"(P[direction correct] = {result['direction_probability']:.1%})"
        )
        print(
            f"  market baseline       : {result['market_drift_pct']:+.2f}%   "
            f"model view vs market: {result['model_excess_return_pct']:+.2f}%"
        )
        print(f"Signal ({result['decision_rule']})  : {result['signal']}")
        print(f"Cost hurdle             : {result['signal_threshold_pct']:.2f}%")

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    print("Important: this is an academic model output, not financial advice.")


if __name__ == "__main__":
    main()
