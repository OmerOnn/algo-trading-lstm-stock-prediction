from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.backtest import BacktestConfig, cost_aware_signal_threshold, return_to_signal
from src.data_download import download_earnings_data, download_macro_data, download_price_data
from src.device import get_best_device
from src.features import (
    add_benchmark_features,
    add_earnings_features,
    add_macro_features,
    add_technical_indicators,
)
from src.model import StockReturnPredictor


ROOT = Path(__file__).resolve().parent


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_available_horizons(config: dict) -> list[int]:
    if "prediction_horizons" in config and config["prediction_horizons"]:
        return [int(h) for h in config["prediction_horizons"]]
    return [int(config.get("prediction_horizon", 10))]


def artifact_path(base_path: str | Path, horizon: int) -> Path:
    path = Path(base_path)
    return ROOT / path.with_name(f"{path.stem}_h{horizon}{path.suffix}")


def resolve_artifact_paths(config: dict, horizon: int) -> tuple[Path, Path, Path]:
    model_path = artifact_path(config["model_output_path"], horizon)
    scaler_path = artifact_path(config["scaler_output_path"], horizon)
    metadata_path = artifact_path(config["metadata_output_path"], horizon)

    if model_path.exists() and scaler_path.exists() and metadata_path.exists():
        return model_path, scaler_path, metadata_path

    default_horizon = int(config.get("default_prediction_horizon", config.get("prediction_horizon", 10)))
    if horizon == default_horizon:
        fallback_model = ROOT / config["model_output_path"]
        fallback_scaler = ROOT / config["scaler_output_path"]
        fallback_metadata = ROOT / config["metadata_output_path"]
        if fallback_model.exists() and fallback_scaler.exists() and fallback_metadata.exists():
            return fallback_model, fallback_scaler, fallback_metadata

    raise FileNotFoundError(
        f"No trained model artifacts were found for horizon={horizon}. "
        f"Run: python3 train.py --horizon {horizon}"
    )


def build_latest_features(ticker: str, config: dict, feature_columns: list[str], horizon: int) -> pd.DataFrame:
    price_df = download_price_data(ticker, config["start_date"], config["end_date"])
    benchmark_df = download_price_data(config["benchmark_ticker"], config["start_date"], config["end_date"])
    macro_df = pd.DataFrame()
    if config.get("macro_tickers"):
        macro_df = download_macro_data(config["macro_tickers"], config["start_date"], config["end_date"])

    earnings_df = download_earnings_data(ticker)
    df = add_technical_indicators(price_df)
    df = add_benchmark_features(df, benchmark_df)
    df = add_macro_features(df, macro_df)
    df = add_earnings_features(df, earnings_df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns).copy()
    return df


def build_backtest_config(config: dict) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        allow_short=bool(config["allow_short"]),
        signal_threshold_multiplier=float(config.get("signal_threshold_multiplier", 1.0)),
        min_signal_edge=float(config.get("min_signal_edge", 0.0)),
    )


def load_model_and_metadata(config: dict, horizon: int | None = None):
    selected_horizon = int(horizon or config.get("default_prediction_horizon", config.get("prediction_horizon", 10)))
    model_path, scaler_path, metadata_path = resolve_artifact_paths(config, selected_horizon)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]
    scaler = joblib.load(scaler_path)
    device_info = get_best_device(config.get("device", "auto"))
    device = device_info.device

    model = StockReturnPredictor(
        input_size=len(feature_columns),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        dropout=float(metadata["dropout"]),
        model_type=str(metadata.get("model_type", config.get("model_type", "lstm"))),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    metadata["runtime_device"] = str(device)
    metadata["runtime_accelerator"] = device_info.accelerator
    metadata["runtime_device_name"] = device_info.device_name
    metadata["artifact_model_path"] = str(model_path)
    metadata["artifact_scaler_path"] = str(scaler_path)
    metadata["artifact_metadata_path"] = str(metadata_path)
    return model, metadata, scaler, feature_columns, device


def predict_ticker_with_artifacts(
    ticker: str,
    config: dict,
    model: StockReturnPredictor,
    metadata: dict,
    scaler,
    feature_columns: list[str],
    device: torch.device,
    horizon: int | None = None,
) -> dict:
    selected_horizon = int(horizon or metadata.get("prediction_horizon", config.get("prediction_horizon", 10)))
    df = build_latest_features(ticker, config, feature_columns, selected_horizon)
    if len(df) < int(metadata["window_size"]):
        raise ValueError("Not enough valid rows for prediction after feature engineering.")

    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.transform(df_scaled[feature_columns])
    latest_window = df_scaled[feature_columns].tail(int(metadata["window_size"])).values.astype(np.float32)
    x = torch.tensor(latest_window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_return = float(model(x.to(device, non_blocking=(device.type == "cuda"))).cpu().numpy()[0])

    backtest_cfg = build_backtest_config(config)
    signal_threshold = cost_aware_signal_threshold(backtest_cfg)
    predicted_signal = return_to_signal(predicted_return, signal_threshold)

    return {
        "ticker": ticker,
        "latest_data_date": str(df.index[-1].date()),
        "prediction_horizon_trading_days": selected_horizon,
        "signal": predicted_signal,
        "predicted_return": predicted_return,
        "expected_return": predicted_return,
        "expected_return_pct": predicted_return * 100.0,
        "signal_threshold": signal_threshold,
        "signal_threshold_pct": signal_threshold * 100.0,
        "signal_strength": 0.0 if signal_threshold <= 0 else abs(predicted_return) / signal_threshold,
    }


def predict_ticker(ticker: str, config: dict, horizon: int | None = None) -> dict:
    model, metadata, scaler, feature_columns, device = load_model_and_metadata(config, horizon=horizon)
    return predict_ticker_with_artifacts(ticker, config, model, metadata, scaler, feature_columns, device, horizon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future return for one or more stock tickers.")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker, for example AAPL")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers, for example AAPL,MSFT,NVDA")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction horizon in trading days, for example 1, 5, 21, 126, 252, 1260, or 2520",
    )
    parser.add_argument("--output", type=str, default="reports/latest_predictions.csv", help="CSV output path")
    return parser.parse_args()


def main() -> None:
    config = load_config()
    args = parse_args()
    selected_horizon = int(args.horizon or config.get("default_prediction_horizon", config.get("prediction_horizon", 10)))

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        tickers = [args.ticker.upper().strip()]
    else:
        ticker = input("Enter stock ticker symbol, for example AAPL: ").upper().strip()
        tickers = [ticker]

    if not tickers:
        raise ValueError("At least one ticker is required.")

    model, metadata, scaler, feature_columns, device = load_model_and_metadata(config, horizon=selected_horizon)
    results = []
    for ticker in tickers:
        result = predict_ticker_with_artifacts(
            ticker,
            config,
            model,
            metadata,
            scaler,
            feature_columns,
            device,
            horizon=selected_horizon,
        )
        results.append(result)

        print("\nPrediction result")
        print("-----------------")
        print(f"Ticker: {result['ticker']}")
        print(f"Latest data date: {result['latest_data_date']}")
        print(f"Prediction horizon: {result['prediction_horizon_trading_days']} trading days")
        print(f"Predicted future return: {result['expected_return_pct']:.2f}%")
        print(f"Derived signal: {result['signal']}")
        print(f"Trade threshold: {result['signal_threshold_pct']:.2f}%")
        print(f"Signal strength: {result['signal_strength']:.2f}x threshold")

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    print("Important: this is an academic model output, not financial advice.")


if __name__ == "__main__":
    main()
