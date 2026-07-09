from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.data_download import download_earnings_data, download_macro_data, download_price_data
from src.features import (
    ID_TO_CLASS,
    add_benchmark_features,
    add_earnings_features,
    add_macro_features,
    add_technical_indicators,
)
from src.device import get_best_device
from src.model import StockSignalModel


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
    """Load horizon-specific artifacts, with fallback to the original default names."""
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
    
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df = df.dropna(subset=feature_columns).copy()
    return df


def load_model_and_metadata(config: dict, horizon: int | None = None):
    selected_horizon = int(horizon or config.get("default_prediction_horizon", config.get("prediction_horizon", 10)))
    model_path, scaler_path, metadata_path = resolve_artifact_paths(config, selected_horizon)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]
    scaler = joblib.load(scaler_path)

    device_info = get_best_device(config.get("device", "auto"))
    device = device_info.device
    model = StockSignalModel(
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


def align_expected_return_with_signal(signal: str, raw_expected_return_pct: float) -> tuple[float, bool]:
    """Create a user-facing expected movement that is consistent with the signal.

    The neural network has two heads: one for classification and one for return regression.
    They can disagree. For the UI, the displayed movement is aligned with the selected
    signal so users do not see confusing outputs such as SELL with +1.80%.
    """
    if signal == "BUY":
        displayed = abs(raw_expected_return_pct)
    elif signal == "SELL":
        displayed = -abs(raw_expected_return_pct)
    else:
        displayed = raw_expected_return_pct
    adjusted = abs(displayed - raw_expected_return_pct) > 1e-9
    return displayed, adjusted


def predict_ticker_with_artifacts(
    ticker: str,
    config: dict,
    model: StockSignalModel,
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
        class_logits, predicted_return = model(x.to(device, non_blocking=(device.type == "cuda")))
        probabilities = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
        predicted_class_id = int(np.argmax(probabilities))
        predicted_signal = ID_TO_CLASS[predicted_class_id]
        raw_expected_return = float(predicted_return.cpu().numpy()[0])

    raw_expected_return_pct = raw_expected_return * 100
    displayed_expected_return_pct, was_aligned = align_expected_return_with_signal(
        predicted_signal,
        raw_expected_return_pct,
    )

    return {
        "ticker": ticker,
        "latest_data_date": str(df.index[-1].date()),
        "prediction_horizon_trading_days": selected_horizon,
        "signal": predicted_signal,
        "expected_return": displayed_expected_return_pct / 100,
        "expected_return_pct": displayed_expected_return_pct,
        "raw_expected_return": raw_expected_return,
        "raw_expected_return_pct": raw_expected_return_pct,
        "return_was_aligned_to_signal": bool(was_aligned),
        "prob_sell": float(probabilities[0]),
        "prob_hold": float(probabilities[1]),
        "prob_buy": float(probabilities[2]),
        "confidence": float(np.max(probabilities)),
    }


def predict_ticker(ticker: str, config: dict, horizon: int | None = None) -> dict:
    model, metadata, scaler, feature_columns, device = load_model_and_metadata(config, horizon=horizon)
    return predict_ticker_with_artifacts(ticker, config, model, metadata, scaler, feature_columns, device, horizon=horizon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict BUY / HOLD / SELL signal for one or more stock tickers.")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker, for example AAPL")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers, for example AAPL,MSFT,NVDA")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon in trading days, for example 5, 10, 20, or 30")
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
        print(f"Signal: {result['signal']}")
        print(f"Expected movement: {result['expected_return_pct']:.2f}%")
        print("Class probabilities:")
        print(f"  SELL: {result['prob_sell'] * 100:.2f}%")
        print(f"  HOLD: {result['prob_hold'] * 100:.2f}%")
        print(f"  BUY : {result['prob_buy'] * 100:.2f}%")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")
    print("Important: this is an academic model output, not financial advice.")


if __name__ == "__main__":
    main()
