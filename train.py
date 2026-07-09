from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame, cost_aware_signal_threshold
from src.dataset import StockSequenceDataset
from src.device import dataloader_device_kwargs, get_best_device, move_batch_to_device
from src.model import StockReturnPredictor
from src.pipeline import build_or_load_dataset_for_tickers
from src.plots import plot_backtest_equity, plot_training_history
from src.training_logger import training_log_context


ROOT = Path(__file__).resolve().parent


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM future-return regression models.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Train only one prediction horizon. If omitted, trains all horizons in config.yaml.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated horizons to train, for example 21,252.",
    )
    return parser.parse_args()


def get_horizons(
    config: dict,
    selected_horizon: int | None = None,
    selected_horizons: list[int] | None = None,
) -> list[int]:
    if selected_horizons is not None:
        return [int(h) for h in selected_horizons]
    if selected_horizon is not None:
        return [int(selected_horizon)]
    if "prediction_horizons" in config and config["prediction_horizons"]:
        return [int(h) for h in config["prediction_horizons"]]
    return [int(config.get("prediction_horizon", 10))]


def parse_horizon_list(raw_horizons: str | None) -> list[int] | None:
    if raw_horizons is None:
        return None
    items = [item.strip() for item in str(raw_horizons).split(",")]
    horizons = [int(item) for item in items if item]
    if not horizons:
        raise ValueError("At least one horizon must be provided in --horizons.")
    ordered_unique: list[int] = []
    for horizon in horizons:
        if horizon not in ordered_unique:
            ordered_unique.append(horizon)
    return ordered_unique


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def artifact_path(base_path: str | Path, horizon: int) -> Path:
    path = Path(base_path)
    return ROOT / path.with_name(f"{path.stem}_h{horizon}{path.suffix}")


def copy_default_artifacts(
    config: dict,
    horizon: int,
    model_path: Path,
    scaler_path: Path,
    metadata_path: Path,
    metrics_path: Path,
    backtest_path: Path,
) -> None:
    default_horizon = int(config.get("default_prediction_horizon", config.get("prediction_horizon", 10)))
    if horizon != default_horizon:
        return

    copies = [
        (model_path, ROOT / config["model_output_path"]),
        (scaler_path, ROOT / config["scaler_output_path"]),
        (metadata_path, ROOT / config["metadata_output_path"]),
        (metrics_path, ROOT / config["metrics_output_path"]),
        (backtest_path, ROOT / config["backtest_output_path"]),
    ]
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, target)


def chronological_train_validation_test_split(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df.index.unique())
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates for chronological split.")

    train_idx = int(len(unique_dates) * train_ratio)
    validation_idx = int(len(unique_dates) * (train_ratio + validation_ratio))
    train_idx = max(1, min(train_idx, len(unique_dates) - 2))
    validation_idx = max(train_idx + 1, min(validation_idx, len(unique_dates) - 1))

    train_end_date = unique_dates[train_idx]
    validation_end_date = unique_dates[validation_idx]
    train_df = df[df.index < train_end_date].copy()
    validation_df = df[(df.index >= train_end_date) & (df.index < validation_end_date)].copy()
    test_df = df[df.index >= validation_end_date].copy()
    return train_df, validation_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    scaler_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    validation_df[feature_columns] = scaler.transform(validation_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return train_df, validation_df, test_df, scaler


def regression_metrics(true_return: np.ndarray, predicted_return: np.ndarray) -> dict:
    errors = predicted_return - true_return
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    direction_accuracy = float(np.mean(np.sign(predicted_return) == np.sign(true_return)))

    if len(true_return) > 1 and np.std(true_return) > 0 and np.std(predicted_return) > 0:
        correlation = float(np.corrcoef(true_return, predicted_return)[0, 1])
    else:
        correlation = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "direction_accuracy": direction_accuracy,
        "return_correlation": correlation,
    }


def derived_signal_metrics(signal_df: pd.DataFrame) -> dict:
    true_signal = signal_df["true_signal"].astype(str)
    predicted_signal = signal_df["predicted_signal"].astype(str)
    labels = ["SELL", "HOLD", "BUY"]
    return {
        "accuracy": float(accuracy_score(true_signal, predicted_signal)),
        "classification_report": classification_report(
            true_signal,
            predicted_signal,
            labels=labels,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true_signal, predicted_signal, labels=labels).tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_grad_norm: float | None,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y_return in loader:
        x, y_return = move_batch_to_device((x, y_return), device)
        optimizer.zero_grad()
        predicted_return = model(x)
        loss = loss_fn(predicted_return, y_return)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    all_true_return: list[float] = []
    all_pred_return: list[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y_return in loader:
            x, y_return_device = move_batch_to_device((x, y_return), device)
            predicted_return = model(x)
            loss = loss_fn(predicted_return, y_return_device)
            total_loss += loss.item() * x.size(0)
            all_true_return.extend(y_return_device.cpu().numpy().tolist())
            all_pred_return.extend(predicted_return.cpu().numpy().tolist())

    true_return_array = np.asarray(all_true_return, dtype=float)
    pred_return_array = np.asarray(all_pred_return, dtype=float)
    metrics = regression_metrics(true_return_array, pred_return_array)
    return {
        "loss": float(total_loss / len(loader.dataset)),
        **metrics,
        "true_return": true_return_array,
        "predicted_return": pred_return_array,
    }


def strip_arrays(metrics: dict) -> dict:
    skipped = {"true_return", "predicted_return"}
    return {key: value for key, value in metrics.items() if key not in skipped}


def train_for_horizon(config: dict, horizon: int, device_info) -> None:
    print("\n" + "=" * 72)
    print(f"Training LSTM regressor for {horizon} trading days ahead")
    print("=" * 72)

    processed_dir = ROOT / "data" / "processed"
    cache_dir = ROOT / "data" / "cache"
    report_dir = ROOT / "reports"
    plots_root = ROOT / config["plots_output_dir"]
    plots_dir = plots_root / f"h{horizon}"

    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_path(config["model_output_path"], horizon)
    scaler_path = artifact_path(config["scaler_output_path"], horizon)
    metadata_path = artifact_path(config["metadata_output_path"], horizon)
    metrics_path = artifact_path(config["metrics_output_path"], horizon)
    backtest_path = artifact_path(config["backtest_output_path"], horizon)

    dataset_cache_path = cache_dir / f"full_dataset_h{horizon}.csv"
    full_df, feature_columns = build_or_load_dataset_for_tickers(
        tickers=config["tickers"],
        benchmark_ticker=config["benchmark_ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        prediction_horizon=horizon,
        buy_threshold=float(config["buy_threshold"]),
        sell_threshold=float(config["sell_threshold"]),
        macro_tickers=config.get("macro_tickers"),
        cache_path=dataset_cache_path,
        use_cache=bool(config.get("use_dataset_cache", True)),
        force_rebuild=bool(config.get("force_rebuild_dataset_cache", False)),
    )

    full_df.to_csv(processed_dir / f"full_dataset_h{horizon}.csv")

    train_df, validation_df, test_df = chronological_train_validation_test_split(
        full_df,
        train_ratio=float(config["train_ratio"]),
        validation_ratio=float(config["validation_ratio"]),
    )
    train_df, validation_df, test_df, _ = scale_features(
        train_df,
        validation_df,
        test_df,
        feature_columns,
        scaler_path,
    )

    train_dataset = StockSequenceDataset(train_df, feature_columns, int(config["window_size"]))
    validation_dataset = StockSequenceDataset(validation_df, feature_columns, int(config["window_size"]))
    test_dataset = StockSequenceDataset(test_df, feature_columns, int(config["window_size"]))

    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Dataset is too small after preprocessing. Try reducing window_size or using more data.")

    device = device_info.device
    loader_kwargs = dataloader_device_kwargs(
        device=device,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True)),
    )
    print(f"Using device: {device} ({device_info.accelerator} - {device_info.device_name})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )

    model = StockReturnPredictor(
        input_size=len(feature_columns),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        model_type=str(config.get("model_type", "lstm")),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    loss_fn = nn.SmoothL1Loss()

    best_validation_mae = np.inf
    epochs_without_improvement = 0
    history: list[dict] = []
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            max_grad_norm=float(config["max_grad_norm"]),
        )
        validation_metrics = evaluate(model, validation_loader, loss_fn, device)

        history.append(
            {
                "horizon": horizon,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_metrics["loss"],
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_direction_accuracy": validation_metrics["direction_accuracy"],
                "validation_return_correlation": validation_metrics["return_correlation"],
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MAE: {validation_metrics['mae']:.4f} | "
            f"Val RMSE: {validation_metrics['rmse']:.4f} | "
            f"Val Dir Acc: {validation_metrics['direction_accuracy']:.4f} | "
            f"Val Corr: {validation_metrics['return_correlation']:.4f}"
        )

        if validation_metrics["mae"] < best_validation_mae:
            best_validation_mae = validation_metrics["mae"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["early_stopping_patience"]):
                print(
                    f"Early stopping activated after {epoch} epochs "
                    f"because validation MAE did not improve for "
                    f"{config['early_stopping_patience']} straight epochs."
                )
                break

    pd.DataFrame(history).to_csv(report_dir / f"training_history_h{horizon}.csv", index=False)
    plot_training_history(history, plots_dir)

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, loss_fn, device)

    backtest_cfg = BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        allow_short=bool(config["allow_short"]),
        signal_threshold_multiplier=float(config.get("signal_threshold_multiplier", 1.0)),
        min_signal_edge=float(config.get("min_signal_edge", 0.0)),
    )

    signal_df = build_signal_frame(
        metadata=test_dataset.metadata,
        true_return=test_metrics["true_return"],
        predicted_return=test_metrics["predicted_return"],
        cfg=backtest_cfg,
    )
    signal_metrics = derived_signal_metrics(signal_df)
    predictions_path = report_dir / f"test_predictions_h{horizon}.csv"
    signal_df.to_csv(predictions_path, index=False)

    backtest_df, backtest_metrics = backtest_signals(signal_df, backtest_cfg)
    backtest_df.to_csv(backtest_path, index=False)
    plot_backtest_equity(backtest_df, plots_dir)

    print("\nFinal test regression metrics:")
    print(
        f"MAE: {test_metrics['mae']:.4f} | "
        f"RMSE: {test_metrics['rmse']:.4f} | "
        f"Direction Acc: {test_metrics['direction_accuracy']:.4f} | "
        f"Corr: {test_metrics['return_correlation']:.4f}"
    )
    print("\nDerived signal classification report:")
    print(signal_metrics["classification_report"])

    metrics_payload = {
        "model_name": "LSTMRegressor",
        "prediction_horizon": horizon,
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "test_size": len(test_dataset),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "test_metrics": strip_arrays(test_metrics),
        "signal_metrics": signal_metrics,
        "backtest_metrics": backtest_metrics,
        "signal_threshold": cost_aware_signal_threshold(backtest_cfg),
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    metadata = {
        "tickers": config["tickers"],
        "benchmark_ticker": config["benchmark_ticker"],
        "feature_columns": feature_columns,
        "window_size": int(config["window_size"]),
        "prediction_horizon": horizon,
        "available_prediction_horizons": get_horizons(config),
        "model_type": str(config.get("model_type", "lstm")),
        "hidden_size": int(config["hidden_size"]),
        "num_layers": int(config["num_layers"]),
        "dropout": float(config["dropout"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "final_test_metrics": strip_arrays(test_metrics),
        "signal_metrics": signal_metrics,
        "device_used": str(device),
        "device_accelerator": device_info.accelerator,
        "device_name": device_info.device_name,
        "backtest_metrics": backtest_metrics,
        "signal_threshold": cost_aware_signal_threshold(backtest_cfg),
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    copy_default_artifacts(config, horizon, model_path, scaler_path, metadata_path, metrics_path, backtest_path)

    print(f"\nSaved best model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved backtest results to: {backtest_path}")
    print(f"Saved plots to: {plots_dir}")


def main() -> None:
    config = load_config()
    set_seed(int(config.get("random_seed", 42)))
    args = parse_args()
    selected_horizons = parse_horizon_list(args.horizons)
    train_models_for_horizons(config, get_horizons(config, args.horizon, selected_horizons))


def train_models_for_horizons(config: dict, horizons: list[int]) -> None:
    set_seed(int(config.get("random_seed", 42)))
    logs_dir = ROOT / "logs"

    with training_log_context(logs_dir, horizons) as log_path:
        device_info = get_best_device(config.get("device", "auto"))
        print("Training configuration")
        print("----------------------")
        print(f"Model type: {config.get('model_type', 'lstm').upper()} Regressor")
        print(f"Horizons: {horizons}")
        print(f"Maximum epochs: {config['epochs']}")
        print(f"Early stopping patience: {config['early_stopping_patience']} epochs without validation MAE improvement")
        print(f"Logs will be saved to: {log_path}")

        for horizon in horizons:
            train_for_horizon(config, int(horizon), device_info)

        print("\nTraining finished.")
        print("Important: this project is for academic research and simulation only, not financial advice.")


if __name__ == "__main__":
    main()
