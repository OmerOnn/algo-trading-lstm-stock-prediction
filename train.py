from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.backtest import BacktestConfig, backtest_signals, build_signal_frame
from src.baselines import add_sma_crossover_baseline, baseline_accuracy
from src.dataset import StockSequenceDataset
from src.device import dataloader_device_kwargs, get_best_device, move_batch_to_device
from src.features import ID_TO_CLASS
from src.model import StockSignalModel
from src.pipeline import build_dataset_for_tickers
from src.plots import plot_backtest_equity, plot_training_history
from src.training_logger import training_log_context


ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM stock signal models.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Train only one prediction horizon, for example 10. If omitted, trains all horizons in config.yaml.",
    )
    return parser.parse_args()


def get_horizons(config: dict, selected_horizon: int | None = None) -> list[int]:
    if selected_horizon is not None:
        return [int(selected_horizon)]

    if "prediction_horizons" in config and config["prediction_horizons"]:
        return [int(h) for h in config["prediction_horizons"]]

    return [int(config.get("prediction_horizon", 10))]


def artifact_path(base_path: str | Path, horizon: int) -> Path:
    """
    Create horizon-specific artifact paths.

    Example:
    models/stock_advanced_model.pt -> models/stock_advanced_model_h10.pt
    reports/metrics.json -> reports/metrics_h10.json
    """
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
    """
    Also save default artifact names for backward compatibility and simple UI loading.
    """
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
    """
    Split by date to avoid training on the future and testing on the past.
    """
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
    """
    Fit scaler on train only to avoid data leakage.
    """
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


def make_class_weights(dataset: StockSequenceDataset, device: torch.device) -> torch.Tensor:
    labels = np.asarray(dataset.y_class, dtype=np.int64)
    counts = np.bincount(labels, minlength=3).astype(np.float32)

    counts[counts == 0] = 1.0

    weights = counts.sum() / (len(counts) * counts)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_loss_fn: nn.Module,
    return_loss_fn: nn.Module,
    device: torch.device,
    return_loss_weight: float,
    max_grad_norm: float | None,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y_class, y_return in loader:
        x, y_class, y_return = move_batch_to_device((x, y_class, y_return), device)

        optimizer.zero_grad()

        class_logits, predicted_return = model(x)

        class_loss = class_loss_fn(class_logits, y_class)
        return_loss = return_loss_fn(predicted_return, y_return)
        loss = class_loss + return_loss_weight * return_loss

        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_loss_fn: nn.Module,
    return_loss_fn: nn.Module,
    device: torch.device,
    return_loss_weight: float,
) -> dict:
    model.eval()

    all_true_class: list[int] = []
    all_pred_class: list[int] = []
    all_true_return: list[float] = []
    all_pred_return: list[float] = []
    all_probabilities: list[np.ndarray] = []

    total_loss = 0.0

    with torch.no_grad():
        for x, y_class, y_return in loader:
            x, y_class_device, y_return_device = move_batch_to_device(
                (x, y_class, y_return),
                device,
            )

            class_logits, predicted_return = model(x)

            class_loss = class_loss_fn(class_logits, y_class_device)
            return_loss = return_loss_fn(predicted_return, y_return_device)
            loss = class_loss + return_loss_weight * return_loss

            total_loss += loss.item() * x.size(0)

            probabilities = torch.softmax(class_logits, dim=1).cpu().numpy()
            pred_class = np.argmax(probabilities, axis=1)

            all_true_class.extend(y_class_device.cpu().numpy().tolist())
            all_pred_class.extend(pred_class.tolist())
            all_true_return.extend(y_return_device.cpu().numpy().tolist())
            all_pred_return.extend(predicted_return.cpu().numpy().tolist())
            all_probabilities.extend(list(probabilities))

    accuracy = accuracy_score(all_true_class, all_pred_class)
    mae = mean_absolute_error(all_true_return, all_pred_return)

    report = classification_report(
        all_true_class,
        all_pred_class,
        labels=[0, 1, 2],
        target_names=[ID_TO_CLASS[i] for i in range(3)],
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_true_class,
        all_pred_class,
        labels=[0, 1, 2],
    ).tolist()

    return {
        "loss": float(total_loss / len(loader.dataset)),
        "accuracy": float(accuracy),
        "return_mae": float(mae),
        "classification_report": report,
        "confusion_matrix": matrix,
        "true_class": np.asarray(all_true_class),
        "predicted_class": np.asarray(all_pred_class),
        "true_return": np.asarray(all_true_return),
        "predicted_return": np.asarray(all_pred_return),
        "probabilities": np.asarray(all_probabilities),
    }


def strip_arrays(metrics: dict) -> dict:
    skipped = {
        "true_class",
        "predicted_class",
        "true_return",
        "predicted_return",
        "probabilities",
    }

    return {key: value for key, value in metrics.items() if key not in skipped}


def train_for_horizon(config: dict, horizon: int, device_info) -> None:
    print("\n" + "=" * 72)
    print(f"Training LSTM model for {horizon} trading days ahead")
    print("=" * 72)

    processed_dir = ROOT / "data" / "processed"
    report_dir = ROOT / "reports"
    plots_root = ROOT / config["plots_output_dir"]
    plots_dir = plots_root / f"h{horizon}"

    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_path(config["model_output_path"], horizon)
    scaler_path = artifact_path(config["scaler_output_path"], horizon)
    metadata_path = artifact_path(config["metadata_output_path"], horizon)
    metrics_path = artifact_path(config["metrics_output_path"], horizon)
    backtest_path = artifact_path(config["backtest_output_path"], horizon)

    full_df, feature_columns = build_dataset_for_tickers(
        tickers=config["tickers"],
        benchmark_ticker=config["benchmark_ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        prediction_horizon=horizon,
        buy_threshold=float(config["buy_threshold"]),
        sell_threshold=float(config["sell_threshold"]),
        macro_tickers=config.get("macro_tickers"),
    )

    full_df = add_sma_crossover_baseline(full_df)
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

    train_dataset = StockSequenceDataset(
        train_df,
        feature_columns,
        int(config["window_size"]),
    )

    validation_dataset = StockSequenceDataset(
        validation_df,
        feature_columns,
        int(config["window_size"]),
    )

    test_dataset = StockSequenceDataset(
        test_df,
        feature_columns,
        int(config["window_size"]),
    )

    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(
            "Dataset is too small after preprocessing. "
            "Try reducing window_size or using more years/tickers."
        )

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

    model = StockSignalModel(
        input_size=len(feature_columns),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        model_type=str(config.get("model_type", "lstm")),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )

    class_weights = (
        make_class_weights(train_dataset, device)
        if config.get("use_class_weights", True)
        else None
    )

    class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    return_loss_fn = nn.SmoothL1Loss()

    best_validation_loss = np.inf
    epochs_without_improvement = 0
    history: list[dict] = []

    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            class_loss_fn,
            return_loss_fn,
            device,
            return_loss_weight=float(config["return_loss_weight"]),
            max_grad_norm=float(config["max_grad_norm"]),
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            class_loss_fn,
            return_loss_fn,
            device,
            return_loss_weight=float(config["return_loss_weight"]),
        )

        row = {
            "horizon": horizon,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_return_mae": validation_metrics["return_mae"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_metrics['loss']:.4f} | "
            f"Val Acc: {validation_metrics['accuracy']:.4f} | "
            f"Val Return MAE: {validation_metrics['return_mae']:.4f}"
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= int(config["early_stopping_patience"]):
                print(
                    f"Early stopping activated after {epoch} epochs "
                    f"because validation loss did not improve for "
                    f"{config['early_stopping_patience']} straight epochs."
                )
                break

    history_path = report_dir / f"training_history_h{horizon}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    plot_training_history(history, plots_dir)

    model.load_state_dict(torch.load(model_path, map_location=device))

    test_metrics = evaluate(
        model,
        test_loader,
        class_loss_fn,
        return_loss_fn,
        device,
        return_loss_weight=float(config["return_loss_weight"]),
    )

    print("\nFinal test classification report:")
    print(test_metrics["classification_report"])

    signal_df = build_signal_frame(
        metadata=test_dataset.metadata,
        true_class=test_metrics["true_class"],
        predicted_class=test_metrics["predicted_class"],
        true_return=test_metrics["true_return"],
        predicted_return=test_metrics["predicted_return"],
        probabilities=test_metrics["probabilities"],
    )

    test_predictions_path = report_dir / f"test_predictions_h{horizon}.csv"
    signal_df.to_csv(test_predictions_path, index=False)

    backtest_cfg = BacktestConfig(
        initial_cash=float(config["initial_cash"]),
        transaction_cost_pct=float(config["transaction_cost_pct"]),
        slippage_pct=float(config["slippage_pct"]),
        min_signal_confidence=float(config["min_signal_confidence"]),
        allow_short=bool(config["allow_short"]),
    )

    backtest_df, backtest_metrics = backtest_signals(signal_df, backtest_cfg)
    backtest_df.to_csv(backtest_path, index=False)

    plot_backtest_equity(backtest_df, plots_dir)

    baselines = baseline_accuracy(test_df)

    all_metrics = {
        "model_name": "LSTM",
        "prediction_horizon": horizon,
        "validation_best_loss": float(best_validation_loss),
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "test_metrics": strip_arrays(test_metrics),
        "baseline_metrics": baselines,
        "backtest_metrics": backtest_metrics,
        "class_mapping": ID_TO_CLASS,
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    metadata = {
        "tickers": config["tickers"],
        "benchmark_ticker": config["benchmark_ticker"],
        "feature_columns": feature_columns,
        "window_size": int(config["window_size"]),
        "prediction_horizon": horizon,
        "available_prediction_horizons": get_horizons(config),
        "buy_threshold": float(config["buy_threshold"]),
        "sell_threshold": float(config["sell_threshold"]),
        "model_type": str(config.get("model_type", "lstm")),
        "hidden_size": int(config["hidden_size"]),
        "num_layers": int(config["num_layers"]),
        "dropout": float(config["dropout"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "final_test_metrics": strip_arrays(test_metrics),
        "device_used": str(device),
        "device_accelerator": device_info.accelerator,
        "device_name": device_info.device_name,
        "backtest_metrics": backtest_metrics,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    copy_default_artifacts(
        config=config,
        horizon=horizon,
        model_path=model_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        backtest_path=backtest_path,
    )

    print(f"\nSaved best model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved backtest results to: {backtest_path}")
    print(f"Saved plots to: {plots_dir}")


def main() -> None:
    config = load_config()
    args = parse_args()
    set_seed(config.get("random_seed", 42))

    horizons = get_horizons(config, args.horizon)
    logs_dir = ROOT / "logs"

    with training_log_context(logs_dir, horizons) as log_path:
        device_info = get_best_device(config.get("device", "auto"))

        print("Training configuration")
        print("----------------------")
        print(f"Model type: {config.get('model_type', 'lstm').upper()}")
        print(f"Horizons: {horizons}")
        print(f"Maximum epochs: {config['epochs']}")
        print(
            f"Early stopping patience: "
            f"{config['early_stopping_patience']} straight epochs without validation improvement"
        )
        print(f"Logs will be saved to: {log_path}")

        for horizon in horizons:
            train_for_horizon(config, int(horizon), device_info)

        print("\nTraining finished.")
        print("Important: this project is for academic research and simulation only, not financial advice.")


if __name__ == "__main__":
    main()