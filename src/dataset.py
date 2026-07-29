from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockSequenceDataset(Dataset):
    """PyTorch dataset for stock time-window regression sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        window_size: int,
        target_dates: set[pd.Timestamp] | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.window_size = window_size

        self.x_sequences: list[np.ndarray] = []
        self.y_return: list[float] = []
        self.y_model_target: list[float] = []
        self.target_scales: list[float] = []
        self.metadata: list[dict] = []

        for ticker, group in df.groupby("Ticker"):
            group = group.sort_index().copy()
            features = group[feature_columns].values.astype(np.float32)
            return_targets = group["future_return"].values.astype(np.float32)
            model_targets = group.get("model_target", group["future_return"]).values.astype(np.float32)
            target_scales = group.get(
                "target_scale",
                pd.Series(1.0, index=group.index),
            ).values.astype(np.float32)
            dates = group.index.to_list()

            for i in range(window_size - 1, len(group)):
                target_date = pd.Timestamp(dates[i])
                if target_dates is not None and target_date not in target_dates:
                    continue
                # Include the target-date features. Inference also uses a window
                # ending on the latest date, so this removes an off-by-one mismatch.
                self.x_sequences.append(features[i - window_size + 1 : i + 1])
                self.y_return.append(float(return_targets[i]))
                self.y_model_target.append(float(model_targets[i]))
                self.target_scales.append(float(target_scales[i]))
                self.metadata.append({"ticker": ticker, "date": str(dates[i].date())})

    def __len__(self) -> int:
        return len(self.x_sequences)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.x_sequences[idx], dtype=torch.float32)
        y_model_target = torch.tensor(self.y_model_target[idx], dtype=torch.float32)
        y_return = torch.tensor(self.y_return[idx], dtype=torch.float32)
        target_scale = torch.tensor(self.target_scales[idx], dtype=torch.float32)
        return x, y_model_target, y_return, target_scale
