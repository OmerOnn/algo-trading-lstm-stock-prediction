from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockSequenceDataset(Dataset):
    """PyTorch dataset for stock time-window sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        window_size: int,
    ) -> None:
        self.feature_columns = feature_columns
        self.window_size = window_size

        self.x_sequences: list[np.ndarray] = []
        self.y_class: list[int] = []
        self.y_return: list[float] = []
        self.metadata: list[dict] = []

        for ticker, group in df.groupby("Ticker"):
            group = group.sort_index().copy()
            features = group[feature_columns].values.astype(np.float32)
            class_targets = group["signal_label"].values.astype(np.int64)
            return_targets = group["future_return"].values.astype(np.float32)
            dates = group.index.to_list()

            for i in range(window_size, len(group)):
                self.x_sequences.append(features[i - window_size : i])
                self.y_class.append(int(class_targets[i]))
                self.y_return.append(float(return_targets[i]))
                self.metadata.append({"ticker": ticker, "date": str(dates[i].date())})

    def __len__(self) -> int:
        return len(self.x_sequences)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.x_sequences[idx], dtype=torch.float32)
        y_class = torch.tensor(self.y_class[idx], dtype=torch.long)
        y_return = torch.tensor(self.y_return[idx], dtype=torch.float32)
        return x, y_class, y_return
