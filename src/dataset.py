from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class StockSequenceDataset(Dataset):
    """
    PyTorch dataset for stock time-window regression sequences.

    Each item carries an integer ``date_code`` alongside the features and
    targets. The code identifies which prediction date the row belongs to, which
    is what lets the training loss compute a *per-date* cross-sectional
    correlation instead of one correlation over a randomly mixed batch.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        window_size: int,
        target_dates: set[pd.Timestamp] | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.window_size = window_size

        # Windows are sliced lazily in ``__getitem__`` rather than materialised
        # here. Every window overlaps its neighbours by ``window_size - 1`` rows,
        # so storing them expanded costs ``window_size`` times the memory of the
        # panel itself: on this dataset (about 480k rows, 102 features, window 30)
        # that is roughly 5.9 GB of float32 versus about 200 MB for the panel.
        # Keeping one contiguous array per ticker and recording positions into it
        # gives identical batches at a fraction of the memory, and builds faster
        # because nothing is copied.
        self._ticker_features: list[np.ndarray] = []
        self._items: list[tuple[int, int]] = []
        self.y_return: list[float] = []
        self.y_model_target: list[float] = []
        self.target_scales: list[float] = []
        self.metadata: list[dict] = []
        raw_dates: list[pd.Timestamp] = []

        for ticker, group in df.groupby("Ticker"):
            group = group.sort_index()
            features = np.ascontiguousarray(
                group[feature_columns].to_numpy(dtype=np.float32)
            )
            return_targets = group["future_return"].to_numpy(dtype=np.float32)
            model_targets = (
                group["model_target"] if "model_target" in group.columns else group["future_return"]
            ).to_numpy(dtype=np.float32)
            target_scales = (
                group["target_scale"]
                if "target_scale" in group.columns
                else pd.Series(1.0, index=group.index)
            ).to_numpy(dtype=np.float32)
            dates = group.index.to_list()

            ticker_index = len(self._ticker_features)
            self._ticker_features.append(features)

            for i in range(window_size - 1, len(group)):
                target_date = pd.Timestamp(dates[i])
                if target_dates is not None and target_date not in target_dates:
                    continue
                # The window ends on the target date inclusive. Inference also uses
                # a window ending on the latest available date, so this removes an
                # off-by-one mismatch between training and serving.
                self._items.append((ticker_index, i))
                self.y_return.append(float(return_targets[i]))
                self.y_model_target.append(float(model_targets[i]))
                self.target_scales.append(float(target_scales[i]))
                self.metadata.append({"ticker": ticker, "date": str(dates[i].date())})
                raw_dates.append(target_date)

        # Factorised once, sorted, so codes are chronologically ordered and can
        # be used directly as group keys by the loss and the sampler.
        if raw_dates:
            codes, uniques = pd.factorize(pd.DatetimeIndex(raw_dates), sort=True)
            self.date_codes = np.asarray(codes, dtype=np.int64)
            self.unique_dates = pd.DatetimeIndex(uniques)
        else:
            self.date_codes = np.zeros(0, dtype=np.int64)
            self.unique_dates = pd.DatetimeIndex([])

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        ticker_index, row = self._items[idx]
        window = self._ticker_features[ticker_index][row - self.window_size + 1 : row + 1]
        x = torch.from_numpy(np.ascontiguousarray(window))
        y_model_target = torch.tensor(self.y_model_target[idx], dtype=torch.float32)
        y_return = torch.tensor(self.y_return[idx], dtype=torch.float32)
        target_scale = torch.tensor(self.target_scales[idx], dtype=torch.float32)
        date_code = torch.tensor(int(self.date_codes[idx]), dtype=torch.long)
        return x, y_model_target, y_return, target_scale, date_code

    def indices_by_date(self) -> list[np.ndarray]:
        """Row positions grouped by prediction date, in chronological order."""
        if len(self.date_codes) == 0:
            return []
        order = np.argsort(self.date_codes, kind="stable")
        sorted_codes = self.date_codes[order]
        boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
        return np.split(order, boundaries)


class DateGroupedBatchSampler(Sampler[list[int]]):
    """
    Yield batches made of whole date cross-sections.

    A batch built from complete dates is what makes a per-date ranking loss
    possible: every date in the batch contains enough of that day's universe for
    its internal correlation to mean something. Sampling rows independently would
    scatter each date across many batches and leave two or three names per date
    per batch, from which no ordering can be learned.

    The *order* of dates is shuffled every epoch, so the model still sees
    de-correlated batches; only the grouping is fixed.

    ``max_rows_per_batch`` is a memory guard. Batches are formed from
    ``dates_per_batch`` dates, but a date holding the full universe can be large,
    so a batch stops early rather than growing without bound.
    """

    def __init__(
        self,
        dataset: StockSequenceDataset,
        dates_per_batch: int = 3,
        shuffle: bool = True,
        seed: int = 42,
        max_rows_per_batch: int = 4096,
        drop_last: bool = False,
    ) -> None:
        self.date_groups = [group for group in dataset.indices_by_date() if len(group) > 0]
        self.dates_per_batch = max(1, int(dates_per_batch))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_rows_per_batch = max(1, int(max_rows_per_batch))
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the shuffle stream so successive epochs see different orders."""
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        order = list(range(len(self.date_groups)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)

        batches: list[list[int]] = []
        current: list[int] = []
        dates_in_current = 0
        for position in order:
            group = self.date_groups[position]
            if dates_in_current >= self.dates_per_batch or (
                current and len(current) + len(group) > self.max_rows_per_batch
            ):
                batches.append(current)
                current = []
                dates_in_current = 0
            current.extend(int(index) for index in group)
            dates_in_current += 1
        if current and not (self.drop_last and dates_in_current < self.dates_per_batch):
            batches.append(current)
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches())
