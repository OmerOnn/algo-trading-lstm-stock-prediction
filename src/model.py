from __future__ import annotations

import torch
import torch.nn as nn


class StockReturnPredictor(nn.Module):
    """Sequence model that predicts future return directly."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
        model_type: str = "gru",
    ) -> None:
        super().__init__()

        model_type = model_type.lower().strip()
        if model_type not in {"gru", "lstm"}:
            raise ValueError("model_type must be either 'gru' or 'lstm'")

        rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.return_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        features = self.shared(last_hidden)
        return self.return_head(features).squeeze(-1)
