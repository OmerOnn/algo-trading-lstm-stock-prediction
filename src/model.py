from __future__ import annotations

import torch
import torch.nn as nn


class StockReturnPredictor(nn.Module):
    """
    Sequence model that predicts a scaled future return.

    Three details matter for a low signal-to-noise financial panel:

    * **Input dropout** randomly removes whole features per step. With ~70
      correlated indicators this is a far stronger regulariser than weight decay
      and it is what makes Monte Carlo dropout meaningful at the input level.
    * **Dual pooling** concatenates the final hidden state with the mean over the
      sequence, so a single noisy last step cannot dominate the forecast.
    * **Layer normalisation** before the head keeps the scale of the pooled
      representation stable across volatility regimes.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.35,
        model_type: str = "lstm",
        input_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        model_type = model_type.lower().strip()
        if model_type not in {"gru", "lstm"}:
            raise ValueError("model_type must be either 'gru' or 'lstm'")

        self.model_type = model_type
        self.input_dropout = nn.Dropout(float(input_dropout))

        rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        rnn_out, _ = self.rnn(x)
        pooled = torch.cat([rnn_out[:, -1, :], rnn_out.mean(dim=1)], dim=-1)
        features = self.shared(self.norm(pooled))
        return self.return_head(features).squeeze(-1)
