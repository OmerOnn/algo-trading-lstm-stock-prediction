from __future__ import annotations

import torch
import torch.nn as nn


class VariationalSequenceDropout(nn.Module):
    """
    Dropout with one mask shared across every timestep of a sequence.

    Ordinary dropout resamples its mask at each step, which injects fresh noise
    into the recurrence at every timestep and destroys the state a recurrent
    layer is trying to carry forward. Sharing the mask across the time axis
    drops the same units for the whole sequence, so the recurrence stays
    coherent while still being regularised.

    This is the practical stand-in for true recurrent dropout (a mask inside the
    gate computation). The fused cuDNN and MPS LSTM kernels do not expose their
    internal gates, so a mask cannot be injected there without giving up the
    fused kernel and the large speed-up it provides.
    """

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)
        if not 0.0 <= self.probability < 1.0:
            raise ValueError("recurrent_dropout must be in [0, 1)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0.0:
            return x
        # Mask shape (batch, 1, features): broadcast over the time axis.
        keep = 1.0 - self.probability
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(keep) / keep
        return x * mask


class StockReturnPredictor(nn.Module):
    """
    Sequence model that predicts a scaled future return.

    Design points that matter on a low signal-to-noise financial panel:

    * **Input dropout** randomly removes whole features per step. With dozens of
      correlated indicators this is a far stronger regulariser than weight decay,
      and it is what makes Monte Carlo dropout meaningful at the input level.
    * **Variational recurrent dropout** regularises the recurrent representation
      with a timestep-shared mask.
    * **Dual pooling** concatenates the final hidden state with the mean over the
      sequence, so a single noisy last step cannot dominate the forecast.
    * **Layer normalisation** before the head keeps the scale of the pooled
      representation stable across volatility regimes.
    * **Auxiliary regression heads** optionally predict the same residual return
      at other horizons. They share the encoder, so the extra supervision
      constrains the representation without changing the reported forecast: the
      main head is still the only output that is saved or served.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.35,
        model_type: str = "lstm",
        input_dropout: float = 0.10,
        recurrent_dropout: float = 0.0,
        auxiliary_horizons: list[int] | tuple[int, ...] = (),
    ) -> None:
        super().__init__()

        model_type = model_type.lower().strip()
        if model_type not in {"gru", "lstm"}:
            raise ValueError("model_type must be either 'gru' or 'lstm'")

        self.model_type = model_type
        self.auxiliary_horizons = [int(h) for h in (auxiliary_horizons or [])]
        self.input_dropout = nn.Dropout(float(input_dropout))
        self.recurrent_dropout = VariationalSequenceDropout(float(recurrent_dropout))

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
        # One extra regression head per auxiliary horizon. Registered by name so
        # a checkpoint records exactly which horizons a model was trained with.
        self.auxiliary_heads = nn.ModuleDict(
            {f"h{horizon}": nn.Linear(hidden_size, 1) for horizon in self.auxiliary_horizons}
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        rnn_out, _ = self.rnn(x)
        rnn_out = self.recurrent_dropout(rnn_out)
        pooled = torch.cat([rnn_out[:, -1, :], rnn_out.mean(dim=1)], dim=-1)
        return self.shared(self.norm(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The reported forecast: the main horizon's scaled return."""
        return self.return_head(self.encode(x)).squeeze(-1)

    def forward_with_auxiliaries(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Main forecast plus every auxiliary head, for multi-task training."""
        features = self.encode(x)
        main = self.return_head(features).squeeze(-1)
        auxiliaries = {
            name: head(features).squeeze(-1) for name, head in self.auxiliary_heads.items()
        }
        return main, auxiliaries
