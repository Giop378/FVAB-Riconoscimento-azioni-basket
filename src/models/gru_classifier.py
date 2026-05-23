import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUActionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_classes: int = 9,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, num_classes),
        )

    def forward(self, features, lengths):
        """
        features: [B, Tmax, 512]
        lengths:  [B]
        """
        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)

        if self.gru.bidirectional:
            # hidden[-2] = ultimo stato direzione forward
            # hidden[-1] = ultimo stato direzione backward
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]

        logits = self.classifier(final_hidden)
        return logits