import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, outputs, lengths, return_weights: bool = False):
        """
        outputs: [B, Tmax, D]
        lengths: [B]
        """
        lengths = lengths.to(outputs.device)

        _, max_len, _ = outputs.shape

        time_idx = torch.arange(max_len, device=outputs.device).unsqueeze(0)
        mask = time_idx < lengths.unsqueeze(1)   # True sui frame reali, False sul padding

        scores = self.attention(outputs).squeeze(-1)  # [B, Tmax]
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)        # [B, Tmax]
        pooled = torch.sum(outputs * weights.unsqueeze(-1), dim=1)

        if return_weights:
            return pooled, weights

        return pooled


class GRUActionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
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

        self.pooling = AttentionPooling(gru_out_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, num_classes),
        )

    def forward(self, features, lengths, return_attention: bool = False):
        """
        features: [B, Tmax, input_dim]
        lengths:  [B]
        """
        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_outputs, _ = self.gru(packed)

        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
        )

        if return_attention:
            pooled, attention_weights = self.pooling(
                outputs,
                lengths,
                return_weights=True,
            )

            logits = self.classifier(pooled)

            return logits, attention_weights

        pooled = self.pooling(outputs, lengths)
        logits = self.classifier(pooled)

        return logits