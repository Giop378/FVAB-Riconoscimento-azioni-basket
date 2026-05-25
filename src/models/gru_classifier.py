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
        mask = time_idx < lengths.unsqueeze(1)

        scores = self.attention(outputs).squeeze(-1)  # [B, Tmax]
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)        # [B, Tmax]
        pooled = torch.sum(outputs * weights.unsqueeze(-1), dim=1)

        if return_weights:
            return pooled, weights

        return pooled


class TemporalPyramidPooling(nn.Module):
    """
    Divide ogni clip in tre parti temporali:
    - inizio
    - centro
    - fine

    Per ogni parte calcola la media degli output della BiGRU.
    In questo modo il classificatore riceve sempre anche informazione
    dalla parte finale della clip, utile per distinguere tiri segnati/sbagliati.
    """

    def __init__(self):
        super().__init__()

    def forward(self, outputs, lengths):
        """
        outputs: [B, Tmax, D]
        lengths: [B]

        return: [B, D * 3]
        """
        lengths = lengths.to(outputs.device)

        pooled_segments = []

        for i in range(outputs.size(0)):
            T = int(lengths[i].item())

            valid_outputs = outputs[i, :T]  # [T, D]

            # Se la clip è molto corta, evitiamo segmenti vuoti.
            if T < 3:
                segment_pooled = valid_outputs.mean(dim=0)
                pooled = torch.cat(
                    [segment_pooled, segment_pooled, segment_pooled],
                    dim=0,
                )
                pooled_segments.append(pooled)
                continue

            first_end = max(1, T // 3)
            second_end = max(first_end + 1, (2 * T) // 3)

            early = valid_outputs[:first_end]
            middle = valid_outputs[first_end:second_end]
            late = valid_outputs[second_end:T]

            # Sicurezza nel caso qualche segmento risultasse vuoto.
            if early.size(0) == 0:
                early = valid_outputs
            if middle.size(0) == 0:
                middle = valid_outputs
            if late.size(0) == 0:
                late = valid_outputs

            early_pooled = early.mean(dim=0)
            middle_pooled = middle.mean(dim=0)
            late_pooled = late.mean(dim=0)

            pooled = torch.cat(
                [early_pooled, middle_pooled, late_pooled],
                dim=0,
            )  # [D * 3]

            pooled_segments.append(pooled)

        return torch.stack(pooled_segments, dim=0)


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

        self.attention_pooling = AttentionPooling(gru_out_dim)
        self.temporal_pyramid_pooling = TemporalPyramidPooling()

        # Rappresentazione finale:
        # attention pooling:            gru_out_dim
        # temporal pyramid: 3 segmenti * gru_out_dim
        # totale: 4 * gru_out_dim
        final_dim = gru_out_dim * 4

        self.classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Dropout(dropout),
            nn.Linear(final_dim, num_classes),
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
            attention_pooled, attention_weights = self.attention_pooling(
                outputs,
                lengths,
                return_weights=True,
            )
        else:
            attention_pooled = self.attention_pooling(
                outputs,
                lengths,
                return_weights=False,
            )
            attention_weights = None

        pyramid_pooled = self.temporal_pyramid_pooling(outputs, lengths)

        final_representation = torch.cat(
            [attention_pooled, pyramid_pooled],
            dim=1,
        )

        logits = self.classifier(final_representation)

        if return_attention:
            return logits, attention_weights

        return logits