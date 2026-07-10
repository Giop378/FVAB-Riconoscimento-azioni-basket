"""
Modello Transformer temporale per classificazione di clip video.

Il modello riceve sequenze di feature frame-level già estratte, ad esempio
DINOv3 o DINOv3 concatenato con feature di tracking palla/canestro, e produce
una predizione di classe per l'intera clip.

È usato sia durante il training dei singoli livelli gerarchici sia durante
la valutazione end-to-end dell'esperimento finale.
"""
# Collegamenti con la pipeline:
# - riceve i batch [B, T, D] creati da data/feature_dataset.py;
# - D può includere sia DINOv3 sia tracking concatenato da training/train.py o
#   evaluation/evaluate_hierarchical.py;
# - la configurazione del modello viene salvata nel checkpoint e usata dalla
#   valutazione per ricostruire esattamente ciascun livello L1/L2/L3.


import math

import torch
import torch.nn as nn


# Codifica deterministica della posizione, registrata come buffer e quindi spostata
# automaticamente sul device senza essere ottimizzata.
class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding sinusoidale.

    Il Transformer, a differenza della GRU, non ha una nozione naturale
    dell'ordine temporale dei frame. Per questo aggiungiamo una codifica
    posizionale alle feature temporali.
    """

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequenza troppo lunga: T={seq_len}, max_len={self.pe.size(1)}. "
                "Aumenta il parametro --max-len."
            )

        return x + self.pe[:, :seq_len, :]


# Il classificatore separa proiezione dell’input, modellazione temporale, pooling
# della clip e testa lineare, consentendo tre strategie di aggregazione.
class TemporalTransformerActionClassifier(nn.Module):
    """
    Classificatore temporale basato su Transformer Encoder.

    Input:
        features: [B, Tmax, input_dim]
            Feature già estratte dai frame, ad esempio ConvNeXt-Tiny da 768 dimensioni.

        lengths: [B]
            Numero reale di frame per ogni clip, senza padding.

    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 9,
        dropout: float = 0.3,
        pooling: str = "cls",
        max_len: int = 1024,
        last_mean_ratio: float = 0.30,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model deve essere divisibile per num_heads. "
                f"Ricevuto d_model={d_model}, num_heads={num_heads}."
            )

        if pooling not in {"cls", "mean", "last_mean"}:
            raise ValueError("pooling deve essere 'cls', 'mean' oppure 'last_mean'.")

        if not 0.0 < last_mean_ratio <= 1.0:
            raise ValueError(
                f"last_mean_ratio deve essere compreso tra 0 e 1. "
                f"Ricevuto: {last_mean_ratio}."
            )

        self.pooling = pooling
        self.d_model = d_model
        self.last_mean_ratio = last_mean_ratio

        # Proietta le feature ConvNeXt da input_dim, es. 768,
        # alla dimensione interna del Transformer, es. 256.
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Token CLS opzionale: rappresenta tutta la clip.
        # Con mean/last_mean non aggiungiamo token: il pooling viene fatto
        # direttamente sulle feature temporali dei frame reali.
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        # batch_first evita permutazioni [T, B, D]; norm_first usa la variante
        # pre-norm, generalmente più stabile durante l’ottimizzazione.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_parameters()

    def _init_parameters(self):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    # Traduce le lunghezze reali nella convenzione PyTorch True=padding ignorato.
    def _make_padding_mask(self, lengths, max_len, device):
        """
        Restituisce una maschera [B, Tmax].

        In PyTorch Transformer:
            True  = posizione da ignorare
            False = posizione valida
        """
        lengths = lengths.to(device)

        time_idx = torch.arange(max_len, device=device).unsqueeze(0)
        padding_mask = time_idx >= lengths.unsqueeze(1)

        return padding_mask

    def _masked_mean_pooling(self, encoded, padding_mask):
        """
        Media temporale ignorando il padding.
        
        Usa tutti i frame reali della clip. È adatta quando l'informazione
        utile è distribuita lungo tutta la sequenza.
        """
        valid_mask = ~padding_mask  # True sui timestep reali

        encoded = encoded * valid_mask.unsqueeze(-1)

        denom = valid_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

        return encoded.sum(dim=1) / denom

    def _last_mean_pooling(self, encoded, lengths):
        """
        Media solo sugli ultimi frame reali della clip.

        È pensata per task in cui l'informazione discriminante è concentrata
        nella parte finale della sequenza, ad esempio l'esito del tiro:
        la preparazione e il rilascio avvengono prima, mentre la palla che
        entra/non entra nel canestro è visibile soprattutto negli ultimi frame.

        encoded: [B, T, d_model], senza token CLS.
        lengths: [B], numero reale di frame prima del padding.
        """
        pooled_outputs = []
        max_seq_len = encoded.size(1)

        for sample_idx in range(encoded.size(0)):
            length = int(lengths[sample_idx].item())
            length = max(1, min(length, max_seq_len))

            last_k = max(1, int(round(length * self.last_mean_ratio)))
            start = max(0, length - last_k)

            pooled_outputs.append(encoded[sample_idx, start:length].mean(dim=0))

        return torch.stack(pooled_outputs, dim=0)

    # Flusso completo: maschera, proiezione, eventuale CLS, positional encoding,
    # Transformer Encoder, pooling selezionato e logits di classificazione.
    def forward(self, features, lengths):
        """
        features: [B, Tmax, input_dim]
        lengths: [B]

        Restituisce:
            logits: [B, num_classes]
        """
        device = features.device
        batch_size, max_len, _ = features.shape

        # Maschera del padding per i frame reali.
        padding_mask = self._make_padding_mask(
            lengths=lengths,
            max_len=max_len,
            device=device,
        )  # [B, Tmax]

        # Proiezione delle feature nel d_model del Transformer.
        x = self.input_projection(features)  # [B, Tmax, d_model]
        x = self.input_norm(x)

        # Se usiamo CLS, lo aggiungiamo all'inizio della sequenza.
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, Tmax + 1, d_model]

            cls_padding_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=device,
            )

            padding_mask = torch.cat(
                [cls_padding_mask, padding_mask],
                dim=1,
            )  # [B, Tmax + 1]

        # Aggiunge informazione temporale/posizionale.
        x = self.positional_encoding(x)

        # Transformer Encoder.
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask,
        )  # [B, T, d_model]

        # Pooling finale.
        if self.pooling == "cls":
            final_representation = encoded[:, 0, :]
        elif self.pooling == "mean":
            final_representation = self._masked_mean_pooling(
                encoded,
                padding_mask,
            )
        elif self.pooling == "last_mean":
            final_representation = self._last_mean_pooling(
                encoded,
                lengths,
            )
        else:
            raise ValueError(f"Pooling non supportato: {self.pooling}")

        logits = self.classifier(final_representation)
        return logits
