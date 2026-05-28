import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPooling(nn.Module):
    """
    Modulo di attention pooling temporale.

    Riceve in input tutti gli output temporali della GRU/BiGRU e impara
    un peso per ogni frame/timestep. In questo modo il modello può dare
    più importanza ai frame più informativi della clip.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        # Piccola rete feed-forward che assegna uno score scalare a ogni timestep.
        # Input:  vettore GRU del singolo timestep, dimensione input_dim
        # Output: score di attenzione, dimensione 1
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, outputs, lengths, return_weights: bool = False):
        """
        outputs: [B, Tmax, D]
            Output della GRU/BiGRU per ogni elemento del batch.
            B    = batch size
            Tmax = lunghezza massima della sequenza nel batch
            D    = dimensione dell'output GRU/BiGRU

        lengths: [B]
            Lunghezze reali delle clip, senza considerare il padding.

        return_weights:
            Se True restituisce anche i pesi di attenzione, utili per capire
            quali frame sono stati considerati più importanti dal modello.
        """
        # Portiamo lengths sullo stesso device di outputs, necessario quando si usa CUDA.
        lengths = lengths.to(outputs.device)

        _, max_len, _ = outputs.shape

        # Costruzione della maschera per ignorare i timestep di padding.
        # mask[b, t] è True solo se t appartiene alla parte reale della clip b.
        time_idx = torch.arange(max_len, device=outputs.device).unsqueeze(0)
        mask = time_idx < lengths.unsqueeze(1)

        # Calcola uno score di attenzione per ogni timestep.
        scores = self.attention(outputs).squeeze(-1)  # [B, Tmax]

        # I timestep di padding vengono forzati a uno score molto basso,
        # così dopo la softmax avranno peso praticamente nullo.
        scores = scores.masked_fill(~mask, -1e9)

        # Converte gli score in pesi normalizzati lungo la dimensione temporale.
        weights = torch.softmax(scores, dim=1)  # [B, Tmax]

        # Media pesata degli output temporali della GRU/BiGRU.
        # Risultato: un unico vettore per ogni clip.
        pooled = torch.sum(outputs * weights.unsqueeze(-1), dim=1)  # [B, D]

        if return_weights:
            return pooled, weights

        return pooled


class GRUActionClassifier(nn.Module):
    """
    Classificatore per action recognition basato su GRU/BiGRU + attention pooling.

    Input atteso:
        features: [B, Tmax, input_dim]
            Sequenza di feature già estratte dai frame della clip.
            Ad esempio, se si usa ConvNeXt-Tiny, input_dim è tipicamente 768.

        lengths: [B]
            Numero reale di frame/timestep per ogni clip, prima del padding.

    Output:
        logits: [B, num_classes]
            Punteggi grezzi per ciascuna classe. La softmax non viene applicata qui,
            perché CrossEntropyLoss la applica internamente.
    """

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

        # GRU temporale.
        # batch_first=True indica che gli input hanno forma [B, T, D].
        # Se bidirectional=True, la GRU legge la sequenza sia in avanti sia all'indietro.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dimensione dell'output della GRU.
        # Con BiGRU si concatena direzione forward e backward, quindi hidden_dim * 2.
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.attention_pooling = AttentionPooling(gru_out_dim)

        # La rappresentazione finale coincide con il vettore prodotto dall'attention pooling.
        # Prima, con il pooling a piramide temporale, era gru_out_dim * 4.
        # Ora è solo gru_out_dim, quindi il classificatore ha meno parametri.
        final_dim = gru_out_dim

        # Classificatore finale.
        # LayerNorm stabilizza le feature, Dropout riduce overfitting,
        # Linear produce i logits per le classi.
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Dropout(dropout),
            nn.Linear(final_dim, num_classes),
        )

    def forward(self, features, lengths, return_attention: bool = False):
        """
        features: [B, Tmax, input_dim]
            Feature dei frame/segmenti, già paddate alla stessa lunghezza nel batch.

        lengths: [B]
            Lunghezze reali delle clip, usate per non far processare il padding alla GRU.

        return_attention:
            Se True restituisce anche i pesi di attention.
        """
        # pack_padded_sequence permette alla GRU di ignorare i timestep di padding.
        # enforce_sorted=False evita di dover ordinare manualmente il batch per lunghezza.
        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Esecuzione della GRU sulla sequenza compressa.
        packed_outputs, _ = self.gru(packed)

        # Riconverte l'output packed in tensore paddato [B, Tmax, D].
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
        )

        # Attention pooling sugli output temporali della GRU/BiGRU.
        if return_attention:
            final_representation, attention_weights = self.attention_pooling(
                outputs,
                lengths,
                return_weights=True,
            )
        else:
            final_representation = self.attention_pooling(
                outputs,
                lengths,
                return_weights=False,
            )
            attention_weights = None

        # Produzione dei logits finali per le classi.
        logits = self.classifier(final_representation)

        if return_attention:
            return logits, attention_weights

        return logits
