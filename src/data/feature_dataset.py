from pathlib import Path

import torch
from torch.utils.data import Dataset


# Dizionario che associa ogni nome di classe a un indice numerico.
LABEL_TO_IDX = {
    "passaggio": 0,
    "tiroDaDue0": 1,
    "tiroDaDue1": 2,
    "tiroDaTre0": 3,
    "tiroDaTre1": 4,
    "tiroLibero0": 5,
    "tiroLibero1": 6,
    "idle": 7,
    "non-gioco": 8,
}

# Dizionario inverso: permette di convertire una predizione numerica
# nel nome testuale della classe corrispondente.
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


class FeatureDataset(Dataset):
    """
    Dataset PyTorch per caricare feature già estratte dalle clip video.

    Questo dataset non legge direttamente i file .mp4. Si aspetta invece file .pt
    generati da uno script di feature extraction.

    Ogni file .pt deve contenere almeno:
        - "features": tensore di forma [T, D]
        - "label" oppure "label_name"
        - "clip_id"

    Compatibilità label:
        - formato vecchio: "label" = nome testuale, es. "passaggio"
        - formato nuovo:  "label" = indice numerico, es. 0
        - formato nuovo DINOv2: "label" = indice numerico e "label_name" = nome testuale
    """

    def __init__(self, features_root: str | Path, split: str):
        self.features_root = Path(features_root)
        self.split = split
        split_dir = self.features_root / split

        if not split_dir.exists():
            raise RuntimeError(f"Cartella split non trovata: {split_dir}")

        # Cerca tutti i file .pt contenuti nelle sottocartelle delle classi.
        self.items = sorted(split_dir.glob("*/*.pt"))

        if len(self.items) == 0:
            raise RuntimeError(f"Nessuna feature trovata in: {split_dir}")

    def __len__(self):
        return len(self.items)

    def _parse_label(self, item):
        """
        Converte la label in indice numerico in modo robusto.

        Supporta:
            - item["label"] come stringa: "passaggio"
            - item["label"] come intero: 0
            - item["label"] come tensore: tensor(0)
            - item["label_name"] come stringa, se presente
        """

        # Se esiste label_name, preferiamo quella perché è esplicita.
        # Nel nuovo extract_features.py:
        #   "label": 0
        #   "label_name": "passaggio"
        raw_label = item.get("label_name", item["label"])

        if isinstance(raw_label, str):
            if raw_label not in LABEL_TO_IDX:
                raise ValueError(f"Label non riconosciuta: {raw_label}")
            return LABEL_TO_IDX[raw_label]

        if torch.is_tensor(raw_label):
            raw_label = raw_label.item()

        label = int(raw_label)

        if label not in IDX_TO_LABEL:
            raise ValueError(f"Indice label non valido: {label}")

        return label

    def __getitem__(self, idx: int):
        path = self.items[idx]

        item = torch.load(path, map_location="cpu")

        # Estrae le feature della clip e le converte in float.
        # Shape attesa: [T, D]
        features = item["features"].float()

        if features.ndim != 2:
            raise ValueError(
                f"Feature con shape non valida nel file {path}: {features.shape}"
            )

        # Converte la label in formato numerico long.
        label = self._parse_label(item)

        return {
            "features": features,
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(features.shape[0], dtype=torch.long),
            "clip_id": item["clip_id"],
        }


def collate_features(batch):
    """
    Funzione di collate personalizzata per il DataLoader.

    Serve a costruire un batch partendo da clip con lunghezze diverse.
    Applica padding temporale fino alla lunghezza massima del batch.

    Output:
        features: [B, Tmax, D]
        mask:     [B, Tmax]
        lengths:  [B]
        labels:   [B]
    """

    features_list = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.stack([item["length"] for item in batch])

    batch_size = len(features_list)
    max_len = max(x.shape[0] for x in features_list)
    feature_dim = features_list[0].shape[1]

    padded = torch.zeros(batch_size, max_len, feature_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, features in enumerate(features_list):
        T = features.shape[0]
        padded[i, :T] = features
        mask[i, :T] = True

    return {
        "features": padded,
        "mask": mask,
        "lengths": lengths,
        "labels": labels,
    }