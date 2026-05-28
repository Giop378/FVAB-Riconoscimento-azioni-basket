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
    generati da uno script di feature extraction

    Struttura attesa delle cartelle:
    features_root/
    |-- train/
    |   |-- passaggio/
    |   |   |-- clip_000001.pt
    |   |   `-- ...
    |   |-- tiroDaDue0/
    |   `-- ...
    |-- val/
    `-- test/

    Ogni file .pt deve contenere almeno:
        - "features": tensore di forma [T, D]
        - "label": nome testuale della classe
        - "clip_id": identificativo della clip

    Dove:
        T = numero di frame/feature della clip
        D = dimensione del vettore di feature per ogni frame
    """

    def __init__(self, features_root: str | Path, split: str):
        """
        Inizializza il dataset caricando la lista dei file .pt dello split richiesto.

        Parametri:
            features_root: cartella principale contenente le feature estratte.
            split: split da caricare, ad esempio "train", "val" o "test".
        """
        
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
        """
        Restituisce il numero totale di clip presenti nel dataset.
        Viene usato automaticamente dal DataLoader di PyTorch.
        """
        return len(self.items)

    def __getitem__(self, idx: int):
        """
        Carica un singolo elemento del dataset.

        Parametri:
            idx: indice della clip da caricare.

        Restituisce:
            Un dizionario contenente:
                - features: tensore [T, D]
                - label: etichetta numerica della classe
                - length: numero reale di frame/feature della clip
                - clip_id: identificativo della clip
        """
        
        path = self.items[idx]

        
        item = torch.load(path, map_location="cpu")

        # Estrae le feature della clip e le converte in float.
        # Shape attesa: [T, D], dove T è la lunghezza temporale della clip
        # e D è la dimensione delle feature estratte dal backbone.
        features = item["features"].float()  # [T, D]

        # Legge il nome testuale della label e lo converte nell'indice numerico.
        label_name = item["label"]
        label = LABEL_TO_IDX[label_name]

        # Restituisce tutte le informazioni necessarie per il training.
        # length è fondamentale perché le clip hanno lunghezze variabili.
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
    Poiché PyTorch non può impilare direttamente tensori [T, D] con T diverso,
    questa funzione applica padding temporale fino alla lunghezza massima del batch.

    Output:
        features: [B, Tmax, D]
        mask:     [B, Tmax]
        lengths:  [B]
        labels:   [B]

    Dove:
        B = batch size
        Tmax = lunghezza della clip più lunga nel batch
        D = dimensione delle feature
    """
    # Estrae dal batch le sequenze di feature, le label e le lunghezze reali.
    features_list = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.stack([item["length"] for item in batch])

    # Numero di elementi nel batch.
    batch_size = len(features_list)

    # Lunghezza massima tra le clip del batch.
    # Tutte le altre clip verranno paddate fino a questa lunghezza.
    max_len = max(x.shape[0] for x in features_list)

    # Dimensione del vettore di feature per ogni frame.
    # Viene letta automaticamente, quindi il codice può funzionare con backbone diversi.
    feature_dim = features_list[0].shape[1]

    # Tensore finale che conterrà le feature paddate.
    # Le posizioni non usate rimangono a zero.
    padded = torch.zeros(batch_size, max_len, feature_dim)

    # Mask booleana che indica quali posizioni sono frame reali e quali sono padding.
    # True  = frame reale
    # False = padding
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Copia ogni sequenza nella parte iniziale del tensore padded.
    for i, features in enumerate(features_list):
        # Lunghezza reale della clip i-esima.
        T = features.shape[0]

        # Inserisce le feature reali nelle prime T posizioni.
        padded[i, :T] = features

        # Segna come valide solo le prime T posizioni della mask.
        mask[i, :T] = True

    # Restituisce il batch pronto per essere dato al modello.
    return {
        "features": padded,
        "mask": mask,
        "lengths": lengths,
        "labels": labels,
    }