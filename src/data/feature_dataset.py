from pathlib import Path

import torch
from torch.utils.data import Dataset


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

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


class FeatureDataset(Dataset):
    def __init__(self, features_root: str | Path, split: str):
        self.features_root = Path(features_root)
        self.split = split

        split_dir = self.features_root / split

        if not split_dir.exists():
            raise RuntimeError(f"Cartella split non trovata: {split_dir}")

        self.items = sorted(split_dir.glob("*/*.pt"))

        if len(self.items) == 0:
            raise RuntimeError(f"Nessuna feature trovata in: {split_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path = self.items[idx]
        item = torch.load(path, map_location="cpu")

        features = item["features"].float()  # [T, 512]
        label_name = item["label"]
        label = LABEL_TO_IDX[label_name]

        return {
            "features": features,
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(features.shape[0], dtype=torch.long),
            "clip_id": item["clip_id"],
        }


def collate_features(batch):
    """
    Padding temporaneo dentro il batch.

    Output:
        padded:  [B, Tmax, 512]
        mask:    [B, Tmax]
        lengths: [B]
        labels:  [B]
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