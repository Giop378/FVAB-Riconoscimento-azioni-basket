"""
Utility condivise per caricare e usare le sequenze temporali di tracking.

Il file gestisce le feature palla/canestro salvate da
extract_ball_rim_tracking_features.py nei file:
- tracking_sequences.npz
- tracking_sequence_index.json

Le sequenze [S, K] vengono recuperate tramite la chiave normalizzata della clip,
interpolate alla lunghezza richiesta [T, K] e, se necessario, normalizzate con
mean/std salvate nel checkpoint del modello.
"""

from pathlib import Path
import json

import numpy as np


def normalize_clip_key(path_value) -> str:
    """
    Normalizza il path di una clip/feature in una chiave confrontabile.

    Esempi:
    - train/tiroDaDue0/clip_000001.mp4 -> train/tiroDaDue0/clip_000001
    - data/features/.../train/tiroDaDue0/clip_000001.pt -> train/tiroDaDue0/clip_000001
    """
    path_str = str(path_value).replace("\\", "/")
    parts = [p for p in Path(path_str).parts if p not in {"", "."}]

    split_idx = None
    for idx, part in enumerate(parts):
        if part in {"train", "val", "test"}:
            split_idx = idx
            break

    if split_idx is not None:
        parts = parts[split_idx:]

    return Path(*parts).with_suffix("").as_posix()


def interpolate_sequence_array(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Interpola una sequenza tracking [S, K] alla lunghezza target [T, K]."""
    target_len = int(target_len)

    if target_len <= 0:
        return np.zeros((0, sequence.shape[1] if sequence.ndim == 2 else 0), dtype=np.float32)

    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"La sequenza tracking deve avere forma [S, K], ricevuta {sequence.shape}.")

    source_len, num_features = sequence.shape

    if source_len == 0:
        return np.zeros((target_len, num_features), dtype=np.float32)

    if source_len == target_len:
        return sequence.astype(np.float32)

    if source_len == 1:
        return np.repeat(sequence, repeats=target_len, axis=0).astype(np.float32)

    source_x = np.linspace(0.0, 1.0, source_len, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)

    resized = np.empty((target_len, num_features), dtype=np.float32)
    for feature_idx in range(num_features):
        resized[:, feature_idx] = np.interp(target_x, source_x, sequence[:, feature_idx])

    return resized.astype(np.float32)


class TrackingSequenceFeatureStore:
    """
    Store delle feature tracking temporali palla/canestro.

    Ogni clip è associata a una sequenza [S, K] salvata in tracking_sequences.npz
    e indicizzata da tracking_sequence_index.json. Le sequenze possono essere
    interpolate alla lunghezza temporale delle feature DINOv3 della clip.
    """

    def __init__(
        self,
        npz_path: str,
        index_path: str | None = None,
        feature_names=None,
        mean=None,
        std=None,
        normalized: bool = False,
    ):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"File NPZ tracking temporale non trovato: {self.npz_path}")

        if index_path is None:
            index_path = self.npz_path.with_name("tracking_sequence_index.json")

        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Indice tracking temporale non trovato: {self.index_path}. "
                "Passa --tracking-sequence-index oppure genera l'indice con lo script di estrazione."
            )

        with open(self.index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.feature_names = feature_names or index_data.get("feature_names")
        if not self.feature_names:
            raise ValueError("Nomi feature tracking temporali non presenti nell'indice.")

        self.rows_by_key = {}
        sequences = index_data.get("sequences", {})
        for key, value in sequences.items():
            if isinstance(value, dict):
                self.rows_by_key[key] = value.get("array_key")
            else:
                self.rows_by_key[key] = str(value)

        if not self.rows_by_key:
            raise ValueError(f"Nessuna sequenza tracking indicizzata in {self.index_path}.")

        self.data = np.load(self.npz_path)
        self.normalized = bool(normalized)
        self.mean = None if mean is None else np.array(mean, dtype=np.float32)
        self.std = None if std is None else np.array(std, dtype=np.float32)

        if self.mean is None:
            self.mean = np.zeros(self.num_features, dtype=np.float32)
        if self.std is None:
            self.std = np.ones(self.num_features, dtype=np.float32)

        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def has(self, path_value) -> bool:
        return normalize_clip_key(path_value) in self.rows_by_key

    def get_raw(self, path_value, target_len=None, missing_policy: str = "zeros") -> np.ndarray:
        key = normalize_clip_key(path_value)

        if key in self.rows_by_key:
            array_key = self.rows_by_key[key]
            if array_key not in self.data:
                raise KeyError(
                    f"Array '{array_key}' non trovato in {self.npz_path} "
                    f"per path='{path_value}'."
                )
            sequence = np.asarray(self.data[array_key], dtype=np.float32)
        elif missing_policy == "zeros":
            length = int(target_len) if target_len is not None else 1
            sequence = np.zeros((max(1, length), self.num_features), dtype=np.float32)
        else:
            raise KeyError(
                f"Sequenza tracking non trovata per path='{path_value}' "
                f"con chiave normalizzata='{key}'."
            )

        if target_len is not None:
            sequence = interpolate_sequence_array(sequence, int(target_len))

        return sequence.astype(np.float32)

    def get(self, path_value, target_len=None, missing_policy: str = "zeros") -> np.ndarray:
        sequence = self.get_raw(path_value, target_len=target_len, missing_policy=missing_policy)
        if self.normalized:
            sequence = (sequence - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
        return sequence.astype(np.float32)

    def fit_normalizer_from_paths(self, paths) -> None:
        sequences = []
        missing = 0

        for path in paths:
            if self.has(path):
                sequences.append(self.get_raw(path, missing_policy="error"))
            else:
                missing += 1

        if not sequences:
            raise RuntimeError(
                "Impossibile normalizzare le sequenze tracking: "
                "nessun campione del training set ha feature tracking associate."
            )

        matrix = np.concatenate(sequences, axis=0).astype(np.float32)
        self.mean = matrix.mean(axis=0).astype(np.float32)
        self.std = matrix.std(axis=0).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        self.normalized = True

        print("\n# Normalizzazione sequenze tracking")
        print(f"Frame usati per stimare mean/std: {matrix.shape[0]}")
        print(f"Campioni train senza sequenze tracking: {missing}")

    def get_config(self) -> dict:
        return {
            "enabled": True,
            "type": "temporal_sequence",
            "npz_path": str(self.npz_path),
            "index_path": str(self.index_path),
            "num_features": self.num_features,
            "feature_names": list(self.feature_names),
            "normalized": bool(self.normalized),
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }
