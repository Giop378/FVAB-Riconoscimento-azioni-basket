from pathlib import Path
import argparse
import random
import sys
import shlex
import traceback
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.data.feature_dataset import FeatureDataset, collate_features, IDX_TO_LABEL, LABEL_TO_IDX
from src.models.temporal_transformer_classifier import TemporalTransformerActionClassifier


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_reconstructed_command() -> str:
    parts = [sys.executable] + sys.argv
    return " ".join(shlex.quote(str(part)) for part in parts)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_original_idx_to_label():
    """
    Restituisce il mapping originale idx -> label in forma di dict.
    IDX_TO_LABEL nel progetto è già usato come mapping indicizzabile.
    """
    return {idx: IDX_TO_LABEL[idx] for idx in range(len(IDX_TO_LABEL))}


def build_label_mapping(label_mode: str):
    """
    Costruisce:
    - label_mapping: label originale -> label usata dal modello
    - idx_to_label: indice di output del modello -> label usata dal modello

    Modalità previste:
    - original: 9 classi originali.
    - action_noaction: passaggio / tiro / no-action.
    - shot_type: passaggio / tipo tiro / idle / non-gioco.
    - shot_outcome: passaggio / esito tiro / idle / non-gioco.
    - action_group: passaggio / tiro / idle / non-gioco.
    - shot_type_only: solo clip di tiro, classificate per tipo.
    - shot_outcome_only: solo clip di tiro, classificate per esito.
    - passaggio_noaction_only: solo clip passaggio/idle/non-gioco,
      con idle e non-gioco collassati in no-action.
    """
    original_idx_to_label = get_original_idx_to_label()

    if label_mode == "original":
        idx_to_label = original_idx_to_label
        label_mapping = {label: label for label in original_idx_to_label.values()}

    elif label_mode == "action_noaction":
        idx_to_label = {
            0: "passaggio",
            1: "tiro",
            2: "no-action",
        }
        label_mapping = {
            "passaggio": "passaggio",
            "tiroDaDue0": "tiro",
            "tiroDaDue1": "tiro",
            "tiroDaTre0": "tiro",
            "tiroDaTre1": "tiro",
            "tiroLibero0": "tiro",
            "tiroLibero1": "tiro",
            "idle": "no-action",
            "non-gioco": "no-action",
        }

    elif label_mode == "shot_type":
        idx_to_label = {
            0: "passaggio",
            1: "tiroDaDue",
            2: "tiroDaTre",
            3: "tiroLibero",
            4: "idle",
            5: "non-gioco",
        }
        label_mapping = {
            "passaggio": "passaggio",
            "tiroDaDue0": "tiroDaDue",
            "tiroDaDue1": "tiroDaDue",
            "tiroDaTre0": "tiroDaTre",
            "tiroDaTre1": "tiroDaTre",
            "tiroLibero0": "tiroLibero",
            "tiroLibero1": "tiroLibero",
            "idle": "idle",
            "non-gioco": "non-gioco",
        }

    elif label_mode == "shot_outcome":
        idx_to_label = {
            0: "passaggio",
            1: "tiro0",
            2: "tiro1",
            3: "idle",
            4: "non-gioco",
        }
        label_mapping = {
            "passaggio": "passaggio",
            "tiroDaDue0": "tiro0",
            "tiroDaTre0": "tiro0",
            "tiroLibero0": "tiro0",
            "tiroDaDue1": "tiro1",
            "tiroDaTre1": "tiro1",
            "tiroLibero1": "tiro1",
            "idle": "idle",
            "non-gioco": "non-gioco",
        }

    elif label_mode == "action_group":
        idx_to_label = {
            0: "passaggio",
            1: "tiro",
            2: "idle",
            3: "non-gioco",
        }
        label_mapping = {
            "passaggio": "passaggio",
            "tiroDaDue0": "tiro",
            "tiroDaDue1": "tiro",
            "tiroDaTre0": "tiro",
            "tiroDaTre1": "tiro",
            "tiroLibero0": "tiro",
            "tiroLibero1": "tiro",
            "idle": "idle",
            "non-gioco": "non-gioco",
        }

    elif label_mode == "shot_type_only":
        idx_to_label = {
            0: "tiroDaDue",
            1: "tiroDaTre",
            2: "tiroLibero",
        }
        label_mapping = {
            "tiroDaDue0": "tiroDaDue",
            "tiroDaDue1": "tiroDaDue",
            "tiroDaTre0": "tiroDaTre",
            "tiroDaTre1": "tiroDaTre",
            "tiroLibero0": "tiroLibero",
            "tiroLibero1": "tiroLibero",
        }

    elif label_mode == "shot_outcome_only":
        idx_to_label = {
            0: "tiro0",
            1: "tiro1",
        }
        label_mapping = {
            "tiroDaDue0": "tiro0",
            "tiroDaTre0": "tiro0",
            "tiroLibero0": "tiro0",
            "tiroDaDue1": "tiro1",
            "tiroDaTre1": "tiro1",
            "tiroLibero1": "tiro1",
        }

    elif label_mode == "passaggio_noaction_only":
        idx_to_label = {
            0: "passaggio",
            1: "no-action",
        }
        label_mapping = {
            "passaggio": "passaggio",
            "idle": "no-action",
            "non-gioco": "no-action",
        }

    else:
        raise ValueError(f"Label mode non supportata: {label_mode}")

    return label_mapping, idx_to_label


def get_label_to_idx(idx_to_label):
    return {label: idx for idx, label in idx_to_label.items()}


def normalize_sample_label(label_value) -> int:
    if isinstance(label_value, torch.Tensor):
        return int(label_value.item())
    return int(label_value)


def infer_original_label_from_item(item):
    """
    Cerca di ricavare la label originale senza caricare il tensore delle feature.
    Nel FeatureDataset corrente gli item sono path e la label è il nome della cartella padre.
    """
    original_idx_to_label = get_original_idx_to_label()

    if isinstance(item, dict):
        if "label" in item:
            label = item["label"]
            if isinstance(label, str):
                return label
            return original_idx_to_label[normalize_sample_label(label)]

        if "path" in item:
            return Path(item["path"]).parent.name

    return Path(item).parent.name


class LabelMappedDataset(Dataset):
    """
    Wrapper per FeatureDataset che permette di:
    - rimappare le label originali in label aggregate;
    - filtrare alcune classi quando serve addestrare un sotto-modello.

    Esempio:
    - shot_type_only tiene solo le clip di tiro e le rimappa in:
      tiroDaDue / tiroDaTre / tiroLibero.
    - shot_outcome_only tiene solo le clip di tiro e le rimappa in:
      tiro0 / tiro1.
    """

    def __init__(self, base_dataset: Dataset, label_mode: str):
        self.base_dataset = base_dataset
        self.label_mode = label_mode

        self.label_mapping, self.idx_to_label = build_label_mapping(label_mode)
        self.label_to_idx = get_label_to_idx(self.idx_to_label)
        self.original_idx_to_label = get_original_idx_to_label()

        self.indices = []
        self.mapped_labels = []
        self.original_labels = []

        if hasattr(base_dataset, "items"):
            for idx, item in enumerate(base_dataset.items):
                original_label = infer_original_label_from_item(item)
                self._try_add_item(idx, original_label)
        else:
            for idx in range(len(base_dataset)):
                sample = base_dataset[idx]
                if "label" not in sample:
                    raise KeyError("Il sample del dataset non contiene la chiave 'label'.")

                original_label_idx = normalize_sample_label(sample["label"])
                original_label = self.original_idx_to_label[original_label_idx]
                self._try_add_item(idx, original_label)

        if len(self.indices) == 0:
            raise ValueError(
                f"Nessun campione disponibile per label_mode='{label_mode}'. "
                "Controlla mapping, nomi delle cartelle e split del dataset."
            )

    def _try_add_item(self, idx: int, original_label: str):
        if original_label not in self.label_mapping:
            return

        mapped_label = self.label_mapping[original_label]
        mapped_idx = self.label_to_idx[mapped_label]

        self.indices.append(idx)
        self.mapped_labels.append(mapped_idx)
        self.original_labels.append(original_label)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        sample = self.base_dataset[base_idx]

        # Copia il dizionario per evitare modifiche indesiderate al sample originale.
        sample = dict(sample)

        mapped_label = self.mapped_labels[idx]

        # collate_features si aspetta che item["label"] sia un Tensor,
        # perché poi fa torch.stack([item["label"] for item in batch]).
        sample["label"] = torch.tensor(mapped_label, dtype=torch.long)

        return sample



TRACKING_METADATA_COLUMNS = {
    "clip_id",
    "split",
    "label",
    "path",
    "video_frames",
    "fps",
    "sampled_frames",
    "video_width",
    "video_height",
}


def normalize_clip_key(path_value) -> str:
    """
    Normalizza il path di una clip/feature in una chiave confrontabile.

    Esempi convertiti nella stessa forma:
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

    normalized = Path(*parts).with_suffix("").as_posix()
    return normalized


def get_base_item_path_from_label_dataset(label_dataset, idx: int) -> str:
    """
    Recupera il path originale associato a un elemento di LabelMappedDataset.
    Serve per associare ogni feature video alla riga corrispondente del CSV tracking.
    """
    if hasattr(label_dataset, "indices") and hasattr(label_dataset, "base_dataset"):
        base_idx = label_dataset.indices[idx]
        base_dataset = label_dataset.base_dataset

        if hasattr(base_dataset, "items"):
            item = base_dataset.items[base_idx]

            if isinstance(item, dict):
                if "path" in item:
                    return str(item["path"])
                return str(item)

            return str(item)

    if hasattr(label_dataset, "base_dataset") and hasattr(label_dataset.base_dataset, "items"):
        item = label_dataset.base_dataset.items[idx]
        if isinstance(item, dict):
            return str(item.get("path", item))
        return str(item)

    return ""


class TrackingFeatureStore:
    def __init__(self, csv_path: str, feature_names=None):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV feature tracking non trovato: {self.csv_path}")

        self.rows_by_key = {}
        self.feature_names = feature_names
        self.mean = None
        self.std = None
        self.normalized = False

        self._load()

    def _load(self):
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(f"CSV tracking vuoto o non valido: {self.csv_path}")

            if "path" not in reader.fieldnames:
                raise ValueError(
                    f"Il CSV tracking deve contenere la colonna 'path'. "
                    f"Colonne presenti: {reader.fieldnames}"
                )

            if self.feature_names is None:
                self.feature_names = [
                    col for col in reader.fieldnames
                    if col not in TRACKING_METADATA_COLUMNS
                ]

            if not self.feature_names:
                raise ValueError("Nessuna feature tracking trovata nel CSV.")

            for row in reader:
                key = normalize_clip_key(row["path"])

                try:
                    vector = np.array(
                        [float(row[name]) for name in self.feature_names],
                        dtype=np.float32,
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"Feature mancante nel CSV tracking: {exc}. "
                        f"Feature attese: {self.feature_names}"
                    ) from exc
                except ValueError as exc:
                    raise ValueError(
                        f"Valore non numerico nel CSV tracking per path={row.get('path')}."
                    ) from exc

                self.rows_by_key[key] = vector

        self.mean = np.zeros(self.num_features, dtype=np.float32)
        self.std = np.ones(self.num_features, dtype=np.float32)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def has(self, path_value) -> bool:
        return normalize_clip_key(path_value) in self.rows_by_key

    def get_raw(self, path_value, missing_policy="zeros"):
        key = normalize_clip_key(path_value)

        if key in self.rows_by_key:
            return self.rows_by_key[key].copy()

        if missing_policy == "zeros":
            return np.zeros(self.num_features, dtype=np.float32)

        raise KeyError(
            f"Feature tracking non trovate per path='{path_value}' "
            f"con chiave normalizzata='{key}'."
        )

    def get(self, path_value, missing_policy="zeros"):
        vector = self.get_raw(path_value, missing_policy=missing_policy)
        if self.normalized:
            vector = (vector - self.mean) / self.std
        return vector.astype(np.float32)

    def fit_normalizer_from_label_dataset(self, label_dataset):
        vectors = []
        missing = 0

        for idx in range(len(label_dataset)):
            path = get_base_item_path_from_label_dataset(label_dataset, idx)
            if self.has(path):
                vectors.append(self.get_raw(path, missing_policy="error"))
            else:
                missing += 1

        if not vectors:
            raise RuntimeError(
                "Impossibile normalizzare le feature tracking: "
                "nessun campione del training set ha feature tracking associate."
            )

        matrix = np.stack(vectors, axis=0).astype(np.float32)
        self.mean = matrix.mean(axis=0).astype(np.float32)
        self.std = matrix.std(axis=0).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        self.normalized = True

        print("\n# Normalizzazione feature tracking")
        print(f"Campioni usati per stimare mean/std: {len(vectors)}")
        print(f"Campioni train senza feature tracking: {missing}")

    def get_config(self):
        return {
            "enabled": True,
            "type": "aggregate",
            "csv_path": str(self.csv_path),
            "num_features": self.num_features,
            "feature_names": list(self.feature_names),
            "normalized": bool(self.normalized),
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }


def interpolate_sequence_array(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ridimensiona una sequenza [S, K] a [target_len, K] con interpolazione lineare.
    Serve per allineare le feature tracking temporali alla lunghezza delle feature DINOv3.
    """
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
    Carica sequenze temporali di tracking palla/canestro salvate in un file NPZ.

    Ogni clip è associata a una matrice [S, K], dove S è il numero di frame
    campionati dal detector e K il numero di feature per frame. In training la
    sequenza viene interpolata alla lunghezza reale della sequenza DINOv3 [T, D].
    """

    def __init__(
        self,
        npz_path: str,
        index_path: str = None,
        feature_names=None,
        mean=None,
        std=None,
        normalized=False,
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

    def get_raw(self, path_value, target_len=None, missing_policy="zeros"):
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

    def get(self, path_value, target_len=None, missing_policy="zeros"):
        sequence = self.get_raw(path_value, target_len=target_len, missing_policy=missing_policy)
        if self.normalized:
            sequence = (sequence - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
        return sequence.astype(np.float32)

    def fit_normalizer_from_label_dataset(self, label_dataset):
        sequences = []
        missing = 0

        for idx in range(len(label_dataset)):
            path = get_base_item_path_from_label_dataset(label_dataset, idx)
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
        print(f"Frame/sequenze usati per stimare mean/std: {matrix.shape[0]}")
        print(f"Campioni train senza sequenze tracking: {missing}")

    def get_config(self):
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


class TrackingAugmentedDataset(Dataset):
    """
    Concatena a ogni timestep delle feature video un vettore globale di feature tracking.

    Input originale:
      features: [T, D]

    Output:
      features: [T, D + K]

    dove K è il numero di feature estratte da YOLO palla/canestro.
    """

    def __init__(self, label_dataset: Dataset, tracking_store: TrackingFeatureStore, missing_policy="zeros"):
        self.label_dataset = label_dataset
        self.tracking_store = tracking_store
        self.missing_policy = missing_policy

        self.missing_count = 0
        for idx in range(len(label_dataset)):
            path = get_base_item_path_from_label_dataset(label_dataset, idx)
            if not tracking_store.has(path):
                self.missing_count += 1

        print("\n# TrackingAugmentedDataset")
        print(f"Campioni: {len(label_dataset)}")
        print(f"Feature tracking: {tracking_store.num_features}")
        print(f"Campioni senza tracking features: {self.missing_count}")
        print(f"Missing policy: {missing_policy}")

    def __len__(self):
        return len(self.label_dataset)

    def __getitem__(self, idx):
        sample = dict(self.label_dataset[idx])

        features = sample["features"]
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.float()

        path = get_base_item_path_from_label_dataset(self.label_dataset, idx)
        tracking_vector = self.tracking_store.get(path, missing_policy=self.missing_policy)
        tracking_tensor = torch.tensor(tracking_vector, dtype=features.dtype)

        if features.ndim != 2:
            raise ValueError(
                f"Le feature video devono avere forma [T, D], "
                f"ma per {path} hanno forma {tuple(features.shape)}."
            )

        tracking_sequence = tracking_tensor.unsqueeze(0).repeat(features.shape[0], 1)
        sample["features"] = torch.cat([features, tracking_sequence], dim=1)

        return sample


class TemporalTrackingAugmentedDataset(Dataset):
    """
    Concatena a ogni timestep delle feature video una sequenza temporale di
    feature palla/canestro allineata alla lunghezza della clip.

    Input originale:
      features: [T, D]
      tracking sequence: [S, K]

    Output:
      features: [T, D + K]
    """

    def __init__(self, label_dataset: Dataset, tracking_store: TrackingSequenceFeatureStore, missing_policy="zeros"):
        self.label_dataset = label_dataset
        self.tracking_store = tracking_store
        self.missing_policy = missing_policy

        self.missing_count = 0
        for idx in range(len(label_dataset)):
            path = get_base_item_path_from_label_dataset(label_dataset, idx)
            if not tracking_store.has(path):
                self.missing_count += 1

        print("\n# TemporalTrackingAugmentedDataset")
        print(f"Campioni: {len(label_dataset)}")
        print(f"Feature tracking temporali per frame: {tracking_store.num_features}")
        print(f"Campioni senza tracking sequences: {self.missing_count}")
        print(f"Missing policy: {missing_policy}")

    def __len__(self):
        return len(self.label_dataset)

    def __getitem__(self, idx):
        sample = dict(self.label_dataset[idx])

        features = sample["features"]
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.float()

        if features.ndim != 2:
            path = get_base_item_path_from_label_dataset(self.label_dataset, idx)
            raise ValueError(
                f"Le feature video devono avere forma [T, D], "
                f"ma per {path} hanno forma {tuple(features.shape)}."
            )

        path = get_base_item_path_from_label_dataset(self.label_dataset, idx)
        tracking_sequence = self.tracking_store.get(
            path,
            target_len=features.shape[0],
            missing_policy=self.missing_policy,
        )
        tracking_tensor = torch.tensor(tracking_sequence, dtype=features.dtype)

        if tracking_tensor.shape[0] != features.shape[0]:
            raise ValueError(
                f"Lunghezza tracking non coerente per {path}: "
                f"features T={features.shape[0]}, tracking T={tracking_tensor.shape[0]}."
            )

        sample["features"] = torch.cat([features, tracking_tensor], dim=1)
        return sample


def get_dataset_labels_and_counts(dataset, num_classes: int):
    labels = []
    counts = torch.zeros(num_classes, dtype=torch.float)

    if hasattr(dataset, "mapped_labels"):
        iterable_labels = dataset.mapped_labels
    else:
        iterable_labels = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            iterable_labels.append(normalize_sample_label(sample["label"]))

    for label_idx in iterable_labels:
        label_idx = int(label_idx)
        if label_idx < 0 or label_idx >= num_classes:
            raise ValueError(
                f"Label index fuori range: {label_idx}. "
                f"Numero classi corrente: {num_classes}."
            )
        labels.append(label_idx)
        counts[label_idx] += 1

    labels = torch.tensor(labels, dtype=torch.long)
    return labels, counts


def compute_class_weights_from_counts(counts: torch.Tensor, power: float = 0.5):
    weights = 1.0 / torch.pow(counts.clamp(min=1.0), power)
    weights = weights / weights.mean()
    return weights


def build_weighted_sampler(labels: torch.Tensor, counts: torch.Tensor, power: float, seed: int):
    class_sample_weights = 1.0 / torch.pow(counts.clamp(min=1.0), power)
    sample_weights = class_sample_weights[labels]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def print_label_mode_info(label_mode: str, label_mapping, idx_to_label):
    print("\n# Label mode")
    print(f"Label mode: {label_mode}")

    print("Classi usate nel training:")
    for idx in range(len(idx_to_label)):
        print(f"  {idx}: {idx_to_label[idx]}")

    print("\nMapping label originali -> label usate nel training:")
    for idx in range(len(IDX_TO_LABEL)):
        original_label = IDX_TO_LABEL[idx]
        if original_label in label_mapping:
            print(f"  {original_label} -> {label_mapping[original_label]}")
        else:
            print(f"  {original_label} -> esclusa")


def print_class_stats(counts, idx_to_label, class_weights=None):
    print("Class counts:")
    for idx in range(len(counts)):
        print(f"  {idx_to_label[idx]}: {int(counts[idx].item())}")

    if class_weights is not None:
        print("\nClass weights:")
        for idx in range(len(class_weights)):
            print(f"  {idx_to_label[idx]}: {class_weights[idx].item():.4f}")


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float = 1.0):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(features, lengths)
        loss = criterion(logits, labels)

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, macro_f1, weighted_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(features, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, macro_f1, weighted_f1, all_labels, all_preds


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/temporal_transformer")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "last_mean"])
    parser.add_argument(
        "--last-mean-ratio",
        type=float,
        default=0.30,
        help=(
            "Percentuale finale della clip da usare con --pooling last_mean. "
            "Esempio: 0.30 usa circa l'ultimo 30% dei frame reali."
        ),
    )
    parser.add_argument("--max-len", type=int, default=1024)

    parser.add_argument(
        "--label-mode",
        type=str,
        default="original",
        choices=[
            "original",
            "action_noaction",
            "shot_type",
            "shot_outcome",
            "action_group",
            "shot_type_only",
            "shot_outcome_only",
            "passaggio_noaction_only",
        ],
        help=(
            "Modalità di etichettatura. "
            "Per la gerarchia usare: action_noaction, shot_type_only, shot_outcome_only. "
            "Per il correttore binario L1 usare: passaggio_noaction_only."
        ),
    )

    parser.add_argument(
        "--tracking-features-csv",
        type=str,
        default=None,
        help=(
            "CSV prodotto da extract_ball_rim_tracking_features.py. "
            "Se indicato, le feature tracking aggregate vengono concatenate alle feature video."
        ),
    )
    parser.add_argument(
        "--tracking-sequences-npz",
        type=str,
        default=None,
        help=(
            "File NPZ con sequenze temporali di tracking palla/canestro. "
            "Se indicato, ogni sequenza [S, K] viene interpolata a [T, K] "
            "e concatenata alle feature video frame per frame."
        ),
    )
    parser.add_argument(
        "--tracking-sequence-index",
        type=str,
        default=None,
        help=(
            "JSON indice associato a --tracking-sequences-npz. "
            "Default: tracking_sequence_index.json nella stessa cartella del file NPZ."
        ),
    )
    parser.add_argument(
        "--tracking-missing-policy",
        type=str,
        default="zeros",
        choices=["zeros", "error"],
        help=(
            "Comportamento se una clip non ha feature tracking: "
            "zeros = vettore nullo, error = interrompe il training."
        ),
    )
    parser.add_argument(
        "--no-normalize-tracking-features",
        action="store_true",
        help="Disattiva la normalizzazione z-score delle feature tracking calcolata sul train set.",
    )

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disattiva i pesi di classe nella CrossEntropyLoss.",
    )

    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument(
        "--no-weighted-sampler",
        action="store_true",
        help="Disattiva il WeightedRandomSampler e usa shuffle=True.",
    )

    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)

    return parser.parse_args()


def build_model(args, device, num_classes: int, tracking_dim: int = 0, tracking_config=None):
    input_dim = int(args.input_dim)
    actual_input_dim = input_dim + int(tracking_dim)

    model = TemporalTransformerActionClassifier(
        input_dim=actual_input_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.ff_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        pooling=args.pooling,
        max_len=args.max_len,
        last_mean_ratio=args.last_mean_ratio,
    ).to(device)

    model_config = {
        "model_type": "temporal_transformer",
        "input_dim": actual_input_dim,
        "base_input_dim": input_dim,
        "tracking_input_dim": int(tracking_dim),
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dim_feedforward": args.ff_dim,
        "num_classes": num_classes,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "last_mean_ratio": args.last_mean_ratio,
        "max_len": args.max_len,
        "tracking_config": tracking_config,
    }

    return model, model_config

def run_training(args):
    print("# Comando utilizzato")
    print(get_reconstructed_command())
    print("\n" + "=" * 80 + "\n")

    print("# Configurazione esperimento")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("\n" + "=" * 80 + "\n")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    label_mapping, idx_to_label = build_label_mapping(args.label_mode)
    label_to_idx = get_label_to_idx(idx_to_label)
    num_classes = len(idx_to_label)

    print_label_mode_info(args.label_mode, label_mapping, idx_to_label)

    base_train_dataset = FeatureDataset(args.features_root, split="train")
    base_val_dataset = FeatureDataset(args.features_root, split="val")

    train_dataset = LabelMappedDataset(base_train_dataset, args.label_mode)
    val_dataset = LabelMappedDataset(base_val_dataset, args.label_mode)

    print("\n# Dataset")
    print(f"Train samples originali: {len(base_train_dataset)}")
    print(f"Train samples usati: {len(train_dataset)}")
    print(f"Val samples originali: {len(base_val_dataset)}")
    print(f"Val samples usati: {len(val_dataset)}")

    train_labels, train_counts = get_dataset_labels_and_counts(
        train_dataset,
        num_classes=num_classes,
    )

    print("\n# Distribuzione classi")

    if args.no_class_weights:
        class_weights = None
        criterion = nn.CrossEntropyLoss()
        print_class_stats(train_counts, idx_to_label)
        print("\nWeighted CrossEntropyLoss disattivata.")
    else:
        class_weights_cpu = compute_class_weights_from_counts(
            train_counts,
            power=args.class_weight_power,
        )
        class_weights = class_weights_cpu.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print_class_stats(train_counts, idx_to_label, class_weights_cpu)
        print("\nWeighted CrossEntropyLoss attiva.")

    if args.no_weighted_sampler:
        train_sampler = None
        train_shuffle = True
        sampler_enabled = False
        print("\nWeightedRandomSampler disattivato. Uso shuffle=True.")
    else:
        train_sampler = build_weighted_sampler(
            labels=train_labels,
            counts=train_counts,
            power=args.sampler_power,
            seed=args.seed,
        )
        train_shuffle = False
        sampler_enabled = True
        print("\nWeightedRandomSampler attivato.")

    tracking_dim = 0
    tracking_config = None

    if args.tracking_features_csv is not None and args.tracking_sequences_npz is not None:
        raise ValueError(
            "Usare una sola modalità tracking: --tracking-features-csv "
            "oppure --tracking-sequences-npz, non entrambe."
        )

    if args.tracking_features_csv is not None:
        print("\n# Feature tracking palla/canestro aggregate")
        tracking_store = TrackingFeatureStore(args.tracking_features_csv)

        if args.no_normalize_tracking_features:
            print("Normalizzazione feature tracking disattivata.")
        else:
            tracking_store.fit_normalizer_from_label_dataset(train_dataset)

        tracking_dim = tracking_store.num_features
        tracking_config = tracking_store.get_config()

        train_dataset = TrackingAugmentedDataset(
            train_dataset,
            tracking_store,
            missing_policy=args.tracking_missing_policy,
        )
        val_dataset = TrackingAugmentedDataset(
            val_dataset,
            tracking_store,
            missing_policy=args.tracking_missing_policy,
        )

        print(f"Input dim feature video: {args.input_dim}")
        print(f"Input dim feature tracking aggregate: {tracking_dim}")
        print(f"Input dim totale modello: {args.input_dim + tracking_dim}")

    elif args.tracking_sequences_npz is not None:
        print("\n# Feature tracking palla/canestro temporali")
        tracking_store = TrackingSequenceFeatureStore(
            args.tracking_sequences_npz,
            index_path=args.tracking_sequence_index,
        )

        if args.no_normalize_tracking_features:
            print("Normalizzazione sequenze tracking disattivata.")
        else:
            tracking_store.fit_normalizer_from_label_dataset(train_dataset)

        tracking_dim = tracking_store.num_features
        tracking_config = tracking_store.get_config()

        train_dataset = TemporalTrackingAugmentedDataset(
            train_dataset,
            tracking_store,
            missing_policy=args.tracking_missing_policy,
        )
        val_dataset = TemporalTrackingAugmentedDataset(
            val_dataset,
            tracking_store,
            missing_policy=args.tracking_missing_policy,
        )

        print(f"Input dim feature video: {args.input_dim}")
        print(f"Input dim feature tracking temporali: {tracking_dim}")
        print(f"Input dim totale modello: {args.input_dim + tracking_dim}")
    else:
        print("\n# Feature tracking palla/canestro")
        print("Feature tracking non usate.")

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=data_loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=data_loader_generator,
    )

    model, model_config = build_model(
        args,
        device,
        num_classes=num_classes,
        tracking_dim=tracking_dim,
        tracking_config=tracking_config,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    print("\n# Modello")
    print(model)

    print("\n# Training")
    print(
        f"Scheduler: ReduceLROnPlateau(mode='max', "
        f"factor={args.scheduler_factor}, "
        f"patience={args.scheduler_patience}, "
        f"min_lr={args.scheduler_min_lr})"
    )

    best_macro_f1 = -1.0
    best_weighted_f1 = 0.0
    best_val_loss = None
    best_val_acc = None
    best_epoch = None
    best_val_labels = None
    best_val_preds = None

    output_dir = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        current_lr = get_current_lr(optimizer)

        train_loss, train_acc, train_f1, train_weighted_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_loss, val_acc, val_f1, val_weighted_f1, val_labels, val_preds = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"lr {current_lr:.8f} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"macroF1 {train_f1:.4f} weightedF1 {train_weighted_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} "
            f"macroF1 {val_f1:.4f} weightedF1 {val_weighted_f1:.4f}"
        )

        scheduler.step(val_f1)

        new_lr = get_current_lr(optimizer)
        if new_lr != current_lr:
            print(f"Learning rate aggiornato: {current_lr:.8f} -> {new_lr:.8f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_weighted_f1 = val_weighted_f1
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_labels = val_labels
            best_val_preds = val_preds

            checkpoint_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "best_weighted_f1": best_weighted_f1,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "epoch": best_epoch,
                    "idx_to_label": idx_to_label,
                    "label_to_idx": label_to_idx,
                    "label_mode": args.label_mode,
                    "label_mapping": label_mapping,
                    "original_idx_to_label": get_original_idx_to_label(),
                    "model_config": model_config,
                    "tracking_config": tracking_config,
                    "training_config": vars(args),
                    "class_weights": class_weights.detach().cpu()
                    if class_weights is not None
                    else None,
                    "weighted_sampler": sampler_enabled,
                    "scheduler": {
                        "name": "ReduceLROnPlateau",
                        "mode": "max",
                        "factor": args.scheduler_factor,
                        "patience": args.scheduler_patience,
                        "min_lr": args.scheduler_min_lr,
                    },
                    "seed": args.seed,
                    "command_reconstructed": get_reconstructed_command(),
                    "argv": sys.argv,
                },
                checkpoint_path,
            )

            print(f"Salvato nuovo best model: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("\nValutazione finale su validation usando il miglior modello salvato:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best val macro-F1: {best_macro_f1:.4f}")
    print(f"Best val weighted-F1: {best_weighted_f1:.4f}")

    labels = list(range(num_classes))
    target_names = [idx_to_label[i] for i in labels]

    print(f"\nClassification report - {num_classes} classi ({args.label_mode}):")
    print(
        classification_report(
            best_val_labels,
            best_val_preds,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    print(f"Confusion matrix - {num_classes} classi ({args.label_mode}):")
    print(
        confusion_matrix(
            best_val_labels,
            best_val_preds,
            labels=labels,
        )
    )

    if args.label_mode == "original":
        print("\nClassification report - solo 7 azioni reali:")
        print(
            classification_report(
                best_val_labels,
                best_val_preds,
                labels=list(range(7)),
                target_names=[IDX_TO_LABEL[i] for i in range(7)],
                zero_division=0,
            )
        )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.txt"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(results_path, "w", encoding="utf-8") as results_file:
        sys.stdout = Tee(original_stdout, results_file)
        sys.stderr = Tee(original_stderr, results_file)

        try:
            print(f"File results.txt: {results_path}")
            print()
            run_training(args)

        except Exception:
            print("\nERRORE DURANTE L'ESECUZIONE:", file=sys.stderr)
            traceback.print_exc()
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Risultati salvati in: {results_path}")


if __name__ == "__main__":
    main()
