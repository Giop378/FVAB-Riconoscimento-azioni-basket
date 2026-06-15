from pathlib import Path
import argparse
import csv
import json
import shlex
import sys
import traceback
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.data.feature_dataset import FeatureDataset, collate_features, IDX_TO_LABEL
from src.models.temporal_transformer_classifier import TemporalTransformerActionClassifier


FINAL_LABELS = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
    "no-action",
]

FINAL_ACTION_LABELS = FINAL_LABELS[:-1]

FINAL_TYPE_LABELS = [
    "passaggio",
    "tiroDaDue",
    "tiroDaTre",
    "tiroLibero",
    "no-action",
]


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


class TrackingFeatureStore:
    def __init__(self, csv_path: str, feature_names=None, mean=None, std=None, normalized=False):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV feature tracking non trovato: {self.csv_path}")

        self.feature_names = feature_names
        self.rows_by_key = {}
        self.normalized = bool(normalized)
        self.mean = None if mean is None else np.array(mean, dtype=np.float32)
        self.std = None if std is None else np.array(std, dtype=np.float32)

        self._load()

        if self.mean is None:
            self.mean = np.zeros(self.num_features, dtype=np.float32)
        if self.std is None:
            self.std = np.ones(self.num_features, dtype=np.float32)

        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)

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
                vector = np.array(
                    [float(row[name]) for name in self.feature_names],
                    dtype=np.float32,
                )
                self.rows_by_key[key] = vector

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def has(self, path_value) -> bool:
        return normalize_clip_key(path_value) in self.rows_by_key

    def get(self, path_value, missing_policy="zeros"):
        key = normalize_clip_key(path_value)

        if key in self.rows_by_key:
            vector = self.rows_by_key[key].copy()
        elif missing_policy == "zeros":
            vector = np.zeros(self.num_features, dtype=np.float32)
        else:
            raise KeyError(
                f"Feature tracking non trovate per path='{path_value}' "
                f"con chiave normalizzata='{key}'."
            )

        if self.normalized:
            vector = (vector - self.mean) / self.std

        return vector.astype(np.float32)


def interpolate_sequence_array(sequence: np.ndarray, target_len: int) -> np.ndarray:
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


def append_tracking_to_features(features: torch.Tensor, tracking_features: torch.Tensor) -> torch.Tensor:
    if tracking_features is None:
        return features

    if tracking_features.ndim != 2:
        raise ValueError(
            f"tracking_features deve avere forma [B, K], "
            f"ricevuta {tuple(tracking_features.shape)}."
        )

    batch_size, seq_len, _ = features.shape

    if tracking_features.shape[0] != batch_size:
        raise ValueError(
            f"Batch size tracking non coerente: features B={batch_size}, "
            f"tracking B={tracking_features.shape[0]}."
        )

    tracking_sequence = tracking_features.unsqueeze(1).repeat(1, seq_len, 1)
    return torch.cat([features, tracking_sequence.to(features.device, dtype=features.dtype)], dim=2)


def append_tracking_sequence_to_features(features: torch.Tensor, tracking_sequences: torch.Tensor) -> torch.Tensor:
    if tracking_sequences is None:
        return features

    if tracking_sequences.ndim != 3:
        raise ValueError(
            f"tracking_sequences deve avere forma [B, T, K], "
            f"ricevuta {tuple(tracking_sequences.shape)}."
        )

    if tracking_sequences.shape[:2] != features.shape[:2]:
        raise ValueError(
            f"Shape tracking temporale non coerente: features {tuple(features.shape)}, "
            f"tracking {tuple(tracking_sequences.shape)}."
        )

    return torch.cat([features, tracking_sequences.to(features.device, dtype=features.dtype)], dim=2)


def append_level_tracking(
    features: torch.Tensor,
    tracking_features: torch.Tensor = None,
    tracking_sequences: torch.Tensor = None,
) -> torch.Tensor:
    if tracking_features is not None and tracking_sequences is not None:
        raise ValueError("tracking_features e tracking_sequences sono mutuamente esclusivi.")

    if tracking_sequences is not None:
        return append_tracking_sequence_to_features(features, tracking_sequences)

    if tracking_features is not None:
        return append_tracking_to_features(features, tracking_features)

    return features


def normalize_idx_to_label(idx_to_label):
    if isinstance(idx_to_label, dict):
        return {int(k): str(v) for k, v in idx_to_label.items()}

    if isinstance(idx_to_label, (list, tuple)):
        return {idx: str(label) for idx, label in enumerate(idx_to_label)}

    raise TypeError(f"Formato idx_to_label non supportato: {type(idx_to_label)}")


def fallback_idx_to_label(label_mode: str):
    mappings = {
        "action_noaction": {
            0: "passaggio",
            1: "tiro",
            2: "no-action",
        },
        "shot_type_only": {
            0: "tiroDaDue",
            1: "tiroDaTre",
            2: "tiroLibero",
        },
        "shot_outcome_only": {
            0: "tiro0",
            1: "tiro1",
        },
        "passaggio_noaction_only": {
            0: "passaggio",
            1: "no-action",
        },
    }

    if label_mode not in mappings:
        raise ValueError(f"Label mode non gestita per fallback: {label_mode}")

    return mappings[label_mode]


def original_idx_to_label():
    return {idx: IDX_TO_LABEL[idx] for idx in range(len(IDX_TO_LABEL))}


def original_to_final_label(original_label: str) -> str:
    if original_label in {"idle", "non-gioco"}:
        return "no-action"

    if original_label in FINAL_ACTION_LABELS:
        return original_label

    raise ValueError(f"Label originale non riconosciuta: {original_label}")


def final_to_type_label(final_label: str) -> str:
    if final_label in {"passaggio", "no-action"}:
        return final_label

    if final_label.startswith("tiroDaDue"):
        return "tiroDaDue"

    if final_label.startswith("tiroDaTre"):
        return "tiroDaTre"

    if final_label.startswith("tiroLibero"):
        return "tiroLibero"

    raise ValueError(f"Label finale non riconosciuta: {final_label}")


def get_sample_path(dataset, idx: int) -> str:
    if hasattr(dataset, "items"):
        item = dataset.items[idx]

        if isinstance(item, dict):
            if "path" in item:
                return str(item["path"])
            return str(item)

        return str(item)

    return ""


def load_checkpoint_model(checkpoint_path: str, device: torch.device, label_mode: str):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_config" not in checkpoint:
        raise KeyError(
            f"Il checkpoint {checkpoint_path} non contiene 'model_config'. "
            "Serve un checkpoint salvato da train.py."
        )

    config = checkpoint["model_config"]
    num_classes = int(config["num_classes"])

    idx_to_label = checkpoint.get("idx_to_label")
    if idx_to_label is None:
        idx_to_label = fallback_idx_to_label(label_mode)
    else:
        idx_to_label = normalize_idx_to_label(idx_to_label)

    if len(idx_to_label) != num_classes:
        idx_to_label = fallback_idx_to_label(label_mode)

    model = TemporalTransformerActionClassifier(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config.get("dim_feedforward", config.get("ff_dim"))),
        num_classes=num_classes,
        dropout=float(config["dropout"]),
        pooling=str(config["pooling"]),
        max_len=int(config.get("max_len", 1024)),
        last_mean_ratio=float(config.get("last_mean_ratio", 0.30)),
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model, idx_to_label, checkpoint, config


def get_checkpoint_tracking_requirements(checkpoint, config):
    tracking_config = checkpoint.get("tracking_config") or config.get("tracking_config")
    tracking_input_dim = int(config.get("tracking_input_dim", 0))

    if tracking_config:
        tracking_type = tracking_config.get("type", "aggregate")
    elif tracking_input_dim > 0:
        tracking_type = "aggregate"
    else:
        tracking_type = "none"

    return tracking_config, tracking_type, tracking_input_dim


def load_tracking_store_for_level(
    level_name: str,
    checkpoint,
    config,
    tracking_features_csv: str = None,
    tracking_sequences_npz: str = None,
    tracking_sequence_index: str = None,
    missing_policy: str = "zeros",
):
    if tracking_features_csv is not None and tracking_sequences_npz is not None:
        raise ValueError(
            f"Usare una sola sorgente tracking per {level_name}: "
            f"feature aggregate oppure sequenze temporali, non entrambe."
        )

    tracking_config, tracking_type, tracking_input_dim = get_checkpoint_tracking_requirements(
        checkpoint=checkpoint,
        config=config,
    )

    if tracking_type == "none" or tracking_input_dim <= 0:
        print(f"\n# Feature tracking per {level_name}")
        print("Non richieste dal checkpoint.")
        return "none", None

    if tracking_type == "aggregate":
        csv_path = tracking_features_csv
        if csv_path is None and tracking_config:
            csv_path = tracking_config.get("csv_path")

        if csv_path is None:
            raise ValueError(
                f"Il checkpoint {level_name} richiede feature tracking aggregate, "
                f"ma non è stato passato il relativo CSV."
            )

        feature_names = tracking_config.get("feature_names") if tracking_config else None
        mean = tracking_config.get("mean") if tracking_config else None
        std = tracking_config.get("std") if tracking_config else None
        normalized = bool(tracking_config.get("normalized", False)) if tracking_config else False

        store = TrackingFeatureStore(
            csv_path,
            feature_names=feature_names,
            mean=mean,
            std=std,
            normalized=normalized,
        )

        print(f"\n# Feature tracking aggregate per {level_name}")
        print(f"CSV tracking: {csv_path}")
        print(f"Numero feature tracking: {store.num_features}")
        print(f"Normalizzate con statistiche del checkpoint {level_name}: {normalized}")
        print(f"Missing policy: {missing_policy}")
        return "aggregate", store

    if tracking_type == "temporal_sequence":
        npz_path = tracking_sequences_npz
        index_path = tracking_sequence_index

        if tracking_config:
            npz_path = npz_path or tracking_config.get("npz_path")
            index_path = index_path or tracking_config.get("index_path")

        if npz_path is None:
            raise ValueError(
                f"Il checkpoint {level_name} richiede sequenze tracking temporali, "
                f"ma non è stato passato il file NPZ."
            )

        feature_names = tracking_config.get("feature_names") if tracking_config else None
        mean = tracking_config.get("mean") if tracking_config else None
        std = tracking_config.get("std") if tracking_config else None
        normalized = bool(tracking_config.get("normalized", False)) if tracking_config else False

        store = TrackingSequenceFeatureStore(
            npz_path,
            index_path=index_path,
            feature_names=feature_names,
            mean=mean,
            std=std,
            normalized=normalized,
        )

        print(f"\n# Feature tracking temporali per {level_name}")
        print(f"NPZ tracking: {npz_path}")
        print(f"Indice tracking: {store.index_path}")
        print(f"Numero feature tracking per frame: {store.num_features}")
        print(f"Normalizzate con statistiche del checkpoint {level_name}: {normalized}")
        print(f"Missing policy: {missing_policy}")
        return "temporal_sequence", store

    raise ValueError(f"Tipo tracking non supportato per {level_name}: {tracking_type}.")


def build_tracking_batch(
    tracking_type: str,
    tracking_store,
    dataset,
    sample_offset: int,
    labels,
    features: torch.Tensor,
    lengths: torch.Tensor,
    missing_policy: str,
    device: torch.device,
):
    batch_size = len(labels)
    tracking_available = [False] * batch_size

    if tracking_store is None or tracking_type == "none":
        return None, None, tracking_available

    if tracking_type == "aggregate":
        tracking_vectors = []

        for i in range(batch_size):
            global_idx = sample_offset + i
            sample_path = get_sample_path(dataset, global_idx)
            tracking_available[i] = tracking_store.has(sample_path)
            tracking_vectors.append(
                tracking_store.get(
                    sample_path,
                    missing_policy=missing_policy,
                )
            )

        tracking_features = torch.tensor(
            np.stack(tracking_vectors, axis=0),
            dtype=features.dtype,
            device=device,
        )
        return tracking_features, None, tracking_available

    if tracking_type == "temporal_sequence":
        max_seq_len = features.shape[1]
        tracking_sequence_vectors = []

        for i in range(batch_size):
            global_idx = sample_offset + i
            sample_path = get_sample_path(dataset, global_idx)
            tracking_available[i] = tracking_store.has(sample_path)

            real_len = int(lengths[i].item())
            sequence = tracking_store.get(
                sample_path,
                target_len=real_len,
                missing_policy=missing_policy,
            )

            padded = np.zeros((max_seq_len, tracking_store.num_features), dtype=np.float32)
            padded[:real_len] = sequence[:real_len]
            tracking_sequence_vectors.append(padded)

        tracking_sequences = torch.tensor(
            np.stack(tracking_sequence_vectors, axis=0),
            dtype=features.dtype,
            device=device,
        )
        return None, tracking_sequences, tracking_available

    raise ValueError(f"Tipo tracking non supportato nel batch: {tracking_type}.")


@torch.no_grad()
def predict_hierarchical_batch(
    features,
    lengths,
    model_l1,
    model_l2,
    model_l3,
    idx_to_label_l1,
    idx_to_label_l2,
    idx_to_label_l3,
    l1_tracking_features=None,
    l1_tracking_sequences=None,
    l2_tracking_features=None,
    l2_tracking_sequences=None,
    l3_tracking_features=None,
    l3_tracking_sequences=None,
    model_l1_binary_corrector=None,
    idx_to_label_l1_binary_corrector=None,
    l1_binary_corrector_tracking_features=None,
    l1_binary_corrector_tracking_sequences=None,
):
    features_l1 = append_level_tracking(
        features,
        tracking_features=l1_tracking_features,
        tracking_sequences=l1_tracking_sequences,
    )

    logits_l1 = model_l1(features_l1, lengths)
    probs_l1 = torch.softmax(logits_l1, dim=1)
    preds_l1 = probs_l1.argmax(dim=1)

    batch_size = features.size(0)

    pred_l1_base_labels = [idx_to_label_l1[int(idx)] for idx in preds_l1.cpu().tolist()]
    pred_l1_labels = list(pred_l1_base_labels)
    pred_l1_corrector_labels = [""] * batch_size
    corrector_used = [False] * batch_size

    pred_l2_labels = [""] * batch_size
    pred_l3_labels = [""] * batch_size
    final_preds = [""] * batch_size

    p_l1 = probs_l1.max(dim=1).values.detach().cpu().tolist()
    p_l1_corrector = [None] * batch_size
    p_l2 = [None] * batch_size
    p_l3 = [None] * batch_size

    non_shot_indices = [
        idx for idx, label in enumerate(pred_l1_base_labels)
        if label in {"passaggio", "no-action"}
    ]

    if model_l1_binary_corrector is not None and non_shot_indices:
        if idx_to_label_l1_binary_corrector is None:
            raise ValueError("idx_to_label_l1_binary_corrector mancante.")

        non_shot_indices_tensor = torch.tensor(
            non_shot_indices,
            dtype=torch.long,
            device=features.device,
        )

        corrector_features = features.index_select(0, non_shot_indices_tensor)
        corrector_lengths = lengths.index_select(0, non_shot_indices_tensor)

        corrector_tracking_features = None
        corrector_tracking_sequences = None

        if l1_binary_corrector_tracking_features is not None:
            corrector_tracking_features = l1_binary_corrector_tracking_features.index_select(
                0,
                non_shot_indices_tensor,
            )

        if l1_binary_corrector_tracking_sequences is not None:
            corrector_tracking_sequences = l1_binary_corrector_tracking_sequences.index_select(
                0,
                non_shot_indices_tensor,
            )

        corrector_features = append_level_tracking(
            corrector_features,
            tracking_features=corrector_tracking_features,
            tracking_sequences=corrector_tracking_sequences,
        )

        logits_corrector = model_l1_binary_corrector(corrector_features, corrector_lengths)
        probs_corrector = torch.softmax(logits_corrector, dim=1)
        preds_corrector = probs_corrector.argmax(dim=1)

        corrector_labels = [
            idx_to_label_l1_binary_corrector[int(idx)]
            for idx in preds_corrector.cpu().tolist()
        ]
        corrector_probs = probs_corrector.max(dim=1).values.detach().cpu().tolist()

        for local_idx, global_idx in enumerate(non_shot_indices):
            corrected_label = corrector_labels[local_idx]

            if corrected_label not in {"passaggio", "no-action"}:
                raise ValueError(
                    f"Il correttore binario deve predire solo passaggio/no-action, "
                    f"ma ha predetto: {corrected_label}"
                )

            pred_l1_labels[global_idx] = corrected_label
            pred_l1_corrector_labels[global_idx] = corrected_label
            p_l1_corrector[global_idx] = corrector_probs[local_idx]
            corrector_used[global_idx] = True

    shot_indices = [
        idx for idx, label in enumerate(pred_l1_base_labels)
        if label == "tiro"
    ]

    for idx, label_l1 in enumerate(pred_l1_labels):
        if label_l1 == "passaggio":
            final_preds[idx] = "passaggio"
        elif label_l1 == "no-action":
            final_preds[idx] = "no-action"
        elif pred_l1_base_labels[idx] == "tiro":
            final_preds[idx] = "__pending_shot__"
        else:
            raise ValueError(f"Predizione L1 non riconosciuta: {label_l1}")

    if shot_indices:
        shot_indices_tensor = torch.tensor(
            shot_indices,
            dtype=torch.long,
            device=features.device,
        )

        shot_features = features.index_select(0, shot_indices_tensor)
        shot_lengths = lengths.index_select(0, shot_indices_tensor)

        shot_l2_tracking_features = None
        shot_l2_tracking_sequences = None

        if l2_tracking_features is not None:
            shot_l2_tracking_features = l2_tracking_features.index_select(0, shot_indices_tensor)
        if l2_tracking_sequences is not None:
            shot_l2_tracking_sequences = l2_tracking_sequences.index_select(0, shot_indices_tensor)

        shot_features_l2 = append_level_tracking(
            shot_features,
            tracking_features=shot_l2_tracking_features,
            tracking_sequences=shot_l2_tracking_sequences,
        )

        logits_l2 = model_l2(shot_features_l2, shot_lengths)
        probs_l2 = torch.softmax(logits_l2, dim=1)
        preds_l2 = probs_l2.argmax(dim=1)

        shot_l3_tracking_features = None
        shot_l3_tracking_sequences = None

        if l3_tracking_features is not None:
            shot_l3_tracking_features = l3_tracking_features.index_select(0, shot_indices_tensor)
        if l3_tracking_sequences is not None:
            shot_l3_tracking_sequences = l3_tracking_sequences.index_select(0, shot_indices_tensor)

        shot_features_l3 = append_level_tracking(
            shot_features,
            tracking_features=shot_l3_tracking_features,
            tracking_sequences=shot_l3_tracking_sequences,
        )

        logits_l3 = model_l3(shot_features_l3, shot_lengths)
        probs_l3 = torch.softmax(logits_l3, dim=1)
        preds_l3 = probs_l3.argmax(dim=1)

        pred_l2_shots = [idx_to_label_l2[int(idx)] for idx in preds_l2.cpu().tolist()]
        pred_l3_shots = [idx_to_label_l3[int(idx)] for idx in preds_l3.cpu().tolist()]

        p_l2_shots = probs_l2.max(dim=1).values.detach().cpu().tolist()
        p_l3_shots = probs_l3.max(dim=1).values.detach().cpu().tolist()

        for local_idx, global_idx in enumerate(shot_indices):
            shot_type = pred_l2_shots[local_idx]
            shot_outcome = pred_l3_shots[local_idx]

            if shot_outcome == "tiro0":
                suffix = "0"
            elif shot_outcome == "tiro1":
                suffix = "1"
            else:
                raise ValueError(f"Predizione L3 non riconosciuta: {shot_outcome}")

            final_preds[global_idx] = f"{shot_type}{suffix}"
            pred_l2_labels[global_idx] = shot_type
            pred_l3_labels[global_idx] = shot_outcome
            p_l2[global_idx] = p_l2_shots[local_idx]
            p_l3[global_idx] = p_l3_shots[local_idx]

    return (
        final_preds,
        pred_l1_labels,
        pred_l1_base_labels,
        pred_l1_corrector_labels,
        corrector_used,
        pred_l2_labels,
        pred_l3_labels,
        p_l1,
        p_l1_corrector,
        p_l2,
        p_l3,
    )


def print_report(title: str, y_true, y_pred, labels):
    print(f"\n{title}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            zero_division=0,
        )
    )

    print(f"Confusion matrix - {title}:")
    print(confusion_matrix(y_true, y_pred, labels=labels))


def run_evaluation(args):
    print("# Comando utilizzato")
    print(get_reconstructed_command())
    print("\n" + "=" * 80 + "\n")

    print("# Configurazione esperimento")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("\n" + "=" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    dataset = FeatureDataset(args.features_root, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Split: {args.split}")
    print(f"Numero campioni: {len(dataset)}")

    model_l1, idx_to_label_l1, ckpt_l1, config_l1 = load_checkpoint_model(
        args.l1_checkpoint,
        device,
        label_mode="action_noaction",
    )
    model_l2, idx_to_label_l2, ckpt_l2, config_l2 = load_checkpoint_model(
        args.l2_checkpoint,
        device,
        label_mode="shot_type_only",
    )
    model_l3, idx_to_label_l3, ckpt_l3, config_l3 = load_checkpoint_model(
        args.l3_checkpoint,
        device,
        label_mode="shot_outcome_only",
    )

    model_l1_binary_corrector = None
    idx_to_label_l1_binary_corrector = None
    ckpt_l1_binary_corrector = None
    config_l1_binary_corrector = None

    if args.l1_binary_corrector_checkpoint is not None:
        (
            model_l1_binary_corrector,
            idx_to_label_l1_binary_corrector,
            ckpt_l1_binary_corrector,
            config_l1_binary_corrector,
        ) = load_checkpoint_model(
            args.l1_binary_corrector_checkpoint,
            device,
            label_mode="passaggio_noaction_only",
        )

    print("\n# Modelli caricati")
    print(f"L1 checkpoint: {args.l1_checkpoint}")
    print(f"L1 idx_to_label: {idx_to_label_l1}")
    print(f"L2 checkpoint: {args.l2_checkpoint}")
    print(f"L2 idx_to_label: {idx_to_label_l2}")
    print(f"L3 checkpoint: {args.l3_checkpoint}")
    print(f"L3 idx_to_label: {idx_to_label_l3}")

    if model_l1_binary_corrector is not None:
        print(f"L1 binary corrector checkpoint: {args.l1_binary_corrector_checkpoint}")
        print(f"L1 binary corrector idx_to_label: {idx_to_label_l1_binary_corrector}")
    else:
        print("L1 binary corrector: non usato")

    l3_tracking_features_csv = args.l3_tracking_features_csv or args.tracking_features_csv
    l3_tracking_sequences_npz = args.l3_tracking_sequences_npz or args.tracking_sequences_npz
    l3_tracking_sequence_index = args.l3_tracking_sequence_index or args.tracking_sequence_index

    l1_tracking_type, l1_tracking_store = load_tracking_store_for_level(
        level_name="L1",
        checkpoint=ckpt_l1,
        config=config_l1,
        tracking_features_csv=args.l1_tracking_features_csv,
        tracking_sequences_npz=args.l1_tracking_sequences_npz,
        tracking_sequence_index=args.l1_tracking_sequence_index,
        missing_policy=args.tracking_missing_policy,
    )
    l2_tracking_type, l2_tracking_store = load_tracking_store_for_level(
        level_name="L2",
        checkpoint=ckpt_l2,
        config=config_l2,
        tracking_features_csv=args.l2_tracking_features_csv,
        tracking_sequences_npz=args.l2_tracking_sequences_npz,
        tracking_sequence_index=args.l2_tracking_sequence_index,
        missing_policy=args.tracking_missing_policy,
    )
    l3_tracking_type, l3_tracking_store = load_tracking_store_for_level(
        level_name="L3",
        checkpoint=ckpt_l3,
        config=config_l3,
        tracking_features_csv=l3_tracking_features_csv,
        tracking_sequences_npz=l3_tracking_sequences_npz,
        tracking_sequence_index=l3_tracking_sequence_index,
        missing_policy=args.tracking_missing_policy,
    )

    l1_binary_corrector_tracking_type = "none"
    l1_binary_corrector_tracking_store = None

    if model_l1_binary_corrector is not None:
        l1_binary_corrector_tracking_type, l1_binary_corrector_tracking_store = load_tracking_store_for_level(
            level_name="L1 binary corrector",
            checkpoint=ckpt_l1_binary_corrector,
            config=config_l1_binary_corrector,
            tracking_features_csv=args.l1_binary_corrector_tracking_features_csv,
            tracking_sequences_npz=args.l1_binary_corrector_tracking_sequences_npz,
            tracking_sequence_index=args.l1_binary_corrector_tracking_sequence_index,
            missing_policy=args.tracking_missing_policy,
        )

    original_mapping = original_idx_to_label()

    y_true_final = []
    y_pred_final = []
    rows = []

    corrector_total_used = 0
    corrector_changed = 0
    corrector_passaggio_to_noaction = 0
    corrector_noaction_to_passaggio = 0

    sample_offset = 0

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].cpu().tolist()

        (
            l1_tracking_features,
            l1_tracking_sequences,
            l1_tracking_available,
        ) = build_tracking_batch(
            tracking_type=l1_tracking_type,
            tracking_store=l1_tracking_store,
            dataset=dataset,
            sample_offset=sample_offset,
            labels=labels,
            features=features,
            lengths=lengths,
            missing_policy=args.tracking_missing_policy,
            device=device,
        )

        (
            l2_tracking_features,
            l2_tracking_sequences,
            l2_tracking_available,
        ) = build_tracking_batch(
            tracking_type=l2_tracking_type,
            tracking_store=l2_tracking_store,
            dataset=dataset,
            sample_offset=sample_offset,
            labels=labels,
            features=features,
            lengths=lengths,
            missing_policy=args.tracking_missing_policy,
            device=device,
        )

        (
            l3_tracking_features,
            l3_tracking_sequences,
            l3_tracking_available,
        ) = build_tracking_batch(
            tracking_type=l3_tracking_type,
            tracking_store=l3_tracking_store,
            dataset=dataset,
            sample_offset=sample_offset,
            labels=labels,
            features=features,
            lengths=lengths,
            missing_policy=args.tracking_missing_policy,
            device=device,
        )

        (
            l1_binary_corrector_tracking_features,
            l1_binary_corrector_tracking_sequences,
            l1_binary_corrector_tracking_available,
        ) = build_tracking_batch(
            tracking_type=l1_binary_corrector_tracking_type,
            tracking_store=l1_binary_corrector_tracking_store,
            dataset=dataset,
            sample_offset=sample_offset,
            labels=labels,
            features=features,
            lengths=lengths,
            missing_policy=args.tracking_missing_policy,
            device=device,
        )

        (
            final_preds,
            pred_l1_labels,
            pred_l1_base_labels,
            pred_l1_corrector_labels,
            batch_corrector_used,
            pred_l2_labels,
            pred_l3_labels,
            p_l1,
            p_l1_corrector,
            p_l2,
            p_l3,
        ) = predict_hierarchical_batch(
            features=features,
            lengths=lengths,
            model_l1=model_l1,
            model_l2=model_l2,
            model_l3=model_l3,
            idx_to_label_l1=idx_to_label_l1,
            idx_to_label_l2=idx_to_label_l2,
            idx_to_label_l3=idx_to_label_l3,
            l1_tracking_features=l1_tracking_features,
            l1_tracking_sequences=l1_tracking_sequences,
            l2_tracking_features=l2_tracking_features,
            l2_tracking_sequences=l2_tracking_sequences,
            l3_tracking_features=l3_tracking_features,
            l3_tracking_sequences=l3_tracking_sequences,
            model_l1_binary_corrector=model_l1_binary_corrector,
            idx_to_label_l1_binary_corrector=idx_to_label_l1_binary_corrector,
            l1_binary_corrector_tracking_features=l1_binary_corrector_tracking_features,
            l1_binary_corrector_tracking_sequences=l1_binary_corrector_tracking_sequences,
        )

        for i, original_idx in enumerate(labels):
            global_idx = sample_offset + i
            original_label = original_mapping[int(original_idx)]
            true_final = original_to_final_label(original_label)
            pred_final = final_preds[i]

            if batch_corrector_used[i]:
                corrector_total_used += 1

                if pred_l1_base_labels[i] != pred_l1_labels[i]:
                    corrector_changed += 1

                    if pred_l1_base_labels[i] == "passaggio" and pred_l1_labels[i] == "no-action":
                        corrector_passaggio_to_noaction += 1
                    elif pred_l1_base_labels[i] == "no-action" and pred_l1_labels[i] == "passaggio":
                        corrector_noaction_to_passaggio += 1

            y_true_final.append(true_final)
            y_pred_final.append(pred_final)

            rows.append(
                {
                    "sample_idx": global_idx,
                    "path": get_sample_path(dataset, global_idx),
                    "original_label": original_label,
                    "true_final": true_final,
                    "pred_l1": pred_l1_labels[i],
                    "pred_l1_base": pred_l1_base_labels[i],
                    "pred_l1_corrector": pred_l1_corrector_labels[i],
                    "pred_l1_after_corrector": pred_l1_labels[i],
                    "corrector_used": int(batch_corrector_used[i]),
                    "pred_l2": pred_l2_labels[i],
                    "pred_l3": pred_l3_labels[i],
                    "pred_final": pred_final,
                    "p_l1": f"{p_l1[i]:.6f}",
                    "p_l1_corrector": "" if p_l1_corrector[i] is None else f"{p_l1_corrector[i]:.6f}",
                    "p_l2": "" if p_l2[i] is None else f"{p_l2[i]:.6f}",
                    "p_l3": "" if p_l3[i] is None else f"{p_l3[i]:.6f}",
                    "tracking_used_l1": int(l1_tracking_store is not None),
                    "tracking_available_l1": int(l1_tracking_available[i]),
                    "tracking_used_l1_binary_corrector": int(l1_binary_corrector_tracking_store is not None),
                    "tracking_available_l1_binary_corrector": int(l1_binary_corrector_tracking_available[i]),
                    "tracking_used_l2": int(l2_tracking_store is not None),
                    "tracking_available_l2": int(l2_tracking_available[i]),
                    "tracking_used_l3": int(l3_tracking_store is not None),
                    "tracking_available_l3": int(l3_tracking_available[i]),
                    "correct": int(true_final == pred_final),
                }
            )

        sample_offset += len(labels)

    if model_l1_binary_corrector is not None:
        print("\n" + "=" * 80)
        print("\n# Correttore binario L1 passaggio/no-action")
        print(f"Campioni passati al correttore: {corrector_total_used}")
        print(f"Predizioni modificate dal correttore: {corrector_changed}")
        print(f"Conversioni passaggio -> no-action: {corrector_passaggio_to_noaction}")
        print(f"Conversioni no-action -> passaggio: {corrector_noaction_to_passaggio}")

    acc = accuracy_score(y_true_final, y_pred_final)
    macro_f1 = f1_score(
        y_true_final,
        y_pred_final,
        labels=FINAL_LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true_final,
        y_pred_final,
        labels=FINAL_LABELS,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("\n# Valutazione gerarchica end-to-end")
    print(f"Accuracy 8 classi: {acc:.4f}")
    print(f"Macro F1 8 classi: {macro_f1:.4f}")
    print(f"Weighted F1 8 classi: {weighted_f1:.4f}")

    print_report(
        title="Classification report - 8 classi finali con no-action",
        y_true=y_true_final,
        y_pred=y_pred_final,
        labels=FINAL_LABELS,
    )

    print_report(
        title="Classification report - solo 7 azioni finali",
        y_true=y_true_final,
        y_pred=y_pred_final,
        labels=FINAL_ACTION_LABELS,
    )

    y_true_type = [final_to_type_label(label) for label in y_true_final]
    y_pred_type = [final_to_type_label(label) for label in y_pred_final]

    type_acc = accuracy_score(y_true_type, y_pred_type)
    type_macro_f1 = f1_score(
        y_true_type,
        y_pred_type,
        labels=FINAL_TYPE_LABELS,
        average="macro",
        zero_division=0,
    )
    type_weighted_f1 = f1_score(
        y_true_type,
        y_pred_type,
        labels=FINAL_TYPE_LABELS,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("\n# Valutazione collassata senza esito del tiro")
    print(f"Accuracy tipo azione: {type_acc:.4f}")
    print(f"Macro F1 tipo azione: {type_macro_f1:.4f}")
    print(f"Weighted F1 tipo azione: {type_weighted_f1:.4f}")

    print_report(
        title="Classification report - tipo azione senza esito",
        y_true=y_true_type,
        y_pred=y_pred_type,
        labels=FINAL_TYPE_LABELS,
    )

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "predictions.csv"

    with open(predictions_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPredizioni salvate in: {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs/exp_31_hierarchical_end_to_end")

    parser.add_argument("--l1-checkpoint", type=str, required=True)
    parser.add_argument("--l2-checkpoint", type=str, required=True)
    parser.add_argument("--l3-checkpoint", type=str, required=True)

    parser.add_argument(
        "--tracking-features-csv",
        type=str,
        default=None,
        help=(
            "[Legacy/L3] CSV prodotto da extract_ball_rim_tracking_features.py. "
            "Usato come fallback per L3 se --l3-tracking-features-csv non è specificato."
        ),
    )
    parser.add_argument(
        "--tracking-sequences-npz",
        type=str,
        default=None,
        help=(
            "[Legacy/L3] File NPZ con sequenze temporali di tracking palla/canestro. "
            "Usato come fallback per L3 se --l3-tracking-sequences-npz non è specificato."
        ),
    )
    parser.add_argument(
        "--tracking-sequence-index",
        type=str,
        default=None,
        help=(
            "[Legacy/L3] JSON indice associato a --tracking-sequences-npz. "
            "Usato come fallback per L3 se --l3-tracking-sequence-index non è specificato."
        ),
    )

    for level in ["l1", "l2", "l3"]:
        parser.add_argument(
            f"--{level}-tracking-features-csv",
            type=str,
            default=None,
            help=(
                f"CSV con feature tracking aggregate da usare per {level.upper()} "
                "se il checkpoint di quel livello è stato addestrato con tracking aggregato."
            ),
        )
        parser.add_argument(
            f"--{level}-tracking-sequences-npz",
            type=str,
            default=None,
            help=(
                f"NPZ con sequenze tracking temporali da usare per {level.upper()} "
                "se il checkpoint di quel livello è stato addestrato con tracking temporale."
            ),
        )
        parser.add_argument(
            f"--{level}-tracking-sequence-index",
            type=str,
            default=None,
            help=(
                f"JSON indice associato al file NPZ tracking temporale per {level.upper()}. "
                "Default: tracking_sequence_index.json nella stessa cartella del file NPZ."
            ),
        )

    parser.add_argument(
        "--l1-binary-corrector-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint opzionale del correttore binario L1 passaggio/no-action. "
            "Se specificato, viene applicato solo ai campioni che L1 predice come "
            "passaggio o no-action; i campioni predetti come tiro non vengono modificati."
        ),
    )
    parser.add_argument(
        "--l1-binary-corrector-tracking-features-csv",
        type=str,
        default=None,
        help=(
            "CSV con feature tracking aggregate per il correttore binario L1, "
            "se il checkpoint è stato addestrato con tracking aggregato."
        ),
    )
    parser.add_argument(
        "--l1-binary-corrector-tracking-sequences-npz",
        type=str,
        default=None,
        help=(
            "NPZ con sequenze tracking temporali per il correttore binario L1, "
            "se il checkpoint è stato addestrato con tracking temporale."
        ),
    )
    parser.add_argument(
        "--l1-binary-corrector-tracking-sequence-index",
        type=str,
        default=None,
        help=(
            "JSON indice associato al file NPZ del correttore binario L1. "
            "Default: tracking_sequence_index.json nella stessa cartella del file NPZ."
        ),
    )

    parser.add_argument(
        "--tracking-missing-policy",
        type=str,
        default="zeros",
        choices=["zeros", "error"],
        help="Comportamento se una clip non ha feature tracking associate.",
    )

    parser.add_argument("--cpu", action="store_true", help="Forza l'esecuzione su CPU.")

    return parser.parse_args()


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
            run_evaluation(args)

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