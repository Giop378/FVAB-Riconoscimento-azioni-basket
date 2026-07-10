"""
Utility di input/output per l'estrazione del tracking palla/canestro.

Raccoglie le funzioni per leggere il manifest, campionare i frame delle clip,
scrivere file CSV di debug e salvare le sequenze temporali in formato NPZ con
indice JSON.
"""
# Collegamenti con la pipeline:
# - è usato da extract_ball_rim_tracking_features.py per leggere clip e manifest;
# - importa l’ordine temp43 da tracking_geometry.py;
# - produce i file NPZ/JSON che tracking_sequence_store.py carica durante training
#   e valutazione dei classificatori gerarchici.


from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np

from src.features.tracking_geometry import TEMPORAL_TRACKING_FEATURE_NAMES_TEMP43


class Tee:
    """Duplica stdout/stderr su più stream, ad esempio terminale e file di log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def normalize_clip_key(path_value) -> str:
    """
    Normalizza il path di una clip in una chiave stabile per associare
    feature video e sequenze temporali.
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


# Filtra il manifest prima dell’inferenza YOLO e risolve il path assoluto di ogni clip.
def read_manifest(manifest_path: Path, dataset_root: Path, splits, labels, max_clips=None):
    rows = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {"path", "label", "split"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Il manifest non contiene le colonne richieste: {sorted(missing_columns)}. "
                f"Colonne presenti: {reader.fieldnames}"
            )

        for row in reader:
            split = row["split"].strip()
            label = row["label"].strip()
            rel_path = row["path"].strip()

            if split not in splits:
                continue

            if label not in labels:
                continue

            video_path = dataset_root / rel_path

            rows.append(
                {
                    "clip_id": row.get("clip_id", Path(rel_path).stem),
                    "split": split,
                    "label": label,
                    "path": rel_path,
                    "video_path": video_path,
                }
            )

    rows = sorted(rows, key=lambda x: (x["split"], x["label"], x["path"]))

    if max_clips is not None and max_clips > 0:
        rows = rows[:max_clips]

    return rows



# Legge i metadati necessari per campionamento e conversione frame->tempo.
def get_video_metadata(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        fps = 25.0

    return frame_count, fps, width, height


# Genera indici unici e ordinati, sull’intera clip oppure nella sola porzione finale.
def build_frame_indices(frame_count: int, num_frames: int, sample_mode: str, last_ratio: float):
    if frame_count <= 0:
        return []

    if num_frames <= 0 or num_frames >= frame_count:
        return list(range(frame_count))

    if sample_mode == "uniform":
        start = 0
        end = frame_count - 1

    elif sample_mode == "last":
        start = int(max(0, math.floor(frame_count * (1.0 - last_ratio))))
        end = frame_count - 1

    else:
        raise ValueError(f"sample_mode non supportato: {sample_mode}")

    if end < start:
        start = 0
        end = frame_count - 1

    indices = np.linspace(start, end, num_frames)
    indices = sorted({int(round(x)) for x in indices})
    indices = [min(max(0, idx), frame_count - 1) for idx in indices]

    return indices


# Effettua accessi casuali ai frame richiesti e tenta il precedente in caso di
# decodifica fallita vicino alla fine della clip.
def read_frames(video_path: Path, frame_indices):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    frames = []
    valid_indices = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok and frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ok, frame = cap.read()

        if not ok or frame is None:
            continue

        frames.append(frame)
        valid_indices.append(frame_idx)

    cap.release()

    return frames, valid_indices



# Serializza output di debug/metadata con formato numerico uniforme.
def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key, "")

                if isinstance(value, float):
                    cleaned[key] = f"{value:.8f}"
                else:
                    cleaned[key] = value

            writer.writerow(cleaned)


# Salva gli array compressi separatamente dall’indice, così il caricamento può
# recuperare una clip tramite chiave normalizzata senza scandire tutto il dataset.
def write_temporal_sequences(output_dir: Path, sequence_entries, temporal_feature_names=None):
    """
    Salva le sequenze tracking in formato NPZ più un indice JSON path -> array.
    """
    if temporal_feature_names is None:
        temporal_feature_names = TEMPORAL_TRACKING_FEATURE_NAMES_TEMP43

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "tracking_sequences.npz"
    index_path = output_dir / "tracking_sequence_index.json"
    feature_names_path = output_dir / "tracking_sequence_feature_names.json"

    arrays = {}
    index = {
        "type": "temporal_sequence",
        "npz_path": str(npz_path),
        "feature_names": temporal_feature_names,
        "num_features": len(temporal_feature_names),
        "sequences": {},
    }

    # Ogni path normalizzato punta alla chiave interna dell’array nel contenitore NPZ.
    for array_idx, entry in enumerate(sequence_entries):
        array_key = f"seq_{array_idx:06d}"
        arrays[array_key] = entry["sequence"].astype(np.float32)

        normalized_key = normalize_clip_key(entry["path"])
        index["sequences"][normalized_key] = {
            "array_key": array_key,
            "clip_id": entry.get("clip_id", ""),
            "split": entry.get("split", ""),
            "label": entry.get("label", ""),
            "path": entry.get("path", ""),
            "sampled_frames": int(entry["sequence"].shape[0]),
        }

    np.savez_compressed(npz_path, **arrays)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": temporal_feature_names,
                "num_features": len(temporal_feature_names),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return npz_path, index_path, feature_names_path

