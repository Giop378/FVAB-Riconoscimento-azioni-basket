from __future__ import annotations

import argparse
import csv
import inspect
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.features.extract_ball_rim_tracking_features import (
    compute_pair_features,
    compute_temporal_sequence_features,
)
from src.long_video import defaults
from src.long_video.utils import (
    as_path,
    check_output_files,
    ensure_exists,
    parse_device_for_torch,
    read_json,
    write_json,
)
from src.models.temporal_transformer_classifier import TemporalTransformerActionClassifier


# =============================================================================
# Configurazione fissa exp_long_13 / exp_46
# =============================================================================

L1_LABELS = ["passaggio", "tiro", "no-action"]
L2_LABELS = ["tiroDaDue", "tiroDaTre", "tiroLibero"]
L3_LABELS = ["0", "1"]

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

# Coerente con gli esperimenti clip-level exp_46 usati dalla pipeline long-video.
TRACKING_MAX_FRAMES_PER_WINDOW = 48
TRACKING_NEAR_THRESHOLD = 0.12
TRACKING_RIM_INSIDE_MARGIN = 0.15
TEMPORAL_POLICY = "exp13_train_like_dino_all_frames_tracking_max48_checkpoint_zscore"

# Struttura feature tracking dei tre livelli exp_46.
L1_EXPECTED_TRACKING_FEATURES = 43  # YOLO v2, temp43
L2_EXPECTED_TRACKING_FEATURES = 29  # YOLO v2, temp29
L3_EXPECTED_TRACKING_FEATURES = 43  # YOLO v1, temp43


# =============================================================================
# Dataclass
# =============================================================================


@dataclass(frozen=True)
class WindowRow:
    window_id: str
    scale_index: int
    scale_sec: float
    start_time: float
    end_time: float
    center_time: float
    start_rel_sec: float
    end_rel_sec: float
    store_start_index: int
    store_end_index: int
    num_store_samples: int


@dataclass
class LevelBundle:
    name: str
    checkpoint_path: Path
    model_config: dict[str, Any]
    tracking_config: dict[str, Any]
    feature_names: list[str]
    labels: list[str]
    model: nn.Module
    tracking_normalized: bool
    tracking_mean: np.ndarray
    tracking_std: np.ndarray


@dataclass
class FeatureStore:
    root: Path
    metadata: dict[str, Any]
    timestamps: np.ndarray
    dino_features: np.ndarray
    yolo_v1_primitives: dict[str, np.ndarray]
    yolo_v2_primitives: dict[str, np.ndarray]
    dino_dim: int
    store_start_sec: float
    store_end_sec: float
    feature_fps: float | None


# =============================================================================
# Lettura feature store e finestre
# =============================================================================


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def npz_to_dict(npz_path: Path) -> dict[str, np.ndarray]:
    ensure_exists(npz_path, npz_path.name, must_be_file=True)
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def load_feature_store(feature_store_dir: Path) -> FeatureStore:
    ensure_exists(feature_store_dir, "Feature store", must_be_file=False)

    metadata_path = feature_store_dir / defaults.METADATA_FILENAME
    timestamps_path = feature_store_dir / defaults.TIMESTAMPS_FILENAME
    dino_path = feature_store_dir / defaults.DINOV3_FEATURES_FILENAME
    yolo_v1_path = feature_store_dir / defaults.YOLO_V1_PRIMITIVES_FILENAME
    yolo_v2_path = feature_store_dir / defaults.YOLO_V2_PRIMITIVES_FILENAME

    ensure_exists(timestamps_path, "timestamps.npy", must_be_file=True)
    ensure_exists(dino_path, "dinov3_features.npy", must_be_file=True)
    ensure_exists(yolo_v1_path, "yolo_v1_primitives.npz", must_be_file=True)
    ensure_exists(yolo_v2_path, "yolo_v2_primitives.npz", must_be_file=True)

    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    timestamps = np.load(timestamps_path).astype(np.float64)
    dino_features = np.load(dino_path, mmap_mode="r")

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps.npy deve avere shape [N], trovato {timestamps.shape}")
    if timestamps.size == 0:
        raise ValueError("timestamps.npy è vuoto")
    if dino_features.ndim != 2:
        raise ValueError(f"dinov3_features.npy deve avere shape [N, D], trovato {dino_features.shape}")
    if dino_features.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"timestamps e DINO hanno N diverso: {timestamps.shape[0]} vs {dino_features.shape[0]}"
        )
    if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps.npy contiene valori non finiti oppure non è monotono crescente")

    video_meta = metadata.get("video", {}) if isinstance(metadata, dict) else {}
    sampling_meta = metadata.get("sampling", {}) if isinstance(metadata, dict) else {}

    store_start = float(video_meta.get("start_sec", float(timestamps[0])))
    store_end = float(video_meta.get("end_sec", float(timestamps[-1])))
    feature_fps_value = sampling_meta.get("feature_fps")
    feature_fps = None if feature_fps_value is None else float(feature_fps_value)

    yolo_v1 = npz_to_dict(yolo_v1_path)
    yolo_v2 = npz_to_dict(yolo_v2_path)

    required_primitive_fields = [
        "ball_detected",
        "ball_conf",
        "ball_xc",
        "ball_yc",
        "ball_w",
        "ball_h",
        "rim_detected",
        "rim_conf",
        "rim_xc",
        "rim_yc",
        "rim_w",
        "rim_h",
    ]
    for source_name, primitives in [("yolo_v1", yolo_v1), ("yolo_v2", yolo_v2)]:
        for required in required_primitive_fields:
            if required not in primitives:
                raise KeyError(f"{source_name}_primitives.npz non contiene '{required}'")
        n = primitives["ball_detected"].shape[0]
        if n != timestamps.shape[0]:
            raise ValueError(
                f"{source_name}_primitives ha N={n}, ma timestamps ha N={timestamps.shape[0]}"
            )

    return FeatureStore(
        root=feature_store_dir,
        metadata=metadata,
        timestamps=timestamps,
        dino_features=dino_features,
        yolo_v1_primitives=yolo_v1,
        yolo_v2_primitives=yolo_v2,
        dino_dim=int(dino_features.shape[1]),
        store_start_sec=store_start,
        store_end_sec=store_end,
        feature_fps=feature_fps,
    )


def read_windows_csv(path: Path) -> list[WindowRow]:
    ensure_exists(path, "windows_manifest.csv", must_be_file=True)
    rows: list[WindowRow] = []

    required = [
        "window_id",
        "scale_index",
        "scale_sec",
        "start_time",
        "end_time",
        "center_time",
        "start_rel_sec",
        "end_rel_sec",
        "store_start_index",
        "store_end_index",
        "num_store_samples",
    ]

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"windows_manifest.csv non contiene colonne richieste: {missing}")

        for row in reader:
            rows.append(
                WindowRow(
                    window_id=row["window_id"],
                    scale_index=int(row["scale_index"]),
                    scale_sec=float(row["scale_sec"]),
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                    center_time=float(row["center_time"]),
                    start_rel_sec=float(row["start_rel_sec"]),
                    end_rel_sec=float(row["end_rel_sec"]),
                    store_start_index=int(row["store_start_index"]),
                    store_end_index=int(row["store_end_index"]),
                    num_store_samples=int(row["num_store_samples"]),
                )
            )

    if not rows:
        raise RuntimeError(f"Nessuna finestra letta da {path}")
    return rows


# =============================================================================
# Checkpoint e modello exp_46
# =============================================================================


def find_model_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    value = ckpt.get("model_config")
    return dict(value) if isinstance(value, dict) else {}


def find_tracking_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    tracking_config = ckpt.get("tracking_config")
    if isinstance(tracking_config, dict):
        return dict(tracking_config)

    model_config = find_model_config(ckpt)
    tracking_config = model_config.get("tracking_config")
    if isinstance(tracking_config, dict):
        return dict(tracking_config)

    raise KeyError("tracking_config non trovato nel checkpoint")


def get_state_dict_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ["model_state_dict", "state_dict", "model_state", "net_state_dict"]:
        value = ckpt.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value

    raise KeyError("State dict del modello non trovato nel checkpoint exp_46")


def strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ["module.", "model.", "net."]
    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value

    return cleaned


def normalize_model_config(model_config: dict[str, Any], num_classes: int) -> dict[str, Any]:
    config = dict(model_config)

    if "dim_feedforward" not in config and "ff_dim" in config:
        config["dim_feedforward"] = config["ff_dim"]
    if "ff_dim" not in config and "dim_feedforward" in config:
        config["ff_dim"] = config["dim_feedforward"]

    config["num_classes"] = int(num_classes)
    config["n_classes"] = int(num_classes)
    config["num_labels"] = int(num_classes)
    return config


def instantiate_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    config = normalize_model_config(model_config, num_classes=num_classes)
    cls = TemporalTransformerActionClassifier
    signature = inspect.signature(cls)

    alias_values: dict[str, Any] = {
        "input_dim": config.get("input_dim"),
        "num_classes": num_classes,
        "n_classes": num_classes,
        "num_labels": num_classes,
        "d_model": config.get("d_model"),
        "hidden_dim": config.get("d_model", config.get("hidden_dim")),
        "num_layers": config.get("num_layers"),
        "n_layers": config.get("num_layers"),
        "num_heads": config.get("num_heads"),
        "nhead": config.get("num_heads"),
        "n_heads": config.get("num_heads"),
        "ff_dim": config.get("ff_dim"),
        "dim_feedforward": config.get("dim_feedforward", config.get("ff_dim")),
        "dropout": config.get("dropout"),
        "pooling": config.get("pooling"),
        "pooling_mode": config.get("pooling"),
        "max_len": config.get("max_len"),
        "last_mean_ratio": config.get("last_mean_ratio"),
    }

    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if name == "self":
            continue
        if name in alias_values and alias_values[name] is not None:
            kwargs[name] = alias_values[name]
        elif name in config and config[name] is not None:
            kwargs[name] = config[name]

    return cls(**kwargs)


def get_tracking_normalizer(
    name: str,
    tracking_config: dict[str, Any],
    num_features: int,
) -> tuple[bool, np.ndarray, np.ndarray]:
    normalized = bool(tracking_config.get("normalized", False))
    mean_value = tracking_config.get("mean")
    std_value = tracking_config.get("std")

    if normalized:
        if mean_value is None or std_value is None:
            raise ValueError(
                f"{name}: tracking_config.normalized=True ma mean/std sono assenti nel checkpoint."
            )
        mean = np.asarray(mean_value, dtype=np.float32)
        std = np.asarray(std_value, dtype=np.float32)
    else:
        mean = np.zeros((num_features,), dtype=np.float32)
        std = np.ones((num_features,), dtype=np.float32)

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(f"{name}: mean/std tracking devono essere vettori 1D")
    if mean.shape[0] != num_features or std.shape[0] != num_features:
        raise ValueError(
            f"{name}: dimensione mean/std tracking non coerente. "
            f"Feature={num_features}, mean={mean.shape[0]}, std={std.shape[0]}."
        )

    std = np.where(np.abs(std) < 1e-6, 1.0, std).astype(np.float32)
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def load_level_bundle(
    name: str,
    checkpoint_path: Path,
    labels: list[str],
    expected_tracking_features: int,
    device: torch.device,
) -> LevelBundle:
    ensure_exists(checkpoint_path, f"Checkpoint {name}", must_be_file=True)
    ckpt = safe_torch_load(checkpoint_path)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint {name} non valido: atteso dict, ottenuto {type(ckpt)}")

    model_config = find_model_config(ckpt)
    tracking_config = find_tracking_config(ckpt)
    state_dict = strip_state_dict_prefixes(get_state_dict_from_checkpoint(ckpt))

    feature_names_raw = tracking_config.get("feature_names")
    if not isinstance(feature_names_raw, (list, tuple)):
        raise TypeError(f"{name}: tracking_config.feature_names assente o non lista")
    feature_names = [str(v) for v in feature_names_raw]

    if len(feature_names) != expected_tracking_features:
        raise ValueError(
            f"{name}: numero feature tracking inatteso. "
            f"Trovate {len(feature_names)}, attese {expected_tracking_features}."
        )

    model = instantiate_model(model_config=model_config, num_classes=len(labels))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    tracking_normalized, tracking_mean, tracking_std = get_tracking_normalizer(
        name=name,
        tracking_config=tracking_config,
        num_features=len(feature_names),
    )

    input_dim = model_config.get("input_dim")
    if input_dim is not None:
        expected_input_dim = int(defaults.DINOV3_FEATURE_DIM + len(feature_names))
        if int(input_dim) != expected_input_dim:
            print(
                f"[WARN] {name}: model_config.input_dim={input_dim}, "
                f"atteso {expected_input_dim}."
            )

    print(f"[OK] {name}: {checkpoint_path}")
    print(f"     labels: {labels}")
    print(f"     tracking features: {len(feature_names)}")
    print(f"     tracking normalized: {tracking_normalized}")

    return LevelBundle(
        name=name,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        tracking_config=tracking_config,
        feature_names=feature_names,
        labels=list(labels),
        model=model,
        tracking_normalized=tracking_normalized,
        tracking_mean=tracking_mean,
        tracking_std=tracking_std,
    )


def output_to_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ["logits", "out", "output", "pred", "prediction"]:
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        tensors = [v for v in output.values() if torch.is_tensor(v)]
        if tensors:
            return tensors[0]
        raise TypeError("Output dict senza tensori utilizzabili")

    if isinstance(output, (list, tuple)):
        if not output:
            raise TypeError("Output tuple/list vuoto")
        return output_to_logits(output[0])

    if torch.is_tensor(output):
        return output

    raise TypeError(f"Output modello non supportato: {type(output)}")


def model_forward_logits(model: nn.Module, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
        raise ValueError(
            f"lengths deve avere shape [B], trovato {tuple(lengths.shape)} per x={tuple(x.shape)}"
        )

    time_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    padding_mask = time_positions >= lengths.unsqueeze(1)

    attempts: list[Callable[[], Any]] = [
        lambda: model(x, lengths=lengths),
        lambda: model(x, lengths),
        lambda: model(x, padding_mask=padding_mask),
        lambda: model(x, src_key_padding_mask=padding_mask),
        lambda: model(x, mask=padding_mask),
        lambda: model(x),
    ]

    errors: list[str] = []
    for attempt in attempts:
        try:
            return output_to_logits(attempt())
        except TypeError as exc:
            errors.append(str(exc))

    raise RuntimeError("Forward del modello fallito. Errori TypeError:\n" + "\n".join(errors[-3:]))


# =============================================================================
# Costruzione feature tracking train-like
# =============================================================================


def select_uniform_local_indices(num_items: int, max_items: int) -> np.ndarray:
    if num_items <= 0:
        return np.empty((0,), dtype=np.int64)
    if max_items <= 0 or num_items <= max_items:
        return np.arange(num_items, dtype=np.int64)

    indices = np.linspace(0, num_items - 1, max_items)
    indices = sorted({int(round(v)) for v in indices})
    indices = [min(max(0, idx), num_items - 1) for idx in indices]
    return np.asarray(indices, dtype=np.int64)


def interpolate_sequence_array(sequence: np.ndarray, target_len: int) -> np.ndarray:
    target_len = int(target_len)
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(f"La sequenza tracking deve avere forma [S, K], ricevuta {sequence.shape}.")
    if target_len <= 0:
        return np.zeros((0, sequence.shape[1]), dtype=np.float32)

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


def primitive_value(primitives: dict[str, np.ndarray], name: str, idx: int, default: float = 0.0) -> float:
    values = primitives.get(name)
    if values is None:
        return float(default)
    return float(values[int(idx)])


def primitive_detection(primitives: dict[str, np.ndarray], prefix: str, idx: int) -> dict[str, float | int]:
    detected = int(primitive_value(primitives, f"{prefix}_detected", idx, 0.0) >= 0.5)
    conf = primitive_value(primitives, f"{prefix}_conf", idx, 0.0) if detected else 0.0
    xc = primitive_value(primitives, f"{prefix}_xc", idx, 0.0) if detected else 0.0
    yc = primitive_value(primitives, f"{prefix}_yc", idx, 0.0) if detected else 0.0
    w = primitive_value(primitives, f"{prefix}_w", idx, 0.0) if detected else 0.0
    h = primitive_value(primitives, f"{prefix}_h", idx, 0.0) if detected else 0.0

    return {
        "detected": detected,
        "conf": float(conf),
        "xc": float(xc),
        "yc": float(yc),
        "w": float(w),
        "h": float(h),
        "area": float(max(0.0, w) * max(0.0, h)) if detected else 0.0,
    }


def build_frame_rows_trainlike(
    feature_store: FeatureStore,
    row: WindowRow,
    primitives: dict[str, np.ndarray],
    selected_local_indices: np.ndarray,
) -> list[dict[str, Any]]:
    dino_len = int(row.store_end_index - row.store_start_index)
    fps = float(feature_store.feature_fps or 29.97002997002997)
    frame_rows: list[dict[str, Any]] = []

    for frame_order, local_idx_value in enumerate(selected_local_indices.tolist()):
        local_idx = int(local_idx_value)
        global_idx = int(row.store_start_index + local_idx)

        ball = primitive_detection(primitives, "ball", global_idx)
        rim = primitive_detection(primitives, "rim", global_idx)
        pair = compute_pair_features(
            ball=ball,
            rim=rim,
            near_threshold=TRACKING_NEAR_THRESHOLD,
            rim_inside_margin=TRACKING_RIM_INSIDE_MARGIN,
        )

        t_rel = local_idx / max(1, dino_len - 1)
        time_sec = local_idx / fps if fps > 0 else 0.0

        frame_rows.append(
            {
                "clip_id": row.window_id,
                "split": "long-video",
                "label": "unknown",
                "path": row.window_id,
                "frame_order": int(frame_order),
                "frame_idx": int(local_idx),
                "time_sec": float(time_sec),
                "t_rel": float(t_rel),
                "width": 1,
                "height": 1,
                "ball_detected": int(ball["detected"]),
                "ball_conf": float(ball["conf"]),
                "ball_xc": float(ball["xc"]),
                "ball_yc": float(ball["yc"]),
                "ball_w": float(ball["w"]),
                "ball_h": float(ball["h"]),
                "ball_area": float(ball["area"]),
                "rim_detected": int(rim["detected"]),
                "rim_conf": float(rim["conf"]),
                "rim_xc": float(rim["xc"]),
                "rim_yc": float(rim["yc"]),
                "rim_w": float(rim["w"]),
                "rim_h": float(rim["h"]),
                "rim_area": float(rim["area"]),
                "both_detected": int(pair["both_detected"]),
                "dx": float(pair["dx"]),
                "dy": float(pair["dy"]),
                "ball_rim_dist": float(pair["dist"]),
                "ball_above_rim": int(pair["ball_above_rim"]),
                "ball_below_rim": int(pair["ball_below_rim"]),
                "ball_near_rim": int(pair["ball_near_rim"]),
                "ball_center_inside_rim": int(pair["ball_center_inside_rim"]),
                "ball_center_inside_expanded_rim": int(pair["ball_center_inside_expanded_rim"]),
                "ball_rim_iou": float(pair["ball_rim_iou"]),
                "ball_passes_close_to_rim": int(pair["ball_passes_close_to_rim"]),
            }
        )

    return frame_rows


def build_tracking_sequence_trainlike(
    feature_store: FeatureStore,
    row: WindowRow,
    level: LevelBundle,
    primitives: dict[str, np.ndarray],
    selected_local_indices: np.ndarray,
) -> np.ndarray:
    frame_rows = build_frame_rows_trainlike(
        feature_store=feature_store,
        row=row,
        primitives=primitives,
        selected_local_indices=selected_local_indices,
    )
    fps = float(feature_store.feature_fps or 29.97002997002997)
    sequence = compute_temporal_sequence_features(
        frame_rows=frame_rows,
        fps=fps,
        temporal_feature_names=level.feature_names,
    )
    if sequence.ndim != 2 or sequence.shape[1] != len(level.feature_names):
        raise RuntimeError(
            f"{level.name}: sequenza tracking inattesa {sequence.shape}, "
            f"attesa [S, {len(level.feature_names)}]."
        )
    return sequence.astype(np.float32)


def apply_tracking_normalization(tracking_seq: np.ndarray, level: LevelBundle) -> np.ndarray:
    tracking_seq = np.asarray(tracking_seq, dtype=np.float32)
    if tracking_seq.ndim != 2:
        raise ValueError(f"{level.name}: tracking_seq deve avere shape [T, K], trovato {tracking_seq.shape}.")
    if tracking_seq.shape[1] != len(level.feature_names):
        raise ValueError(
            f"{level.name}: tracking_seq ha K={tracking_seq.shape[1]}, "
            f"ma il checkpoint richiede K={len(level.feature_names)}."
        )
    if not level.tracking_normalized:
        return tracking_seq.astype(np.float32)
    return (
        (tracking_seq - level.tracking_mean.reshape(1, -1))
        / level.tracking_std.reshape(1, -1)
    ).astype(np.float32)


def build_input_for_window(
    feature_store: FeatureStore,
    row: WindowRow,
    level: LevelBundle,
    primitives: dict[str, np.ndarray],
) -> np.ndarray:
    start_idx = int(row.store_start_index)
    end_idx = int(row.store_end_index)

    if start_idx < 0 or end_idx > feature_store.timestamps.shape[0] or end_idx <= start_idx:
        raise ValueError(
            f"Indici finestra non validi per {row.window_id}: "
            f"{start_idx}:{end_idx} su N={feature_store.timestamps.shape[0]}"
        )

    dino_seq = np.asarray(feature_store.dino_features[start_idx:end_idx], dtype=np.float32)
    dino_len = int(dino_seq.shape[0])
    if dino_len <= 0:
        raise ValueError(f"Finestra senza feature DINO: {row.window_id}")

    tracking_local_indices = select_uniform_local_indices(
        num_items=dino_len,
        max_items=TRACKING_MAX_FRAMES_PER_WINDOW,
    )
    tracking_raw_seq = build_tracking_sequence_trainlike(
        feature_store=feature_store,
        row=row,
        level=level,
        primitives=primitives,
        selected_local_indices=tracking_local_indices,
    )
    tracking_seq = interpolate_sequence_array(tracking_raw_seq, target_len=dino_len)
    tracking_seq = apply_tracking_normalization(tracking_seq, level=level)

    return np.concatenate([dino_seq, tracking_seq], axis=1).astype(np.float32)


def pad_sequences(sequences: list[np.ndarray], expected_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("Batch vuoto: nessuna sequenza da paddare")

    lengths = np.asarray([seq.shape[0] for seq in sequences], dtype=np.int64)
    max_len = int(lengths.max())
    padded = np.zeros((len(sequences), max_len, expected_dim), dtype=np.float32)

    for i, seq in enumerate(sequences):
        if seq.ndim != 2 or seq.shape[1] != expected_dim:
            raise ValueError(
                f"Sequenza {i} con shape inattesa: {seq.shape}; attesa [T, {expected_dim}]"
            )
        padded[i, : seq.shape[0], :] = seq

    return padded, lengths


def build_batch_inputs(
    feature_store: FeatureStore,
    rows: list[WindowRow],
    l1: LevelBundle,
    l2: LevelBundle,
    l3: LevelBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l1_dim = feature_store.dino_dim + len(l1.feature_names)
    l2_dim = feature_store.dino_dim + len(l2.feature_names)
    l3_dim = feature_store.dino_dim + len(l3.feature_names)

    seqs_l1 = [
        build_input_for_window(feature_store, row, l1, feature_store.yolo_v2_primitives)
        for row in rows
    ]
    seqs_l2 = [
        build_input_for_window(feature_store, row, l2, feature_store.yolo_v2_primitives)
        for row in rows
    ]
    seqs_l3 = [
        build_input_for_window(feature_store, row, l3, feature_store.yolo_v1_primitives)
        for row in rows
    ]

    x_l1, lengths = pad_sequences(seqs_l1, expected_dim=l1_dim)
    x_l2, lengths_l2 = pad_sequences(seqs_l2, expected_dim=l2_dim)
    x_l3, lengths_l3 = pad_sequences(seqs_l3, expected_dim=l3_dim)

    if not (np.array_equal(lengths, lengths_l2) and np.array_equal(lengths, lengths_l3)):
        raise RuntimeError("Lunghezze diverse tra L1/L2/L3 nello stesso batch")

    return x_l1, x_l2, x_l3, lengths


# =============================================================================
# Probabilità e output CSV
# =============================================================================


def compute_final_scores(p_l1: np.ndarray, p_l2: np.ndarray, p_l3: np.ndarray) -> dict[str, np.ndarray]:
    p_passaggio = p_l1[:, 0]
    p_tiro = p_l1[:, 1]
    p_noaction = p_l1[:, 2]

    p_due = p_l2[:, 0]
    p_tre = p_l2[:, 1]
    p_libero = p_l2[:, 2]

    p_0 = p_l3[:, 0]
    p_1 = p_l3[:, 1]

    return {
        "passaggio": p_passaggio,
        "tiroDaDue0": p_tiro * p_due * p_0,
        "tiroDaDue1": p_tiro * p_due * p_1,
        "tiroDaTre0": p_tiro * p_tre * p_0,
        "tiroDaTre1": p_tiro * p_tre * p_1,
        "tiroLibero0": p_tiro * p_libero * p_0,
        "tiroLibero1": p_tiro * p_libero * p_1,
        "no-action": p_noaction,
    }


def argmax_final_labels(scores: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    matrix = np.stack([scores[label] for label in FINAL_LABELS], axis=1)
    indices = matrix.argmax(axis=1)
    confidences = matrix[np.arange(matrix.shape[0]), indices]
    labels = [FINAL_LABELS[int(i)] for i in indices]
    return labels, confidences.astype(np.float32)


def output_fieldnames() -> list[str]:
    return [
        "window_id",
        "scale_index",
        "scale_sec",
        "start_time",
        "end_time",
        "center_time",
        "start_rel_sec",
        "end_rel_sec",
        "pred_label",
        "confidence",
        "score_passaggio",
        "score_tiroDaDue0",
        "score_tiroDaDue1",
        "score_tiroDaTre0",
        "score_tiroDaTre1",
        "score_tiroLibero0",
        "score_tiroLibero1",
        "score_noaction",
    ]


def write_prediction_rows(
    writer: csv.DictWriter,
    rows: list[WindowRow],
    scores: dict[str, np.ndarray],
    pred_labels: list[str],
    confidences: np.ndarray,
) -> None:
    for i, row in enumerate(rows):
        writer.writerow(
            {
                "window_id": row.window_id,
                "scale_index": row.scale_index,
                "scale_sec": f"{row.scale_sec:.6f}",
                "start_time": f"{row.start_time:.6f}",
                "end_time": f"{row.end_time:.6f}",
                "center_time": f"{row.center_time:.6f}",
                "start_rel_sec": f"{row.start_rel_sec:.6f}",
                "end_rel_sec": f"{row.end_rel_sec:.6f}",
                "pred_label": pred_labels[i],
                "confidence": f"{float(confidences[i]):.8f}",
                "score_passaggio": f"{float(scores['passaggio'][i]):.8f}",
                "score_tiroDaDue0": f"{float(scores['tiroDaDue0'][i]):.8f}",
                "score_tiroDaDue1": f"{float(scores['tiroDaDue1'][i]):.8f}",
                "score_tiroDaTre0": f"{float(scores['tiroDaTre0'][i]):.8f}",
                "score_tiroDaTre1": f"{float(scores['tiroDaTre1'][i]):.8f}",
                "score_tiroLibero0": f"{float(scores['tiroLibero0'][i]):.8f}",
                "score_tiroLibero1": f"{float(scores['tiroLibero1'][i]):.8f}",
                "score_noaction": f"{float(scores['no-action'][i]):.8f}",
            }
        )


# =============================================================================
# Inferenza
# =============================================================================


def predict_batch(
    x_np: np.ndarray,
    lengths_np: np.ndarray,
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    x = torch.from_numpy(x_np).to(device, non_blocking=True)
    lengths = torch.from_numpy(lengths_np.astype(np.int64)).to(device, non_blocking=True)
    amp_enabled = bool(use_amp and device.type == "cuda")

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model_forward_logits(model, x, lengths=lengths)
        if logits.ndim != 2:
            raise RuntimeError(f"Logits attesi [B, C], trovati {tuple(logits.shape)}")
        probs = torch.softmax(logits.float(), dim=1)

    return probs.detach().cpu().numpy().astype(np.float32)


def infer_all_windows(
    feature_store: FeatureStore,
    windows: list[WindowRow],
    l1: LevelBundle,
    l2: LevelBundle,
    l3: LevelBundle,
    output_csv: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError(f"batch_size deve essere > 0, trovato {batch_size}")

    label_counts: dict[str, int] = {label: 0 for label in FINAL_LABELS}
    input_lengths: list[int] = []
    num_batches = math.ceil(len(windows) / batch_size)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames())
        writer.writeheader()

        progress = tqdm(range(0, len(windows), batch_size), total=num_batches, desc="Inferenza exp_long_13")
        for start in progress:
            batch_rows = windows[start : start + batch_size]
            x_l1, x_l2, x_l3, lengths = build_batch_inputs(feature_store, batch_rows, l1, l2, l3)

            p_l1 = predict_batch(x_l1, lengths, l1.model, device=device, use_amp=True)
            p_l2 = predict_batch(x_l2, lengths, l2.model, device=device, use_amp=True)
            p_l3 = predict_batch(x_l3, lengths, l3.model, device=device, use_amp=True)

            scores = compute_final_scores(p_l1, p_l2, p_l3)
            pred_labels, confidences = argmax_final_labels(scores)

            for label in pred_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            input_lengths.extend(int(v) for v in lengths.tolist())

            write_prediction_rows(writer, batch_rows, scores, pred_labels, confidences)

    input_lengths_arr = np.asarray(input_lengths, dtype=np.int64)
    return {
        "num_windows": int(len(windows)),
        "label_counts": label_counts,
        "temporal_policy": TEMPORAL_POLICY,
        "input_lengths": {
            "min": int(input_lengths_arr.min()) if input_lengths_arr.size else 0,
            "max": int(input_lengths_arr.max()) if input_lengths_arr.size else 0,
            "mean": float(input_lengths_arr.mean()) if input_lengths_arr.size else 0.0,
        },
    }


def write_inference_metadata(
    path: Path,
    feature_store: FeatureStore,
    windows_csv: Path,
    windows: list[WindowRow],
    batch_size: int,
    device: torch.device,
    started_at: float,
    summary: dict[str, Any],
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started_at, 3),
        "experiment": "exp_long_13",
        "feature_store": {
            "dir": str(feature_store.root),
            "num_samples": int(feature_store.timestamps.shape[0]),
            "dino_dim": int(feature_store.dino_dim),
            "store_start_sec": float(feature_store.store_start_sec),
            "store_end_sec": float(feature_store.store_end_sec),
            "feature_fps": feature_store.feature_fps,
        },
        "windows": {
            "csv": str(windows_csv),
            "num_windows": int(len(windows)),
            "first_start_time": float(windows[0].start_time),
            "last_end_time": float(windows[-1].end_time),
        },
        "runtime": {
            "device": str(device),
            "batch_size": int(batch_size),
            "amp": True,
            "temporal_policy": TEMPORAL_POLICY,
            "dino_policy": "all_store_samples_in_window",
            "tracking_policy": "compute_temporal_sequence_features_max48_then_interpolate_then_checkpoint_zscore",
        },
        "levels": {
            "L1": {
                "checkpoint": str(defaults.EXP46_L1_CHECKPOINT),
                "tracking_source": "yolo_v2",
                "labels": L1_LABELS,
                "num_tracking_features": L1_EXPECTED_TRACKING_FEATURES,
            },
            "L2": {
                "checkpoint": str(defaults.EXP46_L2_CHECKPOINT),
                "tracking_source": "yolo_v2",
                "labels": L2_LABELS,
                "num_tracking_features": L2_EXPECTED_TRACKING_FEATURES,
            },
            "L3": {
                "checkpoint": str(defaults.EXP46_L3_CHECKPOINT),
                "tracking_source": "yolo_v1",
                "labels": L3_LABELS,
                "num_tracking_features": L3_EXPECTED_TRACKING_FEATURES,
            },
        },
        "outputs": {
            "window_predictions_raw": defaults.WINDOW_PREDICTIONS_FILENAME,
            "metadata": defaults.INFERENCE_METADATA_FILENAME,
        },
        "summary": summary,
    }
    write_json(path, metadata)


# =============================================================================
# CLI
# =============================================================================


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inferenza raw per exp_long_13 da feature store long-video. "
            "Usa i checkpoint exp_46 definiti in defaults.py e produce window_predictions_raw.csv."
        )
    )
    parser.add_argument("--feature-store-dir", type=Path, default=defaults.VAL_FEATURE_STORE_DIR)
    parser.add_argument("--windows-csv", type=Path, default=defaults.VAL_OUTPUT_DIR / defaults.WINDOWS_MANIFEST_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=defaults.VAL_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    started_at = time.time()

    args.feature_store_dir = as_path(args.feature_store_dir)
    args.windows_csv = as_path(args.windows_csv)
    args.output_dir = as_path(args.output_dir)
    assert args.feature_store_dir is not None
    assert args.windows_csv is not None
    assert args.output_dir is not None

    if args.batch_size <= 0:
        raise ValueError("--batch-size deve essere > 0")

    check_output_files(
        args.output_dir,
        output_files=[defaults.WINDOW_PREDICTIONS_FILENAME, defaults.INFERENCE_METADATA_FILENAME],
        overwrite=bool(args.overwrite),
        remove_existing=True,
    )

    device = parse_device_for_torch(args.device)

    print("=== Caricamento feature store ===")
    feature_store = load_feature_store(args.feature_store_dir)
    print(f"feature_store_dir: {args.feature_store_dir}")
    print(f"timestamps:         {feature_store.timestamps.shape}")
    print(f"dinov3_features:    {feature_store.dino_features.shape}")
    print(f"segmento store:     {feature_store.store_start_sec:.3f}s -> {feature_store.store_end_sec:.3f}s")

    print("\n=== Caricamento finestre ===")
    windows = read_windows_csv(args.windows_csv)
    print(f"windows_csv:        {args.windows_csv}")
    print(f"num_windows:        {len(windows)}")
    print(f"prima finestra:     {windows[0].start_time:.3f}s -> {windows[0].end_time:.3f}s")
    print(f"ultima finestra:    {windows[-1].start_time:.3f}s -> {windows[-1].end_time:.3f}s")

    print("\n=== Caricamento modelli exp_46 per exp_long_13 ===")
    l1 = load_level_bundle(
        name="L1",
        checkpoint_path=defaults.EXP46_L1_CHECKPOINT,
        labels=L1_LABELS,
        expected_tracking_features=L1_EXPECTED_TRACKING_FEATURES,
        device=device,
    )
    l2 = load_level_bundle(
        name="L2",
        checkpoint_path=defaults.EXP46_L2_CHECKPOINT,
        labels=L2_LABELS,
        expected_tracking_features=L2_EXPECTED_TRACKING_FEATURES,
        device=device,
    )
    l3 = load_level_bundle(
        name="L3",
        checkpoint_path=defaults.EXP46_L3_CHECKPOINT,
        labels=L3_LABELS,
        expected_tracking_features=L3_EXPECTED_TRACKING_FEATURES,
        device=device,
    )

    output_csv = args.output_dir / defaults.WINDOW_PREDICTIONS_FILENAME
    print("\n=== Inferenza raw finestre ===")
    print(f"output_csv:         {output_csv}")
    print(f"device:             {device}")
    print(f"batch_size:         {args.batch_size}")
    print(f"temporal_policy:    {TEMPORAL_POLICY}")

    summary = infer_all_windows(
        feature_store=feature_store,
        windows=windows,
        l1=l1,
        l2=l2,
        l3=l3,
        output_csv=output_csv,
        batch_size=int(args.batch_size),
        device=device,
    )

    metadata_path = args.output_dir / defaults.INFERENCE_METADATA_FILENAME
    write_inference_metadata(
        path=metadata_path,
        feature_store=feature_store,
        windows_csv=args.windows_csv,
        windows=windows,
        batch_size=int(args.batch_size),
        device=device,
        started_at=started_at,
        summary=summary,
    )

    print("\n=== Inferenza completata ===")
    print(f"predizioni: {output_csv}")
    print(f"metadata:   {metadata_path}")
    print("Predizioni per label finale:")
    for label, count in summary["label_counts"].items():
        print(f"- {label}: {count}")
    print(f"tempo totale: {time.time() - started_at:.1f}s")


if __name__ == "__main__":
    main()
