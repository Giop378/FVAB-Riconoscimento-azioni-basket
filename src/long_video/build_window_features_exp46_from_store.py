from __future__ import annotations

import argparse
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.long_video import defaults
from src.long_video.infer_exp46_from_store import (
    TEMPORAL_POLICY,
    TRACKING_MAX_FRAMES_PER_WINDOW,
    as_path,
    build_input_for_window,
    check_output_files,
    ensure_exists,
    find_model_config,
    find_tracking_config,
    get_tracking_normalizer,
    load_feature_store,
    read_windows_csv,
    safe_torch_load,
    write_json,
)


OUTPUT_FILES = [
    "window_features_l1.npz",
    "window_features_l2.npz",
    "window_features_l3.npz",
    "window_features_index.json",
    "window_features_metadata.json",
]


@dataclass
class TrackingLevel:
    """Configurazione tracking minima richiesta da build_input_for_window."""

    name: str
    checkpoint_path: Path
    tracking_config: dict[str, Any]
    feature_names: list[str]
    tracking_normalized: bool
    tracking_mean: np.ndarray
    tracking_std: np.ndarray
    expected_tracking_features: int
    tracking_source: str
    checkpoint_input_dim: int | None = None


def load_tracking_level(
    name: str,
    checkpoint_path: Path,
    expected_tracking_features: int,
    tracking_source: str,
) -> TrackingLevel:
    ensure_exists(checkpoint_path, f"Checkpoint {name}", must_be_file=True)
    ckpt = safe_torch_load(checkpoint_path)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint {name} non valido: atteso dict, ottenuto {type(ckpt)}")

    tracking_config = find_tracking_config(ckpt)
    feature_names = tracking_config.get("feature_names")
    if not isinstance(feature_names, (list, tuple)):
        raise TypeError(f"{name}: tracking_config.feature_names assente o non lista")

    feature_names = [str(value) for value in feature_names]
    if len(feature_names) != int(expected_tracking_features):
        raise ValueError(
            f"{name}: numero feature tracking inatteso. "
            f"Trovate {len(feature_names)}, attese {expected_tracking_features}."
        )

    tracking_normalized, tracking_mean, tracking_std = get_tracking_normalizer(
        name=name,
        tracking_config=tracking_config,
        num_features=len(feature_names),
    )

    model_config = find_model_config(ckpt)
    checkpoint_input_dim = model_config.get("input_dim")
    checkpoint_input_dim = None if checkpoint_input_dim is None else int(checkpoint_input_dim)

    print(f"[OK] {name}: {checkpoint_path}")
    print(f"     tracking source: {tracking_source}")
    print(f"     tracking features: {len(feature_names)}")
    print(f"     tracking normalized: {tracking_normalized}")

    return TrackingLevel(
        name=name,
        checkpoint_path=checkpoint_path,
        tracking_config=tracking_config,
        feature_names=feature_names,
        tracking_normalized=tracking_normalized,
        tracking_mean=tracking_mean,
        tracking_std=tracking_std,
        expected_tracking_features=int(expected_tracking_features),
        tracking_source=str(tracking_source),
        checkpoint_input_dim=checkpoint_input_dim,
    )


def window_to_index_row(row: Any, array_key: str, shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "array_key": array_key,
        "shape": [int(shape[0]), int(shape[1])],
        "scale_index": int(row.scale_index),
        "scale_sec": float(row.scale_sec),
        "start_time": float(row.start_time),
        "end_time": float(row.end_time),
        "center_time": float(row.center_time),
        "start_rel_sec": float(row.start_rel_sec),
        "end_rel_sec": float(row.end_rel_sec),
        "store_start_index": int(row.store_start_index),
        "store_end_index": int(row.store_end_index),
        "num_store_samples": int(row.num_store_samples),
        "dino_num_frames": None if row.dino_num_frames is None else int(row.dino_num_frames),
        "tracking_raw_num_frames": None
        if row.tracking_raw_num_frames is None
        else int(row.tracking_raw_num_frames),
        "first_sample_time": None if row.first_sample_time is None else float(row.first_sample_time),
        "last_sample_time": None if row.last_sample_time is None else float(row.last_sample_time),
    }


def write_array_to_npz(
    zf: zipfile.ZipFile,
    array_key: str,
    array: np.ndarray,
) -> None:
    """Scrive una singola matrice dentro un NPZ senza accumulare tutto in RAM."""
    array = np.asarray(array, dtype=np.float32)
    with zf.open(f"{array_key}.npy", mode="w") as handle:
        np.lib.format.write_array(handle, array, allow_pickle=False)


def primitives_for_level(feature_store: Any, level: TrackingLevel) -> dict[str, np.ndarray]:
    if level.tracking_source == "yolo_v1":
        return feature_store.yolo_v1_primitives
    if level.tracking_source == "yolo_v2":
        return feature_store.yolo_v2_primitives
    raise ValueError(f"Tracking source non supportata per {level.name}: {level.tracking_source}")


def materialize_level_features(
    output_path: Path,
    feature_store: Any,
    windows: list[Any],
    level: TrackingLevel,
    compression: int,
) -> dict[str, Any]:
    sequences_index: dict[str, dict[str, Any]] = {}
    lengths: list[int] = []
    dims: list[int] = []
    primitives = primitives_for_level(feature_store, level)

    print(f"\n=== Materializzazione {level.name} ===")
    print(f"output: {output_path}")
    print(f"tracking source: {level.tracking_source}")

    with zipfile.ZipFile(output_path, mode="w", compression=compression) as zf:
        for i, row in enumerate(tqdm(windows, desc=f"{level.name} windows")):
            array_key = f"arr_{i:07d}"
            features = build_input_for_window(
                feature_store=feature_store,
                row=row,
                level=level,
                primitives=primitives,
            )

            if features.ndim != 2:
                raise RuntimeError(
                    f"{level.name} {row.window_id}: feature con shape non valida {features.shape}"
                )

            write_array_to_npz(zf, array_key=array_key, array=features)
            sequences_index[row.window_id] = window_to_index_row(
                row=row,
                array_key=array_key,
                shape=features.shape,
            )
            lengths.append(int(features.shape[0]))
            dims.append(int(features.shape[1]))

    if not lengths:
        raise RuntimeError(f"Nessuna feature materializzata per {level.name}")

    lengths_np = np.asarray(lengths, dtype=np.int64)
    dims_np = np.asarray(dims, dtype=np.int64)

    if int(dims_np.min()) != int(dims_np.max()):
        raise RuntimeError(f"{level.name}: dimensione feature non costante: {sorted(set(dims))}")

    return {
        "file": output_path.name,
        "tracking_source": level.tracking_source,
        "checkpoint_path": str(level.checkpoint_path),
        "checkpoint_input_dim": level.checkpoint_input_dim,
        "num_windows": int(len(lengths)),
        "feature_dim": int(dims_np[0]),
        "dino_dim": int(feature_store.dino_dim),
        "tracking_dim": int(len(level.feature_names)),
        "tracking_feature_names": list(level.feature_names),
        "tracking_normalized": bool(level.tracking_normalized),
        "min_len": int(lengths_np.min()),
        "max_len": int(lengths_np.max()),
        "mean_len": float(lengths_np.mean()),
        "sequences": sequences_index,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materializza le feature finestra exp_46 da una feature store long-video. "
            "L'output è un NPZ per livello L1/L2/L3 con sequenze [T, DINO+tracking] "
            "già costruite con la policy train-like usata dall'inferenza."
        )
    )

    parser.add_argument("--feature-store-dir", type=Path, default=defaults.VAL_FEATURE_STORE_DIR)
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=defaults.VAL_OUTPUT_DIR / "windows_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.VAL_OUTPUT_DIR / "window_features_exp46",
    )

    parser.add_argument("--l1-checkpoint", type=Path, default=defaults.EXP46_L1_CHECKPOINT)
    parser.add_argument("--l2-checkpoint", type=Path, default=defaults.EXP46_L2_CHECKPOINT)
    parser.add_argument("--l3-checkpoint", type=Path, default=defaults.EXP46_L3_CHECKPOINT)

    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Salva NPZ non compressi. Più veloce ma occupa più spazio.",
    )

    return parser


def main() -> None:
    args = make_parser().parse_args()
    started_at = time.time()

    args.feature_store_dir = as_path(args.feature_store_dir)
    args.windows_csv = as_path(args.windows_csv)
    args.output_dir = as_path(args.output_dir)
    args.l1_checkpoint = as_path(args.l1_checkpoint)
    args.l2_checkpoint = as_path(args.l2_checkpoint)
    args.l3_checkpoint = as_path(args.l3_checkpoint)

    assert args.feature_store_dir is not None
    assert args.windows_csv is not None
    assert args.output_dir is not None
    assert args.l1_checkpoint is not None
    assert args.l2_checkpoint is not None
    assert args.l3_checkpoint is not None

    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max-windows deve essere > 0 se specificato")

    check_output_files(args.output_dir, output_files=OUTPUT_FILES, overwrite=args.overwrite)

    print("=== Caricamento feature store ===")
    feature_store = load_feature_store(args.feature_store_dir)
    print(f"feature_store_dir: {args.feature_store_dir}")
    print(f"timestamps:         {feature_store.timestamps.shape}")
    print(f"dinov3_features:    {feature_store.dino_features.shape}")
    print(f"segmento store:     {feature_store.store_start_sec:.3f}s -> {feature_store.store_end_sec:.3f}s")

    print("\n=== Caricamento finestre ===")
    windows = read_windows_csv(args.windows_csv, max_windows=args.max_windows)
    print(f"windows_csv:        {args.windows_csv}")
    print(f"num_windows:        {len(windows)}")
    print(f"prima finestra:     {windows[0].start_time:.3f}s -> {windows[0].end_time:.3f}s")
    print(f"ultima finestra:    {windows[-1].start_time:.3f}s -> {windows[-1].end_time:.3f}s")

    print("\n=== Caricamento configurazioni tracking exp_46 ===")
    l1 = load_tracking_level(
        name="L1",
        checkpoint_path=args.l1_checkpoint,
        expected_tracking_features=43,
        tracking_source="yolo_v2",
    )
    l2 = load_tracking_level(
        name="L2",
        checkpoint_path=args.l2_checkpoint,
        expected_tracking_features=29,
        tracking_source="yolo_v2",
    )
    l3 = load_tracking_level(
        name="L3",
        checkpoint_path=args.l3_checkpoint,
        expected_tracking_features=43,
        tracking_source="yolo_v1",
    )

    expected_dims = {
        "L1": int(feature_store.dino_dim + len(l1.feature_names)),
        "L2": int(feature_store.dino_dim + len(l2.feature_names)),
        "L3": int(feature_store.dino_dim + len(l3.feature_names)),
    }
    print("\nInput dimension attese:")
    for level_name, dim in expected_dims.items():
        print(f"{level_name}: {dim}")

    compression = zipfile.ZIP_STORED if args.no_compress else zipfile.ZIP_DEFLATED

    args.output_dir.mkdir(parents=True, exist_ok=True)
    levels = {
        "L1": l1,
        "L2": l2,
        "L3": l3,
    }
    output_paths = {
        "L1": args.output_dir / "window_features_l1.npz",
        "L2": args.output_dir / "window_features_l2.npz",
        "L3": args.output_dir / "window_features_l3.npz",
    }

    index: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_store_dir": str(args.feature_store_dir),
        "windows_csv": str(args.windows_csv),
        "num_windows": int(len(windows)),
        "temporal_policy": TEMPORAL_POLICY,
        "tracking_max_frames_per_window": int(TRACKING_MAX_FRAMES_PER_WINDOW),
        "levels": {},
    }

    for level_name, level in levels.items():
        level_index = materialize_level_features(
            output_path=output_paths[level_name],
            feature_store=feature_store,
            windows=windows,
            level=level,
            compression=compression,
        )
        index["levels"][level_name] = level_index

    index_path = args.output_dir / "window_features_index.json"
    metadata_path = args.output_dir / "window_features_metadata.json"

    write_json(index_path, index)

    metadata = {
        "created_at": index["created_at"],
        "elapsed_sec": round(time.time() - started_at, 3),
        "feature_store": {
            "dir": str(args.feature_store_dir),
            "store_start_sec": float(feature_store.store_start_sec),
            "store_end_sec": float(feature_store.store_end_sec),
            "feature_fps": None if feature_store.feature_fps is None else float(feature_store.feature_fps),
            "num_samples": int(feature_store.timestamps.shape[0]),
            "dino_shape": [int(v) for v in feature_store.dino_features.shape],
            "dino_dim": int(feature_store.dino_dim),
        },
        "windows": {
            "csv": str(args.windows_csv),
            "num_windows": int(len(windows)),
            "first_window": {
                "start_time": float(windows[0].start_time),
                "end_time": float(windows[0].end_time),
            },
            "last_window": {
                "start_time": float(windows[-1].start_time),
                "end_time": float(windows[-1].end_time),
            },
        },
        "policy": {
            "temporal_policy": TEMPORAL_POLICY,
            "dino_policy": "usa tutti i sample/frame della finestra",
            "tracking_policy": (
                f"compute_temporal_sequence_features su max {TRACKING_MAX_FRAMES_PER_WINDOW} "
                "frame uniformi, poi interpolazione a T DINO"
            ),
        },
        "outputs": {
            "l1": output_paths["L1"].name,
            "l2": output_paths["L2"].name,
            "l3": output_paths["L3"].name,
            "index": index_path.name,
            "metadata": metadata_path.name,
        },
        "levels_summary": {
            level_name: {
                key: value
                for key, value in level_data.items()
                if key != "sequences"
            }
            for level_name, level_data in index["levels"].items()
        },
    }
    write_json(metadata_path, metadata)

    print("\n=== Completato ===")
    print(f"Output dir: {args.output_dir}")
    print(f"Index:      {index_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"Tempo:      {metadata['elapsed_sec']}s")


if __name__ == "__main__":
    main()
