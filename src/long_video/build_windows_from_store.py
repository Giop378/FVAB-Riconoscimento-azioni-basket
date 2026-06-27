from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.long_video import defaults
from src.long_video.utils import as_path, check_output_files, ensure_exists, read_json, write_json


# =============================================================================
# Configurazione train-like long-video
# =============================================================================

# La feature store train-like contiene una riga per ogni frame reale del video.
# Per ogni finestra, DINOv3 userà tutti i sample della finestra; il tracking
# palla/canestro verrà invece ricostruito nello step di inferenza usando al
# massimo 48 frame, come nello script clip-level extract_ball_rim_tracking_features.py.
TRACKING_MAX_FRAMES_PER_WINDOW = 48
DEFAULT_WINDOW_SIZES_SEC = [0.5, 0.75, 1.0, 1.5, 2.0]


# =============================================================================
# Lettura feature store
# =============================================================================


@dataclass(frozen=True)
class FeatureStoreInfo:
    feature_store_dir: Path
    timestamps_path: Path
    metadata_path: Path | None
    timestamps: np.ndarray
    metadata: dict[str, Any]
    store_start_sec: float
    store_end_sec: float
    feature_fps: float | None


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
    dino_num_frames: int
    tracking_raw_num_frames: int
    first_sample_time: float | None
    last_sample_time: float | None


CSV_FIELDNAMES = [
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
    "dino_num_frames",
    "tracking_raw_num_frames",
    "first_sample_time",
    "last_sample_time",
]


def estimate_feature_fps(timestamps: np.ndarray) -> float | None:
    if timestamps.size < 2:
        return None
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None
    return float(1.0 / median_dt)


def load_feature_store(feature_store_dir: Path) -> FeatureStoreInfo:
    ensure_exists(feature_store_dir, "Feature store", must_be_file=False)

    timestamps_path = feature_store_dir / "timestamps.npy"
    ensure_exists(timestamps_path, "timestamps.npy", must_be_file=True)

    timestamps = np.load(timestamps_path)
    if timestamps.ndim != 1:
        raise ValueError(f"timestamps.npy deve essere 1D, shape trovata: {timestamps.shape}")
    if timestamps.size == 0:
        raise ValueError(f"timestamps.npy è vuoto: {timestamps_path}")
    timestamps = timestamps.astype(np.float64)

    if np.any(~np.isfinite(timestamps)):
        raise ValueError("timestamps.npy contiene valori non finiti")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps.npy non è ordinato in modo crescente")

    metadata_path = feature_store_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = read_json(metadata_path)
    else:
        metadata_path = None
        print("[WARN] metadata.json non trovato: uso i timestamp per ricavare il range.")

    feature_fps = None
    if metadata:
        feature_fps = metadata.get("sampling", {}).get("feature_fps")
        if feature_fps is not None:
            feature_fps = float(feature_fps)
    if feature_fps is None:
        feature_fps = estimate_feature_fps(timestamps)

    if metadata:
        video_meta = metadata.get("video", {})
        store_start_sec = float(video_meta.get("start_sec", timestamps[0]))
        store_end_sec = float(video_meta.get("end_sec", timestamps[-1]))
    else:
        store_start_sec = float(timestamps[0])
        if feature_fps is not None and feature_fps > 0:
            store_end_sec = float(timestamps[-1] + 1.0 / feature_fps)
        else:
            store_end_sec = float(timestamps[-1])

    if store_end_sec <= store_start_sec:
        raise ValueError(
            f"Range feature store non valido: {store_start_sec:.6f} -> {store_end_sec:.6f}"
        )

    return FeatureStoreInfo(
        feature_store_dir=feature_store_dir,
        timestamps_path=timestamps_path,
        metadata_path=metadata_path,
        timestamps=timestamps,
        metadata=metadata,
        store_start_sec=store_start_sec,
        store_end_sec=store_end_sec,
        feature_fps=feature_fps,
    )


# =============================================================================
# Costruzione finestre
# =============================================================================


def format_float_for_id(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def validate_args_values(
    start_sec: float,
    end_sec: float,
    window_sizes: list[float],
    stride_sec: float,
    min_store_samples: int,
) -> None:
    if start_sec < 0:
        raise ValueError(f"start_sec deve essere >= 0, trovato {start_sec}")
    if end_sec <= start_sec:
        raise ValueError(f"end_sec deve essere > start_sec, trovato {start_sec} -> {end_sec}")
    if not window_sizes:
        raise ValueError("window_sizes non può essere vuoto")
    for w in window_sizes:
        if w <= 0:
            raise ValueError(f"Tutte le window_sizes devono essere > 0, trovato {w}")
    if stride_sec <= 0:
        raise ValueError(f"stride_sec deve essere > 0, trovato {stride_sec}")
    if min_store_samples < 1:
        raise ValueError(f"min_store_samples deve essere >= 1, trovato {min_store_samples}")


def make_start_times(segment_start: float, segment_end: float, window_sec: float, stride_sec: float) -> np.ndarray:
    available = segment_end - segment_start - window_sec
    eps = 1e-9
    if available < -eps:
        return np.empty((0,), dtype=np.float64)

    num = int(math.floor(max(0.0, available) / stride_sec + eps)) + 1
    starts = segment_start + np.arange(num, dtype=np.float64) * stride_sec

    # Protezione contro piccoli errori floating point.
    starts = starts[starts + window_sec <= segment_end + 1e-7]
    return starts.astype(np.float64)


def compute_tracking_raw_num_frames(num_store_samples: int) -> int:
    """Numero di frame ball/rim grezzi da usare per una finestra.

    Replica la logica clip-level: se una clip/finestra ha al massimo 48 frame,
    il tracking usa tutti i frame; se ne ha di più, nello step di inferenza il
    tracking viene campionato uniformemente a 48 frame e poi interpolato alla
    lunghezza DINO.
    """
    return int(min(int(num_store_samples), TRACKING_MAX_FRAMES_PER_WINDOW))


def build_windows(
    timestamps: np.ndarray,
    segment_start: float,
    segment_end: float,
    window_sizes: list[float],
    stride_sec: float,
    min_store_samples: int,
) -> list[WindowRow]:
    rows: list[WindowRow] = []
    global_index = 0

    for scale_index, window_sec in enumerate(window_sizes):
        starts = make_start_times(segment_start, segment_end, window_sec, stride_sec)
        scale_id = format_float_for_id(window_sec)

        for start_time in starts:
            end_time = float(start_time + window_sec)

            # Finestra semichiusa [start_time, end_time), coerente con timestamp uniformi.
            store_start_index = int(np.searchsorted(timestamps, start_time, side="left"))
            store_end_index = int(np.searchsorted(timestamps, end_time, side="left"))
            num_store_samples = int(store_end_index - store_start_index)

            if num_store_samples < min_store_samples:
                continue

            first_sample_time = float(timestamps[store_start_index]) if num_store_samples > 0 else None
            last_sample_time = float(timestamps[store_end_index - 1]) if num_store_samples > 0 else None
            dino_num_frames = int(num_store_samples)
            tracking_raw_num_frames = compute_tracking_raw_num_frames(num_store_samples)

            rows.append(
                WindowRow(
                    window_id=f"win_{global_index:07d}_s{scale_id}",
                    scale_index=int(scale_index),
                    scale_sec=float(window_sec),
                    start_time=float(start_time),
                    end_time=end_time,
                    center_time=float(start_time + 0.5 * window_sec),
                    start_rel_sec=float(start_time - segment_start),
                    end_rel_sec=float(end_time - segment_start),
                    store_start_index=store_start_index,
                    store_end_index=store_end_index,
                    num_store_samples=num_store_samples,
                    dino_num_frames=dino_num_frames,
                    tracking_raw_num_frames=tracking_raw_num_frames,
                    first_sample_time=first_sample_time,
                    last_sample_time=last_sample_time,
                )
            )
            global_index += 1

    rows.sort(key=lambda r: (r.start_time, r.scale_sec, r.end_time))
    return rows


def write_windows_csv(path: Path, rows: list[WindowRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
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
                    "store_start_index": row.store_start_index,
                    "store_end_index": row.store_end_index,
                    "num_store_samples": row.num_store_samples,
                    "dino_num_frames": row.dino_num_frames,
                    "tracking_raw_num_frames": row.tracking_raw_num_frames,
                    "first_sample_time": "" if row.first_sample_time is None else f"{row.first_sample_time:.6f}",
                    "last_sample_time": "" if row.last_sample_time is None else f"{row.last_sample_time:.6f}",
                }
            )


def summarize_rows(rows: list[WindowRow]) -> dict[str, Any]:
    by_scale: dict[str, dict[str, Any]] = {}
    scale_counts: dict[float, list[WindowRow]] = defaultdict(list)
    for row in rows:
        scale_counts[row.scale_sec].append(row)

    for scale_sec, scale_rows in sorted(scale_counts.items(), key=lambda kv: kv[0]):
        counts = np.asarray([r.num_store_samples for r in scale_rows], dtype=np.int64)
        tracking_counts = np.asarray([r.tracking_raw_num_frames for r in scale_rows], dtype=np.int64)
        by_scale[f"{scale_sec:.6f}"] = {
            "scale_sec": float(scale_sec),
            "num_windows": int(len(scale_rows)),
            "min_store_samples": int(counts.min()) if counts.size else 0,
            "max_store_samples": int(counts.max()) if counts.size else 0,
            "mean_store_samples": float(counts.mean()) if counts.size else 0.0,
            "min_dino_frames": int(counts.min()) if counts.size else 0,
            "max_dino_frames": int(counts.max()) if counts.size else 0,
            "mean_dino_frames": float(counts.mean()) if counts.size else 0.0,
            "min_tracking_raw_frames": int(tracking_counts.min()) if tracking_counts.size else 0,
            "max_tracking_raw_frames": int(tracking_counts.max()) if tracking_counts.size else 0,
            "mean_tracking_raw_frames": float(tracking_counts.mean()) if tracking_counts.size else 0.0,
            "first_window_start": float(scale_rows[0].start_time) if scale_rows else None,
            "last_window_end": float(scale_rows[-1].end_time) if scale_rows else None,
        }

    counts_all = np.asarray([r.num_store_samples for r in rows], dtype=np.int64)
    tracking_counts_all = np.asarray([r.tracking_raw_num_frames for r in rows], dtype=np.int64)
    return {
        "num_windows": int(len(rows)),
        "min_store_samples": int(counts_all.min()) if counts_all.size else 0,
        "max_store_samples": int(counts_all.max()) if counts_all.size else 0,
        "mean_store_samples": float(counts_all.mean()) if counts_all.size else 0.0,
        "min_dino_frames": int(counts_all.min()) if counts_all.size else 0,
        "max_dino_frames": int(counts_all.max()) if counts_all.size else 0,
        "mean_dino_frames": float(counts_all.mean()) if counts_all.size else 0.0,
        "min_tracking_raw_frames": int(tracking_counts_all.min()) if tracking_counts_all.size else 0,
        "max_tracking_raw_frames": int(tracking_counts_all.max()) if tracking_counts_all.size else 0,
        "mean_tracking_raw_frames": float(tracking_counts_all.mean()) if tracking_counts_all.size else 0.0,
        "by_scale": by_scale,
    }


def write_windows_metadata(
    path: Path,
    feature_store: FeatureStoreInfo,
    args: argparse.Namespace,
    rows: list[WindowRow],
    segment_start: float,
    segment_end: float,
) -> None:
    summary = summarize_rows(rows)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_store": {
            "dir": str(feature_store.feature_store_dir),
            "timestamps_path": str(feature_store.timestamps_path),
            "metadata_path": str(feature_store.metadata_path) if feature_store.metadata_path is not None else None,
            "store_start_sec": feature_store.store_start_sec,
            "store_end_sec": feature_store.store_end_sec,
            "num_store_samples": int(feature_store.timestamps.shape[0]),
            "feature_fps": feature_store.feature_fps,
        },
        "window_generation": {
            "segment_start_sec": float(segment_start),
            "segment_end_sec": float(segment_end),
            "segment_duration_sec": float(segment_end - segment_start),
            "window_sizes_sec": [float(v) for v in args.window_sizes],
            "stride_sec": float(args.stride_sec),
            "min_store_samples": int(args.min_store_samples),
            "interval_policy": "[start_time, end_time)",
            "partial_windows": False,
            "sort_order": "start_time, scale_sec, end_time",
            "dino_policy": "usa tutti i sample della feature store contenuti nella finestra",
            "tracking_policy": "nello step di inferenza usa min(num_store_samples, 48) frame ball/rim campionati uniformemente e poi interpolati alla lunghezza DINO",
            "tracking_max_frames_per_window": TRACKING_MAX_FRAMES_PER_WINDOW,
        },
        "outputs": {
            "windows_csv": "windows_manifest.csv",
            "windows_metadata": "windows_metadata.json",
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
            "Costruisce finestre temporali virtuali da una feature store long-video. "
            "Non crea clip .mp4: salva solo un manifest CSV con intervalli e indici "
            "per accedere alle feature già estratte."
        )
    )

    parser.add_argument("--feature-store-dir", type=Path, default=defaults.VAL_FEATURE_STORE_DIR)
    parser.add_argument("--output-dir", type=Path, default=defaults.VAL_OUTPUT_DIR)

    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help=(
            "Inizio del segmento da finestrare. Se omesso, usa start_sec salvato "
            "nel metadata.json della feature store."
        ),
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help=(
            "Fine del segmento da finestrare. Se omesso, usa end_sec salvato "
            "nel metadata.json della feature store."
        ),
    )
    parser.add_argument(
        "--window-sizes",
        type=float,
        nargs="+",
        default=DEFAULT_WINDOW_SIZES_SEC,
        help=(
            "Durate delle finestre in secondi. Default train-like: "
            "0.5 0.75 1.0 1.5 2.0."
        ),
    )
    parser.add_argument("--stride-sec", type=float, default=0.25)
    parser.add_argument(
        "--min-store-samples",
        type=int,
        default=2,
        help="Scarta finestre con meno di questo numero di campioni nella feature store.",
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser


def main() -> None:
    args = make_parser().parse_args()

    args.feature_store_dir = as_path(args.feature_store_dir)
    args.output_dir = as_path(args.output_dir)
    assert args.feature_store_dir is not None
    assert args.output_dir is not None

    feature_store = load_feature_store(args.feature_store_dir)

    segment_start = float(feature_store.store_start_sec if args.start_sec is None else args.start_sec)
    segment_end = float(feature_store.store_end_sec if args.end_sec is None else args.end_sec)

    validate_args_values(
        start_sec=segment_start,
        end_sec=segment_end,
        window_sizes=args.window_sizes,
        stride_sec=args.stride_sec,
        min_store_samples=args.min_store_samples,
    )

    eps = 1e-6
    if segment_start < feature_store.store_start_sec - eps:
        raise ValueError(
            f"start_sec={segment_start:.6f} è prima dell'inizio della feature store "
            f"({feature_store.store_start_sec:.6f})."
        )
    if segment_end > feature_store.store_end_sec + eps:
        raise ValueError(
            f"end_sec={segment_end:.6f} supera la fine della feature store "
            f"({feature_store.store_end_sec:.6f})."
        )

    check_output_files(
        args.output_dir,
        output_files=["windows_manifest.csv", "windows_metadata.json"],
        overwrite=args.overwrite,
    )

    rows = build_windows(
        timestamps=feature_store.timestamps,
        segment_start=segment_start,
        segment_end=segment_end,
        window_sizes=[float(v) for v in args.window_sizes],
        stride_sec=float(args.stride_sec),
        min_store_samples=int(args.min_store_samples),
    )

    if not rows:
        raise RuntimeError(
            "Nessuna finestra generata. Controlla window_sizes, stride_sec, "
            "segmento temporale e min_store_samples."
        )

    windows_csv = args.output_dir / "windows_manifest.csv"
    windows_metadata = args.output_dir / "windows_metadata.json"

    write_windows_csv(windows_csv, rows)
    write_windows_metadata(
        path=windows_metadata,
        feature_store=feature_store,
        args=args,
        rows=rows,
        segment_start=segment_start,
        segment_end=segment_end,
    )

    summary = summarize_rows(rows)

    print("=== Build windows da feature store ===")
    print(f"feature_store_dir: {args.feature_store_dir}")
    print(f"output_dir:        {args.output_dir}")
    print(f"segmento:          {segment_start:.3f}s -> {segment_end:.3f}s")
    print(f"window_sizes:      {', '.join(str(float(v)) for v in args.window_sizes)}")
    print(f"stride_sec:        {args.stride_sec:.3f}")
    print(f"tracking max raw:  {TRACKING_MAX_FRAMES_PER_WINDOW} frame per finestra")
    print(f"num_windows:       {summary['num_windows']}")
    print("\nFinestre per scala:")
    for scale_key, scale_summary in summary["by_scale"].items():
        print(
            f"- {float(scale_key):.3f}s: {scale_summary['num_windows']} finestre, "
            f"DINO frame min/mean/max = "
            f"{scale_summary['min_dino_frames']}/"
            f"{scale_summary['mean_dino_frames']:.1f}/"
            f"{scale_summary['max_dino_frames']}, "
            f"ball-rim raw min/mean/max = "
            f"{scale_summary['min_tracking_raw_frames']}/"
            f"{scale_summary['mean_tracking_raw_frames']:.1f}/"
            f"{scale_summary['max_tracking_raw_frames']}"
        )

    print("\nOutput creati:")
    print(f"- {windows_csv}")
    print(f"- {windows_metadata}")
    print("\nWindows manifest creato correttamente.")


if __name__ == "__main__":
    main()
