from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from src.long_video import defaults
except Exception:  # noqa: BLE001
    defaults = None


# =============================================================================
# Utility
# =============================================================================


def ensure_exists(path: Path, name: str, must_be_file: bool | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} non trovato: {path}")
    if must_be_file is True and not path.is_file():
        raise FileNotFoundError(f"{name} dovrebbe essere un file: {path}")
    if must_be_file is False and not path.is_dir():
        raise NotADirectoryError(f"{name} dovrebbe essere una cartella: {path}")


def read_json(path: Path) -> dict[str, Any]:
    ensure_exists(path, path.name, must_be_file=True)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_video_info(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"FPS non valido: {fps}")
    if num_frames <= 0:
        raise RuntimeError(f"Numero frame non valido: {num_frames}")

    return {
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "duration_sec": num_frames / fps,
    }


def read_frame_by_index(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Impossibile leggere frame_index={frame_index}")
        return frame
    finally:
        cap.release()


def resize_keep_aspect(image: np.ndarray, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    scale = target_w / float(w)
    target_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def draw_text_box(
    image: np.ndarray,
    lines: list[str],
    x: int = 12,
    y: int = 12,
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = int(22 * font_scale / 0.55)
    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, tw)
    box_h = line_h * len(lines) + 12
    box_w = max_w + 16
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, dst=image)
    yy = y + line_h
    for line in lines:
        cv2.putText(image, line, (x + 8, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        yy += line_h


def draw_box(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    color: tuple[int, int, int],
) -> None:
    h, w = image.shape[:2]
    p1 = (int(round(x1 * w)), int(round(y1 * h)))
    p2 = (int(round(x2 * w)), int(round(y2 * h)))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(
        image,
        label,
        (p1[0], max(16, p1[1] - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    ensure_exists(path, path.name, must_be_file=True)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def find_active_events(events: list[dict[str, str]], timestamp: float) -> list[str]:
    active: list[str] = []
    for row in events:
        try:
            s = float(row["start_time"])
            e = float(row["end_time"])
        except Exception:
            continue
        if s <= timestamp <= e:
            label = row.get("label", row.get("pred_label", "event"))
            conf = row.get("confidence", row.get("score", ""))
            if conf != "":
                try:
                    active.append(f"{label} {float(conf):.2f}")
                except Exception:
                    active.append(f"{label} {conf}")
            else:
                active.append(str(label))
    return active


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_predictions(predictions_csv: Path | None) -> dict[str, Any]:
    if predictions_csv is None or not predictions_csv.exists():
        return {}
    rows = read_csv_rows(predictions_csv)
    counts: dict[str, int] = {}
    action_rows: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("pred_label", "")
        counts[label] = counts.get(label, 0) + 1
        if label and label != "no-action":
            try:
                conf = float(row.get("confidence", "nan"))
            except Exception:
                conf = float("nan")
            action_rows.append(
                {
                    "window_id": row.get("window_id", ""),
                    "label": label,
                    "confidence": conf,
                    "start_time": row.get("start_time", ""),
                    "end_time": row.get("end_time", ""),
                    "scale": row.get("scale_sec", row.get("scale", "")),
                }
            )
    action_rows.sort(key=lambda r: -float(r["confidence"]) if np.isfinite(r["confidence"]) else 9999)
    return {
        "num_rows": len(rows),
        "counts_by_pred_label": counts,
        "top_action_windows": action_rows[:30],
    }


# =============================================================================
# Analisi feature store
# =============================================================================


def compute_alignment_table(
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    video_fps: float,
    sample_indices: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in sample_indices:
        idx = int(idx)
        ts = float(timestamps[idx])
        stored = int(frame_indices[idx])
        expected_round = int(round(ts * video_fps))
        expected_floor = int(math.floor(ts * video_fps))
        expected_ceil = int(math.ceil(ts * video_fps))
        stored_time = stored / video_fps
        rows.append(
            {
                "sample_index": idx,
                "timestamp": ts,
                "stored_frame_index": stored,
                "stored_frame_time": stored_time,
                "delta_ms_frame_time_minus_timestamp": (stored_time - ts) * 1000.0,
                "expected_round_index": expected_round,
                "diff_vs_round": stored - expected_round,
                "expected_floor_index": expected_floor,
                "diff_vs_floor": stored - expected_floor,
                "expected_ceil_index": expected_ceil,
                "diff_vs_ceil": stored - expected_ceil,
            }
        )
    return rows


def write_alignment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(
    video_path: Path,
    output_path: Path,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    sample_indices: np.ndarray,
    video_fps: float,
    primitives_by_source: dict[str, dict[str, np.ndarray]],
    events: list[dict[str, str]],
    tile_width: int,
    cols: int,
) -> None:
    tiles: list[np.ndarray] = []
    for idx in sample_indices:
        idx = int(idx)
        ts = float(timestamps[idx])
        frame_index = int(frame_indices[idx])
        frame = read_frame_by_index(video_path, frame_index)

        for source_name, primitives in primitives_by_source.items():
            prefix = source_name.replace("yolo_", "")
            if primitives.get("ball_detected", np.zeros(1))[idx] >= 0.5:
                draw_box(
                    frame,
                    float(primitives["ball_x1"][idx]),
                    float(primitives["ball_y1"][idx]),
                    float(primitives["ball_x2"][idx]),
                    float(primitives["ball_y2"][idx]),
                    f"ball {prefix} {float(primitives['ball_conf'][idx]):.2f}",
                    (0, 255, 255),
                )
            if primitives.get("rim_detected", np.zeros(1))[idx] >= 0.5:
                draw_box(
                    frame,
                    float(primitives["rim_x1"][idx]),
                    float(primitives["rim_y1"][idx]),
                    float(primitives["rim_x2"][idx]),
                    float(primitives["rim_y2"][idx]),
                    f"rim {prefix} {float(primitives['rim_conf'][idx]):.2f}",
                    (255, 0, 255),
                )

        active = find_active_events(events, ts)
        lines = [
            f"sample={idx}",
            f"t={ts:.3f}s  frame={frame_index}",
            f"frame/fps={frame_index / video_fps:.3f}s  delta={(frame_index / video_fps - ts) * 1000:.1f}ms",
        ]
        if active:
            lines.append("event: " + ", ".join(active[:2]))
        draw_text_box(frame, lines)
        tiles.append(resize_keep_aspect(frame, tile_width))

    if not tiles:
        return

    max_h = max(t.shape[0] for t in tiles)
    padded: list[np.ndarray] = []
    for tile in tiles:
        h, w = tile.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)

    cols = max(1, int(cols))
    rows = int(math.ceil(len(padded) / cols))
    blank = np.zeros_like(padded[0])
    grid_rows: list[np.ndarray] = []
    for r in range(rows):
        row_tiles = padded[r * cols : (r + 1) * cols]
        while len(row_tiles) < cols:
            row_tiles.append(blank.copy())
        grid_rows.append(np.hstack(row_tiles))
    sheet = np.vstack(grid_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


# =============================================================================
# CLI
# =============================================================================


def make_parser() -> argparse.ArgumentParser:
    val_video_path = getattr(defaults, "VAL_VIDEO_PATH", Path("data/datasets/dataset_basket_v1/videos/PrimaParte.mp4")) if defaults else Path("data/datasets/dataset_basket_v1/videos/PrimaParte.mp4")
    val_feature_dir = getattr(defaults, "VAL_DEBUG_FEATURE_STORE_DIR", Path("data/features_long/primaparte_0215_0245_exp46_debug")) if defaults else Path("data/features_long/primaparte_0215_0245_exp46_debug")
    val_output_dir = getattr(defaults, "VAL_DEBUG_OUTPUT_DIR", Path("outputs/long_video/primaparte_0215_0245_exp46_debug")) if defaults else Path("outputs/long_video/primaparte_0215_0245_exp46_debug")

    parser = argparse.ArgumentParser(
        description=(
            "Diagnostica l'allineamento tra feature store long-video, timestamp, frame reali "
            "del video e detection YOLO salvate. Genera un CSV di controllo e una contact sheet."
        )
    )
    parser.add_argument("--input-video", type=Path, default=val_video_path)
    parser.add_argument("--feature-store-dir", type=Path, default=val_feature_dir)
    parser.add_argument("--output-dir", type=Path, default=val_output_dir)
    parser.add_argument("--events-csv", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--sample-times",
        type=float,
        nargs="+",
        default=None,
        help="Timestamp assoluti da controllare. Se omessi, campiona uniformemente il segmento.",
    )
    parser.add_argument(
        "--yolo-source",
        choices=["yolo_v1", "yolo_v2", "both", "none"],
        default="both",
    )
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--cols", type=int, default=2)
    return parser


def main() -> None:
    args = make_parser().parse_args()

    ensure_exists(args.input_video, "Video input", must_be_file=True)
    ensure_exists(args.feature_store_dir, "Feature store", must_be_file=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.feature_store_dir / "metadata.json"
    timestamps_path = args.feature_store_dir / "timestamps.npy"
    frame_indices_path = args.feature_store_dir / "frame_indices.npy"
    dino_path = args.feature_store_dir / "dinov3_features.npy"

    ensure_exists(metadata_path, "metadata.json", must_be_file=True)
    ensure_exists(timestamps_path, "timestamps.npy", must_be_file=True)
    ensure_exists(frame_indices_path, "frame_indices.npy", must_be_file=True)

    metadata = read_json(metadata_path)
    timestamps = np.load(timestamps_path).astype(np.float64)
    frame_indices = np.load(frame_indices_path).astype(np.int64)
    video_info = get_video_info(args.input_video)

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps.npy deve essere 1D, trovato {timestamps.shape}")
    if frame_indices.ndim != 1:
        raise ValueError(f"frame_indices.npy deve essere 1D, trovato {frame_indices.shape}")
    if len(timestamps) != len(frame_indices):
        raise ValueError(f"timestamps e frame_indices hanno lunghezze diverse: {len(timestamps)} vs {len(frame_indices)}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps.npy non è strettamente crescente")
    if np.any(np.diff(frame_indices) < 0):
        raise ValueError("frame_indices.npy non è monotono crescente")

    dino_shape = None
    if dino_path.exists():
        dino = np.load(dino_path, mmap_mode="r")
        dino_shape = tuple(int(v) for v in dino.shape)
        if dino.shape[0] != len(timestamps):
            raise ValueError(f"dinov3_features ha N diverso: {dino.shape[0]} vs {len(timestamps)}")

    if args.sample_times:
        sample_indices = []
        for t in args.sample_times:
            idx = int(np.argmin(np.abs(timestamps - float(t))))
            sample_indices.append(idx)
        sample_indices = np.asarray(sorted(set(sample_indices)), dtype=np.int64)
    else:
        n = max(1, min(int(args.num_samples), len(timestamps)))
        sample_indices = np.linspace(0, len(timestamps) - 1, n, dtype=np.int64)

    align_rows = compute_alignment_table(
        timestamps=timestamps,
        frame_indices=frame_indices,
        video_fps=float(video_info["fps"]),
        sample_indices=sample_indices,
    )
    alignment_csv = args.output_dir / "feature_store_alignment_report.csv"
    write_alignment_csv(alignment_csv, align_rows)

    primitives_by_source: dict[str, dict[str, np.ndarray]] = {}
    if args.yolo_source in {"yolo_v1", "both"}:
        p = args.feature_store_dir / "yolo_v1_primitives.npz"
        if p.exists():
            primitives_by_source["yolo_v1"] = load_npz_dict(p)
    if args.yolo_source in {"yolo_v2", "both"}:
        p = args.feature_store_dir / "yolo_v2_primitives.npz"
        if p.exists():
            primitives_by_source["yolo_v2"] = load_npz_dict(p)

    # Controllo lunghezze primitive.
    primitive_summary: dict[str, Any] = {}
    for source, prim in primitives_by_source.items():
        source_summary: dict[str, Any] = {"num_fields": len(prim)}
        for key in ["ball_detected", "rim_detected", "both_detected"]:
            if key in prim:
                if len(prim[key]) != len(timestamps):
                    raise ValueError(f"{source}.{key} ha lunghezza {len(prim[key])}, attesa {len(timestamps)}")
                source_summary[f"{key}_mean"] = float(np.mean(prim[key].astype(np.float32)))
                source_summary[f"{key}_count"] = int(np.sum(prim[key].astype(np.float32) >= 0.5))
        primitive_summary[source] = source_summary

    events = read_csv_rows(args.events_csv)
    predictions_summary = summarize_predictions(args.predictions_csv)

    contact_sheet = args.output_dir / "feature_store_alignment_contact_sheet.jpg"
    make_contact_sheet(
        video_path=args.input_video,
        output_path=contact_sheet,
        timestamps=timestamps,
        frame_indices=frame_indices,
        sample_indices=sample_indices,
        video_fps=float(video_info["fps"]),
        primitives_by_source=primitives_by_source,
        events=events,
        tile_width=int(args.tile_width),
        cols=int(args.cols),
    )

    # Statistiche globali sull'errore timestamp <-> frame_index/fps.
    frame_times = frame_indices.astype(np.float64) / float(video_info["fps"])
    delta_ms = (frame_times - timestamps) * 1000.0
    max_abs_delta_ms = float(np.max(np.abs(delta_ms))) if delta_ms.size else 0.0
    mean_abs_delta_ms = float(np.mean(np.abs(delta_ms))) if delta_ms.size else 0.0
    expected_half_frame_ms = 500.0 / float(video_info["fps"])

    report = {
        "input_video": str(args.input_video),
        "feature_store_dir": str(args.feature_store_dir),
        "video_info": video_info,
        "metadata_video": metadata.get("video", {}),
        "metadata_sampling": metadata.get("sampling", {}),
        "num_timestamps": int(len(timestamps)),
        "first_timestamp": float(timestamps[0]),
        "last_timestamp": float(timestamps[-1]),
        "first_frame_index": int(frame_indices[0]),
        "last_frame_index": int(frame_indices[-1]),
        "dino_shape": dino_shape,
        "timestamp_frame_alignment": {
            "mean_abs_delta_ms": mean_abs_delta_ms,
            "max_abs_delta_ms": max_abs_delta_ms,
            "expected_half_frame_ms": expected_half_frame_ms,
            "ok_if_max_abs_delta_le_half_frame_plus_tolerance": bool(max_abs_delta_ms <= expected_half_frame_ms + 2.0),
        },
        "primitive_summary": primitive_summary,
        "predictions_summary": predictions_summary,
        "events_count": len(events),
        "outputs": {
            "alignment_csv": str(alignment_csv),
            "contact_sheet": str(contact_sheet),
        },
    }
    report_path = args.output_dir / "feature_store_alignment_summary.json"
    write_json(report_path, report)

    print("\n=== Feature store alignment debug ===")
    print(f"video:               {args.input_video}")
    print(f"feature_store_dir:   {args.feature_store_dir}")
    print(f"video fps:           {video_info['fps']:.6f}")
    print(f"num timestamps:      {len(timestamps)}")
    print(f"range timestamps:    {timestamps[0]:.6f}s -> {timestamps[-1]:.6f}s")
    print(f"range frame_indices: {frame_indices[0]} -> {frame_indices[-1]}")
    print(f"DINO shape:          {dino_shape}")
    print(f"mean abs delta:      {mean_abs_delta_ms:.3f} ms")
    print(f"max abs delta:       {max_abs_delta_ms:.3f} ms")
    print(f"half frame:          {expected_half_frame_ms:.3f} ms")
    print(f"alignment OK:        {report['timestamp_frame_alignment']['ok_if_max_abs_delta_le_half_frame_plus_tolerance']}")
    print("\nPrimitive summary:")
    for source, source_summary in primitive_summary.items():
        print(f"  {source}: {source_summary}")
    if predictions_summary:
        print("\nPrediction counts:")
        print(predictions_summary.get("counts_by_pred_label", {}))
        print("Top action windows salvate nel JSON.")
    print("\nOutput creati:")
    print(f"- {alignment_csv}")
    print(f"- {contact_sheet}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
