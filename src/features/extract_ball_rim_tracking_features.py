from pathlib import Path
import argparse
import csv
import json
import math
import sys
import traceback
from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


SHOT_LABELS = {
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
}

CONTEXT_LABELS = {
    "passaggio",
    "idle",
    "non-gioco",
}

ALL_LABELS = sorted(SHOT_LABELS | CONTEXT_LABELS)
ALL_SPLITS = ["train", "val", "test"]

DEFAULT_DATASET_ROOT = "data/datasets/dataset_basket_v1"
DEFAULT_MANIFEST = "data/datasets/dataset_basket_v1/manifest.csv"
DEFAULT_YOLO_WEIGHTS = (
    "runs/detect/outputs/ball_rim_detector/"
    "yolo11m_1280_v2/weights/best.pt"
)
DEFAULT_OUTPUT_DIR = "data/features/ball_rim_tracking_features_clip_complete"


TRACKING_FEATURE_NAMES = [
    "ball_detect_rate",
    "rim_detect_rate",
    "both_detect_rate",
    "ball_conf_mean",
    "ball_conf_max",
    "rim_conf_mean",
    "rim_conf_max",
    "ball_area_mean",
    "ball_area_max",
    "rim_area_mean",
    "rim_area_max",
    "ball_rim_dist_min",
    "ball_rim_dist_mean",
    "ball_rim_dist_std",
    "ball_rim_dist_last",
    "ball_rim_dist_last_third_min",
    "ball_rim_dist_last_third_mean",
    "ball_near_rim_rate",
    "ball_near_rim_last_third_rate",
    "ball_above_rim_rate",
    "ball_below_rim_rate",
    "ball_above_rim_last",
    "dx_mean",
    "dx_std",
    "dx_last",
    "dy_mean",
    "dy_std",
    "dy_last",
    "abs_dx_min",
    "abs_dy_min",
    "ball_velocity_mean",
    "ball_velocity_max",
    "ball_velocity_last_third_mean",
    "ball_vx_mean",
    "ball_vy_mean",
    "ball_vy_last_third_mean",
    "rim_center_std",
    "rim_area_std",
    "ball_crosses_rim_y",
]


TEMPORAL_TRACKING_FEATURE_NAMES = [
    "t_rel",
    "ball_detected",
    "rim_detected",
    "both_detected",
    "ball_conf",
    "rim_conf",
    "ball_xc",
    "ball_yc",
    "ball_w",
    "ball_h",
    "ball_area",
    "rim_xc",
    "rim_yc",
    "rim_w",
    "rim_h",
    "rim_area",
    "dx",
    "dy",
    "ball_rim_dist",
    "ball_near_rim",
    "ball_above_rim",
    "ball_below_rim",
    "ball_vx",
    "ball_vy",
    "ball_speed",
    "ball_ax",
    "ball_ay",
    "ball_acceleration",
    "ball_rim_dist_delta",
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


def safe_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    return float(value)


def mean_or_default(values, default=0.0):
    values = [safe_float(v) for v in values if v is not None]
    if not values:
        return default
    return float(np.mean(values))


def std_or_default(values, default=0.0):
    values = [safe_float(v) for v in values if v is not None]
    if len(values) <= 1:
        return default
    return float(np.std(values))


def min_or_default(values, default=1.0):
    values = [safe_float(v) for v in values if v is not None]
    if not values:
        return default
    return float(np.min(values))


def max_or_default(values, default=0.0):
    values = [safe_float(v) for v in values if v is not None]
    if not values:
        return default
    return float(np.max(values))


def last_or_default(values, default=0.0):
    values = [v for v in values if v is not None]
    if not values:
        return default
    return safe_float(values[-1], default=default)


def normalize_clip_key(path_value) -> str:
    """
    Normalizza il path di una clip in una chiave stabile per associare
    feature video, feature aggregate e sequenze temporali.
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


def get_model_names(model):
    names = model.names

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(names)}

    raise TypeError(f"Formato model.names non supportato: {type(names)}")


def resolve_class_id(model, class_name, explicit_id=None):
    names = get_model_names(model)

    if explicit_id is not None:
        explicit_id = int(explicit_id)
        if explicit_id not in names:
            raise ValueError(
                f"Class id esplicito {explicit_id} non presente nel modello. "
                f"Classi modello: {names}"
            )
        return explicit_id

    normalized = {name.lower(): class_id for class_id, name in names.items()}
    class_name_lower = class_name.lower()

    if class_name_lower not in normalized:
        raise ValueError(
            f"Classe '{class_name}' non trovata nel modello YOLO. "
            f"Classi disponibili: {names}. "
            f"Usa --ball-class-id e --rim-class-id se necessario."
        )

    return normalized[class_name_lower]


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


def detection_to_dict(box_xyxy, conf, frame_width, frame_height):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    x1 = max(0.0, min(x1, frame_width - 1))
    x2 = max(0.0, min(x2, frame_width - 1))
    y1 = max(0.0, min(y1, frame_height - 1))
    y2 = max(0.0, min(y2, frame_height - 1))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return {
        "detected": 1,
        "conf": float(conf),
        "xc": xc / frame_width if frame_width > 0 else 0.0,
        "yc": yc / frame_height if frame_height > 0 else 0.0,
        "w": bw / frame_width if frame_width > 0 else 0.0,
        "h": bh / frame_height if frame_height > 0 else 0.0,
        "area": (bw * bh) / (frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0,
    }


def empty_detection():
    return {
        "detected": 0,
        "conf": 0.0,
        "xc": 0.0,
        "yc": 0.0,
        "w": 0.0,
        "h": 0.0,
        "area": 0.0,
    }


def parse_yolo_result(result, ball_class_id, rim_class_id, frame_width, frame_height):
    best = {
        "ball": None,
        "rim": None,
    }

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return empty_detection(), empty_detection()

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)

    for box, conf, class_id in zip(xyxy, confs, classes):
        if class_id == ball_class_id:
            current = best["ball"]
            if current is None or conf > current["conf"]:
                best["ball"] = detection_to_dict(box, conf, frame_width, frame_height)

        elif class_id == rim_class_id:
            current = best["rim"]
            if current is None or conf > current["conf"]:
                best["rim"] = detection_to_dict(box, conf, frame_width, frame_height)

    ball = best["ball"] if best["ball"] is not None else empty_detection()
    rim = best["rim"] if best["rim"] is not None else empty_detection()

    return ball, rim


def compute_pair_features(ball, rim, near_threshold):
    both_detected = int(ball["detected"] == 1 and rim["detected"] == 1)

    if not both_detected:
        return {
            "both_detected": 0,
            "dx": 0.0,
            "dy": 0.0,
            "dist": 1.0,
            "ball_above_rim": 0,
            "ball_below_rim": 0,
            "ball_near_rim": 0,
        }

    dx = ball["xc"] - rim["xc"]
    dy = ball["yc"] - rim["yc"]
    dist = math.sqrt(dx * dx + dy * dy)

    return {
        "both_detected": 1,
        "dx": dx,
        "dy": dy,
        "dist": dist,
        "ball_above_rim": int(ball["yc"] < rim["yc"]),
        "ball_below_rim": int(ball["yc"] > rim["yc"]),
        "ball_near_rim": int(dist <= near_threshold),
    }


def compute_velocities(frame_rows, fps):
    detected = [
        row
        for row in frame_rows
        if int(row["ball_detected"]) == 1
    ]

    velocities = []
    vx_values = []
    vy_values = []

    for prev, curr in zip(detected[:-1], detected[1:]):
        delta_frames = int(curr["frame_idx"]) - int(prev["frame_idx"])

        if delta_frames <= 0:
            continue

        dt = delta_frames / fps if fps > 0 else float(delta_frames)

        dx = float(curr["ball_xc"]) - float(prev["ball_xc"])
        dy = float(curr["ball_yc"]) - float(prev["ball_yc"])

        vx = dx / dt
        vy = dy / dt
        v = math.sqrt(vx * vx + vy * vy)

        velocities.append(
            {
                "frame_order": int(curr["frame_order"]),
                "v": v,
                "vx": vx,
                "vy": vy,
            }
        )

        vx_values.append(vx)
        vy_values.append(vy)

    return velocities, vx_values, vy_values


def aggregate_clip_features(frame_rows, fps, near_threshold):
    n = len(frame_rows)

    if n == 0:
        return {name: 0.0 for name in TRACKING_FEATURE_NAMES}

    last_third_start = int(math.floor(n * 2.0 / 3.0))

    ball_rows = [r for r in frame_rows if int(r["ball_detected"]) == 1]
    rim_rows = [r for r in frame_rows if int(r["rim_detected"]) == 1]
    both_rows = [r for r in frame_rows if int(r["both_detected"]) == 1]
    both_last_rows = [r for r in both_rows if int(r["frame_order"]) >= last_third_start]

    ball_conf = [float(r["ball_conf"]) for r in ball_rows]
    rim_conf = [float(r["rim_conf"]) for r in rim_rows]
    ball_area = [float(r["ball_area"]) for r in ball_rows]
    rim_area = [float(r["rim_area"]) for r in rim_rows]

    distances = [float(r["ball_rim_dist"]) for r in both_rows]
    distances_last = [float(r["ball_rim_dist"]) for r in both_last_rows]

    dx_values = [float(r["dx"]) for r in both_rows]
    dy_values = [float(r["dy"]) for r in both_rows]
    abs_dx_values = [abs(float(r["dx"])) for r in both_rows]
    abs_dy_values = [abs(float(r["dy"])) for r in both_rows]

    near_values = [int(r["ball_near_rim"]) for r in both_rows]
    near_last_values = [int(r["ball_near_rim"]) for r in both_last_rows]

    above_values = [int(r["ball_above_rim"]) for r in both_rows]
    below_values = [int(r["ball_below_rim"]) for r in both_rows]

    rim_x = [float(r["rim_xc"]) for r in rim_rows]
    rim_y = [float(r["rim_yc"]) for r in rim_rows]
    rim_center_std = math.sqrt(
        std_or_default(rim_x, default=0.0) ** 2
        + std_or_default(rim_y, default=0.0) ** 2
    )

    velocities, vx_values, vy_values = compute_velocities(frame_rows, fps)
    v_values = [v["v"] for v in velocities]
    v_last_values = [v["v"] for v in velocities if int(v["frame_order"]) >= last_third_start]
    vy_last_values = [v["vy"] for v in velocities if int(v["frame_order"]) >= last_third_start]

    # Attraversamento verticale rispetto al rim:
    # se il segno di dy cambia, la palla è passata da sopra a sotto o viceversa.
    crosses_rim_y = 0
    if len(dy_values) >= 2:
        signs = [np.sign(v) for v in dy_values if abs(v) > 1e-6]
        for a, b in zip(signs[:-1], signs[1:]):
            if a != b:
                crosses_rim_y = 1
                break

    features = {
        "ball_detect_rate": len(ball_rows) / n,
        "rim_detect_rate": len(rim_rows) / n,
        "both_detect_rate": len(both_rows) / n,

        "ball_conf_mean": mean_or_default(ball_conf, default=0.0),
        "ball_conf_max": max_or_default(ball_conf, default=0.0),
        "rim_conf_mean": mean_or_default(rim_conf, default=0.0),
        "rim_conf_max": max_or_default(rim_conf, default=0.0),

        "ball_area_mean": mean_or_default(ball_area, default=0.0),
        "ball_area_max": max_or_default(ball_area, default=0.0),
        "rim_area_mean": mean_or_default(rim_area, default=0.0),
        "rim_area_max": max_or_default(rim_area, default=0.0),

        "ball_rim_dist_min": min_or_default(distances, default=1.0),
        "ball_rim_dist_mean": mean_or_default(distances, default=1.0),
        "ball_rim_dist_std": std_or_default(distances, default=0.0),
        "ball_rim_dist_last": last_or_default(distances, default=1.0),
        "ball_rim_dist_last_third_min": min_or_default(distances_last, default=1.0),
        "ball_rim_dist_last_third_mean": mean_or_default(distances_last, default=1.0),

        "ball_near_rim_rate": mean_or_default(near_values, default=0.0),
        "ball_near_rim_last_third_rate": mean_or_default(near_last_values, default=0.0),

        "ball_above_rim_rate": mean_or_default(above_values, default=0.0),
        "ball_below_rim_rate": mean_or_default(below_values, default=0.0),
        "ball_above_rim_last": last_or_default(above_values, default=0.0),

        "dx_mean": mean_or_default(dx_values, default=0.0),
        "dx_std": std_or_default(dx_values, default=0.0),
        "dx_last": last_or_default(dx_values, default=0.0),

        "dy_mean": mean_or_default(dy_values, default=0.0),
        "dy_std": std_or_default(dy_values, default=0.0),
        "dy_last": last_or_default(dy_values, default=0.0),

        "abs_dx_min": min_or_default(abs_dx_values, default=1.0),
        "abs_dy_min": min_or_default(abs_dy_values, default=1.0),

        "ball_velocity_mean": mean_or_default(v_values, default=0.0),
        "ball_velocity_max": max_or_default(v_values, default=0.0),
        "ball_velocity_last_third_mean": mean_or_default(v_last_values, default=0.0),
        "ball_vx_mean": mean_or_default(vx_values, default=0.0),
        "ball_vy_mean": mean_or_default(vy_values, default=0.0),
        "ball_vy_last_third_mean": mean_or_default(vy_last_values, default=0.0),

        "rim_center_std": rim_center_std,
        "rim_area_std": std_or_default(rim_area, default=0.0),

        "ball_crosses_rim_y": float(crosses_rim_y),
    }

    return {name: safe_float(features.get(name, 0.0), default=0.0) for name in TRACKING_FEATURE_NAMES}


def compute_temporal_sequence_features(frame_rows, fps):
    """
    Converte le detection per-frame in una sequenza [S, K] di feature temporali.

    A differenza delle 39 feature aggregate per clip, questa rappresentazione
    conserva l'ordine temporale e include posizione, distanza palla-canestro,
    velocità e accelerazione della palla.
    """
    rows = sorted(frame_rows, key=lambda r: int(r["frame_order"]))

    if not rows:
        return np.zeros((0, len(TEMPORAL_TRACKING_FEATURE_NAMES)), dtype=np.float32)

    sequence_rows = []
    prev_row = None
    prev_vx = None
    prev_vy = None
    prev_speed = None

    for row in rows:
        ball_detected = int(row["ball_detected"]) == 1
        rim_detected = int(row["rim_detected"]) == 1
        both_detected = int(row["both_detected"]) == 1

        ball_vx = 0.0
        ball_vy = 0.0
        ball_speed = 0.0
        ball_ax = 0.0
        ball_ay = 0.0
        ball_acceleration = 0.0
        dist_delta = 0.0
        current_velocity_valid = False

        if prev_row is not None:
            delta_frames = int(row["frame_idx"]) - int(prev_row["frame_idx"])
            dt = delta_frames / fps if fps > 0 and delta_frames > 0 else 0.0

            prev_ball_detected = int(prev_row["ball_detected"]) == 1
            prev_both_detected = int(prev_row["both_detected"]) == 1

            if dt > 0 and ball_detected and prev_ball_detected:
                ball_vx = (float(row["ball_xc"]) - float(prev_row["ball_xc"])) / dt
                ball_vy = (float(row["ball_yc"]) - float(prev_row["ball_yc"])) / dt
                ball_speed = math.sqrt(ball_vx * ball_vx + ball_vy * ball_vy)
                current_velocity_valid = True

                if prev_vx is not None and prev_vy is not None and prev_speed is not None:
                    ball_ax = (ball_vx - prev_vx) / dt
                    ball_ay = (ball_vy - prev_vy) / dt
                    ball_acceleration = (ball_speed - prev_speed) / dt

            if dt > 0 and both_detected and prev_both_detected:
                dist_delta = (float(row["ball_rim_dist"]) - float(prev_row["ball_rim_dist"])) / dt

        feature_values = {
            "t_rel": float(row["t_rel"]),
            "ball_detected": float(row["ball_detected"]),
            "rim_detected": float(row["rim_detected"]),
            "both_detected": float(row["both_detected"]),
            "ball_conf": float(row["ball_conf"]),
            "rim_conf": float(row["rim_conf"]),
            "ball_xc": float(row["ball_xc"]),
            "ball_yc": float(row["ball_yc"]),
            "ball_w": float(row["ball_w"]),
            "ball_h": float(row["ball_h"]),
            "ball_area": float(row["ball_area"]),
            "rim_xc": float(row["rim_xc"]),
            "rim_yc": float(row["rim_yc"]),
            "rim_w": float(row["rim_w"]),
            "rim_h": float(row["rim_h"]),
            "rim_area": float(row["rim_area"]),
            "dx": float(row["dx"]),
            "dy": float(row["dy"]),
            "ball_rim_dist": float(row["ball_rim_dist"]),
            "ball_near_rim": float(row["ball_near_rim"]),
            "ball_above_rim": float(row["ball_above_rim"]),
            "ball_below_rim": float(row["ball_below_rim"]),
            "ball_vx": ball_vx,
            "ball_vy": ball_vy,
            "ball_speed": ball_speed,
            "ball_ax": ball_ax,
            "ball_ay": ball_ay,
            "ball_acceleration": ball_acceleration,
            "ball_rim_dist_delta": dist_delta,
        }

        sequence_rows.append([
            safe_float(feature_values[name], default=0.0)
            for name in TEMPORAL_TRACKING_FEATURE_NAMES
        ])

        if current_velocity_valid:
            prev_vx = ball_vx
            prev_vy = ball_vy
            prev_speed = ball_speed

        prev_row = row

    return np.asarray(sequence_rows, dtype=np.float32)


def process_clip(
    row,
    model,
    ball_class_id,
    rim_class_id,
    args,
):
    video_path = row["video_path"]

    frame_count, fps, width, height = get_video_metadata(video_path)

    frame_indices = build_frame_indices(
        frame_count=frame_count,
        num_frames=args.num_frames,
        sample_mode=args.sample_mode,
        last_ratio=args.last_ratio,
    )

    frames, valid_indices = read_frames(video_path, frame_indices)

    if not frames:
        raise RuntimeError(f"Nessun frame leggibile per clip: {video_path}")

    frame_rows = []

    for start in range(0, len(frames), args.batch_size):
        batch_frames = frames[start:start + args.batch_size]
        batch_indices = valid_indices[start:start + args.batch_size]

        results = model.predict(
            source=batch_frames,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            classes=[ball_class_id, rim_class_id],
            verbose=False,
        )

        for local_idx, result in enumerate(results):
            frame_order = start + local_idx
            frame_idx = int(batch_indices[local_idx])

            frame_height, frame_width = batch_frames[local_idx].shape[:2]

            ball, rim = parse_yolo_result(
                result=result,
                ball_class_id=ball_class_id,
                rim_class_id=rim_class_id,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            pair = compute_pair_features(
                ball=ball,
                rim=rim,
                near_threshold=args.near_threshold,
            )

            time_sec = frame_idx / fps if fps > 0 else 0.0
            t_rel = frame_idx / max(1, frame_count - 1)

            frame_row = {
                "clip_id": row["clip_id"],
                "split": row["split"],
                "label": row["label"],
                "path": row["path"],
                "frame_order": frame_order,
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "t_rel": t_rel,
                "width": frame_width,
                "height": frame_height,

                "ball_detected": ball["detected"],
                "ball_conf": ball["conf"],
                "ball_xc": ball["xc"],
                "ball_yc": ball["yc"],
                "ball_w": ball["w"],
                "ball_h": ball["h"],
                "ball_area": ball["area"],

                "rim_detected": rim["detected"],
                "rim_conf": rim["conf"],
                "rim_xc": rim["xc"],
                "rim_yc": rim["yc"],
                "rim_w": rim["w"],
                "rim_h": rim["h"],
                "rim_area": rim["area"],

                "both_detected": pair["both_detected"],
                "dx": pair["dx"],
                "dy": pair["dy"],
                "ball_rim_dist": pair["dist"],
                "ball_above_rim": pair["ball_above_rim"],
                "ball_below_rim": pair["ball_below_rim"],
                "ball_near_rim": pair["ball_near_rim"],
            }

            frame_rows.append(frame_row)

    clip_features = aggregate_clip_features(
        frame_rows=frame_rows,
        fps=fps,
        near_threshold=args.near_threshold,
    )

    clip_row = {
        "clip_id": row["clip_id"],
        "split": row["split"],
        "label": row["label"],
        "path": row["path"],
        "video_frames": frame_count,
        "fps": fps,
        "sampled_frames": len(frame_rows),
        "video_width": width,
        "video_height": height,
    }

    clip_row.update(clip_features)

    return clip_row, frame_rows


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


def write_temporal_sequences(output_dir: Path, sequence_entries):
    """
    Salva le sequenze tracking in formato NPZ più un indice JSON path -> array.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "tracking_sequences.npz"
    index_path = output_dir / "tracking_sequence_index.json"
    feature_names_path = output_dir / "tracking_sequence_feature_names.json"

    arrays = {}
    index = {
        "type": "temporal_sequence",
        "npz_path": str(npz_path),
        "feature_names": TEMPORAL_TRACKING_FEATURE_NAMES,
        "num_features": len(TEMPORAL_TRACKING_FEATURE_NAMES),
        "sequences": {},
    }

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
                "feature_names": TEMPORAL_TRACKING_FEATURE_NAMES,
                "num_features": len(TEMPORAL_TRACKING_FEATURE_NAMES),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return npz_path, index_path, feature_names_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estrae feature di tracking palla/canestro da tutte le clip del dataset "
            "usando un detector YOLO addestrato su ball/rim. Di default processa "
            "train, val e test e tutte le 9 classi."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root del dataset video.",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST,
        help="Path al manifest.csv del dataset.",
    )

    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=DEFAULT_YOLO_WEIGHTS,
        help="Path al best.pt del detector ball/rim.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella in cui salvare tracking_features.csv e gli altri output.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=ALL_SPLITS,
        choices=["train", "val", "test"],
        help="Split da processare. Default: train val test.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=ALL_LABELS,
        help="Label da includere. Default: tutte le 9 classi del dataset.",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=48,
        help=(
            "Numero di frame campionati per clip. "
            "Default: 48. Usa 0 per processare tutti i frame di ogni clip."
        ),
    )

    parser.add_argument(
        "--sample-mode",
        type=str,
        default="uniform",
        choices=["uniform", "last"],
        help=(
            "uniform = campiona su tutta la clip; "
            "last = campiona solo nella parte finale definita da --last-ratio."
        ),
    )

    parser.add_argument(
        "--last-ratio",
        type=float,
        default=0.50,
        help="Usato solo con --sample-mode last. 0.50 = usa l'ultima metà della clip.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Risoluzione YOLO in inferenza.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Soglia di confidenza YOLO. Bassa per non perdere la palla.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="Soglia IoU per NMS.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device YOLO. Esempio: 0 oppure cpu.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Numero di frame processati insieme da YOLO.",
    )

    parser.add_argument(
        "--ball-class-name",
        type=str,
        default="ball",
        help="Nome classe palla nel modello YOLO.",
    )

    parser.add_argument(
        "--rim-class-name",
        type=str,
        default="rim",
        help="Nome classe canestro/ferro nel modello YOLO.",
    )

    parser.add_argument(
        "--ball-class-id",
        type=int,
        default=None,
        help="ID classe ball. Se non indicato, viene ricavato dal nome.",
    )

    parser.add_argument(
        "--rim-class-id",
        type=int,
        default=None,
        help="ID classe rim. Se non indicato, viene ricavato dal nome.",
    )

    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.12,
        help=(
            "Distanza normalizzata sotto cui la palla è considerata vicina al rim. "
            "La distanza è calcolata sui centri normalizzati."
        ),
    )

    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limita il numero di clip processate. Utile per test rapido.",
    )

    parser.add_argument(
        "--save-per-frame",
        action="store_true",
        help="Salva anche per_frame_detections.csv. Può essere grande.",
    )

    temporal_group = parser.add_mutually_exclusive_group()
    temporal_group.add_argument(
        "--save-temporal-sequences",
        dest="save_temporal_sequences",
        action="store_true",
        help=(
            "Salva tracking_sequences.npz e tracking_sequence_index.json, "
            "cioè feature palla/canestro per-frame utilizzabili come tracking temporale."
        ),
    )
    temporal_group.add_argument(
        "--no-save-temporal-sequences",
        dest="save_temporal_sequences",
        action="store_false",
        help="Disabilita il salvataggio delle sequenze temporali.",
    )
    parser.set_defaults(save_temporal_sequences=True)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permette di sovrascrivere file già esistenti.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    yolo_weights = Path(args.yolo_weights)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "extract_tracking_results.txt"
    tracking_csv_path = output_dir / "tracking_features.csv"
    per_frame_csv_path = output_dir / "per_frame_detections.csv"
    feature_names_path = output_dir / "tracking_feature_names.json"
    tracking_sequences_npz_path = output_dir / "tracking_sequences.npz"
    tracking_sequence_index_path = output_dir / "tracking_sequence_index.json"

    if tracking_csv_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"File già esistente: {tracking_csv_path}. "
            "Usa --overwrite per rigenerarlo."
        )

    if args.save_temporal_sequences and tracking_sequences_npz_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"File già esistente: {tracking_sequences_npz_path}. "
            "Usa --overwrite per rigenerarlo."
        )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(results_path, "w", encoding="utf-8") as results_file:
        sys.stdout = Tee(original_stdout, results_file)
        sys.stderr = Tee(original_stderr, results_file)

        try:
            print(f"File log: {results_path}")
            print("\n# Configurazione")
            for key, value in vars(args).items():
                print(f"{key}: {value}")

            print("\n# Controllo path")
            print(f"dataset_root: {dataset_root}")
            print(f"manifest: {manifest_path}")
            print(f"yolo_weights: {yolo_weights}")
            print(f"output_dir: {output_dir}")

            if not dataset_root.exists():
                raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")

            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

            if not yolo_weights.exists():
                raise FileNotFoundError(f"Pesi YOLO non trovati: {yolo_weights}")

            print("\n# Caricamento YOLO")
            model = YOLO(str(yolo_weights))

            model_names = get_model_names(model)
            print(f"Classi YOLO: {model_names}")

            ball_class_id = resolve_class_id(
                model,
                class_name=args.ball_class_name,
                explicit_id=args.ball_class_id,
            )
            rim_class_id = resolve_class_id(
                model,
                class_name=args.rim_class_name,
                explicit_id=args.rim_class_id,
            )

            print(f"ball_class_id: {ball_class_id}")
            print(f"rim_class_id: {rim_class_id}")

            print("\n# Lettura manifest")
            rows = read_manifest(
                manifest_path=manifest_path,
                dataset_root=dataset_root,
                splits=set(args.splits),
                labels=set(args.labels),
                max_clips=args.max_clips,
            )

            print(f"Clip selezionate: {len(rows)}")

            if not rows:
                raise RuntimeError("Nessuna clip selezionata. Controllare --splits e --labels.")

            counts = Counter((row["split"], row["label"]) for row in rows)
            print("\n# Distribuzione clip selezionate")
            for (split, label), count in sorted(counts.items()):
                print(f"{split:5s} | {label:12s} | {count}")

            clip_rows = []
            all_frame_rows = []
            temporal_sequence_entries = []
            errors = []

            print("\n# Estrazione feature")
            for idx, row in enumerate(rows, start=1):
                print(
                    f"[{idx}/{len(rows)}] "
                    f"{row['split']} | {row['label']} | {row['path']}"
                )

                if not row["video_path"].exists():
                    msg = f"Video non trovato: {row['video_path']}"
                    print(f"[WARN] {msg}")
                    errors.append(
                        {
                            "path": row["path"],
                            "label": row["label"],
                            "split": row["split"],
                            "error": msg,
                        }
                    )
                    continue

                try:
                    clip_row, frame_rows = process_clip(
                        row=row,
                        model=model,
                        ball_class_id=ball_class_id,
                        rim_class_id=rim_class_id,
                        args=args,
                    )

                    clip_rows.append(clip_row)

                    if args.save_per_frame:
                        all_frame_rows.extend(frame_rows)

                    if args.save_temporal_sequences:
                        temporal_sequence = compute_temporal_sequence_features(
                            frame_rows=frame_rows,
                            fps=float(clip_row.get("fps", 25.0)),
                        )
                        temporal_sequence_entries.append(
                            {
                                "clip_id": clip_row["clip_id"],
                                "split": clip_row["split"],
                                "label": clip_row["label"],
                                "path": clip_row["path"],
                                "sequence": temporal_sequence,
                            }
                        )

                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    print(f"[WARN] Errore su clip {row['path']}: {msg}")
                    errors.append(
                        {
                            "path": row["path"],
                            "label": row["label"],
                            "split": row["split"],
                            "error": msg,
                        }
                    )

            if not clip_rows:
                raise RuntimeError("Nessuna feature estratta correttamente.")

            print("\n# Salvataggio output")

            metadata_fields = [
                "clip_id",
                "split",
                "label",
                "path",
                "video_frames",
                "fps",
                "sampled_frames",
                "video_width",
                "video_height",
            ]

            write_csv(
                path=tracking_csv_path,
                rows=clip_rows,
                fieldnames=metadata_fields + TRACKING_FEATURE_NAMES,
            )

            print(f"Feature per clip salvate in: {tracking_csv_path}")

            with open(feature_names_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_names": TRACKING_FEATURE_NAMES,
                        "num_features": len(TRACKING_FEATURE_NAMES),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"Nomi feature salvati in: {feature_names_path}")

            if args.save_per_frame:
                frame_fields = [
                    "clip_id",
                    "split",
                    "label",
                    "path",
                    "frame_order",
                    "frame_idx",
                    "time_sec",
                    "t_rel",
                    "width",
                    "height",
                    "ball_detected",
                    "ball_conf",
                    "ball_xc",
                    "ball_yc",
                    "ball_w",
                    "ball_h",
                    "ball_area",
                    "rim_detected",
                    "rim_conf",
                    "rim_xc",
                    "rim_yc",
                    "rim_w",
                    "rim_h",
                    "rim_area",
                    "both_detected",
                    "dx",
                    "dy",
                    "ball_rim_dist",
                    "ball_above_rim",
                    "ball_below_rim",
                    "ball_near_rim",
                ]

                write_csv(
                    path=per_frame_csv_path,
                    rows=all_frame_rows,
                    fieldnames=frame_fields,
                )

                print(f"Detection per frame salvate in: {per_frame_csv_path}")

            if args.save_temporal_sequences:
                npz_path, index_path, sequence_feature_names_path = write_temporal_sequences(
                    output_dir=output_dir,
                    sequence_entries=temporal_sequence_entries,
                )
                print(f"Sequenze tracking temporali salvate in: {npz_path}")
                print(f"Indice sequenze tracking salvato in: {index_path}")
                print(f"Nomi feature tracking temporali salvati in: {sequence_feature_names_path}")

            if errors:
                errors_path = output_dir / "errors.csv"
                write_csv(
                    path=errors_path,
                    rows=errors,
                    fieldnames=["split", "label", "path", "error"],
                )
                print(f"Errori salvati in: {errors_path}")

            print("\n# Riepilogo")
            print(f"Clip processate correttamente: {len(clip_rows)}")
            print(f"Clip con errore: {len(errors)}")
            print(f"Numero feature tracking aggregate: {len(TRACKING_FEATURE_NAMES)}")
            if args.save_temporal_sequences:
                print(f"Numero feature tracking temporali per frame: {len(TEMPORAL_TRACKING_FEATURE_NAMES)}")
                print(f"Sequenze temporali salvate: {len(temporal_sequence_entries)}")

        except Exception:
            print("\nERRORE DURANTE L'ESECUZIONE:", file=sys.stderr)
            traceback.print_exc()
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Estrazione completata. Log salvato in: {results_path}")


if __name__ == "__main__":
    main()