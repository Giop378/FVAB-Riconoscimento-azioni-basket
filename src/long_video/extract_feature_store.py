from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.features.dinov3_extractor import (
    DINO_FEATURE_DIMS,
    DINOv3FeatureExtractor,
    build_dino_transform,
    extract_dino_features_for_frames,
    get_dino_config,
)
from src.long_video import defaults


# =============================================================================
# Utility generali
# =============================================================================


def as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(value)


def str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def ensure_exists(path: Path, name: str, must_be_file: bool | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} non trovato: {path}")
    if must_be_file is True and not path.is_file():
        raise FileNotFoundError(f"{name} dovrebbe essere un file: {path}")
    if must_be_file is False and not path.is_dir():
        raise NotADirectoryError(f"{name} dovrebbe essere una cartella: {path}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"La cartella di output esiste già e non è vuota: {output_dir}\n"
                f"Usa --overwrite per sostituire i file esistenti."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_device_for_torch(device: str) -> torch.device:
    device = str(device)
    if device.lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if device == "cuda":
            return torch.device("cuda")
        if device.startswith("cuda"):
            return torch.device(device)
        return torch.device(f"cuda:{device}")
    print("[WARN] CUDA non disponibile: uso CPU.")
    return torch.device("cpu")


def parse_device_for_yolo(device: str) -> str:
    device = str(device)
    if device.lower() == "cpu":
        return "cpu"
    return device.replace("cuda:", "")


def file_sha256(path: Path | None, chunk_size: int = 1024 * 1024) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# =============================================================================
# Video e campionamento frame
# =============================================================================


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    num_frames: int
    width: int
    height: int
    duration_sec: float


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"FPS non valido per il video {video_path}: {fps}")
    if num_frames <= 0:
        raise RuntimeError(f"Numero frame non valido per il video {video_path}: {num_frames}")

    return VideoInfo(
        fps=fps,
        num_frames=num_frames,
        width=width,
        height=height,
        duration_sec=float(num_frames / fps),
    )


def make_source_frame_grid(
    start_sec: float,
    end_sec: float,
    source_fps: float,
    num_video_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crea una griglia usando tutti i frame reali del video sorgente.

    È la modalità train-like per la pipeline long-video: come nelle clip usate
    per il training, ogni frame reale compreso nel segmento produce una feature
    DINOv3 e una riga di primitive YOLO palla/canestro.

    La finestra temporale è [start_sec, end_sec): end_sec escluso.
    """
    if start_sec < 0:
        raise ValueError(f"start_sec deve essere >= 0, trovato {start_sec}")
    if end_sec <= start_sec:
        raise ValueError(f"end_sec deve essere > start_sec, trovato {start_sec} -> {end_sec}")
    if source_fps <= 0:
        raise ValueError(f"source_fps deve essere > 0, trovato {source_fps}")
    if num_video_frames <= 0:
        raise ValueError(f"num_video_frames deve essere > 0, trovato {num_video_frames}")

    eps = 1e-9
    start_frame = int(math.ceil(start_sec * float(source_fps) - eps))
    end_frame_exclusive = int(math.ceil(end_sec * float(source_fps) - eps))

    start_frame = max(0, min(start_frame, num_video_frames))
    end_frame_exclusive = max(0, min(end_frame_exclusive, num_video_frames))

    if end_frame_exclusive <= start_frame:
        raise ValueError(
            "Il segmento selezionato non contiene frame reali del video. "
            f"start_sec={start_sec:.6f}, end_sec={end_sec:.6f}, "
            f"start_frame={start_frame}, end_frame_exclusive={end_frame_exclusive}."
        )

    frame_indices = np.arange(start_frame, end_frame_exclusive, dtype=np.int64)
    timestamps = frame_indices.astype(np.float64) / float(source_fps)

    if np.any(np.diff(frame_indices) != 1):
        raise RuntimeError("La griglia source_frames dovrebbe contenere frame consecutivi.")

    return timestamps.astype(np.float64), frame_indices.astype(np.int64)


def iter_sampled_frame_batches(
    video_path: Path,
    sample_positions: np.ndarray,
    frame_indices: np.ndarray,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, list[np.ndarray]]]:
    """Yield batch di frame BGR seguendo gli indici campionati in ordine temporale."""
    if batch_size <= 0:
        raise ValueError(f"batch_size deve essere > 0, trovato {batch_size}")
    if len(sample_positions) != len(frame_indices):
        raise ValueError("sample_positions e frame_indices devono avere la stessa lunghezza")
    if len(sample_positions) == 0:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    first_frame = int(frame_indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    current_frame_idx = first_frame
    last_frame_idx: int | None = None
    last_frame_bgr: np.ndarray | None = None

    batch_positions: list[int] = []
    batch_frames: list[np.ndarray] = []

    try:
        for pos, target_frame_idx in zip(sample_positions, frame_indices):
            target_frame_idx = int(target_frame_idx)
            pos = int(pos)

            if last_frame_idx == target_frame_idx and last_frame_bgr is not None:
                frame_bgr = last_frame_bgr
            else:
                if target_frame_idx < current_frame_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                    current_frame_idx = target_frame_idx

                frame_bgr = None
                while current_frame_idx <= target_frame_idx:
                    ok, decoded = cap.read()
                    if not ok or decoded is None:
                        raise RuntimeError(
                            f"Errore lettura frame {current_frame_idx} da {video_path}"
                        )
                    if current_frame_idx == target_frame_idx:
                        frame_bgr = decoded
                    current_frame_idx += 1

                if frame_bgr is None:
                    raise RuntimeError(f"Frame {target_frame_idx} non decodificato")

                last_frame_idx = target_frame_idx
                last_frame_bgr = frame_bgr

            batch_positions.append(pos)
            batch_frames.append(frame_bgr)

            if len(batch_frames) >= batch_size:
                yield np.asarray(batch_positions, dtype=np.int64), batch_frames
                batch_positions = []
                batch_frames = []

        if batch_frames:
            yield np.asarray(batch_positions, dtype=np.int64), batch_frames
    finally:
        cap.release()


# =============================================================================
# YOLO palla/canestro
# =============================================================================


PRIMITIVE_FIELDS = [
    "ball_detected",
    "rim_detected",
    "both_detected",
    "ball_conf",
    "rim_conf",
    "ball_x1",
    "ball_y1",
    "ball_x2",
    "ball_y2",
    "ball_xc",
    "ball_yc",
    "ball_w",
    "ball_h",
    "rim_x1",
    "rim_y1",
    "rim_x2",
    "rim_y2",
    "rim_xc",
    "rim_yc",
    "rim_w",
    "rim_h",
    "ball_x1_px",
    "ball_y1_px",
    "ball_x2_px",
    "ball_y2_px",
    "rim_x1_px",
    "rim_y1_px",
    "rim_x2_px",
    "rim_y2_px",
    "num_ball_detections",
    "num_rim_detections",
]


def init_primitives(num_samples: int) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}
    for field in PRIMITIVE_FIELDS:
        if field.endswith("_detected") or field.startswith("num_"):
            data[field] = np.zeros((num_samples,), dtype=np.int32)
        else:
            data[field] = np.zeros((num_samples,), dtype=np.float32)
    return data


def update_detection_for_class(
    primitives: dict[str, np.ndarray],
    global_pos: int,
    prefix: str,
    xyxy: np.ndarray,
    conf: float,
    frame_width: int,
    frame_height: int,
) -> None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1_c = max(0.0, min(x1, frame_width - 1.0))
    y1_c = max(0.0, min(y1, frame_height - 1.0))
    x2_c = max(0.0, min(x2, frame_width - 1.0))
    y2_c = max(0.0, min(y2, frame_height - 1.0))

    w_px = max(0.0, x2_c - x1_c)
    h_px = max(0.0, y2_c - y1_c)
    xc_px = x1_c + 0.5 * w_px
    yc_px = y1_c + 0.5 * h_px

    fw = max(float(frame_width), 1.0)
    fh = max(float(frame_height), 1.0)

    primitives[f"{prefix}_detected"][global_pos] = 1
    primitives[f"{prefix}_conf"][global_pos] = float(conf)
    primitives[f"{prefix}_x1"][global_pos] = x1_c / fw
    primitives[f"{prefix}_y1"][global_pos] = y1_c / fh
    primitives[f"{prefix}_x2"][global_pos] = x2_c / fw
    primitives[f"{prefix}_y2"][global_pos] = y2_c / fh
    primitives[f"{prefix}_xc"][global_pos] = xc_px / fw
    primitives[f"{prefix}_yc"][global_pos] = yc_px / fh
    primitives[f"{prefix}_w"][global_pos] = w_px / fw
    primitives[f"{prefix}_h"][global_pos] = h_px / fh
    primitives[f"{prefix}_x1_px"][global_pos] = x1_c
    primitives[f"{prefix}_y1_px"][global_pos] = y1_c
    primitives[f"{prefix}_x2_px"][global_pos] = x2_c
    primitives[f"{prefix}_y2_px"][global_pos] = y2_c


def load_yolo_model(weights_path: Path) -> Any:
    ensure_exists(weights_path, "Pesi YOLO", must_be_file=True)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Modulo ultralytics non installato. Installa/attiva l'ambiente usato per YOLO."
        ) from exc

    return YOLO(str(weights_path))


def run_yolo_on_frames(
    model: Any,
    frames_bgr: list[np.ndarray],
    batch_positions: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    primitives: dict[str, np.ndarray],
    detections_writer: csv.DictWriter,
    source_name: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch_size: int,
    ball_class_id: int,
    rim_class_id: int,
) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size_yolo deve essere > 0, trovato {batch_size}")

    for start in range(0, len(frames_bgr), batch_size):
        sub_frames = frames_bgr[start : start + batch_size]
        sub_positions = batch_positions[start : start + batch_size]

        results = model.predict(
            source=sub_frames,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            classes=[int(ball_class_id), int(rim_class_id)],
            verbose=False,
        )

        if len(results) != len(sub_frames):
            raise RuntimeError(
                f"YOLO {source_name}: numero risultati inatteso: {len(results)} invece di {len(sub_frames)}"
            )

        for local_i, result in enumerate(results):
            global_pos = int(sub_positions[local_i])
            frame = sub_frames[local_i]
            frame_height, frame_width = frame.shape[:2]

            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy_all = boxes.xyxy.detach().cpu().numpy()
            conf_all = boxes.conf.detach().cpu().numpy()
            cls_all = boxes.cls.detach().cpu().numpy().astype(np.int64)

            primitives["num_ball_detections"][global_pos] = int(np.sum(cls_all == ball_class_id))
            primitives["num_rim_detections"][global_pos] = int(np.sum(cls_all == rim_class_id))

            for class_id, prefix in [(ball_class_id, "ball"), (rim_class_id, "rim")]:
                class_mask = cls_all == class_id
                if not np.any(class_mask):
                    continue

                candidate_indices = np.flatnonzero(class_mask)
                best_idx = int(candidate_indices[np.argmax(conf_all[candidate_indices])])
                update_detection_for_class(
                    primitives=primitives,
                    global_pos=global_pos,
                    prefix=prefix,
                    xyxy=xyxy_all[best_idx],
                    conf=float(conf_all[best_idx]),
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

            primitives["both_detected"][global_pos] = int(
                primitives["ball_detected"][global_pos] == 1
                and primitives["rim_detected"][global_pos] == 1
            )

            for det_idx in range(len(cls_all)):
                class_id = int(cls_all[det_idx])
                if class_id not in {ball_class_id, rim_class_id}:
                    continue
                x1, y1, x2, y2 = [float(v) for v in xyxy_all[det_idx]]
                detections_writer.writerow(
                    {
                        "sample_index": global_pos,
                        "timestamp": f"{float(timestamps[global_pos]):.6f}",
                        "frame_index": int(frame_indices[global_pos]),
                        "source": source_name,
                        "class_id": class_id,
                        "class_name": "ball" if class_id == ball_class_id else "rim",
                        "conf": f"{float(conf_all[det_idx]):.6f}",
                        "x1": f"{x1:.3f}",
                        "y1": f"{y1:.3f}",
                        "x2": f"{x2:.3f}",
                        "y2": f"{y2:.3f}",
                        "frame_width": int(frame_width),
                        "frame_height": int(frame_height),
                    }
                )


# =============================================================================
# Output
# =============================================================================


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    video_info: VideoInfo,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    started_at: float,
) -> None:
    dino_meta: dict[str, Any]
    if args.skip_dino:
        dino_meta = {"enabled": False, "output_file": None}
    else:
        dino_meta = get_dino_config(
            model_name=args.dino_model_name,
            weights=args.dino_weights,
            repo_or_dir=args.dino_repo,
            source=args.dino_source,
            image_size=args.dino_input_size,
        )
        dino_meta.update(
            {
                "enabled": True,
                "weights_sha256": file_sha256(args.dino_weights),
                "output_file": "dinov3_features.npy",
            }
        )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started_at, 3),
        "video": {
            "input_path": str(args.input_video),
            "sha256": file_sha256(args.input_video),
            "fps": video_info.fps,
            "num_frames": video_info.num_frames,
            "width": video_info.width,
            "height": video_info.height,
            "duration_sec": video_info.duration_sec,
            "start_sec": args.start_sec,
            "end_sec": args.end_sec,
            "segment_duration_sec": args.end_sec - args.start_sec,
        },
        "sampling": {
            "mode": "source_frames",
            "policy": "all real source frames in [start_sec, end_sec)",
            "source_fps": video_info.fps,
            "feature_fps": video_info.fps,
            "num_samples": int(len(timestamps)),
            "first_timestamp": float(timestamps[0]),
            "last_timestamp": float(timestamps[-1]),
            "first_frame_index": int(frame_indices[0]),
            "last_frame_index": int(frame_indices[-1]),
        },
        "dinov3": dino_meta,
        "yolo": {
            "enabled_v1": not args.skip_yolo_v1,
            "enabled_v2": not args.skip_yolo_v2,
            "v1_weights": str(args.yolo_v1_weights),
            "v2_weights": str(args.yolo_v2_weights),
            "v1_sha256": file_sha256(args.yolo_v1_weights),
            "v2_sha256": file_sha256(args.yolo_v2_weights),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "classes_filter": [int(args.ball_class_id), int(args.rim_class_id)],
            "ball_class_id": args.ball_class_id,
            "rim_class_id": args.rim_class_id,
        },
        "runtime": {
            "device": args.device,
            "batch_size_decode": args.batch_size_decode,
            "batch_size_dino": args.batch_size_dino,
            "batch_size_yolo": args.batch_size_yolo,
            "amp": bool(args.amp),
        },
        "files": {
            "metadata": "metadata.json",
            "timestamps": "timestamps.npy",
            "frame_indices": "frame_indices.npy",
            "dinov3_features": "dinov3_features.npy" if not args.skip_dino else None,
            "yolo_v1_primitives": "yolo_v1_primitives.npz" if not args.skip_yolo_v1 else None,
            "yolo_v2_primitives": "yolo_v2_primitives.npz" if not args.skip_yolo_v2 else None,
            "yolo_v1_detections": "yolo_v1_detections.csv" if not args.skip_yolo_v1 else None,
            "yolo_v2_detections": "yolo_v2_detections.csv" if not args.skip_yolo_v2 else None,
        },
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_primitives_npz(
    path: Path,
    primitives: dict[str, np.ndarray],
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        timestamps=timestamps.astype(np.float64),
        frame_indices=frame_indices.astype(np.int64),
        **primitives,
    )


# =============================================================================
# CLI
# =============================================================================


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estrae una feature store temporale per la pipeline long-video: "
            "DINOv3 frame-level + primitive YOLO palla/canestro su un segmento video. "
            "La logica DINOv3 è la stessa usata per le clip."
        )
    )

    parser.add_argument("--input-video", type=Path, default=defaults.VAL_VIDEO_PATH)
    parser.add_argument("--start-sec", type=float, default=defaults.VAL_START_SEC)
    parser.add_argument("--end-sec", type=float, default=defaults.VAL_END_SEC)
    parser.add_argument("--output-dir", type=Path, default=defaults.VAL_FEATURE_STORE_DIR)

    parser.add_argument("--batch-size-decode", type=int, default=128)
    parser.add_argument("--batch-size-dino", type=int, default=32)
    parser.add_argument("--batch-size-yolo", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--dino-repo", type=Path, default=getattr(defaults, "DINOV3_REPO", Path("third_party/dinov3")))
    parser.add_argument("--dino-weights", type=Path, default=getattr(defaults, "DINOV3_WEIGHTS", None))
    parser.add_argument("--dino-source", type=str, default=getattr(defaults, "DINOV3_SOURCE", "local"), choices=["local", "github"])
    parser.add_argument("--dino-model-name", type=str, default=getattr(defaults, "DINOV3_MODEL_NAME", "dinov3_vitl16"), choices=list(DINO_FEATURE_DIMS.keys()))
    parser.add_argument("--dino-input-size", type=int, default=defaults.DINOV3_INPUT_SIZE)
    parser.add_argument("--dino-feature-dim", type=int, default=defaults.DINOV3_FEATURE_DIM)
    parser.add_argument(
        "--dino-resize-mode",
        type=str,
        default="stretch",
        choices=["stretch"],
        help="Compatibilità CLI: la pipeline supporta solo stretch, come nelle clip.",
    )
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Abilita AMP per DINOv3. Default disattivato per massimizzare la coerenza con le clip.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help=argparse.SUPPRESS,  # accettato per compatibilità con i vecchi comandi
    )

    parser.add_argument("--yolo-v1-weights", type=Path, default=defaults.YOLO_V1_WEIGHTS)
    parser.add_argument("--yolo-v2-weights", type=Path, default=defaults.YOLO_V2_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--ball-class-id", type=int, default=0)
    parser.add_argument("--rim-class-id", type=int, default=1)
    parser.add_argument("--skip-yolo-v1", action="store_true")
    parser.add_argument("--skip-yolo-v2", action="store_true")

    return parser


def main() -> None:
    args = make_parser().parse_args()
    started_at = time.time()

    # Vecchio flag: se presente, forza AMP off.
    if getattr(args, "no_amp", False):
        args.amp = False

    args.input_video = as_path(args.input_video)
    args.output_dir = as_path(args.output_dir)
    args.dino_repo = as_path(args.dino_repo)
    args.dino_weights = as_path(args.dino_weights)
    args.yolo_v1_weights = as_path(args.yolo_v1_weights)
    args.yolo_v2_weights = as_path(args.yolo_v2_weights)

    assert args.input_video is not None
    assert args.output_dir is not None
    assert args.yolo_v1_weights is not None
    assert args.yolo_v2_weights is not None

    ensure_exists(args.input_video, "Video input", must_be_file=True)

    if not args.skip_dino:
        if args.dino_weights is None:
            raise ValueError(
                "I pesi DINOv3 sono obbligatori. Imposta defaults.DINOV3_WEIGHTS "
                "oppure passa --dino-weights."
            )
        assert args.dino_repo is not None
        ensure_exists(args.dino_repo, "Repository DINOv3", must_be_file=False)
        ensure_exists(args.dino_weights, "Pesi DINOv3", must_be_file=True)

        expected_dim = int(DINO_FEATURE_DIMS[args.dino_model_name])
        if int(args.dino_feature_dim) != expected_dim:
            raise ValueError(
                f"--dino-feature-dim={args.dino_feature_dim} non coerente con "
                f"{args.dino_model_name}, che richiede {expected_dim}."
            )
        if args.dino_resize_mode != "stretch":
            raise ValueError("La pipeline supporta solo --dino-resize-mode stretch.")

    if not args.skip_yolo_v1:
        ensure_exists(args.yolo_v1_weights, "Pesi YOLO v1", must_be_file=True)
    if not args.skip_yolo_v2:
        ensure_exists(args.yolo_v2_weights, "Pesi YOLO v2", must_be_file=True)

    prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    video_info = get_video_info(args.input_video)
    if args.end_sec > video_info.duration_sec:
        raise ValueError(
            f"Segmento fuori dal video: end_sec={args.end_sec:.3f}, "
            f"durata={video_info.duration_sec:.3f}"
        )

    timestamps, frame_indices = make_source_frame_grid(
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        source_fps=video_info.fps,
        num_video_frames=video_info.num_frames,
    )

    np.save(args.output_dir / "timestamps.npy", timestamps)
    np.save(args.output_dir / "frame_indices.npy", frame_indices)

    print("=== Feature store long-video ===")
    print(f"video: {args.input_video}")
    print(f"segmento: {args.start_sec:.3f}s -> {args.end_sec:.3f}s")
    print(f"video fps: {video_info.fps:.3f}")
    print("sampling: source_frames, tutti i frame reali in [start_sec, end_sec)")
    print(f"feature_fps effettivo: {video_info.fps:.3f}")
    print(f"frame iniziale/finale: {int(frame_indices[0])} -> {int(frame_indices[-1])}")
    print(f"num samples: {len(timestamps)}")
    print(f"output_dir: {args.output_dir}")

    torch_device = parse_device_for_torch(args.device)
    yolo_device = parse_device_for_yolo(args.device)

    dino_model: torch.nn.Module | None = None
    dino_transform = None
    dino_features = None
    if not args.skip_dino:
        print("\n=== DINOv3 ===")
        print(f"repo:        {args.dino_repo}")
        print(f"source:      {args.dino_source}")
        print(f"weights:     {args.dino_weights}")
        print(f"model:       {args.dino_model_name}")
        print(f"input size:  {args.dino_input_size}")
        print("resize:      stretch, no center crop")
        print("output:      x_norm_clstoken")
        print(f"amp:         {bool(args.amp)}")

        dino_model = DINOv3FeatureExtractor(
            model_name=args.dino_model_name,
            weights=args.dino_weights,
            repo_or_dir=args.dino_repo,
            source=args.dino_source,
        ).to(torch_device)
        dino_model.eval()

        dino_transform = build_dino_transform(args.dino_input_size)
        dino_features = np.lib.format.open_memmap(
            args.output_dir / "dinov3_features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(timestamps), args.dino_feature_dim),
        )

    yolo_v1_model = None
    yolo_v2_model = None
    yolo_v1_primitives = None
    yolo_v2_primitives = None
    yolo_v1_csv = None
    yolo_v2_csv = None
    yolo_v1_writer = None
    yolo_v2_writer = None

    detection_fieldnames = [
        "sample_index",
        "timestamp",
        "frame_index",
        "source",
        "class_id",
        "class_name",
        "conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "frame_width",
        "frame_height",
    ]

    if not args.skip_yolo_v1:
        yolo_v1_model = load_yolo_model(args.yolo_v1_weights)
        yolo_v1_primitives = init_primitives(len(timestamps))
        yolo_v1_csv = (args.output_dir / "yolo_v1_detections.csv").open("w", newline="", encoding="utf-8")
        yolo_v1_writer = csv.DictWriter(yolo_v1_csv, fieldnames=detection_fieldnames)
        yolo_v1_writer.writeheader()

    if not args.skip_yolo_v2:
        yolo_v2_model = load_yolo_model(args.yolo_v2_weights)
        yolo_v2_primitives = init_primitives(len(timestamps))
        yolo_v2_csv = (args.output_dir / "yolo_v2_detections.csv").open("w", newline="", encoding="utf-8")
        yolo_v2_writer = csv.DictWriter(yolo_v2_csv, fieldnames=detection_fieldnames)
        yolo_v2_writer.writeheader()

    sample_positions = np.arange(len(timestamps), dtype=np.int64)

    try:
        iterator = iter_sampled_frame_batches(
            video_path=args.input_video,
            sample_positions=sample_positions,
            frame_indices=frame_indices,
            batch_size=args.batch_size_decode,
        )
        progress = tqdm(iterator, total=math.ceil(len(timestamps) / args.batch_size_decode), desc="Feature store")

        for batch_positions, frames_bgr in progress:
            if dino_model is not None and dino_transform is not None and dino_features is not None:
                feats = extract_dino_features_for_frames(
                    frames_bgr=frames_bgr,
                    model=dino_model,
                    transform=dino_transform,
                    device=torch_device,
                    batch_size=args.batch_size_dino,
                    expected_dim=args.dino_feature_dim,
                    use_amp=bool(args.amp),
                )
                if feats.shape[0] != len(batch_positions):
                    raise RuntimeError(
                        f"Batch DINO con numero righe errato: {feats.shape[0]} vs {len(batch_positions)}"
                    )
                dino_features[batch_positions] = feats

            if yolo_v1_model is not None and yolo_v1_primitives is not None and yolo_v1_writer is not None:
                run_yolo_on_frames(
                    model=yolo_v1_model,
                    frames_bgr=frames_bgr,
                    batch_positions=batch_positions,
                    timestamps=timestamps,
                    frame_indices=frame_indices,
                    primitives=yolo_v1_primitives,
                    detections_writer=yolo_v1_writer,
                    source_name="yolo_v1",
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=yolo_device,
                    batch_size=args.batch_size_yolo,
                    ball_class_id=args.ball_class_id,
                    rim_class_id=args.rim_class_id,
                )

            if yolo_v2_model is not None and yolo_v2_primitives is not None and yolo_v2_writer is not None:
                run_yolo_on_frames(
                    model=yolo_v2_model,
                    frames_bgr=frames_bgr,
                    batch_positions=batch_positions,
                    timestamps=timestamps,
                    frame_indices=frame_indices,
                    primitives=yolo_v2_primitives,
                    detections_writer=yolo_v2_writer,
                    source_name="yolo_v2",
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=yolo_device,
                    batch_size=args.batch_size_yolo,
                    ball_class_id=args.ball_class_id,
                    rim_class_id=args.rim_class_id,
                )
    finally:
        if yolo_v1_csv is not None:
            yolo_v1_csv.close()
        if yolo_v2_csv is not None:
            yolo_v2_csv.close()

    if dino_features is not None:
        dino_features.flush()
        del dino_features

    if yolo_v1_primitives is not None:
        save_primitives_npz(
            args.output_dir / "yolo_v1_primitives.npz",
            yolo_v1_primitives,
            timestamps=timestamps,
            frame_indices=frame_indices,
        )
    if yolo_v2_primitives is not None:
        save_primitives_npz(
            args.output_dir / "yolo_v2_primitives.npz",
            yolo_v2_primitives,
            timestamps=timestamps,
            frame_indices=frame_indices,
        )

    write_metadata(
        output_dir=args.output_dir,
        args=args,
        video_info=video_info,
        timestamps=timestamps,
        frame_indices=frame_indices,
        started_at=started_at,
    )

    print("\n=== Output creati ===")
    for path in sorted(args.output_dir.iterdir()):
        print(f"- {path}")
    print("\nFeature store completato correttamente.")


if __name__ == "__main__":
    main()
