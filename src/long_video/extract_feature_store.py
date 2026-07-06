# =============================================================================
# Questo script crea la feature store utilizzata dalla pipeline long-video
# exp_long_13/exp_46 a partire da un segmento di video continuo. In pratica:
# 1) apre il video e seleziona tutti i frame reali compresi in [start_sec, end_sec);
# 2) salva timestamp e indici frame, così gli step successivi possono ricostruire
#    finestre temporali senza dover rieseguire la decodifica video;
# 3) estrae per ogni frame una feature DINOv3 frame-level, compatibile con il
#    modello gerarchico usato nelle clip;
# 4) esegue YOLO v1 e YOLO v2 sugli stessi frame per salvare primitive normalizzate
#    di palla e canestro;
# 5) scrive file .npy/.npz e metadata.json in output, che saranno poi letti da
#    build_windows_from_store.py e dall'inferenza long-video.
#
# Lo script non produce eventi finali: prepara soltanto le rappresentazioni
# intermedie pesanti, in modo che l'inferenza e il post-processing possano essere
# rieseguiti più volte senza ripetere DINOv3/YOLO.
# =============================================================================

from __future__ import annotations

import argparse
import math
import time
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
)
from src.long_video import defaults
from src.long_video.utils import (
    as_path,
    ensure_exists,
    get_video_info,
    parse_device_for_torch,
    parse_device_for_yolo,
    prepare_output_dir,
    write_json,
)


# =============================================================================
# Configurazione fissa per exp_long_13 / exp_46
# =============================================================================

# exp_long_13 usa la stessa feature store train-like della pipeline exp_46:
# - tutti i frame reali del segmento video;
# - feature DINOv3 frame-level;
# - primitive YOLO v1 e YOLO v2 per palla/canestro.
#
# L'inferenza exp_46 usa YOLO v2 per L1/L2 e YOLO v1 per L3, quindi entrambi
# i file .npz sono necessari.
USE_AMP_DINO = False

PRIMITIVE_FIELDS = [
    "ball_detected",
    "rim_detected",
    "ball_conf",
    "rim_conf",
    "ball_xc",
    "ball_yc",
    "ball_w",
    "ball_h",
    "rim_xc",
    "rim_yc",
    "rim_w",
    "rim_h",
]


# =============================================================================
# Video e campionamento frame
# =============================================================================


# Costruisce la corrispondenza frame <-> timestamp del segmento da processare.
# La scelta train-like è usare tutti i frame sorgente, senza ricampionamento.

def make_source_frame_grid(
    start_sec: float,
    end_sec: float,
    source_fps: float,
    num_video_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crea la griglia train-like usando tutti i frame reali del video.

    La finestra temporale è [start_sec, end_sec), quindi end_sec è escluso.
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
    return timestamps.astype(np.float64), frame_indices.astype(np.int64)


# Decodifica il video in batch seguendo gli indici reali calcolati prima.
# Questo evita di caricare l'intero segmento in RAM e mantiene l'ordine temporale.

def iter_frame_batches(
    video_path: Path,
    frame_indices: np.ndarray,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, list[np.ndarray]]]:
    """Yield di batch di frame BGR in ordine temporale.

    Ritorna:
    - posizioni locali nella feature store;
    - frame BGR decodificati.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size_decode deve essere > 0, trovato {batch_size}")
    if frame_indices.size == 0:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    first_frame = int(frame_indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    current_frame_idx = first_frame

    batch_positions: list[int] = []
    batch_frames: list[np.ndarray] = []

    try:
        for local_pos, target_frame_idx in enumerate(frame_indices.tolist()):
            target_frame_idx = int(target_frame_idx)

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

            batch_positions.append(int(local_pos))
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


# Inizializza gli array delle primitive YOLO per tutti i frame del segmento.

def init_primitives(num_samples: int) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}
    for field in PRIMITIVE_FIELDS:
        if field.endswith("_detected"):
            data[field] = np.zeros((num_samples,), dtype=np.int32)
        else:
            data[field] = np.zeros((num_samples,), dtype=np.float32)
    return data


# Salva la migliore detection di una classe YOLO in formato normalizzato.
# Le coordinate xyxy vengono limitate ai bordi del frame e convertite in
# centro/larghezza/altezza relativi alla dimensione dell'immagine.

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
    primitives[f"{prefix}_xc"][global_pos] = xc_px / fw
    primitives[f"{prefix}_yc"][global_pos] = yc_px / fh
    primitives[f"{prefix}_w"][global_pos] = w_px / fw
    primitives[f"{prefix}_h"][global_pos] = h_px / fh


# Carica un modello YOLO Ultralytics dai pesi indicati e controlla che il file
# esista prima dell'inizializzazione, così gli errori di configurazione emergono subito.

def load_yolo_model(weights_path: Path) -> Any:
    ensure_exists(weights_path, "Pesi YOLO", must_be_file=True)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Modulo ultralytics non installato. Installa/attiva l'ambiente usato per YOLO."
        ) from exc
    return YOLO(str(weights_path))


# Esegue YOLO su un batch di frame e aggiorna le primitive palla/canestro.
# Per ogni frame e per ogni classe viene mantenuta solo la detection con confidence più alta.

def run_yolo_on_frames(
    model: Any,
    frames_bgr: list[np.ndarray],
    batch_positions: np.ndarray,
    primitives: dict[str, np.ndarray],
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
                f"YOLO {source_name}: numero risultati inatteso: "
                f"{len(results)} invece di {len(sub_frames)}"
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


# =============================================================================
# Output
# =============================================================================


# Scrive su disco le primitive YOLO insieme a timestamp e indici frame.
# Il formato compresso .npz riduce lo spazio occupato mantenendo accesso semplice tramite NumPy.

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


# Raccoglie in metadata.json le informazioni necessarie a riprodurre e verificare
# la feature store: video sorgente, campionamento, modelli, parametri runtime e file creati.

def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    video_info: Any,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    started_at: float,
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started_at, 3),
        "experiment": "exp_long_13_feature_store",
        "video": {
            "input_path": str(args.input_video),
            "fps": float(video_info.fps),
            "num_frames": int(video_info.num_frames),
            "width": int(video_info.width),
            "height": int(video_info.height),
            "duration_sec": float(video_info.duration_sec),
            "start_sec": float(args.start_sec),
            "end_sec": float(args.end_sec),
            "segment_duration_sec": float(args.end_sec - args.start_sec),
        },
        "sampling": {
            "mode": "source_frames",
            "policy": "all real source frames in [start_sec, end_sec)",
            "source_fps": float(video_info.fps),
            "feature_fps": float(video_info.fps),
            "num_samples": int(len(timestamps)),
            "first_timestamp": float(timestamps[0]),
            "last_timestamp": float(timestamps[-1]),
            "first_frame_index": int(frame_indices[0]),
            "last_frame_index": int(frame_indices[-1]),
        },
        "dinov3": {
            "enabled": True,
            "repo": str(args.dino_repo),
            "source": str(args.dino_source),
            "weights": str(args.dino_weights),
            "model_name": str(args.dino_model_name),
            "input_size": int(args.dino_input_size),
            "feature_dim": int(args.dino_feature_dim),
            "amp": bool(USE_AMP_DINO),
            "output_file": "dinov3_features.npy",
        },
        "yolo": {
            "enabled_v1": True,
            "enabled_v2": True,
            "v1_weights": str(args.yolo_v1_weights),
            "v2_weights": str(args.yolo_v2_weights),
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "classes_filter": [int(args.ball_class_id), int(args.rim_class_id)],
            "ball_class_id": int(args.ball_class_id),
            "rim_class_id": int(args.rim_class_id),
            "primitive_fields": PRIMITIVE_FIELDS,
        },
        "runtime": {
            "device": str(args.device),
            "batch_size_decode": int(args.batch_size_decode),
            "batch_size_dino": int(args.batch_size_dino),
            "batch_size_yolo": int(args.batch_size_yolo),
        },
        "files": {
            "metadata": "metadata.json",
            "timestamps": "timestamps.npy",
            "frame_indices": "frame_indices.npy",
            "dinov3_features": "dinov3_features.npy",
            "yolo_v1_primitives": "yolo_v1_primitives.npz",
            "yolo_v2_primitives": "yolo_v2_primitives.npz",
        },
    }
    write_json(output_dir / "metadata.json", metadata)


# =============================================================================
# CLI
# =============================================================================


# Definisce l'interfaccia da riga di comando, includendo path, segmento video,
# batch size, device e parametri dei modelli DINOv3/YOLO.

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estrae la feature store necessaria a exp_long_13: "
            "DINOv3 frame-level + primitive YOLO v1/v2 palla/canestro."
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

    parser.add_argument(
        "--dino-repo",
        type=Path,
        default=getattr(defaults, "DINOV3_REPO", Path("third_party/dinov3")),
    )
    parser.add_argument(
        "--dino-weights",
        type=Path,
        default=getattr(defaults, "DINOV3_WEIGHTS", None),
    )
    parser.add_argument(
        "--dino-source",
        type=str,
        default=getattr(defaults, "DINOV3_SOURCE", "local"),
        choices=["local", "github"],
    )
    parser.add_argument(
        "--dino-model-name",
        type=str,
        default=getattr(defaults, "DINOV3_MODEL_NAME", "dinov3_vitl16"),
        choices=list(DINO_FEATURE_DIMS.keys()),
    )
    parser.add_argument("--dino-input-size", type=int, default=defaults.DINOV3_INPUT_SIZE)
    parser.add_argument("--dino-feature-dim", type=int, default=defaults.DINOV3_FEATURE_DIM)

    parser.add_argument("--yolo-v1-weights", type=Path, default=defaults.YOLO_V1_WEIGHTS)
    parser.add_argument("--yolo-v2-weights", type=Path, default=defaults.YOLO_V2_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--ball-class-id", type=int, default=0)
    parser.add_argument("--rim-class-id", type=int, default=1)

    return parser


# Normalizza e valida gli argomenti prima di avviare operazioni costose.
# Controlla esistenza dei file, range temporale, batch size e coerenza della dimensione DINOv3.

def validate_args(args: argparse.Namespace) -> None:
    args.input_video = as_path(args.input_video)
    args.output_dir = as_path(args.output_dir)
    args.dino_repo = as_path(args.dino_repo)
    args.dino_weights = as_path(args.dino_weights)
    args.yolo_v1_weights = as_path(args.yolo_v1_weights)
    args.yolo_v2_weights = as_path(args.yolo_v2_weights)

    assert args.input_video is not None
    assert args.output_dir is not None
    assert args.dino_repo is not None
    assert args.dino_weights is not None
    assert args.yolo_v1_weights is not None
    assert args.yolo_v2_weights is not None

    ensure_exists(args.input_video, "Video input", must_be_file=True)
    ensure_exists(args.dino_repo, "Repository DINOv3", must_be_file=False)
    ensure_exists(args.dino_weights, "Pesi DINOv3", must_be_file=True)
    ensure_exists(args.yolo_v1_weights, "Pesi YOLO v1", must_be_file=True)
    ensure_exists(args.yolo_v2_weights, "Pesi YOLO v2", must_be_file=True)

    if args.start_sec < 0:
        raise ValueError(f"start_sec deve essere >= 0, trovato {args.start_sec}")
    if args.end_sec <= args.start_sec:
        raise ValueError(f"end_sec deve essere > start_sec, trovato {args.start_sec} -> {args.end_sec}")
    if args.batch_size_decode <= 0 or args.batch_size_dino <= 0 or args.batch_size_yolo <= 0:
        raise ValueError("I batch size devono essere > 0")

    expected_dim = int(DINO_FEATURE_DIMS[args.dino_model_name])
    if int(args.dino_feature_dim) != expected_dim:
        raise ValueError(
            f"--dino-feature-dim={args.dino_feature_dim} non coerente con "
            f"{args.dino_model_name}, che richiede {expected_dim}."
        )


# Coordina l'intera pipeline: validazione input, preparazione output, estrazione
# DINOv3/YOLO frame-level, salvataggio degli artefatti e stampa del riepilogo finale.

def main() -> None:
    args = make_parser().parse_args()
    started_at = time.time()
    validate_args(args)

    # Prepara la cartella di output e, se richiesto, elimina risultati precedenti.
    prepare_output_dir(args.output_dir, overwrite=args.overwrite, clear_if_exists=True)

    # Legge metadati del video per validare il segmento e costruire la griglia frame.
    video_info = get_video_info(args.input_video)
    if args.end_sec > video_info.duration_sec:
        raise ValueError(
            f"Segmento fuori dal video: end_sec={args.end_sec:.3f}, "
            f"durata={video_info.duration_sec:.3f}"
        )

    # Crea timestamp e indici dei frame reali da processare nel segmento richiesto.
    timestamps, frame_indices = make_source_frame_grid(
        start_sec=float(args.start_sec),
        end_sec=float(args.end_sec),
        source_fps=float(video_info.fps),
        num_video_frames=int(video_info.num_frames),
    )

    # Salva subito la base temporale della feature store, usata dagli step successivi.
    np.save(args.output_dir / "timestamps.npy", timestamps)
    np.save(args.output_dir / "frame_indices.npy", frame_indices)

    print("=== Feature store long-video | exp_long_13 ===")
    print(f"video: {args.input_video}")
    print(f"segmento: {args.start_sec:.3f}s -> {args.end_sec:.3f}s")
    print(f"video fps: {video_info.fps:.3f}")
    print("sampling: source_frames, tutti i frame reali in [start_sec, end_sec)")
    print(f"feature_fps effettivo: {video_info.fps:.3f}")
    print(f"frame iniziale/finale: {int(frame_indices[0])} -> {int(frame_indices[-1])}")
    print(f"num samples: {len(timestamps)}")
    print(f"output_dir: {args.output_dir}")

    # Converte lo stesso argomento --device nei formati richiesti da PyTorch e YOLO.
    torch_device = parse_device_for_torch(args.device)
    yolo_device = parse_device_for_yolo(args.device)

    print("\n=== DINOv3 ===")
    print(f"repo:        {args.dino_repo}")
    print(f"source:      {args.dino_source}")
    print(f"weights:     {args.dino_weights}")
    print(f"model:       {args.dino_model_name}")
    print(f"input size:  {args.dino_input_size}")
    print("resize:      stretch, no center crop")
    print("output:      x_norm_clstoken")
    print(f"amp:         {USE_AMP_DINO}")

    # Inizializza DINOv3 e crea il memmap delle feature per scrivere progressivamente su disco.
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
        shape=(len(timestamps), int(args.dino_feature_dim)),
    )

    print("\n=== YOLO palla/canestro ===")
    print(f"YOLO v1 weights: {args.yolo_v1_weights}")
    print(f"YOLO v2 weights: {args.yolo_v2_weights}")
    print(f"imgsz/conf/iou:  {args.imgsz} / {args.conf} / {args.iou}")

    # Carica entrambi i detector YOLO e prepara array separati per le rispettive primitive.
    yolo_v1_model = load_yolo_model(args.yolo_v1_weights)
    yolo_v2_model = load_yolo_model(args.yolo_v2_weights)
    yolo_v1_primitives = init_primitives(len(timestamps))
    yolo_v2_primitives = init_primitives(len(timestamps))

    # Crea un iteratore batch-wise dei frame e una progress bar per monitorare l'estrazione.
    iterator = iter_frame_batches(
        video_path=args.input_video,
        frame_indices=frame_indices,
        batch_size=int(args.batch_size_decode),
    )
    progress = tqdm(
        iterator,
        total=math.ceil(len(timestamps) / int(args.batch_size_decode)),
        desc="Feature store exp_long_13",
    )

    # Per ogni batch decodificato, le stesse immagini alimentano DINOv3 e i due YOLO.
    for batch_positions, frames_bgr in progress:
        # Estrae feature DINOv3 frame-level e le scrive nelle righe corrispondenti del memmap.
        feats = extract_dino_features_for_frames(
            frames_bgr=frames_bgr,
            model=dino_model,
            transform=dino_transform,
            device=torch_device,
            batch_size=int(args.batch_size_dino),
            expected_dim=int(args.dino_feature_dim),
            use_amp=USE_AMP_DINO,
        )
        if feats.shape[0] != len(batch_positions):
            raise RuntimeError(
                f"Batch DINO con numero righe errato: {feats.shape[0]} vs {len(batch_positions)}"
            )
        dino_features[batch_positions] = feats

        # Estrae primitive palla/canestro con YOLO v1 per tutti i frame del batch.
        run_yolo_on_frames(
            model=yolo_v1_model,
            frames_bgr=frames_bgr,
            batch_positions=batch_positions,
            primitives=yolo_v1_primitives,
            source_name="yolo_v1",
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            device=yolo_device,
            batch_size=int(args.batch_size_yolo),
            ball_class_id=int(args.ball_class_id),
            rim_class_id=int(args.rim_class_id),
        )
        # Estrae primitive palla/canestro con YOLO v2 sugli stessi frame, per supportare L1/L2.
        run_yolo_on_frames(
            model=yolo_v2_model,
            frames_bgr=frames_bgr,
            batch_positions=batch_positions,
            primitives=yolo_v2_primitives,
            source_name="yolo_v2",
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            device=yolo_device,
            batch_size=int(args.batch_size_yolo),
            ball_class_id=int(args.ball_class_id),
            rim_class_id=int(args.rim_class_id),
        )

    # Forza la scrittura su disco del memmap DINO prima di chiudere il riferimento.
    dino_features.flush()
    del dino_features

    # Salva le primitive YOLO v1/v2 in file separati, mantenendo timestamp e frame_indices comuni.
    save_primitives_npz(
        args.output_dir / "yolo_v1_primitives.npz",
        yolo_v1_primitives,
        timestamps=timestamps,
        frame_indices=frame_indices,
    )
    save_primitives_npz(
        args.output_dir / "yolo_v2_primitives.npz",
        yolo_v2_primitives,
        timestamps=timestamps,
        frame_indices=frame_indices,
    )

    # Scrive il riepilogo completo della feature store per tracciabilità e debugging.
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
    print("\nFeature store exp_long_13 completata correttamente.")


if __name__ == "__main__":
    main()
