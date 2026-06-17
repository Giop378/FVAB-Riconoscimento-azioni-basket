from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import torch

from src.long_video import defaults


EXPECTED_TRACKING = {
    "L1": {
        "checkpoint": defaults.EXP46_L1_CHECKPOINT,
        "source": defaults.L1_TRACKING_SOURCE,
        "type": defaults.L1_TRACKING_TYPE,
        "num_features": 43,
        "expected_input_dim": defaults.DINOV3_FEATURE_DIM + 43,
    },
    "L2": {
        "checkpoint": defaults.EXP46_L2_CHECKPOINT,
        "source": defaults.L2_TRACKING_SOURCE,
        "type": defaults.L2_TRACKING_TYPE,
        "num_features": 29,
        "expected_input_dim": defaults.DINOV3_FEATURE_DIM + 29,
    },
    "L3": {
        "checkpoint": defaults.EXP46_L3_CHECKPOINT,
        "source": defaults.L3_TRACKING_SOURCE,
        "type": defaults.L3_TRACKING_TYPE,
        "num_features": 43,
        "expected_input_dim": defaults.DINOV3_FEATURE_DIM + 43,
    },
}


def _as_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(value)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint in a way that works across torch versions."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint non valido: atteso dict, ottenuto {type(ckpt)}")
    return ckpt


def check_path(path: Path, name: str, must_be_dir: bool | None = None) -> None:
    path = _as_path(path)

    if not path.exists():
        raise FileNotFoundError(f"{name} non trovato: {path}")

    if must_be_dir is True and not path.is_dir():
        raise NotADirectoryError(f"{name} dovrebbe essere una cartella: {path}")

    if must_be_dir is False and not path.is_file():
        raise FileNotFoundError(f"{name} dovrebbe essere un file: {path}")

    print(f"[OK] {name}: {path}")


def get_video_info(video_path: Path) -> dict[str, float | int]:
    video_path = _as_path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"FPS non valido per il video: {video_path} -> {fps}")

    duration_sec = float(num_frames / fps)

    return {
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def check_video_segment(video_path: Path, start_sec: float | None, end_sec: float | None, name: str) -> None:
    info = get_video_info(video_path)

    print(f"\n=== Video {name} ===")
    print(f"path: {video_path}")
    print(f"fps: {info['fps']:.3f}")
    print(f"frames: {info['num_frames']}")
    print(f"resolution: {info['width']}x{info['height']}")
    print(f"duration: {info['duration_sec']:.3f}s")

    if start_sec is None or end_sec is None:
        print("segmento: non impostato, da passare in seguito da riga di comando")
        return

    if start_sec < 0:
        raise ValueError(f"{name}: start_sec negativo: {start_sec}")

    if end_sec <= start_sec:
        raise ValueError(f"{name}: end_sec deve essere > start_sec: {start_sec} -> {end_sec}")

    if end_sec > float(info["duration_sec"]):
        raise ValueError(
            f"{name}: segmento fuori dal video. "
            f"Richiesto end_sec={end_sec:.3f}, durata={float(info['duration_sec']):.3f}"
        )

    print(f"segmento: {start_sec:.3f}s -> {end_sec:.3f}s ({end_sec - start_sec:.3f}s)")
    print("[OK] segmento valido")


def find_tracking_config(ckpt: dict[str, Any]) -> dict[str, Any] | None:
    tracking_config = ckpt.get("tracking_config")
    if tracking_config is not None:
        return tracking_config

    model_config = ckpt.get("model_config")
    if isinstance(model_config, dict):
        tracking_config = model_config.get("tracking_config")
        if tracking_config is not None:
            return tracking_config

    return None


def find_model_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    model_config = ckpt.get("model_config")
    if isinstance(model_config, dict):
        return model_config
    return {}


def inspect_checkpoint(level_name: str, expected: dict[str, Any], print_features: bool) -> None:
    checkpoint_path = _as_path(expected["checkpoint"])

    print(f"\n=== Checkpoint {level_name} ===")
    print(f"path: {checkpoint_path}")
    print(f"tracking atteso: {expected['source']} {expected['type']}")

    ckpt = load_checkpoint(checkpoint_path)
    model_config = find_model_config(ckpt)
    tracking_config = find_tracking_config(ckpt)

    if model_config:
        for key in ["input_dim", "d_model", "num_layers", "num_heads", "ff_dim", "dropout", "pooling"]:
            if key in model_config:
                print(f"model_config.{key}: {model_config[key]}")

        input_dim = model_config.get("input_dim")
        if input_dim is not None and int(input_dim) != int(expected["expected_input_dim"]):
            raise ValueError(
                f"{level_name}: input_dim inatteso. "
                f"Trovato {input_dim}, atteso {expected['expected_input_dim']} "
                f"(DINO {defaults.DINOV3_FEATURE_DIM} + tracking {expected['num_features']})"
            )

    if tracking_config is None:
        raise KeyError(f"{level_name}: tracking_config non trovato nel checkpoint")

    tracking_type = tracking_config.get("type")
    feature_names = tracking_config.get("feature_names", [])

    if not isinstance(feature_names, (list, tuple)):
        raise TypeError(f"{level_name}: tracking_config.feature_names non è una lista")

    print(f"tracking_config.type: {tracking_type}")
    print(f"num tracking feature: {len(feature_names)}")

    if len(feature_names) != int(expected["num_features"]):
        raise ValueError(
            f"{level_name}: numero feature tracking errato. "
            f"Trovate {len(feature_names)}, attese {expected['num_features']}"
        )

    if tracking_type is not None and str(expected["type"]) not in str(tracking_type):
        print(
            f"[WARN] {level_name}: tracking_config.type='{tracking_type}' non contiene "
            f"esplicitamente '{expected['type']}'. Controlla che sia corretto."
        )

    if print_features:
        print("feature_names:")
        for i, name in enumerate(feature_names, start=1):
            print(f"  {i:02d}. {name}")

    print(f"[OK] {level_name}: checkpoint coerente")


def write_summary(output_json: Path) -> None:
    summary = {
        "validation": {
            "video_id": defaults.VAL_VIDEO_ID,
            "video_path": str(defaults.VAL_VIDEO_PATH),
            "start_sec": defaults.VAL_START_SEC,
            "end_sec": defaults.VAL_END_SEC,
            "feature_store_dir": str(defaults.VAL_FEATURE_STORE_DIR),
            "output_dir": str(defaults.VAL_OUTPUT_DIR),
        },
        "test": {
            "video_id": defaults.TEST_VIDEO_ID,
            "video_path": str(defaults.TEST_VIDEO_PATH),
            "start_sec": defaults.TEST_START_SEC,
            "end_sec": defaults.TEST_END_SEC,
            "feature_store_dir": str(defaults.TEST_FEATURE_STORE_DIR),
            "output_dir": str(defaults.TEST_OUTPUT_DIR),
        },
        "dinov3": {
            "repo": str(defaults.DINOV3_REPO),
            "input_size": defaults.DINOV3_INPUT_SIZE,
            "feature_dim": defaults.DINOV3_FEATURE_DIM,
        },
        "yolo": {
            "v1_weights": str(defaults.YOLO_V1_WEIGHTS),
            "v2_weights": str(defaults.YOLO_V2_WEIGHTS),
        },
        "exp46": {
            level: {
                "checkpoint": str(cfg["checkpoint"]),
                "tracking_source": cfg["source"],
                "tracking_type": cfg["type"],
                "num_tracking_features": cfg["num_features"],
                "expected_input_dim": cfg["expected_input_dim"],
            }
            for level, cfg in EXPECTED_TRACKING.items()
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[OK] Summary salvato in: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlla path, video e checkpoint necessari per la pipeline long-video basata su exp_46."
    )
    parser.add_argument(
        "--check-video-duration",
        action="store_true",
        help="Controlla durata e segmenti dei video con OpenCV.",
    )
    parser.add_argument(
        "--print-features",
        action="store_true",
        help="Stampa l'ordine completo delle feature tracking salvate nei checkpoint.",
    )
    parser.add_argument(
        "--write-summary-json",
        type=Path,
        default=None,
        help="Opzionale: salva un riepilogo JSON dei path e della configurazione controllata.",
    )
    args = parser.parse_args()

    print("=== Controllo path principali ===")
    check_path(defaults.VAL_VIDEO_PATH, "Video validation", must_be_dir=False)
    check_path(defaults.TEST_VIDEO_PATH, "Video test", must_be_dir=False)
    check_path(defaults.DINOV3_REPO, "DINOv3 repo", must_be_dir=True)
    check_path(defaults.YOLO_V1_WEIGHTS, "YOLO v1 weights", must_be_dir=False)
    check_path(defaults.YOLO_V2_WEIGHTS, "YOLO v2 weights", must_be_dir=False)
    check_path(defaults.EXP46_L1_CHECKPOINT, "Checkpoint L1 exp_46", must_be_dir=False)
    check_path(defaults.EXP46_L2_CHECKPOINT, "Checkpoint L2 exp_46", must_be_dir=False)
    check_path(defaults.EXP46_L3_CHECKPOINT, "Checkpoint L3 exp_46", must_be_dir=False)

    print("\n=== Cartelle output previste ===")
    print(f"VAL feature store: {defaults.VAL_FEATURE_STORE_DIR}")
    print(f"VAL output dir:    {defaults.VAL_OUTPUT_DIR}")
    print(f"TEST feature store:{defaults.TEST_FEATURE_STORE_DIR}")
    print(f"TEST output dir:   {defaults.TEST_OUTPUT_DIR}")

    if args.check_video_duration:
        check_video_segment(
            defaults.VAL_VIDEO_PATH,
            defaults.VAL_START_SEC,
            defaults.VAL_END_SEC,
            "validation",
        )
        check_video_segment(
            defaults.TEST_VIDEO_PATH,
            defaults.TEST_START_SEC,
            defaults.TEST_END_SEC,
            "test",
        )

    print("\n=== Controllo checkpoint exp_46 ===")
    for level_name, expected in EXPECTED_TRACKING.items():
        inspect_checkpoint(level_name, expected, print_features=args.print_features)

    if args.write_summary_json is not None:
        write_summary(args.write_summary_json)

    print("\nTutti i controlli sono stati completati correttamente.")


if __name__ == "__main__":
    main()
