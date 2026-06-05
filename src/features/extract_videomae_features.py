from pathlib import Path
import argparse
import json
import random
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, VideoMAEModel

from src.data.feature_dataset import LABEL_TO_IDX


SHOT_LABELS = {
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_video_rgb_frames(video_path: Path) -> List[np.ndarray]:
    """
    Legge tutti i frame del video e li restituisce in RGB uint8 [H, W, 3].
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    frames: List[np.ndarray] = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video senza frame leggibili: {video_path}")

    return frames


def sample_fixed_num_frames(
    frames: List[np.ndarray],
    num_frames: int,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Campiona num_frames frame uniformemente lungo tutta la clip.

    Se la clip ha meno di num_frames, ripete l'ultimo frame. Questo permette di
    usare VideoMAE a 16 frame senza scartare le clip più corte.
    """
    total_frames = len(frames)

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int)
        sampled_frames = [frames[int(idx)] for idx in indices]
        sampled_indices = [int(idx) for idx in indices]
    else:
        sampled_frames = list(frames)
        sampled_indices = list(range(total_frames))

        last_frame = frames[-1]
        last_index = total_frames - 1

        while len(sampled_frames) < num_frames:
            sampled_frames.append(last_frame)
            sampled_indices.append(last_index)

    return sampled_frames, sampled_indices


def preprocess_frames_stretched(
    frames: List[np.ndarray],
    image_size: int,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    """
    Preprocessing conservativo senza center crop.

    Ogni frame viene ridimensionato direttamente a image_size x image_size,
    quindi in modalità stretched. Questo evita di tagliare parti laterali del
    campo dove possono trovarsi palla e canestro.

    Output:
        pixel_values [1, T, 3, H, W]
    """
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    processed = []
    for frame in frames:
        resized = cv2.resize(
            frame,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        resized = resized.astype(np.float32) / 255.0
        normalized = (resized - mean_arr) / std_arr
        chw = np.transpose(normalized, (2, 0, 1))
        processed.append(chw)

    pixel_values = torch.from_numpy(np.stack(processed, axis=0)).float()
    pixel_values = pixel_values.unsqueeze(0)  # [1, T, C, H, W]
    return pixel_values


def get_spatial_tokens_per_temporal_step(model: VideoMAEModel, image_size: int) -> int:
    """
    Calcola quanti token spaziali corrispondono a ogni step temporale VideoMAE.

    Per VideoMAE base tipicamente:
        image_size = 224
        patch_size = 16
        spatial_tokens = 14 * 14 = 196
    """
    patch_size = getattr(model.config, "patch_size", 16)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]

    spatial_size = image_size // int(patch_size)
    return spatial_size * spatial_size


@torch.no_grad()
def extract_videomae_temporal_feature(
    model: VideoMAEModel,
    pixel_values: torch.Tensor,
    device: torch.device,
    image_size: int,
    use_fp16: bool = False,
) -> torch.Tensor:
    """
    Estrae una sequenza temporale da VideoMAE, non una singola feature globale.

    VideoMAE produce token spazio-temporali. Invece di fare una media globale
    di tutti i token, li ricostruiamo come:

        [num_temporal_tokens, num_spatial_tokens, hidden_size]

    e poi facciamo mean pooling solo sui token spaziali. In questo modo resta
    una sequenza temporale:

        features.shape = [num_temporal_tokens, hidden_size]

    Con VideoMAE base, 16 frame e tubelet_size=2, solitamente si ottiene:

        features.shape = [8, 768]

    Questa è la correzione rispetto al vecchio esperimento, dove veniva salvata
    una sola feature [1, 768] e il classificatore riceveva una sequenza lunga 1.
    """
    pixel_values = pixel_values.to(device)
    if use_fp16:
        pixel_values = pixel_values.half()

    outputs = model(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state.squeeze(0)  # [N, D]

    spatial_tokens = get_spatial_tokens_per_temporal_step(model, image_size=image_size)

    if tokens.shape[0] % spatial_tokens != 0:
        raise RuntimeError(
            "Numero di token VideoMAE non divisibile per i token spaziali. "
            f"tokens={tokens.shape[0]}, spatial_tokens={spatial_tokens}. "
            "Controlla image_size/patch_size del modello."
        )

    temporal_tokens = tokens.shape[0] // spatial_tokens

    # [temporal_tokens, spatial_tokens, hidden_size]
    tokens = tokens.reshape(temporal_tokens, spatial_tokens, tokens.shape[-1])

    # Mean pooling spaziale, mantenendo la dimensione temporale.
    temporal_features = tokens.mean(dim=1)  # [temporal_tokens, hidden_size]

    return temporal_features.float().cpu()


def get_row_value(row: pd.Series, *names: str, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def build_video_path(dataset_root: Path, row: pd.Series) -> Path:
    rel_path = get_row_value(row, "path", "filepath", "file_path", "video_path")

    if rel_path is None:
        raise KeyError(
            "Nel manifest non trovo una colonna path/filepath/file_path/video_path."
        )

    video_path = Path(str(rel_path))
    if not video_path.is_absolute():
        video_path = dataset_root / video_path

    return video_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estrae feature VideoMAE temporali a 16 frame, senza center crop, "
            "salvandole nello stesso formato atteso da FeatureDataset."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root del dataset contenente i video, es. data/datasets/dataset_basket_v1.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path del manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Cartella in cui salvare le feature, "
            "es. data/features/videomae_kinetics16_temporal_stretch."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="MCG-NJU/videomae-base-finetuned-kinetics",
        help=(
            "Checkpoint Hugging Face VideoMAE. Default: modello fine-tuned "
            "su Kinetics, più adatto di videomae-base puro."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Numero di frame da campionare per clip. Per VideoMAE base usare 16.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Risoluzione quadrata stretched. Default 224, come VideoMAE base.",
    )
    parser.add_argument(
        "--shot-only",
        action="store_true",
        help="Estrae feature solo per le clip di tiro. Consigliato per L3 shot_outcome_only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rigenera anche le feature già esistenti.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Usa il modello in float16 su CUDA per ridurre memoria e accelerare.",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    required_columns = {"split", "label"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise KeyError(f"Colonne mancanti nel manifest: {sorted(missing_columns)}")

    if args.shot_only:
        manifest = manifest[manifest["label"].isin(SHOT_LABELS)].copy()

    if len(manifest) == 0:
        raise RuntimeError("Nessuna clip da processare dopo il filtraggio.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Num frames: {args.num_frames}")
    print(f"Image size: {args.image_size}x{args.image_size} stretched")
    print("Center crop: disattivato")
    print("Output features: sequenza temporale VideoMAE, es. [8, 768]")
    print(f"Shot only: {args.shot_only}")
    print(f"Clip da processare: {len(manifest)}")

    # Carichiamo comunque l'image processor per usare mean/std ufficiali del checkpoint,
    # ma non gli facciamo fare resize/crop: il preprocessing stretched è manuale.
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    mean = getattr(image_processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(image_processor, "image_std", [0.229, 0.224, 0.225])

    model = VideoMAEModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    if args.fp16 and device.type == "cuda":
        model.half()
        use_fp16 = True
    else:
        if args.fp16 and device.type != "cuda":
            print("--fp16 richiesto ma CUDA non disponibile: continuo in float32.")
        use_fp16 = False

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors = []
    feature_shapes = {}

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="VideoMAE temporal features"):
        label_name = str(row["label"])

        if label_name not in LABEL_TO_IDX:
            errors.append({
                "clip_id": get_row_value(row, "clip_id", default="unknown"),
                "error": f"Label non riconosciuta: {label_name}",
            })
            continue

        split = str(row["split"])
        path_value = get_row_value(row, "path", "filepath", "file_path", "video_path")
        clip_id = str(get_row_value(row, "clip_id", default=Path(str(path_value)).stem))
        video_path = build_video_path(dataset_root, row)

        out_path = output_dir / split / label_name / f"{clip_id}.pt"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            frames = read_video_rgb_frames(video_path)
            sampled_frames, sampled_indices = sample_fixed_num_frames(
                frames=frames,
                num_frames=args.num_frames,
            )

            pixel_values = preprocess_frames_stretched(
                frames=sampled_frames,
                image_size=args.image_size,
                mean=mean,
                std=std,
            )

            features = extract_videomae_temporal_feature(
                model=model,
                pixel_values=pixel_values,
                device=device,
                image_size=args.image_size,
                use_fp16=use_fp16,
            )

            shape_key = str(tuple(features.shape))
            feature_shapes[shape_key] = feature_shapes.get(shape_key, 0) + 1

            out_path.parent.mkdir(parents=True, exist_ok=True)

            item = {
                "features": features,  # [T_videomae, hidden_size], es. [8, 768]
                "label": LABEL_TO_IDX[label_name],
                "label_name": label_name,
                "clip_id": clip_id,
                "split": split,
                "video_path": str(video_path),
                "num_original_frames": len(frames),
                "num_sampled_frames": args.num_frames,
                "sampled_indices": sampled_indices,
                "feature_extractor": "VideoMAE",
                "model_name": args.model_name,
                "preprocessing": "stretch_no_center_crop",
                "image_size": args.image_size,
                "token_pooling": "spatial_mean_keep_temporal",
            }

            torch.save(item, out_path)
            processed += 1

        except Exception as exc:
            errors.append({
                "clip_id": clip_id,
                "path": str(video_path),
                "error": repr(exc),
            })

    summary = {
        "model_name": args.model_name,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "shot_only": args.shot_only,
        "preprocessing": "stretch_no_center_crop",
        "output_dir": str(output_dir),
        "processed": processed,
        "skipped": skipped,
        "feature_shapes": feature_shapes,
        "errors": errors,
        "num_errors": len(errors),
    }

    summary_path = output_dir / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEstrazione completata.")
    print(f"Feature create: {processed}")
    print(f"Feature saltate perché già esistenti: {skipped}")
    print(f"Feature shapes: {feature_shapes}")
    print(f"Errori: {len(errors)}")
    print(f"Summary: {summary_path}")

    if errors:
        print("\nPrimi errori:")
        for error in errors[:10]:
            print(error)


if __name__ == "__main__":
    main()
