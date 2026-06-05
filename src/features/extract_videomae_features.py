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
    Legge tutti i frame del video e li restituisce in RGB.

    Output:
        lista di array uint8 con shape [H, W, 3].
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    frames = []

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
    Campiona un numero fisso di frame.

    - Se la clip ha almeno num_frames, campiona uniformemente lungo tutta la clip.
    - Se la clip ha meno di num_frames, usa tutti i frame disponibili e ripete
      l'ultimo frame fino ad arrivare a num_frames.

    Questo evita di scartare le clip corte, mantenendo compatibilità con VideoMAE.
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


@torch.no_grad()
def extract_videomae_feature(
    model: VideoMAEModel,
    image_processor: AutoImageProcessor,
    frames: List[np.ndarray],
    device: torch.device,
    use_fp16: bool = False,
) -> torch.Tensor:
    """
    Estrae una feature clip-level da VideoMAE.

    VideoMAE restituisce una sequenza di token spazio-temporali nel
    last_hidden_state. Per ottenere una singola feature per clip facciamo
    mean pooling sui token.

    Output:
        tensor CPU float32 con shape [1, hidden_size]
    """
    inputs = image_processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    if use_fp16:
        pixel_values = pixel_values.half()

    outputs = model(pixel_values=pixel_values)

    # last_hidden_state: [B, num_tokens, hidden_size]
    clip_feature = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]

    return clip_feature.squeeze(0).float().cpu().unsqueeze(0)  # [1, hidden_size]


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
            "Estrae feature clip-level con VideoMAE usando 16 frame per clip. "
            "Le feature vengono salvate nello stesso formato atteso da FeatureDataset."
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
        help="Cartella in cui salvare le feature, es. data/features/videomae_base_16.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="MCG-NJU/videomae-base",
        help="Checkpoint Hugging Face VideoMAE.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Numero di frame da campionare per clip. Per VideoMAE base usare 16.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Per ora lasciato per compatibilità. L'estrazione è clip-by-clip "
            "per gestire video di durata variabile in modo semplice."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Non usato in questa versione clip-by-clip; lasciato per compatibilità.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

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
    print(f"Shot only: {args.shot_only}")
    print(f"Clip da processare: {len(manifest)}")

    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = VideoMAEModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    if args.fp16:
        if device.type != "cuda":
            print("--fp16 richiesto ma CUDA non disponibile: continuo in float32.")
            use_fp16 = False
        else:
            model.half()
            use_fp16 = True
    else:
        use_fp16 = False

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="VideoMAE features"):
        label_name = str(row["label"])

        if label_name not in LABEL_TO_IDX:
            errors.append({
                "clip_id": get_row_value(row, "clip_id", default="unknown"),
                "error": f"Label non riconosciuta: {label_name}",
            })
            continue

        split = str(row["split"])
        clip_id = str(get_row_value(row, "clip_id", default=Path(str(get_row_value(row, "path"))).stem))
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

            feature = extract_videomae_feature(
                model=model,
                image_processor=image_processor,
                frames=sampled_frames,
                device=device,
                use_fp16=use_fp16,
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)

            item = {
                "features": feature,  # [1, hidden_size]
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
                "pooling": "token_mean",
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
        "shot_only": args.shot_only,
        "output_dir": str(output_dir),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "num_errors": len(errors),
    }

    summary_path = output_dir / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEstrazione completata.")
    print(f"Feature create: {processed}")
    print(f"Feature saltate perché già esistenti: {skipped}")
    print(f"Errori: {len(errors)}")
    print(f"Summary: {summary_path}")

    if errors:
        print("\nPrimi errori:")
        for error in errors[:10]:
            print(error)


if __name__ == "__main__":
    main()
