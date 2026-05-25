from pathlib import Path
import argparse

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models

from src.data.video_io import read_video_frames


# Per eseguire questo codice:
# python -m src.features.extract_features \
#   --dataset-root data/datasets/dataset_basket_v1 \
#   --manifest data/datasets/dataset_basket_v1/manifest.csv \
#   --output-dir data/features/convnext_tiny

LABELS = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
    "idle",
    "non-gioco",
]


def build_convnext_tiny_feature_extractor(device: torch.device) -> nn.Module:
    """
    ConvNeXt-Tiny preaddestrata su ImageNet.
    Rimuoviamo il classificatore finale e teniamo feature da 768 dimensioni.
    """
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)

    # ConvNeXt fa:
    # features -> avgpool -> classifier
    # Manteniamo la normalizzazione finale del classifier e rimuoviamo il Linear finale.
    final_norm = model.classifier[0]
    model.classifier = nn.Sequential(
        final_norm,
        nn.Flatten(1),
    )

    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    return model


def extract_features_for_video(
    video_path: Path,
    model: nn.Module,
    transform,
    device: torch.device,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Restituisce un tensore [T, 768], dove T è il numero reale di frame della clip.
    """
    frames = read_video_frames(video_path)

    all_features = []

    with torch.no_grad():
        for start in range(0, len(frames), chunk_size):
            chunk = frames[start:start + chunk_size]

            batch = torch.stack([transform(frame) for frame in chunk])
            batch = batch.to(device)

            features = model(batch)              # [chunk_size, 768]
            all_features.append(features.cpu())

    features = torch.cat(all_features, dim=0)    # [T, 768]
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Se attivo, ricalcola le feature anche se il file .pt esiste già.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(manifest_path)

    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    transform = weights.transforms()

    model = build_convnext_tiny_feature_extractor(device)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        clip_id = row["clip_id"]
        rel_path = row["path"]
        split = row["split"]
        label = row["label"]

        if label not in LABELS:
            print(f"Label non riconosciuta per {clip_id}: {label}")
            continue

        video_path = dataset_root / rel_path

        save_dir = output_dir / split / label
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{clip_id}.pt"

        if save_path.exists() and not args.overwrite:
            continue

        try:
            features = extract_features_for_video(
                video_path=video_path,
                model=model,
                transform=transform,
                device=device,
                chunk_size=args.chunk_size,
            )

            item = {
                "clip_id": clip_id,
                "features": features,       # [T, 768]
                "label": label,
                "path": rel_path,
                "num_frames": features.shape[0],
                "feature_extractor": "convnext_tiny",
                "feature_dim": features.shape[1],
            }

            torch.save(item, save_path)

        except Exception as e:
            print(f"Errore su {video_path}: {e}")


if __name__ == "__main__":
    main()