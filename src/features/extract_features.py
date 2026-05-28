from pathlib import Path
import argparse

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms

from src.data.video_io import read_video_frames


# Esempio minimale di esecuzione:
#
# python -m src.features.extract_features
#
# Di default:
# - legge il dataset da data/datasets/dataset_basket_v1
# - legge il manifest da data/datasets/dataset_basket_v1/manifest.csv
# - salva le feature in data/features/convnext_tiny_stretched_320
# - usa resize stretched 320x320 senza center crop
#
# Esempio con parametri espliciti:
#
# python -m src.features.extract_features \
#   --dataset-root data/datasets/dataset_basket_v1 \
#   --manifest data/datasets/dataset_basket_v1/manifest.csv \
#   --output-dir data/features/convnext_tiny_stretched_320 \
#   --chunk-size 150


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
    Costruisce un modello ConvNeXt-Tiny preaddestrato su ImageNet
    da usare come estrattore di feature.

    Il classificatore finale viene rimosso, quindi il modello non restituisce
    più una classe ImageNet, ma un vettore di feature da 768 valori per frame.
    """

    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)

    # ConvNeXt originale fa:
    # features -> avgpool -> classifier
    #
    # Noi rimuoviamo il Linear finale del classifier ImageNet
    # e manteniamo il vettore di feature da 768 dimensioni.
    final_norm = model.classifier[0]

    model.classifier = nn.Sequential(
        final_norm,
        nn.Flatten(1),
    )

    model.eval()
    model.to(device)

    # ConvNeXt viene usata come feature extractor frozen.
    for param in model.parameters():
        param.requires_grad = False

    return model


def build_stretched_320_transform():
    """
    Costruisce il preprocessing scelto dopo l'analisi visiva dei frame.

    Invece del preprocessing standard ConvNeXt:
        resize + center crop

    usiamo:
        resize diretto a 320x320 + normalizzazione ImageNet

    Questo significa che:
    - tutto il frame viene mantenuto;
    - non viene fatto center crop;
    - l'immagine 16:9 viene deformata/stretched in un quadrato 320x320.
    """

    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def extract_features_for_video(
    video_path: Path,
    model: nn.Module,
    transform,
    device: torch.device,
    chunk_size: int = 150,
) -> torch.Tensor:
    """
    Estrae le feature da una singola clip video.

    Restituisce un tensore [T, 768], dove T è il numero reale di frame
    della clip.
    """

    frames = read_video_frames(video_path)

    if len(frames) == 0:
        raise ValueError(f"Nessun frame letto dal video: {video_path}")

    all_features = []

    with torch.no_grad():
        for start in range(0, len(frames), chunk_size):
            chunk = frames[start:start + chunk_size]

            # Ogni frame viene trasformato in 320x320 stretched,
            # senza center crop.
            #
            # batch shape:
            # [numero_frame_chunk, 3, 320, 320]
            batch = torch.stack([transform(frame) for frame in chunk])
            batch = batch.to(device)

            # Output:
            # [numero_frame_chunk, 768]
            features = model(batch)

            # Spostiamo le feature su CPU per liberare memoria GPU.
            all_features.append(features.cpu())

    features = torch.cat(all_features, dim=0)

    return features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/datasets/dataset_basket_v1",
        help="Cartella principale del dataset.",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="data/datasets/dataset_basket_v1/manifest.csv",
        help="Percorso del manifest.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features/convnext_tiny_stretched_320",
        help="Cartella in cui salvare le feature estratte.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=150,
        help="Numero di frame processati contemporaneamente.",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Se attivo, NON ricalcola le feature già esistenti.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Di default sovrascriviamo le feature.
    # Se usi --no-overwrite, salta quelle già esistenti.
    overwrite = not args.no_overwrite

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {output_dir}")
    print(f"Chunk size: {args.chunk_size}")
    print("Transform: stretched resize 320x320, no center crop")
    print(f"Overwrite: {overwrite}")

    df = pd.read_csv(manifest_path)

    # Nuovo transform: 320x320 stretched senza center crop.
    transform = build_stretched_320_transform()

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

        if save_path.exists() and not overwrite:
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

                # Metadati sul preprocessing usato.
                "resize_mode": "stretched",
                "input_size": 320,
                "center_crop": False,
                "normalization": "imagenet",
            }

            torch.save(item, save_path)

        except Exception as e:
            print(f"Errore su {video_path}: {e}")


if __name__ == "__main__":
    main()