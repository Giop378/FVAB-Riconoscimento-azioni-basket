from pathlib import Path
import argparse

import pandas as pd
import torch
from tqdm import tqdm

from src.data.video_io import read_video_frames
from src.features.dinov3_extractor import (
    DINO_FEATURE_DIMS,
    DINOv3FeatureExtractor,
    build_dino_transform,
    extract_clip_features,
)


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

LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}


# Applichiamo augmentation solo alle classi di tiro e solo nello split train.
# Le classi passaggio, idle e non-gioco restano non augmentate perché già numerose.
AUGMENT_LABELS = {
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
}


def parse_device(device_arg: str) -> torch.device:
    """Accetta cpu, cuda, cuda:N oppure N."""
    device_arg = str(device_arg)
    if device_arg.lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if device_arg == "cuda":
            return torch.device("cuda")
        if device_arg.startswith("cuda"):
            return torch.device(device_arg)
        return torch.device(f"cuda:{device_arg}")
    print("[WARN] CUDA non disponibile: uso CPU.")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estrae feature DINOv3 clip-level. "
            "La logica DINO è condivisa con la pipeline long-video."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root del dataset video, es. data/datasets/dataset_basket_v1",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path del manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Cartella di output delle feature estratte.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov3_vitl16",
        choices=list(DINO_FEATURE_DIMS.keys()),
        help="Modello DINOv3 da usare.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=(
            "Path locale o URL dei pesi DINOv3. "
            "Esempio: checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ),
    )
    parser.add_argument(
        "--repo-or-dir",
        type=str,
        default="facebookresearch/dinov3",
        help=(
            "Repository GitHub o path locale del repository DINOv3. "
            "Esempio GitHub: facebookresearch/dinov3. "
            "Esempio locale: third_party/dinov3"
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="github",
        choices=["github", "local"],
        help=(
            "Usa 'github' se vuoi caricare da facebookresearch/dinov3; "
            "usa 'local' se hai clonato il repository DINOv3 in locale."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=336,
        help="Dimensione del resize quadrato. Per DINOv3 ViT-* /16 usare multipli di 16.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Numero di frame processati insieme dalla GPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu, cuda, cuda:N oppure N.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Se attivo, sovrascrive feature già esistenti.",
    )
    parser.add_argument(
        "--augment-train-shots",
        action="store_true",
        help="Se attivo, applica augmentation leggera solo alle classi di tiro nello split train.",
    )
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=["hflip", "color"],
        choices=["hflip", "color", "hflip_color"],
        help="Lista di augmentation da applicare alle clip di tiro nel train.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    if args.model_name not in DINO_FEATURE_DIMS:
        raise ValueError(f"Modello DINOv3 non supportato: {args.model_name}")

    feature_dim = int(DINO_FEATURE_DIMS[args.model_name])
    device = parse_device(args.device)

    print(f"Device: {device}")
    print(f"Modello DINOv3: {args.model_name}")
    print(f"Feature dim attesa: {feature_dim}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Output dir: {output_dir}")
    print(f"Repo/dir DINOv3: {args.repo_or_dir}")
    print(f"Source: {args.source}")
    print(f"Weights: {args.weights}")
    print(f"Output token DINO: x_norm_clstoken")
    print(f"Resize policy: stretch 336x336, no center crop")
    print(f"Augment train shots: {args.augment_train_shots}")
    if args.augment_train_shots:
        print(f"Augmentations: {args.augmentations}")
        print(f"Classi augmentate: {sorted(AUGMENT_LABELS)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)

    model = DINOv3FeatureExtractor(
        model_name=args.model_name,
        weights=args.weights,
        repo_or_dir=args.repo_or_dir,
        source=args.source,
    ).to(device)
    model.eval()

    transform = build_dino_transform(args.image_size)

    num_ok = 0
    num_skipped = 0
    num_errors = 0

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        clip_id = str(row.clip_id)
        rel_path = Path(row.path)
        label = str(row.label)
        split = str(row.split)

        if label not in LABEL_TO_IDX:
            raise ValueError(f"Label non riconosciuta: {label}")

        video_path = dataset_root / rel_path

        variants = ["orig"]
        if args.augment_train_shots and split == "train" and label in AUGMENT_LABELS:
            variants.extend(args.augmentations)

        if not args.overwrite:
            existing_variants = 0
            for augmentation in variants:
                out_clip_id = clip_id if augmentation == "orig" else f"{clip_id}_{augmentation}"
                out_path = output_dir / split / label / f"{out_clip_id}.pt"
                if out_path.exists():
                    existing_variants += 1

            if existing_variants == len(variants):
                num_skipped += len(variants)
                continue

        try:
            frames = read_video_frames(video_path)

            for augmentation in variants:
                out_clip_id = clip_id if augmentation == "orig" else f"{clip_id}_{augmentation}"
                out_path = output_dir / split / label / f"{out_clip_id}.pt"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_path.exists() and not args.overwrite:
                    num_skipped += 1
                    continue

                features = extract_clip_features(
                    frames=frames,
                    model=model,
                    transform=transform,
                    device=device,
                    chunk_size=args.chunk_size,
                    augmentation=augmentation,
                )

                if features.ndim != 2:
                    raise ValueError(f"Feature con shape non valida: {features.shape}")
                if features.shape[1] != feature_dim:
                    raise ValueError(
                        f"Feature dim inattesa per {args.model_name}: "
                        f"ottenuto {features.shape[1]}, atteso {feature_dim}"
                    )

                torch.save(
                    {
                        "features": features,
                        "label": LABEL_TO_IDX[label],
                        "label_name": label,
                        "clip_id": out_clip_id,
                        "source_clip_id": clip_id,
                        "path": str(rel_path),
                        "split": split,
                        "model_name": args.model_name,
                        "weights": str(args.weights),
                        "feature_dim": feature_dim,
                        "image_size": args.image_size,
                        "resize_mode": "stretch",
                        "center_crop": False,
                        "normalization": "imagenet",
                        "output_token": "x_norm_clstoken",
                        "augmentation": augmentation,
                    },
                    out_path,
                )

                num_ok += 1

        except Exception as exc:  # noqa: BLE001
            num_errors += 1
            print(f"\nErrore su {video_path}: {exc}")

    print("\nEstrazione completata.")
    print(f"Feature salvate: {num_ok}")
    print(f"Feature saltate perché già esistenti: {num_skipped}")
    print(f"Clip con errore: {num_errors}")
    print(f"Feature salvate in: {output_dir}")


if __name__ == "__main__":
    main()
