from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.video_io import read_video_frames


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


# Per ora teniamo solo i modelli DINOv3 più utili per il progetto.
# DINOv3 ViT-B/16 -> feature da 768
# DINOv3 ViT-L/16 -> feature da 1024
DINO_FEATURE_DIMS = {
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
}


class DINOv3FeatureExtractor(nn.Module):
    """
    Wrapper per DINOv3.

    Usiamo il token globale CLS come feature del singolo frame.

    Input:
        x: [B, 3, H, W]

    Output:
        features: [B, D]

    dove D dipende dal backbone:
        dinov3_vitb16 -> 768
        dinov3_vitl16 -> 1024
    """

    def __init__(
        self,
        model_name: str,
        weights: str,
        repo_or_dir: str,
        source: str,
    ):
        super().__init__()

        self.model_name = model_name

        if source == "local":
            self.backbone = torch.hub.load(
                repo_or_dir,
                model_name,
                source="local",
                weights=weights,
            )
        else:
            self.backbone = torch.hub.load(
                repo_or_dir,
                model_name,
                source="github",
                weights=weights,
                trust_repo=True,
            )

        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        Estrae una feature globale per ogni frame.

        DINOv3, a seconda del modello/versione, può esporre:
        - forward_features con output dizionario
        - forward diretto che restituisce già il CLS token

        Qui gestiamo entrambi i casi.
        """

        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)

            if isinstance(out, dict):
                # Caso più comune per modelli ViT stile DINO.
                if "x_norm_clstoken" in out:
                    return out["x_norm_clstoken"]

                # Fallback: primo token della sequenza.
                if "x_prenorm" in out:
                    return out["x_prenorm"][:, 0]

                if "x_norm_patchtokens" in out:
                    # Ultimo fallback ragionevole: media dei patch tokens.
                    return out["x_norm_patchtokens"].mean(dim=1)

            if torch.is_tensor(out):
                if out.ndim == 2:
                    return out

                if out.ndim == 3:
                    return out[:, 0]

        out = self.backbone(x)

        if torch.is_tensor(out):
            if out.ndim == 2:
                return out

            if out.ndim == 3:
                return out[:, 0]

        raise RuntimeError(
            f"Output DINOv3 non gestito. Tipo: {type(out)}"
        )


def build_transform(image_size: int):
    """
    Preprocessing per DINOv3.

    Manteniamo la stessa scelta usata negli esperimenti DINOv2:
    - resize stretched quadrato
    - no center crop
    - normalizzazione ImageNet

    Per DINOv3 ViT-* /16 è consigliato usare dimensioni multiple di 16.
    336 = 16 * 21, quindi va bene.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def frame_to_pil(frame):
    """
    Converte un frame in PIL RGB.

    Gestisce:
    - PIL.Image
    - numpy array [H, W, C]
    - torch tensor [H, W, C]
    - torch tensor [C, H, W]
    """

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, np.ndarray):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return Image.fromarray(frame).convert("RGB")

    if torch.is_tensor(frame):
        frame = frame.detach().cpu()

        if frame.ndim != 3:
            raise ValueError(f"Frame tensor con shape non valida: {frame.shape}")

        # Se è [H, W, C], lo porto a [C, H, W].
        if frame.shape[-1] in (1, 3):
            frame = frame.permute(2, 0, 1)

        if frame.dtype != torch.uint8:
            if frame.max() <= 1.0:
                frame = frame * 255.0

            frame = frame.clamp(0, 255).to(torch.uint8)

        return transforms.functional.to_pil_image(frame).convert("RGB")

    raise TypeError(f"Tipo frame non supportato: {type(frame)}")


@torch.no_grad()
def extract_clip_features(
    frames,
    model,
    transform,
    device,
    chunk_size: int,
):
    """
    Estrae feature DINOv3 da tutti i frame di una clip.

    Input:
        frames: lista di frame della clip

    Output:
        Tensor [T, D]
        dove:
            T = numero di frame della clip
            D = dimensione della feature DINOv3
    """

    if len(frames) == 0:
        raise ValueError("Clip senza frame.")

    all_features = []

    for start_idx in range(0, len(frames), chunk_size):
        chunk = frames[start_idx : start_idx + chunk_size]

        batch = torch.stack(
            [transform(frame_to_pil(frame)) for frame in chunk],
            dim=0,
        ).to(device)

        features = model(batch)

        if features.ndim != 2:
            raise ValueError(f"Feature con shape non valida: {features.shape}")

        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def parse_args():
    parser = argparse.ArgumentParser()

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
            "Esempio: checkpoints/dinov3_vitl16_pretrain_lvd1689m.pth"
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
        help="cuda oppure cpu.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Se attivo, sovrascrive feature già esistenti.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    if args.model_name not in DINO_FEATURE_DIMS:
        raise ValueError(f"Modello DINOv3 non supportato: {args.model_name}")

    feature_dim = DINO_FEATURE_DIMS[args.model_name]

    if args.image_size % 16 != 0:
        raise ValueError(
            f"image_size={args.image_size} non è multiplo di 16. "
            "Per DINOv3 ViT-* /16 usa una dimensione multipla di 16, ad esempio 224, 320 o 336."
        )

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    print(f"Device: {device}")
    print(f"Modello DINOv3: {args.model_name}")
    print(f"Feature dim attesa: {feature_dim}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Output dir: {output_dir}")
    print(f"Repo/dir DINOv3: {args.repo_or_dir}")
    print(f"Source: {args.source}")
    print(f"Weights: {args.weights}")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)

    model = DINOv3FeatureExtractor(
        model_name=args.model_name,
        weights=args.weights,
        repo_or_dir=args.repo_or_dir,
        source=args.source,
    ).to(device)

    model.eval()

    transform = build_transform(args.image_size)

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

        out_path = output_dir / split / label / f"{clip_id}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            num_skipped += 1
            continue

        try:
            frames = read_video_frames(video_path)

            features = extract_clip_features(
                frames=frames,
                model=model,
                transform=transform,
                device=device,
                chunk_size=args.chunk_size,
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
                    "clip_id": clip_id,
                    "path": str(rel_path),
                    "split": split,
                    "model_name": args.model_name,
                    "feature_dim": feature_dim,
                    "image_size": args.image_size,
                },
                out_path,
            )

            num_ok += 1

        except Exception as exc:
            num_errors += 1
            print(f"\nErrore su {video_path}: {exc}")

    print("\nEstrazione completata.")
    print(f"Clip processate: {num_ok}")
    print(f"Clip saltate perché già esistenti: {num_skipped}")
    print(f"Clip con errore: {num_errors}")
    print(f"Feature salvate in: {output_dir}")


if __name__ == "__main__":
    main()