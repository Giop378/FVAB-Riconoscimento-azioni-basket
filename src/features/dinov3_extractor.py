from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# DINOv3 ViT-B/16 -> 768, DINOv3 ViT-L/16 -> 1024.
DINO_FEATURE_DIMS = {
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
}

# Feature globale usata negli esperimenti sulle clip.
DEFAULT_DINO_OUTPUT_TOKEN = "x_norm_clstoken"


class DINOv3FeatureExtractor(nn.Module):
    """
    Wrapper unico per DINOv3 usato sia sulle clip sia sui video lunghi.

    Scelte fissate per avere feature coerenti tra training/inferenza:
    - caricamento con pesi espliciti;
    - forward_features quando disponibile;
    - preferenza per il CLS token normalizzato x_norm_clstoken;
    - nessun fallback a modello casuale/non pre-addestrato.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        weights: str | Path | None = None,
        repo_or_dir: str | Path = "third_party/dinov3",
        source: str = "local",
        output_token: str = DEFAULT_DINO_OUTPUT_TOKEN,
    ) -> None:
        super().__init__()

        if model_name not in DINO_FEATURE_DIMS:
            raise ValueError(
                f"Modello DINOv3 non supportato: {model_name}. "
                f"Modelli supportati: {sorted(DINO_FEATURE_DIMS)}"
            )
        if weights is None or str(weights).strip() == "":
            raise ValueError(
                "I pesi DINOv3 sono obbligatori. "
                "Passa il checkpoint con --weights / --dino-weights."
            )
        if source not in {"local", "github"}:
            raise ValueError("source deve essere 'local' oppure 'github'.")

        self.model_name = str(model_name)
        self.weights = str(weights)
        self.repo_or_dir = str(repo_or_dir)
        self.source = str(source)
        self.output_token = str(output_token)
        self.feature_dim = int(DINO_FEATURE_DIMS[self.model_name])

        load_kwargs: dict[str, Any] = {
            "weights": self.weights,
        }

        if self.source == "local":
            self.backbone = torch.hub.load(
                self.repo_or_dir,
                self.model_name,
                source="local",
                **load_kwargs,
            )
        else:
            self.backbone = torch.hub.load(
                self.repo_or_dir,
                self.model_name,
                source="github",
                trust_repo=True,
                **load_kwargs,
            )

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: Any

        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
            features = self._features_from_output(out)
        else:
            out = self.backbone(x)
            features = self._features_from_output(out)

        if features.ndim != 2:
            raise RuntimeError(f"Feature DINOv3 non bidimensionali: {tuple(features.shape)}")
        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Dimensione feature DINOv3 inattesa: {features.shape[1]} invece di {self.feature_dim}. "
                "Controlla model_name e pesi."
            )
        return features

    def _features_from_output(self, out: Any) -> torch.Tensor:
        if isinstance(out, dict):
            # Caso previsto per i ViT DINOv3 usati nelle clip.
            if self.output_token in out and torch.is_tensor(out[self.output_token]):
                selected = out[self.output_token]
                if selected.ndim == 2:
                    return selected
                if selected.ndim == 3:
                    return selected[:, 0]

            # Fallback controllati, solo su output già pre-addestrato caricato con weights.
            if "x_prenorm" in out and torch.is_tensor(out["x_prenorm"]):
                selected = out["x_prenorm"]
                if selected.ndim == 3:
                    return selected[:, 0]
                if selected.ndim == 2:
                    return selected

            if "x_norm_patchtokens" in out and torch.is_tensor(out["x_norm_patchtokens"]):
                selected = out["x_norm_patchtokens"]
                if selected.ndim == 3:
                    return selected.mean(dim=1)

            tensor_values = [value for value in out.values() if torch.is_tensor(value)]
            if tensor_values:
                selected = tensor_values[0]
                if selected.ndim == 2:
                    return selected
                if selected.ndim == 3:
                    return selected[:, 0]
                if selected.ndim == 4:
                    return selected.flatten(2).mean(dim=-1)

            raise RuntimeError(
                "Output DINOv3 dict senza tensori utilizzabili. "
                f"Chiavi disponibili: {list(out.keys())}"
            )

        if isinstance(out, (list, tuple)):
            if not out:
                raise RuntimeError("Output DINOv3 tuple/list vuoto.")
            return self._features_from_output(out[0])

        if torch.is_tensor(out):
            if out.ndim == 2:
                return out
            if out.ndim == 3:
                return out[:, 0]
            if out.ndim == 4:
                return out.flatten(2).mean(dim=-1)

        raise RuntimeError(f"Output DINOv3 non gestito. Tipo: {type(out)}")


def build_dino_transform(image_size: int = 336) -> transforms.Compose:
    """
    Preprocessing DINOv3 identico a quello usato sulle clip:
    - resize stretched quadrato image_size x image_size;
    - no center crop;
    - normalizzazione ImageNet.
    """
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError("image_size deve essere > 0")
    if image_size % 16 != 0:
        raise ValueError(
            f"image_size={image_size} non è multiplo di 16. "
            "Per DINOv3 ViT-* /16 usa 224, 320, 336, ..."
        )

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


# Alias utile per rendere minima la modifica in extract_features.py.
build_transform = build_dino_transform


def frame_to_pil(frame: Any) -> Image.Image:
    """
    Converte un frame RGB/PIL/tensor in PIL RGB.

    Da usare per le clip lette con src.data.video_io.read_video_frames, dove il
    frame è già trattato come immagine RGB dal codice clip-level esistente.
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
            raise ValueError(f"Frame tensor con shape non valida: {tuple(frame.shape)}")

        if frame.shape[-1] in (1, 3):
            frame = frame.permute(2, 0, 1)

        if frame.dtype != torch.uint8:
            if float(frame.max()) <= 1.0:
                frame = frame * 255.0
            frame = frame.clamp(0, 255).to(torch.uint8)

        return transforms.functional.to_pil_image(frame).convert("RGB")

    raise TypeError(f"Tipo frame non supportato: {type(frame)}")


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Converte un frame OpenCV BGR in PIL RGB per la pipeline long-video."""
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError(f"Atteso numpy array BGR, ricevuto {type(frame_bgr)}")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Frame BGR con shape non valida: {frame_bgr.shape}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb).convert("RGB")


def apply_frame_augmentation(image: Image.Image, augmentation: str = "orig") -> Image.Image:
    """Augmentation deterministica usata solo per le clip di train, non per video lunghi."""
    if augmentation == "orig":
        return image

    if augmentation in ("hflip", "hflip_color"):
        image = ImageOps.mirror(image)

    if augmentation in ("color", "hflip_color"):
        image = ImageEnhance.Brightness(image).enhance(1.08)
        image = ImageEnhance.Contrast(image).enhance(1.08)
        image = ImageEnhance.Color(image).enhance(1.05)

    return image


@torch.no_grad()
def extract_clip_features(
    frames: Iterable[Any],
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device | str,
    chunk_size: int,
    augmentation: str = "orig",
) -> torch.Tensor:
    """Estrae feature DINOv3 [T, D] da tutti i frame di una clip."""
    frames = list(frames)
    if not frames:
        raise ValueError("Clip senza frame.")
    if chunk_size <= 0:
        raise ValueError("chunk_size deve essere > 0")

    device = torch.device(device)
    all_features: list[torch.Tensor] = []

    for start_idx in range(0, len(frames), chunk_size):
        chunk = frames[start_idx : start_idx + chunk_size]
        batch = torch.stack(
            [
                transform(
                    apply_frame_augmentation(
                        frame_to_pil(frame),
                        augmentation=augmentation,
                    )
                )
                for frame in chunk
            ],
            dim=0,
        ).to(device, non_blocking=True)

        features = model(batch)
        if features.ndim != 2:
            raise ValueError(f"Feature con shape non valida: {tuple(features.shape)}")
        all_features.append(features.detach().float().cpu())

    return torch.cat(all_features, dim=0)


@torch.no_grad()
def extract_dino_features_for_frames(
    frames_bgr: list[np.ndarray],
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device | str,
    batch_size: int,
    expected_dim: int | None = None,
    use_amp: bool = False,
) -> np.ndarray:
    """
    Estrae feature DINOv3 [T, D] da frame OpenCV BGR.

    Questa funzione è pensata per extract_feature_store.py sui video lunghi.
    Di default AMP è disattivato per massimizzare la coerenza numerica con le clip.
    """
    if batch_size <= 0:
        raise ValueError("batch_size deve essere > 0")
    if not frames_bgr:
        return np.zeros((0, int(expected_dim or 0)), dtype=np.float32)

    device = torch.device(device)
    amp_enabled = bool(use_amp and device.type == "cuda")
    all_features: list[np.ndarray] = []

    for start_idx in range(0, len(frames_bgr), batch_size):
        chunk = frames_bgr[start_idx : start_idx + batch_size]
        batch = torch.stack([transform(bgr_to_pil(frame)) for frame in chunk], dim=0)
        batch = batch.to(device, non_blocking=True)

        # Compatibile con versioni PyTorch vecchie e nuove.
        if amp_enabled and hasattr(torch, "amp"):
            with torch.amp.autocast(device_type="cuda", enabled=True):
                features = model(batch)
        elif amp_enabled:
            with torch.cuda.amp.autocast(enabled=True):
                features = model(batch)
        else:
            features = model(batch)

        if features.ndim != 2:
            raise RuntimeError(f"Feature DINOv3 non bidimensionali: {tuple(features.shape)}")
        if expected_dim is not None and int(features.shape[1]) != int(expected_dim):
            raise RuntimeError(
                f"Dimensione feature DINOv3 inattesa: {features.shape[1]} invece di {expected_dim}."
            )

        all_features.append(features.detach().float().cpu().numpy().astype(np.float32))

    return np.concatenate(all_features, axis=0).astype(np.float32)


def get_dino_config(
    model_name: str,
    weights: str | Path,
    repo_or_dir: str | Path,
    source: str,
    image_size: int,
    output_token: str = DEFAULT_DINO_OUTPUT_TOKEN,
) -> dict[str, Any]:
    """Metadata standard da salvare insieme alle feature."""
    return {
        "model_name": str(model_name),
        "weights": str(weights),
        "repo_or_dir": str(repo_or_dir),
        "source": str(source),
        "input_size": int(image_size),
        "resize_mode": "stretch",
        "center_crop": False,
        "normalization": "imagenet",
        "output_token": str(output_token),
        "feature_dim": int(DINO_FEATURE_DIMS[str(model_name)]),
    }
