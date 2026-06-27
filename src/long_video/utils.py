from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# =============================================================================
# Path e file system
# =============================================================================


def as_path(value: str | Path | None) -> Path | None:
    """Converte stringhe/path in pathlib.Path, lasciando None invariato."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(value)


def ensure_exists(path: Path, name: str, must_be_file: bool | None = None) -> None:
    """Verifica che un path esista e, opzionalmente, che sia file o cartella."""
    if not path.exists():
        raise FileNotFoundError(f"{name} non trovato: {path}")
    if must_be_file is True and not path.is_file():
        raise FileNotFoundError(f"{name} dovrebbe essere un file: {path}")
    if must_be_file is False and not path.is_dir():
        raise NotADirectoryError(f"{name} dovrebbe essere una cartella: {path}")


def prepare_output_dir(
    output_dir: Path,
    overwrite: bool = False,
    *,
    clear_if_exists: bool = False,
    require_empty: bool = False,
) -> None:
    """Prepara una cartella di output.

    Parametri:
    - clear_if_exists=True: se la cartella esiste e overwrite=True, la elimina e la ricrea.
    - require_empty=True: se la cartella esiste e contiene file, richiede overwrite=True.

    Uso tipico:
    - feature extraction: clear_if_exists=True, perché conviene ricreare tutta la feature store.
    - valutazione: clear_if_exists=True, perché gli output sono un blocco unico.
    - post-processing: usare check_output_files() per controllare solo i file specifici.
    """
    if output_dir.exists():
        has_content = any(output_dir.iterdir()) if output_dir.is_dir() else True

        if clear_if_exists and has_content:
            if not overwrite:
                raise FileExistsError(
                    f"La cartella di output esiste già e non è vuota: {output_dir}\n"
                    "Usa --overwrite per sostituirla."
                )
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return

        if require_empty and has_content and not overwrite:
            existing = ", ".join(p.name for p in sorted(output_dir.iterdir())[:8])
            raise FileExistsError(
                f"La cartella di output esiste già: {output_dir}\n"
                f"Contenuto iniziale: {existing}\n"
                "Usa --overwrite per sostituirla."
            )

    output_dir.mkdir(parents=True, exist_ok=True)


def check_output_files(
    output_dir: Path,
    output_files: Iterable[str | Path],
    overwrite: bool = False,
    *,
    remove_existing: bool = False,
) -> None:
    """Controlla se alcuni file di output esistono già.

    Se overwrite=False, solleva errore.
    Se overwrite=True e remove_existing=True, elimina i file/cartelle già presenti.
    Se overwrite=True e remove_existing=False, lascia i file esistenti: verranno sovrascritti
    dalle successive operazioni di scrittura.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in output_files if (output_dir / name).exists()]

    if existing and not overwrite:
        joined = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Alcuni file di output esistono già:\n"
            f"{joined}\n"
            "Usa --overwrite per sovrascriverli."
        )

    if overwrite and remove_existing:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def prepare_output_file(output_path: Path, overwrite: bool = False) -> None:
    """Prepara un singolo file di output, creando la cartella padre se necessario."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Il file di output esiste già: {output_path}\n"
                "Usa --overwrite per sostituirlo."
            )
        output_path.unlink()


# =============================================================================
# JSON e conversioni sicure
# =============================================================================


def json_sanitize(value: Any) -> Any:
    """Converte oggetti NumPy/NaN/Inf in valori compatibili con JSON."""
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def read_json(path: Path) -> dict[str, Any]:
    """Legge un JSON e verifica che il contenuto sia un dizionario."""
    ensure_exists(path, "JSON", must_be_file=True)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Il file JSON non contiene un oggetto: {path}")
    return data


def write_json(path: Path, data: dict[str, Any], *, indent: int = 2) -> None:
    """Scrive un dizionario JSON gestendo automaticamente valori NumPy e NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_sanitize(data), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    """Converte un valore in float, tornando default in caso di errore o valore non finito."""
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def safe_int(value: Any, default: int = 0) -> int:
    """Converte un valore in int, tornando default in caso di errore."""
    try:
        return int(value)
    except Exception:
        return int(default)


def round_float(value: Any, ndigits: int = 6, default: float = 0.0) -> float:
    """Converte un valore in float sicuro e lo arrotonda."""
    return round(safe_float(value, default=default), ndigits)


def parse_optional_float(value: str | None) -> float | None:
    """Parsing CLI per float opzionali: '', 'none' e 'null' diventano None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


# =============================================================================
# Video
# =============================================================================


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    num_frames: int
    width: int
    height: int
    duration_sec: float


def get_video_info(video_path: Path) -> VideoInfo:
    """Legge metadati base di un video con OpenCV."""
    ensure_exists(video_path, "Video", must_be_file=True)

    import cv2  # import locale per non obbligare OpenCV negli script che non usano video

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
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Risoluzione non valida per il video {video_path}: {width}x{height}")

    return VideoInfo(
        fps=fps,
        num_frames=num_frames,
        width=width,
        height=height,
        duration_sec=float(num_frames / fps),
    )


def format_time(seconds: float) -> str:
    """Formatta secondi in mm:ss.xx."""
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


# =============================================================================
# Device e riproducibilità artefatti
# =============================================================================


def parse_device_for_torch(device: str):
    """Restituisce torch.device da stringa CLI: 'cpu', 'cuda', 'cuda:0' oppure '0'."""
    import torch  # import locale per non obbligare PyTorch negli script leggeri

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
    """Converte il device CLI nel formato atteso da ultralytics YOLO."""
    device = str(device)
    if device.lower() == "cpu":
        return "cpu"
    return device.replace("cuda:", "")


def file_sha256(path: Path | None, chunk_size: int = 1024 * 1024) -> str | None:
    """Calcola SHA256 di un file. Utile per metadati/debug, ma non necessario alla pipeline."""
    if path is None or not path.exists() or not path.is_file():
        return None

    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# =============================================================================
# Metriche temporali e label
# =============================================================================


def normalize_label(label: Any) -> str:
    """Normalizza alias comuni delle label usate nella pipeline BasketAR."""
    text = str(label).strip()
    aliases = {
        "noaction": "no-action",
        "no_action": "no-action",
        "idle": "no-action",
        "background": "no-action",
        "non-gioco": "no-action",
        "non_gioco": "no-action",
        "nongioco": "no-action",
        "tiro_da_due_0": "tiroDaDue0",
        "tiro_da_due_1": "tiroDaDue1",
        "tiro_da_tre_0": "tiroDaTre0",
        "tiro_da_tre_1": "tiroDaTre1",
        "tiro_libero_0": "tiroLibero0",
        "tiro_libero_1": "tiroLibero1",
    }
    return aliases.get(text, text)


def overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Durata dell'intersezione temporale tra due intervalli."""
    return float(max(0.0, min(end_a, end_b) - max(start_a, start_b)))


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Intersection over Union temporale tra due intervalli."""
    inter = overlap_duration(start_a, end_a, start_b, end_b)
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return float(inter / union)


def f1_score(precision: float, recall: float) -> float:
    """Calcola F1 a partire da precision e recall."""
    if not np.isfinite(precision) or not np.isfinite(recall):
        return float("nan")
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def nanmean(values: Iterable[float]) -> float:
    """Media ignorando NaN/Inf. Ritorna NaN se non ci sono valori validi."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())
