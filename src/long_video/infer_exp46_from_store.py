from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.long_video import defaults


# =============================================================================
# Costanti label exp_46
# =============================================================================

DEFAULT_L1_LABELS = ["passaggio", "tiro", "no-action"]
DEFAULT_L2_LABELS = ["tiroDaDue", "tiroDaTre", "tiroLibero"]
DEFAULT_L3_LABELS = ["0", "1"]

FINAL_LABELS = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
    "no-action",
]

SHOT_TYPE_TO_PREFIX = {
    "tiroDaDue": "tiroDaDue",
    "tiroDaTre": "tiroDaTre",
    "tiroLibero": "tiroLibero",
}

# Candidati import del modello. Il primo dovrebbe essere quello del progetto.
MODEL_CLASS_CANDIDATES = [
    # Nome reale usato nel progetto. Vedi src/models/temporal_transformer_classifier.py.
    ("src.models.temporal_transformer_classifier", "TemporalTransformerActionClassifier"),

    # Alias/fallback mantenuti per compatibilità con eventuali versioni precedenti.
    ("src.models.temporal_transformer_classifier", "TemporalTransformerClassifier"),
    ("src.models.temporal_transformer_classifier", "TemporalTransformer"),
    ("src.models.temporal_transformer_classifier", "TransformerClassifier"),
    ("src.models.temporal_transformer", "TemporalTransformerClassifier"),
    ("src.models.temporal_transformer", "TemporalTransformer"),
]


# =============================================================================
# Dataclass
# =============================================================================


@dataclass(frozen=True)
class WindowRow:
    window_id: str
    scale_index: int
    scale_sec: float
    start_time: float
    end_time: float
    center_time: float
    start_rel_sec: float
    end_rel_sec: float
    store_start_index: int
    store_end_index: int
    num_store_samples: int
    first_sample_time: float | None
    last_sample_time: float | None


@dataclass
class LevelBundle:
    name: str
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    model_config: dict[str, Any]
    tracking_config: dict[str, Any]
    feature_names: list[str]
    labels: list[str]
    model: nn.Module


@dataclass
class FeatureStore:
    root: Path
    metadata: dict[str, Any]
    timestamps: np.ndarray
    dino_features: np.ndarray
    yolo_v1_primitives: dict[str, np.ndarray]
    yolo_v2_primitives: dict[str, np.ndarray]
    dino_dim: int
    store_start_sec: float
    store_end_sec: float
    feature_fps: float | None


# =============================================================================
# Utility generali
# =============================================================================


def as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(value)


def ensure_exists(path: Path, name: str, must_be_file: bool | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} non trovato: {path}")
    if must_be_file is True and not path.is_file():
        raise FileNotFoundError(f"{name} dovrebbe essere un file: {path}")
    if must_be_file is False and not path.is_dir():
        raise NotADirectoryError(f"{name} dovrebbe essere una cartella: {path}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def check_output_files(output_dir: Path, output_files: Iterable[str], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f for f in output_files if (output_dir / f).exists()]
    if existing and not overwrite:
        joined = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "Alcuni file di output esistono già:\n"
            f"{joined}\n"
            "Usa --overwrite per sovrascriverli."
        )
    for p in existing:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def parse_device(device: str) -> torch.device:
    device = str(device)
    if device.lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if device.startswith("cuda"):
            return torch.device(device)
        return torch.device(f"cuda:{device}")
    print("[WARN] CUDA non disponibile: uso CPU.")
    return torch.device("cpu")


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# =============================================================================
# Lettura feature store e finestre
# =============================================================================


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def npz_to_dict(npz_path: Path) -> dict[str, np.ndarray]:
    ensure_exists(npz_path, npz_path.name, must_be_file=True)
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def load_feature_store(feature_store_dir: Path) -> FeatureStore:
    ensure_exists(feature_store_dir, "Feature store", must_be_file=False)

    metadata_path = feature_store_dir / "metadata.json"
    timestamps_path = feature_store_dir / "timestamps.npy"
    dino_path = feature_store_dir / "dinov3_features.npy"
    yolo_v1_path = feature_store_dir / "yolo_v1_primitives.npz"
    yolo_v2_path = feature_store_dir / "yolo_v2_primitives.npz"

    ensure_exists(timestamps_path, "timestamps.npy", must_be_file=True)
    ensure_exists(dino_path, "dinov3_features.npy", must_be_file=True)
    ensure_exists(yolo_v1_path, "yolo_v1_primitives.npz", must_be_file=True)
    ensure_exists(yolo_v2_path, "yolo_v2_primitives.npz", must_be_file=True)

    metadata = load_metadata(metadata_path)
    timestamps = np.load(timestamps_path).astype(np.float64)
    dino_features = np.load(dino_path, mmap_mode="r")

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps.npy deve avere shape [N], trovato {timestamps.shape}")
    if dino_features.ndim != 2:
        raise ValueError(f"dinov3_features.npy deve avere shape [N, D], trovato {dino_features.shape}")
    if dino_features.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"timestamps e DINO hanno N diverso: {timestamps.shape[0]} vs {dino_features.shape[0]}"
        )
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps.npy non è monotono crescente")

    video_meta = metadata.get("video", {}) if isinstance(metadata, dict) else {}
    sampling_meta = metadata.get("sampling", {}) if isinstance(metadata, dict) else {}

    store_start = float(video_meta.get("start_sec", float(timestamps[0])))
    store_end = float(video_meta.get("end_sec", float(timestamps[-1])))
    feature_fps = sampling_meta.get("feature_fps")
    feature_fps = None if feature_fps is None else float(feature_fps)

    yolo_v1 = npz_to_dict(yolo_v1_path)
    yolo_v2 = npz_to_dict(yolo_v2_path)

    for source_name, primitives in [("yolo_v1", yolo_v1), ("yolo_v2", yolo_v2)]:
        for required in ["ball_detected", "rim_detected", "ball_xc", "ball_yc", "rim_xc", "rim_yc"]:
            if required not in primitives:
                raise KeyError(f"{source_name}_primitives.npz non contiene '{required}'")
        n = primitives["ball_detected"].shape[0]
        if n != timestamps.shape[0]:
            raise ValueError(
                f"{source_name}_primitives ha N={n}, ma timestamps ha N={timestamps.shape[0]}"
            )

    return FeatureStore(
        root=feature_store_dir,
        metadata=metadata,
        timestamps=timestamps,
        dino_features=dino_features,
        yolo_v1_primitives=yolo_v1,
        yolo_v2_primitives=yolo_v2,
        dino_dim=int(dino_features.shape[1]),
        store_start_sec=store_start,
        store_end_sec=store_end,
        feature_fps=feature_fps,
    )


def parse_optional_float(value: str) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def read_windows_csv(path: Path, max_windows: int | None = None) -> list[WindowRow]:
    ensure_exists(path, "windows_manifest.csv", must_be_file=True)
    rows: list[WindowRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = [
            "window_id",
            "scale_index",
            "scale_sec",
            "start_time",
            "end_time",
            "center_time",
            "start_rel_sec",
            "end_rel_sec",
            "store_start_index",
            "store_end_index",
            "num_store_samples",
        ]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"windows_manifest.csv non contiene colonne richieste: {missing}")

        for row in reader:
            rows.append(
                WindowRow(
                    window_id=row["window_id"],
                    scale_index=int(row["scale_index"]),
                    scale_sec=float(row["scale_sec"]),
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                    center_time=float(row["center_time"]),
                    start_rel_sec=float(row["start_rel_sec"]),
                    end_rel_sec=float(row["end_rel_sec"]),
                    store_start_index=int(row["store_start_index"]),
                    store_end_index=int(row["store_end_index"]),
                    num_store_samples=int(row["num_store_samples"]),
                    first_sample_time=parse_optional_float(row.get("first_sample_time", "")),
                    last_sample_time=parse_optional_float(row.get("last_sample_time", "")),
                )
            )
            if max_windows is not None and len(rows) >= max_windows:
                break

    if not rows:
        raise RuntimeError(f"Nessuna finestra letta da {path}")
    return rows


# =============================================================================
# Checkpoint e modello
# =============================================================================


def find_model_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    value = ckpt.get("model_config")
    return dict(value) if isinstance(value, dict) else {}


def find_tracking_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    tracking_config = ckpt.get("tracking_config")
    if isinstance(tracking_config, dict):
        return dict(tracking_config)
    model_config = find_model_config(ckpt)
    tracking_config = model_config.get("tracking_config")
    if isinstance(tracking_config, dict):
        return dict(tracking_config)
    raise KeyError("tracking_config non trovato nel checkpoint")


def get_state_dict_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ["model_state_dict", "state_dict", "model_state", "net_state_dict"]:
        value = ckpt.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value

    value = ckpt.get("model")
    if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
        return value

    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt

    raise KeyError(
        "State dict del modello non trovato nel checkpoint. "
        "Chiavi cercate: model_state_dict, state_dict, model_state, net_state_dict, model."
    )


def strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ["module.", "model.", "net."]
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def import_model_class() -> type[nn.Module]:
    """Importa la classe del modello temporale.

    Nel progetto la classe reale si chiama TemporalTransformerActionClassifier.
    Manteniamo comunque un fallback automatico per evitare che la pipeline si rompa
    se in futuro il file viene rinominato o vengono aggiunti alias.
    """
    errors: list[str] = []

    for module_name, class_name in MODEL_CLASS_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            if not inspect.isclass(cls) or not issubclass(cls, nn.Module):
                errors.append(f"{module_name}.{class_name}: non è una sottoclasse nn.Module")
                continue
            print(f"[INFO] Classe modello importata: {module_name}.{class_name}")
            return cls
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}.{class_name}: {type(exc).__name__}: {exc}")

    # Fallback: cerca automaticamente una classe nn.Module definita nel modulo del progetto.
    module_name = "src.models.temporal_transformer_classifier"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            f"Impossibile importare il modulo {module_name}.\n"
            f"Errore: {type(exc).__name__}: {exc}\n\n"
            "Tentativi precedenti:\n" + "\n".join(errors)
        ) from exc

    candidates: list[tuple[int, str, type[nn.Module], Any]] = []

    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        try:
            if not issubclass(obj, nn.Module):
                continue
        except TypeError:
            continue

        # Evita classi importate da torch o classi ausiliarie definite altrove.
        if getattr(obj, "__module__", None) != module.__name__:
            continue

        lower_name = name.lower()
        if "positional" in lower_name or "encoding" in lower_name:
            continue

        try:
            sig = inspect.signature(obj)
        except Exception:  # noqa: BLE001
            sig = None

        score = 0
        for token in ["classifier", "transformer", "temporal", "action", "video"]:
            if token in lower_name:
                score += 5

        if sig is not None:
            params = set(sig.parameters.keys())
            for token in [
                "input_dim",
                "d_model",
                "num_layers",
                "num_heads",
                "dim_feedforward",
                "ff_dim",
                "num_classes",
                "dropout",
                "pooling",
                "max_len",
                "last_mean_ratio",
            ]:
                if token in params:
                    score += 2

        candidates.append((score, name, obj, sig))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, name, cls, sig = candidates[0]
        print(
            f"[INFO] Classe modello rilevata automaticamente: "
            f"{module_name}.{name} | score={score} | signature={sig}"
        )
        return cls

    raise ImportError(
        "Impossibile importare la classe del classificatore temporale.\n"
        "Tentativi espliciti:\n" + "\n".join(errors)
    )

def get_num_classes_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    candidates: list[tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim == 2 and any(token in key.lower() for token in ["classifier", "head", "fc", "linear"]):
            candidates.append((key, value))
    if not candidates:
        return None
    # Di solito l'ultimo layer di classificazione ha shape [num_classes, hidden_dim].
    key, value = candidates[-1]
    return int(value.shape[0])


def normalize_model_config(model_config: dict[str, Any], num_classes: int) -> dict[str, Any]:
    config = dict(model_config)

    # I checkpoint storici possono salvare la feed-forward dimension come ff_dim,
    # mentre TemporalTransformerActionClassifier usa dim_feedforward.
    if "dim_feedforward" not in config and "ff_dim" in config:
        config["dim_feedforward"] = config["ff_dim"]
    if "ff_dim" not in config and "dim_feedforward" in config:
        config["ff_dim"] = config["dim_feedforward"]

    config.setdefault("num_classes", num_classes)
    config.setdefault("n_classes", num_classes)
    config.setdefault("num_labels", num_classes)
    return config


def instantiate_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    cls = import_model_class()
    config = normalize_model_config(model_config, num_classes=num_classes)

    alias_values: dict[str, Any] = {
        "input_dim": config.get("input_dim"),
        "num_classes": num_classes,
        "n_classes": num_classes,
        "num_labels": num_classes,
        "d_model": config.get("d_model"),
        "hidden_dim": config.get("d_model", config.get("hidden_dim")),
        "num_layers": config.get("num_layers"),
        "n_layers": config.get("num_layers"),
        "num_heads": config.get("num_heads"),
        "nhead": config.get("num_heads"),
        "n_heads": config.get("num_heads"),
        "ff_dim": config.get("ff_dim"),
        "dim_feedforward": config.get("ff_dim", config.get("dim_feedforward")),
        "dropout": config.get("dropout"),
        "pooling": config.get("pooling"),
        "pooling_mode": config.get("pooling"),
    }

    try:
        sig = inspect.signature(cls)
        params = sig.parameters
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs: dict[str, Any] = {}
        for name in params:
            if name == "self":
                continue
            if name in alias_values and alias_values[name] is not None:
                kwargs[name] = alias_values[name]
            elif name in config and config[name] is not None:
                kwargs[name] = config[name]
        if has_var_kwargs:
            for key, value in alias_values.items():
                if value is not None:
                    kwargs.setdefault(key, value)
        return cls(**kwargs)
    except Exception as exc_first:  # noqa: BLE001
        # Fallback per firme meno standard.
        fallback_kwargs = {
            "input_dim": config.get("input_dim"),
            "num_classes": num_classes,
            "d_model": config.get("d_model"),
            "num_layers": config.get("num_layers"),
            "num_heads": config.get("num_heads"),
            "ff_dim": config.get("ff_dim"),
            "dropout": config.get("dropout"),
            "pooling": config.get("pooling"),
        }
        fallback_kwargs = {k: v for k, v in fallback_kwargs.items() if v is not None}
        try:
            return cls(**fallback_kwargs)
        except Exception as exc_second:  # noqa: BLE001
            raise RuntimeError(
                f"Impossibile istanziare {cls}.\n"
                f"Primo errore: {type(exc_first).__name__}: {exc_first}\n"
                f"Fallback errore: {type(exc_second).__name__}: {exc_second}\n"
                f"model_config disponibile: {model_config}"
            ) from exc_second


def extract_labels_from_checkpoint(
    ckpt: dict[str, Any],
    default_labels: list[str],
    expected_num_classes: int,
) -> list[str]:
    candidates = [
        ckpt.get("class_names"),
        ckpt.get("classes"),
        ckpt.get("labels"),
        ckpt.get("label_names"),
        ckpt.get("idx_to_class"),
        ckpt.get("idx_to_label"),
    ]
    model_config = find_model_config(ckpt)
    candidates.extend(
        [
            model_config.get("class_names"),
            model_config.get("classes"),
            model_config.get("labels"),
            model_config.get("label_names"),
            model_config.get("idx_to_class"),
            model_config.get("idx_to_label"),
        ]
    )

    for candidate in candidates:
        if isinstance(candidate, dict):
            try:
                labels = [str(candidate[str(i)]) if str(i) in candidate else str(candidate[i]) for i in range(len(candidate))]
                if len(labels) == expected_num_classes:
                    return labels
            except Exception:
                continue
        if isinstance(candidate, (list, tuple)) and len(candidate) == expected_num_classes:
            return [str(v) for v in candidate]

    return list(default_labels)


def load_level_bundle(
    name: str,
    checkpoint_path: Path,
    default_labels: list[str],
    expected_tracking_features: int,
    device: torch.device,
    strict: bool,
) -> LevelBundle:
    ensure_exists(checkpoint_path, f"Checkpoint {name}", must_be_file=True)
    ckpt = safe_torch_load(checkpoint_path)

    # Se il checkpoint contiene direttamente il modello, lo usiamo.
    direct_model = ckpt.get("model") if isinstance(ckpt, dict) else None
    if isinstance(direct_model, nn.Module):
        model = direct_model
        model_config = find_model_config(ckpt)
        tracking_config = find_tracking_config(ckpt)
        state_dict = None
        num_classes = len(default_labels)
    else:
        if not isinstance(ckpt, dict):
            raise TypeError(f"Checkpoint {name} non valido: atteso dict, ottenuto {type(ckpt)}")
        model_config = find_model_config(ckpt)
        tracking_config = find_tracking_config(ckpt)
        state_dict = get_state_dict_from_checkpoint(ckpt)
        num_classes = (
            int(model_config.get("num_classes"))
            if model_config.get("num_classes") is not None
            else get_num_classes_from_state_dict(state_dict) or len(default_labels)
        )
        model = instantiate_model(model_config=model_config, num_classes=num_classes)

    feature_names = tracking_config.get("feature_names")
    if not isinstance(feature_names, (list, tuple)):
        raise TypeError(f"{name}: tracking_config.feature_names assente o non lista")
    feature_names = [str(v) for v in feature_names]

    if len(feature_names) != int(expected_tracking_features):
        raise ValueError(
            f"{name}: numero feature tracking inatteso. "
            f"Trovate {len(feature_names)}, attese {expected_tracking_features}."
        )

    if state_dict is not None:
        cleaned = strip_state_dict_prefixes(state_dict)
        try:
            model.load_state_dict(cleaned, strict=strict)
        except RuntimeError as exc:
            if strict:
                raise
            print(f"[WARN] {name}: load_state_dict strict=False dopo errore: {exc}")
            model.load_state_dict(cleaned, strict=False)

    labels = extract_labels_from_checkpoint(
        ckpt if isinstance(ckpt, dict) else {},
        default_labels=default_labels,
        expected_num_classes=len(default_labels),
    )

    model = model.to(device)
    model.eval()

    input_dim = model_config.get("input_dim")
    if input_dim is not None:
        expected_input_dim = int(defaults.DINOV3_FEATURE_DIM + len(feature_names))
        if int(input_dim) != expected_input_dim:
            print(
                f"[WARN] {name}: model_config.input_dim={input_dim}, "
                f"mentre DINO {defaults.DINOV3_FEATURE_DIM} + tracking {len(feature_names)} = {expected_input_dim}."
            )

    print(f"[OK] {name}: {checkpoint_path}")
    print(f"     labels: {labels}")
    print(f"     tracking features: {len(feature_names)}")

    return LevelBundle(
        name=name,
        checkpoint_path=checkpoint_path,
        checkpoint=ckpt if isinstance(ckpt, dict) else {},
        model_config=model_config,
        tracking_config=tracking_config,
        feature_names=feature_names,
        labels=labels,
        model=model,
    )


def model_forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # I modelli del progetto dovrebbero accettare solo x. I fallback servono se
    # la firma usa lengths o mask.
    lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)
    padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)

    attempts: list[Callable[[], Any]] = [
        # TemporalTransformerActionClassifier del progetto richiede features e lengths.
        lambda: model(x, lengths=lengths),
        lambda: model(x, lengths),
        lambda: model(x),
        lambda: model(x, padding_mask=padding_mask),
        lambda: model(x, src_key_padding_mask=padding_mask),
        lambda: model(x, mask=padding_mask),
    ]

    errors: list[str] = []
    for attempt in attempts:
        try:
            out = attempt()
            return output_to_logits(out)
        except TypeError as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError("Forward del modello fallito. Errori TypeError:\n" + "\n".join(errors[-3:]))


def output_to_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ["logits", "out", "output", "pred", "prediction"]:
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        tensors = [v for v in output.values() if torch.is_tensor(v)]
        if tensors:
            return tensors[0]
        raise TypeError("Output dict senza tensori utilizzabili")
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise TypeError("Output tuple/list vuoto")
        return output_to_logits(output[0])
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Output modello non supportato: {type(output)}")


# =============================================================================
# Campionamento feature da feature store
# =============================================================================


def make_query_times(start_time: float, end_time: float, num_frames: int) -> np.ndarray:
    if num_frames < 2:
        raise ValueError("num_frames deve essere >= 2")
    return np.linspace(float(start_time), float(end_time), num_frames, endpoint=True, dtype=np.float64)


def interpolate_matrix(
    store_timestamps: np.ndarray,
    matrix: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Interpolazione lineare vettoriale da [N, D] a [T, D]."""
    q = np.clip(query_times.astype(np.float64), store_timestamps[0], store_timestamps[-1])
    right = np.searchsorted(store_timestamps, q, side="left")
    right = np.clip(right, 0, len(store_timestamps) - 1)
    left = np.maximum(right - 1, 0)

    exact_or_left = np.isclose(store_timestamps[right], q, rtol=0.0, atol=1e-9)
    left = np.where(exact_or_left, right, left)

    t_left = store_timestamps[left]
    t_right = store_timestamps[right]
    denom = np.maximum(t_right - t_left, 1e-12)
    w = ((q - t_left) / denom).astype(np.float32)
    w = np.where(left == right, 0.0, w).astype(np.float32)

    a = np.asarray(matrix[left], dtype=np.float32)
    b = np.asarray(matrix[right], dtype=np.float32)
    return ((1.0 - w[:, None]) * a + w[:, None] * b).astype(np.float32)


def sample_series_linear(
    store_timestamps: np.ndarray,
    values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    values_f = values.astype(np.float32)
    q = np.clip(query_times.astype(np.float64), store_timestamps[0], store_timestamps[-1])
    return np.interp(q, store_timestamps, values_f).astype(np.float32)


def sample_series_nearest(
    store_timestamps: np.ndarray,
    values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    q = np.clip(query_times.astype(np.float64), store_timestamps[0], store_timestamps[-1])
    right = np.searchsorted(store_timestamps, q, side="left")
    right = np.clip(right, 0, len(store_timestamps) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(store_timestamps[right] - q) < np.abs(q - store_timestamps[left])
    idx = np.where(choose_right, right, left)
    return values[idx].astype(np.float32)


def sample_primitives(
    store_timestamps: np.ndarray,
    primitives: dict[str, np.ndarray],
    query_times: np.ndarray,
) -> dict[str, np.ndarray]:
    sampled: dict[str, np.ndarray] = {}
    for key, values in primitives.items():
        if key in {"timestamps", "frame_indices"}:
            continue
        if values.ndim != 1:
            continue
        if key.endswith("_detected") or key.startswith("num_") or key == "both_detected":
            sampled[key] = sample_series_nearest(store_timestamps, values, query_times)
        else:
            sampled[key] = sample_series_linear(store_timestamps, values, query_times)

    # Ricalcola both_detected dopo nearest sampling per evitare incoerenze.
    if "ball_detected" in sampled and "rim_detected" in sampled:
        sampled["both_detected"] = (
            (sampled["ball_detected"] >= 0.5) & (sampled["rim_detected"] >= 0.5)
        ).astype(np.float32)
    return sampled


# =============================================================================
# Costruzione feature tracking
# =============================================================================


def delta_previous(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    out[1:] = x[1:] - x[:-1]
    return out


def gradient_series(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros_like(x, dtype=np.float32)
    return np.gradient(x.astype(np.float32)).astype(np.float32)


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (a / np.maximum(np.abs(b), eps)).astype(np.float32)


def build_available_tracking_features(
    sampled: dict[str, np.ndarray],
    start_time: float,
    end_time: float,
    velocity_mode: str,
) -> dict[str, np.ndarray]:
    n = next(iter(sampled.values())).shape[0] if sampled else 0
    duration = max(float(end_time - start_time), 1e-6)
    t_rel = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)

    def get(name: str) -> np.ndarray:
        if name in sampled:
            return sampled[name].astype(np.float32)
        return np.zeros((n,), dtype=np.float32)

    ball_detected = (get("ball_detected") >= 0.5).astype(np.float32)
    rim_detected = (get("rim_detected") >= 0.5).astype(np.float32)
    both_detected = ((ball_detected > 0.5) & (rim_detected > 0.5)).astype(np.float32)

    ball_xc = get("ball_xc")
    ball_yc = get("ball_yc")
    ball_w = get("ball_w")
    ball_h = get("ball_h")
    ball_x1 = get("ball_x1")
    ball_y1 = get("ball_y1")
    ball_x2 = get("ball_x2")
    ball_y2 = get("ball_y2")

    rim_xc = get("rim_xc")
    rim_yc = get("rim_yc")
    rim_w = get("rim_w")
    rim_h = get("rim_h")
    rim_x1 = get("rim_x1")
    rim_y1 = get("rim_y1")
    rim_x2 = get("rim_x2")
    rim_y2 = get("rim_y2")

    ball_conf = get("ball_conf") * ball_detected
    rim_conf = get("rim_conf") * rim_detected

    ball_area = (ball_w * ball_h).astype(np.float32)
    rim_area = (rim_w * rim_h).astype(np.float32)
    ball_aspect = safe_divide(ball_w, ball_h)
    rim_aspect = safe_divide(rim_w, rim_h)

    rel_x = (ball_xc - rim_xc).astype(np.float32)
    rel_y = (ball_yc - rim_yc).astype(np.float32)
    rel_dist = np.sqrt(rel_x * rel_x + rel_y * rel_y).astype(np.float32)
    rel_x_masked = rel_x * both_detected
    rel_y_masked = rel_y * both_detected
    rel_dist_masked = rel_dist * both_detected

    if velocity_mode == "per_second":
        # Coordinate normalizzate per secondo assoluto.
        dt = duration / max(n - 1, 1)
        ball_vx = gradient_series(ball_xc) / max(dt, 1e-6)
        ball_vy = gradient_series(ball_yc) / max(dt, 1e-6)
        rim_vx = gradient_series(rim_xc) / max(dt, 1e-6)
        rim_vy = gradient_series(rim_yc) / max(dt, 1e-6)
    elif velocity_mode == "delta":
        # Delta normalizzato per step temporale, più vicino alle sequenze uniformi usate in training.
        ball_vx = delta_previous(ball_xc)
        ball_vy = delta_previous(ball_yc)
        rim_vx = delta_previous(rim_xc)
        rim_vy = delta_previous(rim_yc)
    else:
        raise ValueError(f"velocity_mode non supportato: {velocity_mode}")

    ball_speed = np.sqrt(ball_vx * ball_vx + ball_vy * ball_vy).astype(np.float32)
    rim_speed = np.sqrt(rim_vx * rim_vx + rim_vy * rim_vy).astype(np.float32)

    ball_ax = delta_previous(ball_vx)
    ball_ay = delta_previous(ball_vy)
    ball_acc = np.sqrt(ball_ax * ball_ax + ball_ay * ball_ay).astype(np.float32)

    rel_vx = delta_previous(rel_x)
    rel_vy = delta_previous(rel_y)
    rel_speed = np.sqrt(rel_vx * rel_vx + rel_vy * rel_vy).astype(np.float32)

    ball_inside_rim_x = ((ball_xc >= rim_x1) & (ball_xc <= rim_x2) & (both_detected > 0.5)).astype(np.float32)
    ball_inside_rim_y = ((ball_yc >= rim_y1) & (ball_yc <= rim_y2) & (both_detected > 0.5)).astype(np.float32)
    ball_inside_rim_bbox = (ball_inside_rim_x * ball_inside_rim_y).astype(np.float32)

    # Feature temp43 richieste dai checkpoint exp_46.
    # Sono ricostruite a partire dalle primitive salvate nella feature store
    # long-video, mantenendo una semantica per-frame/finestra coerente con
    # le sequenze temporali usate in training.
    expanded_rim_w = rim_w * 1.5
    expanded_rim_h = rim_h * 1.5
    expanded_rim_x1 = rim_xc - expanded_rim_w * 0.5
    expanded_rim_x2 = rim_xc + expanded_rim_w * 0.5
    expanded_rim_y1 = rim_yc - expanded_rim_h * 0.5
    expanded_rim_y2 = rim_yc + expanded_rim_h * 0.5

    ball_center_inside_expanded_rim = (
        (ball_xc >= expanded_rim_x1)
        & (ball_xc <= expanded_rim_x2)
        & (ball_yc >= expanded_rim_y1)
        & (ball_yc <= expanded_rim_y2)
        & (both_detected > 0.5)
    ).astype(np.float32)

    inter_x1 = np.maximum(ball_x1, rim_x1)
    inter_y1 = np.maximum(ball_y1, rim_y1)
    inter_x2 = np.minimum(ball_x2, rim_x2)
    inter_y2 = np.minimum(ball_y2, rim_y2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = (inter_w * inter_h).astype(np.float32)
    union_area = np.maximum(ball_area + rim_area - inter_area, 1e-6)
    ball_rim_iou = (inter_area / union_area * both_detected).astype(np.float32)

    ball_above_rim = ((ball_yc < rim_yc) & (both_detected > 0.5)).astype(np.float32)
    ball_below_rim = ((ball_yc > rim_yc) & (both_detected > 0.5)).astype(np.float32)
    ball_left_of_rim = ((ball_xc < rim_xc) & (both_detected > 0.5)).astype(np.float32)
    ball_right_of_rim = ((ball_xc > rim_xc) & (both_detected > 0.5)).astype(np.float32)
    ball_near_rim = ((rel_dist <= np.maximum(rim_w, rim_h) * 1.5) & (both_detected > 0.5)).astype(np.float32)
    ball_passes_close_to_rim = np.full((n,), float(ball_near_rim.max() if n > 0 else 0.0), dtype=np.float32)

    motion_den = np.maximum(np.abs(ball_vx) + np.abs(ball_vy), 1e-6)
    ball_motion_horizontal_ratio = (np.abs(ball_vx) / motion_den).astype(np.float32)
    ball_motion_vertical_ratio = (np.abs(ball_vy) / motion_den).astype(np.float32)

    ball_rim_dist_delta = (delta_previous(rel_dist) * both_detected).astype(np.float32)
    ball_relative_vx = (ball_vx - rim_vx).astype(np.float32)
    ball_relative_vy = (ball_vy - rim_vy).astype(np.float32)
    ball_relative_speed = np.sqrt(ball_relative_vx * ball_relative_vx + ball_relative_vy * ball_relative_vy).astype(np.float32)
    ball_rim_approach_speed = np.maximum(-ball_rim_dist_delta, 0.0).astype(np.float32)
    ball_rim_departure_speed = np.maximum(ball_rim_dist_delta, 0.0).astype(np.float32)

    prev_rel_y = np.concatenate([rel_y[:1], rel_y[:-1]]) if n > 0 else rel_y
    prev_both = np.concatenate([both_detected[:1], both_detected[:-1]]) if n > 0 else both_detected
    valid_cross = (both_detected > 0.5) & (prev_both > 0.5)
    crosses_down = ((prev_rel_y < 0.0) & (rel_y >= 0.0) & valid_cross).astype(np.float32)
    crosses_up = ((prev_rel_y > 0.0) & (rel_y <= 0.0) & valid_cross).astype(np.float32)
    crosses_any = np.maximum(crosses_down, crosses_up).astype(np.float32)

    # Dizionario canonico.
    available: dict[str, np.ndarray] = {
        "t_rel": t_rel,
        "time_rel": t_rel,
        "relative_time": t_rel,
        "ball_detected": ball_detected,
        "rim_detected": rim_detected,
        "both_detected": both_detected,
        "ball_conf": ball_conf,
        "rim_conf": rim_conf,
        "ball_x1": ball_x1,
        "ball_y1": ball_y1,
        "ball_x2": ball_x2,
        "ball_y2": ball_y2,
        "ball_xc": ball_xc,
        "ball_yc": ball_yc,
        "ball_cx": ball_xc,
        "ball_cy": ball_yc,
        "ball_center_x": ball_xc,
        "ball_center_y": ball_yc,
        "ball_w": ball_w,
        "ball_h": ball_h,
        "ball_width": ball_w,
        "ball_height": ball_h,
        "ball_area": ball_area,
        "ball_aspect": ball_aspect,
        "ball_aspect_ratio": ball_aspect,
        "rim_x1": rim_x1,
        "rim_y1": rim_y1,
        "rim_x2": rim_x2,
        "rim_y2": rim_y2,
        "rim_xc": rim_xc,
        "rim_yc": rim_yc,
        "rim_cx": rim_xc,
        "rim_cy": rim_yc,
        "rim_center_x": rim_xc,
        "rim_center_y": rim_yc,
        "rim_w": rim_w,
        "rim_h": rim_h,
        "rim_width": rim_w,
        "rim_height": rim_h,
        "rim_area": rim_area,
        "rim_aspect": rim_aspect,
        "rim_aspect_ratio": rim_aspect,
        "dx": rel_x_masked,
        "dy": rel_y_masked,
        "ball_rim_dx": rel_x_masked,
        "ball_rim_dy": rel_y_masked,
        "ball_rim_dist": rel_dist_masked,
        "ball_rim_distance": rel_dist_masked,
        "ball_rim_dist_delta": ball_rim_dist_delta,
        "ball_rim_delta": ball_rim_dist_delta,
        "ball_rim_iou": ball_rim_iou,
        "ball_to_rim_dx": rel_x_masked,
        "ball_to_rim_dy": rel_y_masked,
        "ball_to_rim_dist": rel_dist_masked,
        "ball_to_rim_distance": rel_dist_masked,
        "dx_ball_rim": rel_x_masked,
        "dy_ball_rim": rel_y_masked,
        "dist_ball_rim": rel_dist_masked,
        "rim_ball_dx": -rel_x_masked,
        "rim_ball_dy": -rel_y_masked,
        "rim_ball_dist": rel_dist_masked,
        "rel_x": rel_x_masked,
        "rel_y": rel_y_masked,
        "rel_dist": rel_dist_masked,
        "relative_x": rel_x_masked,
        "relative_y": rel_y_masked,
        "relative_dist": rel_dist_masked,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "ball_vel_x": ball_vx,
        "ball_vel_y": ball_vy,
        "ball_velocity_x": ball_vx,
        "ball_velocity_y": ball_vy,
        "ball_dx": ball_vx,
        "ball_dy": ball_vy,
        "ball_speed": ball_speed,
        "ball_velocity": ball_speed,
        "ball_motion": ball_speed,
        "rim_vx": rim_vx,
        "rim_vy": rim_vy,
        "rim_speed": rim_speed,
        "ball_ax": ball_ax,
        "ball_ay": ball_ay,
        "ball_acc_x": ball_ax,
        "ball_acc_y": ball_ay,
        "ball_acceleration_x": ball_ax,
        "ball_acceleration_y": ball_ay,
        "ball_acc": ball_acc,
        "ball_acceleration": ball_acc,
        "rel_vx": rel_vx,
        "rel_vy": rel_vy,
        "rel_speed": rel_speed,
        "ball_rim_vx": rel_vx,
        "ball_rim_vy": rel_vy,
        "ball_rim_speed": rel_speed,
        "ball_relative_vx": ball_relative_vx,
        "ball_relative_vy": ball_relative_vy,
        "ball_relative_speed": ball_relative_speed,
        "ball_rim_approach_speed": ball_rim_approach_speed,
        "ball_rim_departure_speed": ball_rim_departure_speed,
        "ball_motion_horizontal_ratio": ball_motion_horizontal_ratio,
        "ball_motion_vertical_ratio": ball_motion_vertical_ratio,
        "num_ball_detections": get("num_ball_detections"),
        "num_rim_detections": get("num_rim_detections"),
        "ball_inside_rim_x": ball_inside_rim_x,
        "ball_inside_rim_y": ball_inside_rim_y,
        "ball_inside_rim_bbox": ball_inside_rim_bbox,
        "ball_center_inside_rim": ball_inside_rim_bbox,
        "ball_center_inside_expanded_rim": ball_center_inside_expanded_rim,
        "ball_in_rim_x": ball_inside_rim_x,
        "ball_in_rim_y": ball_inside_rim_y,
        "ball_in_rim": ball_inside_rim_bbox,
        "ball_above_rim": ball_above_rim,
        "ball_below_rim": ball_below_rim,
        "ball_left_of_rim": ball_left_of_rim,
        "ball_right_of_rim": ball_right_of_rim,
        "ball_near_rim": ball_near_rim,
        "ball_passes_close_to_rim": ball_passes_close_to_rim,
        "ball_crosses_rim_y_frame": crosses_any,
        "ball_crosses_rim_y_downward_frame": crosses_down,
        "ball_crosses_rim_y_upward_frame": crosses_up,
    }

    # Aggiunge anche tutte le primitive dirette non già normalizzate.
    for key, value in sampled.items():
        available.setdefault(key, value.astype(np.float32))

    return available


def normalize_feature_name(name: str) -> str:
    return str(name).strip()


def build_tracking_sequence(
    store_timestamps: np.ndarray,
    primitives: dict[str, np.ndarray],
    feature_names: list[str],
    query_times: np.ndarray,
    start_time: float,
    end_time: float,
    velocity_mode: str,
) -> np.ndarray:
    sampled = sample_primitives(store_timestamps, primitives, query_times)
    available = build_available_tracking_features(
        sampled=sampled,
        start_time=start_time,
        end_time=end_time,
        velocity_mode=velocity_mode,
    )

    columns: list[np.ndarray] = []
    missing: list[str] = []
    for raw_name in feature_names:
        name = normalize_feature_name(raw_name)
        if name in available:
            columns.append(available[name].astype(np.float32))
        else:
            # fallback case-insensitive
            lower_map = {k.lower(): v for k, v in available.items()}
            if name.lower() in lower_map:
                columns.append(lower_map[name.lower()].astype(np.float32))
            else:
                missing.append(name)

    if missing:
        available_preview = ", ".join(sorted(available.keys())[:120])
        raise KeyError(
            "Impossibile costruire alcune feature tracking richieste dal checkpoint:\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nFeature disponibili/fallback principali:\n"
            + available_preview
        )

    return np.stack(columns, axis=1).astype(np.float32)


def build_input_for_window(
    feature_store: FeatureStore,
    row: WindowRow,
    feature_names: list[str],
    primitives: dict[str, np.ndarray],
    num_frames: int,
    velocity_mode: str,
) -> np.ndarray:
    query_times = make_query_times(row.start_time, row.end_time, num_frames=num_frames)
    dino_seq = interpolate_matrix(feature_store.timestamps, feature_store.dino_features, query_times)
    tracking_seq = build_tracking_sequence(
        store_timestamps=feature_store.timestamps,
        primitives=primitives,
        feature_names=feature_names,
        query_times=query_times,
        start_time=row.start_time,
        end_time=row.end_time,
        velocity_mode=velocity_mode,
    )
    if dino_seq.shape[0] != tracking_seq.shape[0]:
        raise RuntimeError(f"T diverso tra DINO e tracking: {dino_seq.shape} vs {tracking_seq.shape}")
    return np.concatenate([dino_seq, tracking_seq], axis=1).astype(np.float32)


def build_batch_inputs(
    feature_store: FeatureStore,
    rows: list[WindowRow],
    l1: LevelBundle,
    l2: LevelBundle,
    l3: LevelBundle,
    num_frames: int,
    velocity_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = len(rows)
    l1_dim = feature_store.dino_dim + len(l1.feature_names)
    l2_dim = feature_store.dino_dim + len(l2.feature_names)
    l3_dim = feature_store.dino_dim + len(l3.feature_names)

    x_l1 = np.empty((b, num_frames, l1_dim), dtype=np.float32)
    x_l2 = np.empty((b, num_frames, l2_dim), dtype=np.float32)
    x_l3 = np.empty((b, num_frames, l3_dim), dtype=np.float32)

    for i, row in enumerate(rows):
        x_l1[i] = build_input_for_window(
            feature_store=feature_store,
            row=row,
            feature_names=l1.feature_names,
            primitives=feature_store.yolo_v2_primitives,
            num_frames=num_frames,
            velocity_mode=velocity_mode,
        )
        x_l2[i] = build_input_for_window(
            feature_store=feature_store,
            row=row,
            feature_names=l2.feature_names,
            primitives=feature_store.yolo_v2_primitives,
            num_frames=num_frames,
            velocity_mode=velocity_mode,
        )
        x_l3[i] = build_input_for_window(
            feature_store=feature_store,
            row=row,
            feature_names=l3.feature_names,
            primitives=feature_store.yolo_v1_primitives,
            num_frames=num_frames,
            velocity_mode=velocity_mode,
        )

    return x_l1, x_l2, x_l3


# =============================================================================
# Probabilità e gerarchia
# =============================================================================


def probs_by_label(probs: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    return {label: probs[:, i] for i, label in enumerate(labels)}


def get_prob(label_probs: dict[str, np.ndarray], names: list[str], default: float = 0.0) -> np.ndarray:
    for name in names:
        if name in label_probs:
            return label_probs[name]
    # fallback case-insensitive
    lower = {k.lower(): v for k, v in label_probs.items()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    n = next(iter(label_probs.values())).shape[0]
    return np.full((n,), float(default), dtype=np.float32)


def compute_final_scores(
    p_l1: np.ndarray,
    p_l2: np.ndarray,
    p_l3: np.ndarray,
    l1_labels: list[str],
    l2_labels: list[str],
    l3_labels: list[str],
) -> dict[str, np.ndarray]:
    l1 = probs_by_label(p_l1, l1_labels)
    l2 = probs_by_label(p_l2, l2_labels)
    l3 = probs_by_label(p_l3, l3_labels)

    p_passaggio = get_prob(l1, ["passaggio"])
    p_tiro = get_prob(l1, ["tiro", "shot"])
    p_noaction = get_prob(l1, ["no-action", "noaction", "idle", "non-gioco"])

    p_due = get_prob(l2, ["tiroDaDue", "due", "2", "two", "tiro_da_due"])
    p_tre = get_prob(l2, ["tiroDaTre", "tre", "3", "three", "tiro_da_tre"])
    p_libero = get_prob(l2, ["tiroLibero", "libero", "free", "tiro_libero"])

    p_0 = get_prob(l3, ["0", "tiro0", "miss", "missed", "sbagliato", "fallito"])
    p_1 = get_prob(l3, ["1", "tiro1", "make", "made", "segnato", "successo"])

    return {
        "passaggio": p_passaggio,
        "tiroDaDue0": p_tiro * p_due * p_0,
        "tiroDaDue1": p_tiro * p_due * p_1,
        "tiroDaTre0": p_tiro * p_tre * p_0,
        "tiroDaTre1": p_tiro * p_tre * p_1,
        "tiroLibero0": p_tiro * p_libero * p_0,
        "tiroLibero1": p_tiro * p_libero * p_1,
        "no-action": p_noaction,
    }


def argmax_labels(score_dict: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    matrix = np.stack([score_dict[label] for label in FINAL_LABELS], axis=1)
    indices = matrix.argmax(axis=1)
    confidences = matrix[np.arange(matrix.shape[0]), indices]
    labels = [FINAL_LABELS[int(i)] for i in indices]
    return labels, confidences.astype(np.float32)


def argmax_stage_labels(probs: np.ndarray, labels: list[str]) -> tuple[list[str], np.ndarray]:
    indices = probs.argmax(axis=1)
    conf = probs[np.arange(probs.shape[0]), indices]
    pred = [labels[int(i)] for i in indices]
    return pred, conf.astype(np.float32)


# =============================================================================
# CSV output
# =============================================================================


def output_fieldnames() -> list[str]:
    base = [
        "window_id",
        "scale_index",
        "scale_sec",
        "start_time",
        "end_time",
        "center_time",
        "start_rel_sec",
        "end_rel_sec",
        "store_start_index",
        "store_end_index",
        "num_store_samples",
        "p_l1_passaggio",
        "p_l1_tiro",
        "p_l1_noaction",
        "p_l2_tiroDaDue",
        "p_l2_tiroDaTre",
        "p_l2_tiroLibero",
        "p_l3_0",
        "p_l3_1",
    ]
    score_fields = [f"score_{label.replace('-', '')}" for label in FINAL_LABELS]
    tail = [
        "l1_pred",
        "l1_confidence",
        "l2_pred",
        "l2_confidence",
        "l3_pred",
        "l3_confidence",
        "pred_label",
        "confidence",
    ]
    return base + score_fields + tail


def write_prediction_rows(
    writer: csv.DictWriter,
    rows: list[WindowRow],
    p_l1: np.ndarray,
    p_l2: np.ndarray,
    p_l3: np.ndarray,
    l1: LevelBundle,
    l2: LevelBundle,
    l3: LevelBundle,
) -> None:
    scores = compute_final_scores(p_l1, p_l2, p_l3, l1.labels, l2.labels, l3.labels)
    pred_labels, confidences = argmax_labels(scores)
    l1_pred, l1_conf = argmax_stage_labels(p_l1, l1.labels)
    l2_pred, l2_conf = argmax_stage_labels(p_l2, l2.labels)
    l3_pred, l3_conf = argmax_stage_labels(p_l3, l3.labels)

    l1_probs = probs_by_label(p_l1, l1.labels)
    l2_probs = probs_by_label(p_l2, l2.labels)
    l3_probs = probs_by_label(p_l3, l3.labels)

    p_l1_passaggio = get_prob(l1_probs, ["passaggio"])
    p_l1_tiro = get_prob(l1_probs, ["tiro", "shot"])
    p_l1_noaction = get_prob(l1_probs, ["no-action", "noaction", "idle", "non-gioco"])
    p_l2_due = get_prob(l2_probs, ["tiroDaDue", "due", "2", "two", "tiro_da_due"])
    p_l2_tre = get_prob(l2_probs, ["tiroDaTre", "tre", "3", "three", "tiro_da_tre"])
    p_l2_libero = get_prob(l2_probs, ["tiroLibero", "libero", "free", "tiro_libero"])
    p_l3_0 = get_prob(l3_probs, ["0", "tiro0", "miss", "missed", "sbagliato", "fallito"])
    p_l3_1 = get_prob(l3_probs, ["1", "tiro1", "make", "made", "segnato", "successo"])

    for i, row in enumerate(rows):
        out: dict[str, Any] = {
            "window_id": row.window_id,
            "scale_index": row.scale_index,
            "scale_sec": f"{row.scale_sec:.6f}",
            "start_time": f"{row.start_time:.6f}",
            "end_time": f"{row.end_time:.6f}",
            "center_time": f"{row.center_time:.6f}",
            "start_rel_sec": f"{row.start_rel_sec:.6f}",
            "end_rel_sec": f"{row.end_rel_sec:.6f}",
            "store_start_index": row.store_start_index,
            "store_end_index": row.store_end_index,
            "num_store_samples": row.num_store_samples,
            "p_l1_passaggio": f"{float(p_l1_passaggio[i]):.8f}",
            "p_l1_tiro": f"{float(p_l1_tiro[i]):.8f}",
            "p_l1_noaction": f"{float(p_l1_noaction[i]):.8f}",
            "p_l2_tiroDaDue": f"{float(p_l2_due[i]):.8f}",
            "p_l2_tiroDaTre": f"{float(p_l2_tre[i]):.8f}",
            "p_l2_tiroLibero": f"{float(p_l2_libero[i]):.8f}",
            "p_l3_0": f"{float(p_l3_0[i]):.8f}",
            "p_l3_1": f"{float(p_l3_1[i]):.8f}",
            "l1_pred": l1_pred[i],
            "l1_confidence": f"{float(l1_conf[i]):.8f}",
            "l2_pred": l2_pred[i],
            "l2_confidence": f"{float(l2_conf[i]):.8f}",
            "l3_pred": l3_pred[i],
            "l3_confidence": f"{float(l3_conf[i]):.8f}",
            "pred_label": pred_labels[i],
            "confidence": f"{float(confidences[i]):.8f}",
        }
        for label in FINAL_LABELS:
            out[f"score_{label.replace('-', '')}"] = f"{float(scores[label][i]):.8f}"
        writer.writerow(out)


# =============================================================================
# Main inferenza
# =============================================================================


def predict_batch(
    x_np: np.ndarray,
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    x = torch.from_numpy(x_np).to(device, non_blocking=True)
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model_forward_logits(model, x)
        if logits.ndim != 2:
            raise RuntimeError(f"Logits attesi [B, C], trovati {tuple(logits.shape)}")
        probs = torch.softmax(logits.float(), dim=1)
    return probs.detach().cpu().numpy().astype(np.float32)


def infer_all_windows(
    feature_store: FeatureStore,
    windows: list[WindowRow],
    l1: LevelBundle,
    l2: LevelBundle,
    l3: LevelBundle,
    output_csv: Path,
    batch_size: int,
    num_frames: int,
    device: torch.device,
    velocity_mode: str,
    use_amp: bool,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError(f"batch_size deve essere > 0, trovato {batch_size}")

    n = len(windows)
    num_batches = math.ceil(n / batch_size)
    label_counts: dict[str, int] = {label: 0 for label in FINAL_LABELS}

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames())
        writer.writeheader()

        progress = tqdm(range(0, n, batch_size), total=num_batches, desc="Inferenza exp_46")
        for start in progress:
            batch_rows = windows[start : start + batch_size]
            x_l1, x_l2, x_l3 = build_batch_inputs(
                feature_store=feature_store,
                rows=batch_rows,
                l1=l1,
                l2=l2,
                l3=l3,
                num_frames=num_frames,
                velocity_mode=velocity_mode,
            )

            p_l1 = predict_batch(x_l1, l1.model, device=device, use_amp=use_amp)
            p_l2 = predict_batch(x_l2, l2.model, device=device, use_amp=use_amp)
            p_l3 = predict_batch(x_l3, l3.model, device=device, use_amp=use_amp)

            scores = compute_final_scores(p_l1, p_l2, p_l3, l1.labels, l2.labels, l3.labels)
            pred_labels, _ = argmax_labels(scores)
            for label in pred_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            write_prediction_rows(writer, batch_rows, p_l1, p_l2, p_l3, l1, l2, l3)

    return {
        "num_windows": int(n),
        "label_counts": label_counts,
    }


def write_inference_metadata(
    path: Path,
    args: argparse.Namespace,
    feature_store: FeatureStore,
    windows_csv: Path,
    windows: list[WindowRow],
    l1: LevelBundle,
    l2: LevelBundle,
    l3: LevelBundle,
    started_at: float,
    summary: dict[str, Any],
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started_at, 3),
        "feature_store": {
            "dir": str(feature_store.root),
            "num_samples": int(feature_store.timestamps.shape[0]),
            "dino_dim": int(feature_store.dino_dim),
            "store_start_sec": float(feature_store.store_start_sec),
            "store_end_sec": float(feature_store.store_end_sec),
            "feature_fps": feature_store.feature_fps,
        },
        "windows": {
            "csv": str(windows_csv),
            "num_windows": int(len(windows)),
            "first_start_time": float(windows[0].start_time),
            "last_end_time": float(windows[-1].end_time),
        },
        "runtime": {
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "num_frames": int(args.num_frames),
            "amp": not args.no_amp,
            "velocity_mode": args.velocity_mode,
            "max_windows": args.max_windows,
        },
        "levels": {
            "L1": {
                "checkpoint": str(l1.checkpoint_path),
                "tracking_source": "yolo_v2",
                "num_tracking_features": len(l1.feature_names),
                "labels": l1.labels,
                "feature_names": l1.feature_names,
            },
            "L2": {
                "checkpoint": str(l2.checkpoint_path),
                "tracking_source": "yolo_v2",
                "num_tracking_features": len(l2.feature_names),
                "labels": l2.labels,
                "feature_names": l2.feature_names,
            },
            "L3": {
                "checkpoint": str(l3.checkpoint_path),
                "tracking_source": "yolo_v1",
                "num_tracking_features": len(l3.feature_names),
                "labels": l3.labels,
                "feature_names": l3.feature_names,
            },
        },
        "outputs": {
            "window_predictions_raw": "window_predictions_raw.csv",
            "metadata": "inference_metadata.json",
        },
        "summary": summary,
    }
    write_json(path, metadata)


# =============================================================================
# CLI
# =============================================================================


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inferenza exp_46 dalle feature store long-video. "
            "Costruisce finestre virtuali da DINOv3 + tracking YOLO v1/v2 e salva "
            "le predizioni raw per il post-processing temporale."
        )
    )

    parser.add_argument("--feature-store-dir", type=Path, default=defaults.VAL_FEATURE_STORE_DIR)
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=defaults.VAL_OUTPUT_DIR / "windows_manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=defaults.VAL_OUTPUT_DIR)

    parser.add_argument("--l1-checkpoint", type=Path, default=defaults.EXP46_L1_CHECKPOINT)
    parser.add_argument("--l2-checkpoint", type=Path, default=defaults.EXP46_L2_CHECKPOINT)
    parser.add_argument("--l3-checkpoint", type=Path, default=defaults.EXP46_L3_CHECKPOINT)

    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Usa strict=True nel load_state_dict. Default: strict=False per robustezza.",
    )
    parser.add_argument(
        "--velocity-mode",
        type=str,
        choices=["delta", "per_second"],
        default="delta",
        help=(
            "Come calcolare le feature di velocità se richieste dal checkpoint. "
            "delta = differenza tra timestep uniformi; per_second = delta normalizzato sui secondi."
        ),
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limita il numero di finestre per debug rapido.",
    )

    return parser


def main() -> None:
    args = make_parser().parse_args()
    started_at = time.time()

    args.feature_store_dir = as_path(args.feature_store_dir)
    args.windows_csv = as_path(args.windows_csv)
    args.output_dir = as_path(args.output_dir)
    args.l1_checkpoint = as_path(args.l1_checkpoint)
    args.l2_checkpoint = as_path(args.l2_checkpoint)
    args.l3_checkpoint = as_path(args.l3_checkpoint)

    assert args.feature_store_dir is not None
    assert args.windows_csv is not None
    assert args.output_dir is not None
    assert args.l1_checkpoint is not None
    assert args.l2_checkpoint is not None
    assert args.l3_checkpoint is not None

    if args.num_frames < 2:
        raise ValueError("--num-frames deve essere >= 2")
    if args.batch_size <= 0:
        raise ValueError("--batch-size deve essere > 0")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max-windows deve essere > 0 se specificato")

    check_output_files(
        args.output_dir,
        output_files=["window_predictions_raw.csv", "inference_metadata.json"],
        overwrite=args.overwrite,
    )

    device = parse_device(args.device)

    print("=== Caricamento feature store ===")
    feature_store = load_feature_store(args.feature_store_dir)
    print(f"feature_store_dir: {args.feature_store_dir}")
    print(f"timestamps:         {feature_store.timestamps.shape}")
    print(f"dinov3_features:    {feature_store.dino_features.shape}")
    print(f"segmento store:     {feature_store.store_start_sec:.3f}s -> {feature_store.store_end_sec:.3f}s")

    print("\n=== Caricamento finestre ===")
    windows = read_windows_csv(args.windows_csv, max_windows=args.max_windows)
    print(f"windows_csv:        {args.windows_csv}")
    print(f"num_windows:        {len(windows)}")
    print(f"prima finestra:     {windows[0].start_time:.3f}s -> {windows[0].end_time:.3f}s")
    print(f"ultima finestra:    {windows[-1].start_time:.3f}s -> {windows[-1].end_time:.3f}s")

    print("\n=== Caricamento modelli exp_46 ===")
    l1 = load_level_bundle(
        name="L1",
        checkpoint_path=args.l1_checkpoint,
        default_labels=DEFAULT_L1_LABELS,
        expected_tracking_features=43,
        device=device,
        strict=args.strict_load,
    )
    l2 = load_level_bundle(
        name="L2",
        checkpoint_path=args.l2_checkpoint,
        default_labels=DEFAULT_L2_LABELS,
        expected_tracking_features=29,
        device=device,
        strict=args.strict_load,
    )
    l3 = load_level_bundle(
        name="L3",
        checkpoint_path=args.l3_checkpoint,
        default_labels=DEFAULT_L3_LABELS,
        expected_tracking_features=43,
        device=device,
        strict=args.strict_load,
    )

    expected_dims = {
        "L1": feature_store.dino_dim + len(l1.feature_names),
        "L2": feature_store.dino_dim + len(l2.feature_names),
        "L3": feature_store.dino_dim + len(l3.feature_names),
    }
    print("\nInput dimension attese:")
    print(f"L1: {feature_store.dino_dim} + {len(l1.feature_names)} = {expected_dims['L1']}")
    print(f"L2: {feature_store.dino_dim} + {len(l2.feature_names)} = {expected_dims['L2']}")
    print(f"L3: {feature_store.dino_dim} + {len(l3.feature_names)} = {expected_dims['L3']}")

    output_csv = args.output_dir / "window_predictions_raw.csv"
    print("\n=== Inferenza raw finestre ===")
    print(f"output_csv:         {output_csv}")
    print(f"device:             {device}")
    print(f"batch_size:         {args.batch_size}")
    print(f"num_frames:         {args.num_frames}")
    print(f"velocity_mode:      {args.velocity_mode}")

    summary = infer_all_windows(
        feature_store=feature_store,
        windows=windows,
        l1=l1,
        l2=l2,
        l3=l3,
        output_csv=output_csv,
        batch_size=int(args.batch_size),
        num_frames=int(args.num_frames),
        device=device,
        velocity_mode=args.velocity_mode,
        use_amp=not args.no_amp,
    )

    metadata_path = args.output_dir / "inference_metadata.json"
    write_inference_metadata(
        path=metadata_path,
        args=args,
        feature_store=feature_store,
        windows_csv=args.windows_csv,
        windows=windows,
        l1=l1,
        l2=l2,
        l3=l3,
        started_at=started_at,
        summary=summary,
    )

    print("\n=== Inferenza completata ===")
    print(f"predizioni: {output_csv}")
    print(f"metadata:   {metadata_path}")
    print("Predizioni per label finale:")
    for label, count in summary["label_counts"].items():
        print(f"- {label}: {count}")
    print(f"tempo totale: {time.time() - started_at:.1f}s")


if __name__ == "__main__":
    main()
