from pathlib import Path
import argparse
import random
from typing import Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO


# ============================================================
# Default configuration
# ============================================================

DEFAULT_DATA = "data/datasets/ball_rim_yolo/data.yaml"
DEFAULT_MODEL = "yolo11m.pt"
DEFAULT_IMGSZ = 1280
DEFAULT_EPOCHS = 150
DEFAULT_BATCH = 8
DEFAULT_DEVICE = "0"
DEFAULT_WORKERS = 8
DEFAULT_PROJECT = "runs/detect/outputs/ball_rim_detector"
DEFAULT_NAME = "yolo11m_1280_v2"
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 30
DEFAULT_CACHE = False

EXPECTED_CLASS_NAMES = ["ball", "rim"]


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise RuntimeError(f"Contenuto YAML non valido in: {path}")

    return data


def normalize_names(names_field: Any) -> list[str]:
    if isinstance(names_field, list):
        return [str(x) for x in names_field]

    if isinstance(names_field, dict):
        return [str(names_field[k]) for k in sorted(names_field.keys(), key=lambda x: int(x))]

    raise RuntimeError(
        "Campo 'names' non valido nel data.yaml. "
        "Mi aspetto una lista oppure un dizionario indicizzato."
    )


def check_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"File data.yaml non trovato: {data_yaml}")

    dataset_cfg = load_yaml(data_yaml)
    dataset_root = data_yaml.parent

    expected_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]

    for d in expected_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Cartella mancante nel dataset YOLO: {d}")

    if "names" not in dataset_cfg:
        raise RuntimeError(
            f"Nel data.yaml manca il campo 'names': {data_yaml}"
        )

    class_names = normalize_names(dataset_cfg["names"])
    if class_names != EXPECTED_CLASS_NAMES:
        raise RuntimeError(
            "Ordine classi non valido nel data.yaml.\n"
            f"Atteso: {EXPECTED_CLASS_NAMES}\n"
            f"Trovato: {class_names}"
        )

    if "nc" in dataset_cfg and int(dataset_cfg["nc"]) != 2:
        raise RuntimeError(
            f"Valore 'nc' non valido nel data.yaml: trovato {dataset_cfg['nc']}, atteso 2"
        )

    train_images = sorted(p for p in (dataset_root / "images" / "train").glob("*") if p.is_file())
    val_images = sorted(p for p in (dataset_root / "images" / "val").glob("*") if p.is_file())

    if len(train_images) == 0:
        raise RuntimeError("Nessuna immagine trovata in images/train")

    if len(val_images) == 0:
        raise RuntimeError("Nessuna immagine trovata in images/val")

    # Controllo base: per ogni immagine deve esistere il corrispondente .txt
    missing_train_labels = []
    for img_path in train_images:
        label_path = dataset_root / "labels" / "train" / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_train_labels.append(label_path)

    missing_val_labels = []
    for img_path in val_images:
        label_path = dataset_root / "labels" / "val" / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_val_labels.append(label_path)

    if missing_train_labels:
        raise RuntimeError(
            f"Mancano {len(missing_train_labels)} file label in train. "
            f"Esempio: {missing_train_labels[0]}"
        )

    if missing_val_labels:
        raise RuntimeError(
            f"Mancano {len(missing_val_labels)} file label in val. "
            f"Esempio: {missing_val_labels[0]}"
        )

    print(f"[INFO] Dataset: {dataset_root}")
    print(f"[INFO] Classi: {class_names} (0=ball, 1=rim)")
    print(f"[INFO] Train images: {len(train_images)}")
    print(f"[INFO] Val images:   {len(val_images)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training YOLO per rilevamento palla e canestro."
    )

    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help="Path al file data.yaml del dataset YOLO.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Modello YOLO pretrained di partenza.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Risoluzione di training.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Numero massimo di epoche.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="GPU da usare. Esempio: 0. Usa cpu per CPU.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Numero workers del dataloader.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help="Cartella in cui salvare gli esperimenti.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Nome esperimento.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed per riproducibilità.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stopping patience.",
    )

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache",
        dest="cache",
        action="store_true",
        help="Carica immagini in cache RAM.",
    )
    cache_group.add_argument(
        "--no-cache",
        dest="cache",
        action="store_false",
        help="Disabilita la cache RAM.",
    )
    parser.set_defaults(cache=DEFAULT_CACHE)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_yaml = Path(args.data)
    check_dataset(data_yaml)
    set_seed(args.seed)

    print("\n[INFO] Configurazione training")
    print(f"       model:    {args.model}")
    print(f"       data:     {args.data}")
    print(f"       imgsz:    {args.imgsz}")
    print(f"       epochs:   {args.epochs}")
    print(f"       batch:    {args.batch}")
    print(f"       device:   {args.device}")
    print(f"       workers:  {args.workers}")
    print(f"       patience: {args.patience}")
    print(f"       seed:     {args.seed}")
    print(f"       cache:    {args.cache}")
    print(f"       project:  {args.project}")
    print(f"       name:     {args.name}")

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        task="detect",
        mode="train",

        # Hardware / training
        device=args.device,
        workers=args.workers,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache="ram" if args.cache else False,

        # Output
        project=args.project,
        name=args.name,
        exist_ok=False,
        plots=True,
        save=True,
        save_period=10,

        # Ottimizzazione
        optimizer="auto",
        cos_lr=True,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=0.0,
        translate=0.05,
        scale=0.30,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=15,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"

    print("\n[DONE] Training completato.")
    print(f"[INFO] Best weights: {best_weights}")

    if best_weights.exists():
        print("[INFO] Eseguo validazione finale sul best.pt...")

        best_model = YOLO(str(best_weights))
        best_model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=f"{args.name}_val",
            plots=True,
        )

        print("[DONE] Validazione completata.")
    else:
        print("[WARN] best.pt non trovato. Controllare la cartella dell'esperimento.")


if __name__ == "__main__":
    main()