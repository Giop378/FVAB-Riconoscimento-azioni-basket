from pathlib import Path
import argparse
import random
import numpy as np
import torch
from ultralytics import YOLO


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"File data.yaml non trovato: {data_yaml}")

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

    train_images = list((dataset_root / "images" / "train").glob("*"))
    val_images = list((dataset_root / "images" / "val").glob("*"))

    if len(train_images) == 0:
        raise RuntimeError("Nessuna immagine trovata in images/train")

    if len(val_images) == 0:
        raise RuntimeError("Nessuna immagine trovata in images/val")

    print(f"[INFO] Dataset: {dataset_root}")
    print(f"[INFO] Train images: {len(train_images)}")
    print(f"[INFO] Val images:   {len(val_images)}")


def main():
    parser = argparse.ArgumentParser(
        description="Training YOLO per rilevamento palla e canestro."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/datasets/ball_rim_yolo_v1/data.yaml",
        help="Path al file data.yaml del dataset YOLO.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help=(
            "Modello YOLO pretrained. Consigliato: yolo11m.pt. "
            "Alternative: yolo11s.pt, yolo11l.pt, yolov8m.pt."
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Risoluzione di training. Per la palla piccola conviene 960 o 1280.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Numero massimo di epoche.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size. Su RTX A5000 con imgsz=1280 partire da 8.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU da usare. Esempio: 0. Usa cpu per CPU.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Numero workers dataloader.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="outputs/ball_rim_detector",
        help="Cartella in cui salvare gli esperimenti.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolo11m_1280_v1",
        help="Nome esperimento.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per riproducibilità.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Carica immagini in cache RAM. Usare solo se c'è memoria sufficiente.",
    )

    args = parser.parse_args()

    data_yaml = Path(args.data)
    check_dataset(data_yaml)
    set_seed(args.seed)

    print("[INFO] Configurazione training")
    print(f"       model:   {args.model}")
    print(f"       data:    {args.data}")
    print(f"       imgsz:   {args.imgsz}")
    print(f"       epochs:  {args.epochs}")
    print(f"       batch:   {args.batch}")
    print(f"       device:  {args.device}")
    print(f"       project: {args.project}")
    print(f"       name:    {args.name}")

    model = YOLO(args.model)

    results = model.train(
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

        # Augmentation: abbastanza utile, ma non troppo aggressiva
        # perché palla e canestro sono oggetti piccoli e le box devono restare precise.
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