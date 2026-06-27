"""
Script di estrazione delle sequenze temporali di tracking palla/canestro.

Usa un detector YOLO addestrato per rilevare palla e ferro/canestro nei frame
delle clip del dataset. A partire dalle detection calcola feature per-frame e le
salva nei formati temporali temp29 o temp43, usati poi insieme alle feature
DINOv3 nei modelli gerarchici L1/L2/L3.

La versione è focalizzata solo sulle sequenze temporali: non produce più feature
statiche per clip.
"""

from pathlib import Path
import argparse
import sys
import traceback
from collections import Counter

from ultralytics import YOLO

from src.features.tracking_geometry import (
    TEMPORAL_TRACKING_FEATURE_SETS,
    compute_pair_features,
    compute_temporal_sequence_features,
    parse_yolo_result,
)
from src.features.tracking_io import (
    Tee,
    build_frame_indices,
    get_video_metadata,
    read_frames,
    read_manifest,
    write_csv,
    write_temporal_sequences,
)


SHOT_LABELS = {
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
}

CONTEXT_LABELS = {
    "passaggio",
    "idle",
    "non-gioco",
}

ALL_LABELS = sorted(SHOT_LABELS | CONTEXT_LABELS)
ALL_SPLITS = ["train", "val", "test"]

DEFAULT_DATASET_ROOT = "data/datasets/dataset_basket_v1"
DEFAULT_MANIFEST = "data/datasets/dataset_basket_v1/manifest.csv"
DEFAULT_YOLO_WEIGHTS = (
    "runs/detect/outputs/ball_rim_detector/"
    "yolo11m_1280_v2/weights/best.pt"
)
DEFAULT_OUTPUT_DIR = "data/features/ball_rim_tracking_temporal_clip_complete"


PER_FRAME_FIELDS = [
    "clip_id",
    "split",
    "label",
    "path",
    "frame_order",
    "frame_idx",
    "time_sec",
    "t_rel",
    "width",
    "height",
    "ball_detected",
    "ball_conf",
    "ball_xc",
    "ball_yc",
    "ball_w",
    "ball_h",
    "ball_area",
    "rim_detected",
    "rim_conf",
    "rim_xc",
    "rim_yc",
    "rim_w",
    "rim_h",
    "rim_area",
    "both_detected",
    "dx",
    "dy",
    "ball_rim_dist",
    "ball_above_rim",
    "ball_below_rim",
    "ball_near_rim",
    "ball_center_inside_rim",
    "ball_center_inside_expanded_rim",
    "ball_rim_iou",
    "ball_passes_close_to_rim",
]


def get_model_names(model):
    """Restituisce il dizionario class_id -> class_name del modello YOLO."""
    names = model.names

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(names)}

    raise TypeError(f"Formato model.names non supportato: {type(names)}")


def resolve_class_id(model, class_name, explicit_id=None):
    """Trova l'id di una classe YOLO dal nome o da un id esplicito."""
    names = get_model_names(model)

    if explicit_id is not None:
        explicit_id = int(explicit_id)
        if explicit_id not in names:
            raise ValueError(
                f"Class id esplicito {explicit_id} non presente nel modello. "
                f"Classi modello: {names}"
            )
        return explicit_id

    normalized = {name.lower(): class_id for class_id, name in names.items()}
    class_name_lower = class_name.lower()

    if class_name_lower not in normalized:
        raise ValueError(
            f"Classe '{class_name}' non trovata nel modello YOLO. "
            f"Classi disponibili: {names}. "
            f"Usa --ball-class-id e --rim-class-id se necessario."
        )

    return normalized[class_name_lower]


def process_clip(row, model, ball_class_id, rim_class_id, args):
    """Estrae detection e sequenza temporale di tracking da una singola clip."""
    video_path = row["video_path"]
    frame_count, fps, width, height = get_video_metadata(video_path)

    frame_indices = build_frame_indices(
        frame_count=frame_count,
        num_frames=args.num_frames,
        sample_mode=args.sample_mode,
        last_ratio=args.last_ratio,
    )
    frames, valid_indices = read_frames(video_path, frame_indices)

    if not frames:
        raise RuntimeError(f"Nessun frame leggibile per clip: {video_path}")

    frame_rows = []

    for start in range(0, len(frames), args.batch_size):
        batch_frames = frames[start:start + args.batch_size]
        batch_indices = valid_indices[start:start + args.batch_size]

        results = model.predict(
            source=batch_frames,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            classes=[ball_class_id, rim_class_id],
            verbose=False,
        )

        for local_idx, result in enumerate(results):
            frame_order = start + local_idx
            frame_idx = int(batch_indices[local_idx])
            frame_height, frame_width = batch_frames[local_idx].shape[:2]

            ball, rim = parse_yolo_result(
                result=result,
                ball_class_id=ball_class_id,
                rim_class_id=rim_class_id,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            pair = compute_pair_features(
                ball=ball,
                rim=rim,
                near_threshold=args.near_threshold,
                rim_inside_margin=args.rim_inside_margin,
            )

            frame_rows.append(
                {
                    "clip_id": row["clip_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "path": row["path"],
                    "frame_order": frame_order,
                    "frame_idx": frame_idx,
                    "time_sec": frame_idx / fps if fps > 0 else 0.0,
                    "t_rel": frame_idx / max(1, frame_count - 1),
                    "width": frame_width,
                    "height": frame_height,
                    "ball_detected": ball["detected"],
                    "ball_conf": ball["conf"],
                    "ball_xc": ball["xc"],
                    "ball_yc": ball["yc"],
                    "ball_w": ball["w"],
                    "ball_h": ball["h"],
                    "ball_area": ball["area"],
                    "rim_detected": rim["detected"],
                    "rim_conf": rim["conf"],
                    "rim_xc": rim["xc"],
                    "rim_yc": rim["yc"],
                    "rim_w": rim["w"],
                    "rim_h": rim["h"],
                    "rim_area": rim["area"],
                    "both_detected": pair["both_detected"],
                    "dx": pair["dx"],
                    "dy": pair["dy"],
                    "ball_rim_dist": pair["dist"],
                    "ball_above_rim": pair["ball_above_rim"],
                    "ball_below_rim": pair["ball_below_rim"],
                    "ball_near_rim": pair["ball_near_rim"],
                    "ball_center_inside_rim": pair["ball_center_inside_rim"],
                    "ball_center_inside_expanded_rim": pair["ball_center_inside_expanded_rim"],
                    "ball_rim_iou": pair["ball_rim_iou"],
                    "ball_passes_close_to_rim": pair["ball_passes_close_to_rim"],
                }
            )

    temporal_feature_names = TEMPORAL_TRACKING_FEATURE_SETS[args.temporal_feature_set]
    sequence = compute_temporal_sequence_features(
        frame_rows=frame_rows,
        fps=fps,
        temporal_feature_names=temporal_feature_names,
    )

    sequence_entry = {
        "clip_id": row["clip_id"],
        "split": row["split"],
        "label": row["label"],
        "path": row["path"],
        "sequence": sequence,
    }

    metadata = {
        "clip_id": row["clip_id"],
        "split": row["split"],
        "label": row["label"],
        "path": row["path"],
        "video_frames": frame_count,
        "fps": fps,
        "sampled_frames": len(frame_rows),
        "video_width": width,
        "video_height": height,
    }

    return sequence_entry, frame_rows, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estrae sequenze temporali di tracking palla/canestro da tutte le clip "
            "del dataset usando un detector YOLO addestrato su ball/rim."
        )
    )

    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--yolo-weights", type=str, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    parser.add_argument("--labels", nargs="+", default=ALL_LABELS)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=48,
        help="Numero di frame campionati per clip. Usa 0 per processare tutti i frame.",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="uniform",
        choices=["uniform", "last"],
        help="uniform = tutta la clip; last = solo la parte finale definita da --last-ratio.",
    )
    parser.add_argument("--last-ratio", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ball-class-name", type=str, default="ball")
    parser.add_argument("--rim-class-name", type=str, default="rim")
    parser.add_argument("--ball-class-id", type=int, default=None)
    parser.add_argument("--rim-class-id", type=int, default=None)
    parser.add_argument("--near-threshold", type=float, default=0.12)
    parser.add_argument("--rim-inside-margin", type=float, default=0.15)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--save-per-frame",
        action="store_true",
        help="Salva anche per_frame_detections.csv. Utile per debug, ma può essere grande.",
    )
    parser.add_argument(
        "--temporal-feature-set",
        type=str,
        default="temp43",
        choices=sorted(TEMPORAL_TRACKING_FEATURE_SETS.keys()),
        help="Set di feature temporali da salvare: temp29 o temp43.",
    )
    parser.add_argument("--overwrite", action="store_true")

    # Opzione mantenuta solo per compatibilità con i comandi vecchi: ora le
    # sequenze temporali vengono sempre salvate.
    parser.add_argument("--save-temporal-sequences", action="store_true", help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    args = parse_args()
    temporal_feature_names = TEMPORAL_TRACKING_FEATURE_SETS[args.temporal_feature_set]

    dataset_root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    yolo_weights = Path(args.yolo_weights)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "extract_tracking_results.txt"
    per_frame_csv_path = output_dir / "per_frame_detections.csv"
    tracking_sequences_npz_path = output_dir / "tracking_sequences.npz"

    if tracking_sequences_npz_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"File già esistente: {tracking_sequences_npz_path}. "
            "Usa --overwrite per rigenerarlo."
        )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(results_path, "w", encoding="utf-8") as results_file:
        sys.stdout = Tee(original_stdout, results_file)
        sys.stderr = Tee(original_stderr, results_file)

        try:
            print(f"File log: {results_path}")
            print("\n# Configurazione")
            for key, value in vars(args).items():
                if key != "save_temporal_sequences":
                    print(f"{key}: {value}")
            print(f"temporal_feature_count: {len(temporal_feature_names)}")

            print("\n# Controllo path")
            print(f"dataset_root: {dataset_root}")
            print(f"manifest: {manifest_path}")
            print(f"yolo_weights: {yolo_weights}")
            print(f"output_dir: {output_dir}")

            if not dataset_root.exists():
                raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")
            if not yolo_weights.exists():
                raise FileNotFoundError(f"Pesi YOLO non trovati: {yolo_weights}")

            print("\n# Caricamento YOLO")
            model = YOLO(str(yolo_weights))
            model_names = get_model_names(model)
            print(f"Classi YOLO: {model_names}")

            ball_class_id = resolve_class_id(model, args.ball_class_name, args.ball_class_id)
            rim_class_id = resolve_class_id(model, args.rim_class_name, args.rim_class_id)
            print(f"ball_class_id: {ball_class_id}")
            print(f"rim_class_id: {rim_class_id}")

            print("\n# Lettura manifest")
            rows = read_manifest(
                manifest_path=manifest_path,
                dataset_root=dataset_root,
                splits=set(args.splits),
                labels=set(args.labels),
                max_clips=args.max_clips,
            )
            print(f"Clip selezionate: {len(rows)}")

            if not rows:
                raise RuntimeError("Nessuna clip selezionata. Controllare --splits e --labels.")

            counts = Counter((row["split"], row["label"]) for row in rows)
            print("\n# Distribuzione clip selezionate")
            for (split, label), count in sorted(counts.items()):
                print(f"{split:5s} | {label:12s} | {count}")

            sequence_entries = []
            all_frame_rows = []
            processed_metadata = []
            errors = []

            print("\n# Estrazione sequenze temporali")
            for idx, row in enumerate(rows, start=1):
                print(f"[{idx}/{len(rows)}] {row['split']} | {row['label']} | {row['path']}")

                if not row["video_path"].exists():
                    msg = f"Video non trovato: {row['video_path']}"
                    print(f"[WARN] {msg}")
                    errors.append(
                        {"path": row["path"], "label": row["label"], "split": row["split"], "error": msg}
                    )
                    continue

                try:
                    sequence_entry, frame_rows, metadata = process_clip(
                        row=row,
                        model=model,
                        ball_class_id=ball_class_id,
                        rim_class_id=rim_class_id,
                        args=args,
                    )
                    sequence_entries.append(sequence_entry)
                    processed_metadata.append(metadata)

                    if args.save_per_frame:
                        all_frame_rows.extend(frame_rows)

                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {exc}"
                    print(f"[WARN] Errore su clip {row['path']}: {msg}")
                    errors.append(
                        {"path": row["path"], "label": row["label"], "split": row["split"], "error": msg}
                    )

            if not sequence_entries:
                raise RuntimeError("Nessuna sequenza temporale estratta correttamente.")

            print("\n# Salvataggio output")
            npz_path, index_path, sequence_feature_names_path = write_temporal_sequences(
                output_dir=output_dir,
                sequence_entries=sequence_entries,
                temporal_feature_names=temporal_feature_names,
            )
            print(f"Sequenze tracking temporali salvate in: {npz_path}")
            print(f"Indice sequenze tracking salvato in: {index_path}")
            print(f"Nomi feature tracking temporali salvati in: {sequence_feature_names_path}")

            metadata_path = output_dir / "processed_clips.csv"
            write_csv(
                path=metadata_path,
                rows=processed_metadata,
                fieldnames=[
                    "clip_id",
                    "split",
                    "label",
                    "path",
                    "video_frames",
                    "fps",
                    "sampled_frames",
                    "video_width",
                    "video_height",
                ],
            )
            print(f"Metadata clip processate salvati in: {metadata_path}")

            if args.save_per_frame:
                write_csv(path=per_frame_csv_path, rows=all_frame_rows, fieldnames=PER_FRAME_FIELDS)
                print(f"Detection per frame salvate in: {per_frame_csv_path}")

            if errors:
                errors_path = output_dir / "errors.csv"
                write_csv(path=errors_path, rows=errors, fieldnames=["split", "label", "path", "error"])
                print(f"Errori salvati in: {errors_path}")

            print("\n# Riepilogo")
            print(f"Clip processate correttamente: {len(sequence_entries)}")
            print(f"Clip con errore: {len(errors)}")
            print(f"Set feature tracking temporali: {args.temporal_feature_set}")
            print(f"Numero feature tracking temporali per frame: {len(temporal_feature_names)}")

        except Exception:
            print("\nERRORE DURANTE L'ESECUZIONE:", file=sys.stderr)
            traceback.print_exc()
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Estrazione completata. Log salvato in: {results_path}")


if __name__ == "__main__":
    main()
