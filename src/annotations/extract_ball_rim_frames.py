import argparse
import csv
import random
import re
import zipfile
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


SHOT_LABELS = {
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
}


def safe_name(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("_")


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def zip_images(image_paths, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img_path in sorted(image_paths):
            zf.write(img_path, arcname=img_path.name)


def assign_parts_balanced(df: pd.DataFrame, num_parts: int, seed: int) -> pd.DataFrame:
    """
    Assegna le clip alle parti in modo bilanciato per classe.
    Tutti i frame della stessa clip restano nella stessa parte.
    """
    if num_parts <= 1:
        df["part"] = 1
        return df

    rng = random.Random(seed)
    out = []

    for (split, label), group in df.groupby(["split", "label"], sort=True):
        rows = group.to_dict("records")
        rng.shuffle(rows)

        for i, row in enumerate(rows):
            row["part"] = (i % num_parts) + 1
            out.append(row)

    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(
        description="Estrae frame dalle clip di tiro per annotazioni ball/rim in CVAT."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root del dataset, es. data/datasets/dataset_basket_v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path del manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory di output per frame e mapping CSV",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="Split da processare. Consigliato iniziare con train.",
    )
    parser.add_argument(
        "--percents",
        nargs="+",
        type=float,
        default=[0.70, 0.85, 0.95, 1.0],
        help="Percentuali relative della clip da estrarre.",
    )
    parser.add_argument(
        "--max-clips-per-class",
        type=int,
        default=None,
        help="Numero massimo di clip per classe. Se omesso, usa tutte le clip disponibili.",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help="Numero di parti/zip in cui dividere il lavoro di annotazione.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per campionamento e divisione bilanciata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive immagini già esistenti.",
    )

    args = parser.parse_args()

    if args.num_parts < 1:
        raise ValueError("--num-parts deve essere almeno 1")

    for p in args.percents:
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Percentuale non valida: {p}. Deve stare tra 0 e 1.")

    random.seed(args.seed)

    manifest = pd.read_csv(args.manifest)

    required_cols = {"path", "label", "split"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel manifest: {missing}")

    if "clip_id" not in manifest.columns:
        manifest["clip_id"] = manifest["path"].apply(lambda x: Path(x).stem)

    if "video_id" not in manifest.columns:
        manifest["video_id"] = ""

    selected = manifest[
        manifest["split"].isin(args.splits)
        & manifest["label"].isin(SHOT_LABELS)
    ].copy()

    if args.max_clips_per_class is not None:
        selected = (
            selected.groupby(["split", "label"], group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), args.max_clips_per_class),
                    random_state=args.seed,
                )
            )
            .reset_index(drop=True)
        )

    selected = assign_parts_balanced(selected, args.num_parts, args.seed)

    print(f"Clip di tiro selezionate: {len(selected)}")
    print("\nDistribuzione per split/label:")
    print(selected.groupby(["split", "label"]).size())

    if args.num_parts > 1:
        print("\nDistribuzione per parte:")
        print(selected.groupby(["split", "part", "label"]).size())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_mapping_rows = []

    for split in args.splits:
        split_df = selected[selected["split"] == split].copy()

        split_out_dir = args.output_dir / split
        images_dir = split_out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        mapping_rows = []
        part_to_images = {part: [] for part in range(1, args.num_parts + 1)}

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Split {split}"):
            rel_path = Path(row["path"])
            video_path = args.dataset_root / rel_path
            part = int(row["part"])

            if not video_path.exists():
                print(f"[WARN] Clip non trovata: {video_path}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[WARN] Impossibile aprire: {video_path}")
                continue

            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if n_frames <= 0:
                print(f"[WARN] Numero frame non valido per: {video_path}")
                cap.release()
                continue

            used_indices = set()

            for percent in args.percents:
                frame_idx = int(round((n_frames - 1) * percent))

                if frame_idx in used_indices:
                    continue

                used_indices.add(frame_idx)

                frame = read_frame_at(cap, frame_idx)
                if frame is None:
                    print(f"[WARN] Frame {frame_idx} non leggibile in: {video_path}")
                    continue

                clip_id = safe_name(row["clip_id"])
                label = safe_name(row["label"])

                percent_name = f"p{int(round(percent * 100)):03d}"
                part_name = f"part{part:02d}"

                image_name = (
                    f"{split}__{part_name}__{label}__{clip_id}__{percent_name}"
                    f"__f{frame_idx:06d}.jpg"
                )

                image_path = images_dir / image_name

                if image_path.exists() and not args.overwrite:
                    pass
                else:
                    ok = cv2.imwrite(
                        str(image_path),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    if not ok:
                        print(f"[WARN] Impossibile salvare: {image_path}")
                        continue

                part_to_images[part].append(image_path)

                mapping_row = {
                    "image": image_name,
                    "split": split,
                    "part": part,
                    "label": row["label"],
                    "clip_id": row["clip_id"],
                    "video_id": row.get("video_id", ""),
                    "clip_path": str(rel_path),
                    "percent": percent,
                    "frame_idx": frame_idx,
                    "n_frames": n_frames,
                    "start_time": row.get("start_time", ""),
                    "end_time": row.get("end_time", ""),
                }

                mapping_rows.append(mapping_row)
                all_mapping_rows.append(mapping_row)

            cap.release()

        mapping_path = split_out_dir / f"{split}_frame_mapping.csv"
        fieldnames = [
            "image",
            "split",
            "part",
            "label",
            "clip_id",
            "video_id",
            "clip_path",
            "percent",
            "frame_idx",
            "n_frames",
            "start_time",
            "end_time",
        ]

        with mapping_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mapping_rows)

        print(f"\nSplit {split}")
        print(f"Immagini salvate in: {images_dir}")
        print(f"Mapping salvato in: {mapping_path}")

        for part in range(1, args.num_parts + 1):
            part_mapping_rows = [r for r in mapping_rows if int(r["part"]) == part]

            part_mapping_path = split_out_dir / f"{split}_part_{part:02d}_frame_mapping.csv"
            with part_mapping_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(part_mapping_rows)

            zip_path = split_out_dir / f"{split}_part_{part:02d}_images_for_cvat.zip"
            zip_images(part_to_images[part], zip_path)

            print(f"Parte {part:02d}:")
            print(f"  immagini: {len(part_to_images[part])}")
            print(f"  mapping:   {part_mapping_path}")
            print(f"  zip CVAT:  {zip_path}")

    global_mapping_path = args.output_dir / "all_frame_mapping.csv"
    fieldnames = [
        "image",
        "split",
        "part",
        "label",
        "clip_id",
        "video_id",
        "clip_path",
        "percent",
        "frame_idx",
        "n_frames",
        "start_time",
        "end_time",
    ]

    with global_mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_mapping_rows)

    print(f"\nMapping globale salvato in: {global_mapping_path}")


if __name__ == "__main__":
    main()