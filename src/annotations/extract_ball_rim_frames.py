import csv
import random
import re
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Parametri fissati nel codice
# =============================================================================

DATASET_ROOT = Path("data/datasets/dataset_basket_v1")
MANIFEST_PATH = Path("data/datasets/dataset_basket_v1/manifest.csv")
OUTPUT_DIR = Path("data/annotations/ball_rim_frames_sample")

SEED = 42
OVERWRITE_IMAGES = False

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

ALL_LABELS = SHOT_LABELS | CONTEXT_LABELS

# Stessa estrazione già usata per i tiri.
SHOT_PERCENTS = [0.70, 0.85, 0.95, 1.00]

# Estrazione più uniforme per passaggio, idle e non-gioco.
CONTEXT_PERCENTS = [0.05, 0.35, 0.65, 0.95]

# Train: tiri come prima, divisi in due zip bilanciati per classe.
TRAIN_SHOT_SPLIT = "train"
TRAIN_SHOT_MAX_CLIPS_PER_CLASS = 50
TRAIN_SHOT_NUM_PARTS = 2

# Train: nuovi contesti in un unico zip.
TRAIN_CONTEXT_SPLIT = "train"
TRAIN_CONTEXT_MAX_CLIPS_PER_LABEL = {
    "passaggio": 50,
    "idle": 35,
    "non-gioco": 35,
}

# Validation: 10 clip per ognuna delle 9 classi, in un unico zip.
VAL_SPLIT = "val"
VAL_MAX_CLIPS_PER_LABEL = 10


FIELDNAMES = [
    "image",
    "split",
    "group",
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


# =============================================================================
# Utility
# =============================================================================


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


def zip_images(image_paths: Iterable[Path], zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img_path in sorted(set(image_paths)):
            if img_path.exists():
                zf.write(img_path, arcname=img_path.name)
            else:
                print(f"[WARN] Immagine non trovata, esclusa dallo zip: {img_path}")


def write_mapping(mapping_rows: list[dict], mapping_path: Path):
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(mapping_rows)


def validate_percents(percents: Iterable[float]):
    for p in percents:
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Percentuale non valida: {p}. Deve stare tra 0 e 1.")


def prepare_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)

    required_cols = {"path", "label", "split"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel manifest: {missing}")

    if "clip_id" not in manifest.columns:
        manifest["clip_id"] = manifest["path"].apply(lambda x: Path(x).stem)

    if "video_id" not in manifest.columns:
        manifest["video_id"] = ""

    return manifest


def assign_parts_balanced(df: pd.DataFrame, num_parts: int, seed: int) -> pd.DataFrame:
    """
    Assegna le clip alle parti in modo bilanciato per classe.
    Tutti i frame della stessa clip restano nella stessa parte.
    """
    if num_parts <= 1:
        df = df.copy()
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


def sample_fixed_per_label(
    manifest: pd.DataFrame,
    split: str,
    label_limits: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """
    Seleziona un numero massimo di clip per ciascuna label.
    Se una label ha meno clip del limite richiesto, usa tutte quelle disponibili.
    """
    selected_groups = []

    for label, max_clips in label_limits.items():
        group = manifest[(manifest["split"] == split) & (manifest["label"] == label)].copy()

        if group.empty:
            print(f"[WARN] Nessuna clip trovata per split={split}, label={label}")
            continue

        n = min(len(group), max_clips)
        if len(group) < max_clips:
            print(
                f"[WARN] split={split}, label={label}: richieste {max_clips} clip, "
                f"disponibili solo {len(group)}. Uso tutte le clip disponibili."
            )

        selected_groups.append(group.sample(n=n, random_state=seed))

    if not selected_groups:
        return pd.DataFrame(columns=manifest.columns)

    return pd.concat(selected_groups, ignore_index=True)


def sample_shots_like_original(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Replica la selezione originale:
    - split train;
    - sole label di tiro;
    - massimo 50 clip per classe;
    - divisione in 2 parti bilanciate per classe.
    """
    selected = manifest[
        (manifest["split"] == TRAIN_SHOT_SPLIT)
        & manifest["label"].isin(SHOT_LABELS)
    ].copy()

    selected = (
        selected.groupby(["split", "label"], group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(len(x), TRAIN_SHOT_MAX_CLIPS_PER_CLASS),
                random_state=SEED,
            )
        )
        .reset_index(drop=True)
    )

    selected = assign_parts_balanced(selected, TRAIN_SHOT_NUM_PARTS, SEED)
    selected["group"] = "train_shots"
    return selected


def percent_list_for_label(label: str) -> list[float]:
    if label in SHOT_LABELS:
        return SHOT_PERCENTS
    if label in CONTEXT_LABELS:
        return CONTEXT_PERCENTS
    raise ValueError(f"Label non gestita: {label}")


def part_token(part) -> str:
    if isinstance(part, int):
        return f"part{part:02d}"
    if isinstance(part, float) and part.is_integer():
        return f"part{int(part):02d}"
    return safe_name(str(part))


def extract_frames(
    rows: pd.DataFrame,
    images_dir: Path,
    group_name: str,
    image_group_token: str | None = None,
) -> tuple[list[dict], list[Path]]:
    """
    Estrae i frame per le righe selezionate.

    image_group_token permette di mantenere esattamente il vecchio naming dei tiri:
    train__part01__label__clip__pXXX__fXXXXXX.jpg

    Per i nuovi gruppi viene invece usato un token esplicito:
    train__context__label__clip__pXXX__fXXXXXX.jpg
    val__validation__label__clip__pXXX__fXXXXXX.jpg
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows = []
    image_paths = []

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=group_name):
        rel_path = Path(row["path"])
        video_path = DATASET_ROOT / rel_path

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
        label = str(row["label"])
        percents = percent_list_for_label(label)

        for percent in percents:
            frame_idx = int(round((n_frames - 1) * percent))

            if frame_idx in used_indices:
                continue

            used_indices.add(frame_idx)

            frame = read_frame_at(cap, frame_idx)
            if frame is None:
                print(f"[WARN] Frame {frame_idx} non leggibile in: {video_path}")
                continue

            split = safe_name(row["split"])
            clip_id = safe_name(row["clip_id"])
            safe_label = safe_name(label)
            percent_name = f"p{int(round(percent * 100)):03d}"
            part = row.get("part", "")

            if image_group_token is None:
                token = part_token(part)
            else:
                token = image_group_token

            image_name = (
                f"{split}__{token}__{safe_label}__{clip_id}__{percent_name}"
                f"__f{frame_idx:06d}.jpg"
            )
            image_path = images_dir / image_name

            if image_path.exists() and not OVERWRITE_IMAGES:
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

            image_paths.append(image_path)

            mapping_rows.append(
                {
                    "image": image_name,
                    "split": row["split"],
                    "group": group_name,
                    "part": part,
                    "label": label,
                    "clip_id": row["clip_id"],
                    "video_id": row.get("video_id", ""),
                    "clip_path": str(rel_path),
                    "percent": percent,
                    "frame_idx": frame_idx,
                    "n_frames": n_frames,
                    "start_time": row.get("start_time", ""),
                    "end_time": row.get("end_time", ""),
                }
            )

        cap.release()

    return mapping_rows, image_paths


def print_distribution(title: str, df: pd.DataFrame):
    print(f"\n{title}")
    if df.empty:
        print("Nessuna clip selezionata.")
        return

    print("Distribuzione per split/label:")
    print(df.groupby(["split", "label"]).size())

    if "part" in df.columns:
        print("\nDistribuzione per split/part/label:")
        print(df.groupby(["split", "part", "label"]).size())


def main():
    validate_percents(SHOT_PERCENTS)
    validate_percents(CONTEXT_PERCENTS)
    random.seed(SEED)

    print("Parametri utilizzati:")
    print(f"  DATASET_ROOT: {DATASET_ROOT}")
    print(f"  MANIFEST_PATH: {MANIFEST_PATH}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  SEED: {SEED}")
    print(f"  SHOT_PERCENTS: {SHOT_PERCENTS}")
    print(f"  CONTEXT_PERCENTS: {CONTEXT_PERCENTS}")
    print(f"  TRAIN_SHOT_MAX_CLIPS_PER_CLASS: {TRAIN_SHOT_MAX_CLIPS_PER_CLASS}")
    print(f"  TRAIN_CONTEXT_MAX_CLIPS_PER_LABEL: {TRAIN_CONTEXT_MAX_CLIPS_PER_LABEL}")
    print(f"  VAL_MAX_CLIPS_PER_LABEL: {VAL_MAX_CLIPS_PER_LABEL}")

    manifest = prepare_manifest(MANIFEST_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Train - tiri, invariato rispetto alla versione precedente
    # -------------------------------------------------------------------------
    train_shots = sample_shots_like_original(manifest)
    print_distribution("Train - tiri selezionati", train_shots)

    train_dir = OUTPUT_DIR / "train"
    train_images_dir = train_dir / "images"

    train_shot_mapping_rows, train_shot_image_paths = extract_frames(
        train_shots,
        images_dir=train_images_dir,
        group_name="train_shots",
        image_group_token=None,  # mantiene part01/part02 come nella versione precedente
    )

    train_shot_mapping_path = train_dir / "train_frame_mapping.csv"
    write_mapping(train_shot_mapping_rows, train_shot_mapping_path)
    print(f"\nMapping tiri train salvato in: {train_shot_mapping_path}")

    for part in range(1, TRAIN_SHOT_NUM_PARTS + 1):
        part_mapping_rows = [
            r for r in train_shot_mapping_rows if int(r["part"]) == part
        ]
        part_images = [
            train_images_dir / r["image"] for r in part_mapping_rows
        ]

        part_mapping_path = train_dir / f"train_part_{part:02d}_frame_mapping.csv"
        part_zip_path = train_dir / f"train_part_{part:02d}_images_for_cvat.zip"

        write_mapping(part_mapping_rows, part_mapping_path)
        zip_images(part_images, part_zip_path)

        print(f"Parte train tiri {part:02d}:")
        print(f"  immagini: {len(part_images)}")
        print(f"  mapping:   {part_mapping_path}")
        print(f"  zip CVAT:  {part_zip_path}")

    # -------------------------------------------------------------------------
    # 2) Train - passaggio, idle, non-gioco in un unico zip
    # -------------------------------------------------------------------------
    train_context = sample_fixed_per_label(
        manifest,
        split=TRAIN_CONTEXT_SPLIT,
        label_limits=TRAIN_CONTEXT_MAX_CLIPS_PER_LABEL,
        seed=SEED,
    )
    train_context["part"] = "context"
    train_context["group"] = "train_context"
    print_distribution("Train - passaggio/idle/non-gioco selezionati", train_context)

    train_context_mapping_rows, train_context_image_paths = extract_frames(
        train_context,
        images_dir=train_images_dir,
        group_name="train_context",
        image_group_token="context",
    )

    train_context_mapping_path = train_dir / "train_context_frame_mapping.csv"
    train_context_zip_path = train_dir / "train_context_images_for_cvat.zip"

    write_mapping(train_context_mapping_rows, train_context_mapping_path)
    zip_images(train_context_image_paths, train_context_zip_path)

    print("\nTrain context:")
    print(f"  immagini: {len(train_context_image_paths)}")
    print(f"  mapping:   {train_context_mapping_path}")
    print(f"  zip CVAT:  {train_context_zip_path}")

    # -------------------------------------------------------------------------
    # 3) Validation - 10 clip per ciascuna delle 9 classi in un unico zip
    # -------------------------------------------------------------------------
    val_label_limits = {label: VAL_MAX_CLIPS_PER_LABEL for label in sorted(ALL_LABELS)}
    val_all = sample_fixed_per_label(
        manifest,
        split=VAL_SPLIT,
        label_limits=val_label_limits,
        seed=SEED,
    )
    val_all["part"] = "validation"
    val_all["group"] = "val_all"
    print_distribution("Validation - tutte le classi selezionate", val_all)

    val_dir = OUTPUT_DIR / "val"
    val_images_dir = val_dir / "images"

    val_mapping_rows, val_image_paths = extract_frames(
        val_all,
        images_dir=val_images_dir,
        group_name="val_all",
        image_group_token="validation",
    )

    val_mapping_path = val_dir / "val_frame_mapping.csv"
    val_zip_path = val_dir / "val_images_for_cvat.zip"

    write_mapping(val_mapping_rows, val_mapping_path)
    zip_images(val_image_paths, val_zip_path)

    print("\nValidation:")
    print(f"  immagini: {len(val_image_paths)}")
    print(f"  mapping:   {val_mapping_path}")
    print(f"  zip CVAT:  {val_zip_path}")

    # -------------------------------------------------------------------------
    # 4) Mapping globale
    # -------------------------------------------------------------------------
    all_mapping_rows = (
        train_shot_mapping_rows
        + train_context_mapping_rows
        + val_mapping_rows
    )
    global_mapping_path = OUTPUT_DIR / "all_frame_mapping.csv"
    write_mapping(all_mapping_rows, global_mapping_path)

    print(f"\nMapping globale salvato in: {global_mapping_path}")
    print("\nOutput principali generati:")
    print(f"  {train_dir / 'train_part_01_images_for_cvat.zip'}")
    print(f"  {train_dir / 'train_part_02_images_for_cvat.zip'}")
    print(f"  {train_context_zip_path}")
    print(f"  {val_zip_path}")


if __name__ == "__main__":
    main()
