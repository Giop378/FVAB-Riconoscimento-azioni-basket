import csv
import shutil
import zipfile
from pathlib import Path


# ============================================================
# Parametri fissi
# ============================================================
# Versione v3: dataset YOLO shot-only con split corretti.
# Train: 1183 frame di tiro estratti dalle clip train.
# Val: frame di tiro estratti dalle clip validation, filtrati dal mapping CSV.

EXPORTS_DIR = Path("data/annotations/ball_rim_cvat_exports")
OUTPUT_DIR = Path("data/datasets/ball_rim_yolo_shot_only_v3")

EXPECTED_CLASS_NAMES = ["ball", "rim"]

# Dataset v3: train solo sui 1183 frame di tiro dello split train;
# validation solo sui frame di tiro estratti dalle clip dello split val.
# Il file mapping serve per filtrare dal file val_cvat_yolo.zip i soli frame
# appartenenti alle classi di tiro, escludendo passaggio/idle/non-gioco.
VAL_FRAME_MAPPING = Path("data/annotations/ball_rim_frames_sample/val/val_frame_mapping.csv")
SHOT_LABELS = {
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
}

EXPORTS = {
    "train_part_01": {
        "zip_name": "train_part_01_cvat_yolo.zip",
        "split": "train",
        "filter": None,
    },
    "train_part_02": {
        "zip_name": "train_part_02_cvat_yolo.zip",
        "split": "train",
        "filter": None,
    },
    "val_shots": {
        "zip_name": "val_cvat_yolo.zip",
        "split": "val",
        "filter": "shot_labels_from_mapping",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# Utility
# ============================================================

def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def get_row_image_name(row: dict) -> str | None:
    """
    Recupera in modo robusto il nome file dell'immagine dal mapping CSV
    generato dallo script di estrazione frame.
    """
    candidate_columns = [
        "image_name",
        "image_filename",
        "filename",
        "file_name",
        "image",
        "image_path",
        "frame_path",
        "output_path",
        "saved_path",
        "path",
        "frame_file",
    ]

    for column in candidate_columns:
        value = row.get(column)
        if value:
            return Path(str(value).replace("\\", "/")).name

    # Fallback: prende il primo valore che sembra un'immagine.
    for value in row.values():
        value = str(value).strip()
        if Path(value.replace("\\", "/")).suffix.lower() in IMAGE_EXTENSIONS:
            return Path(value.replace("\\", "/")).name

    return None


def load_allowed_val_shot_images(mapping_path: Path) -> set[str]:
    """
    Legge il mapping dei frame validation e restituisce i nomi delle sole
    immagini appartenenti a clip di tiro.
    """
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping validation non trovato: {mapping_path}\n"
            "Serve per filtrare val_cvat_yolo.zip e mantenere solo i frame di tiro."
        )

    allowed = set()
    total_rows = 0
    skipped_no_image_name = 0

    with mapping_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Mapping CSV vuoto o non valido: {mapping_path}")

        if "label" not in reader.fieldnames:
            raise RuntimeError(
                f"Nel mapping CSV manca la colonna 'label': {mapping_path}\n"
                f"Colonne presenti: {reader.fieldnames}"
            )

        for row in reader:
            total_rows += 1
            label = str(row.get("label", "")).strip()
            if label not in SHOT_LABELS:
                continue

            image_name = get_row_image_name(row)
            if image_name is None:
                skipped_no_image_name += 1
                continue

            allowed.add(image_name)

    if not allowed:
        raise RuntimeError(
            f"Nessuna immagine di tiro trovata nel mapping: {mapping_path}. "
            "Controlla nomi colonne e label."
        )

    print(
        f"[val_shots] Mapping validation: {total_rows} righe, "
        f"{len(allowed)} immagini di tiro selezionate"
    )
    if skipped_no_image_name:
        print(
            f"[val_shots][WARN] Righe tiro senza nome immagine riconosciuto: "
            f"{skipped_no_image_name}"
        )

    return allowed


def extract_zip(zip_path: Path, extract_dir: Path):
    clean_dir(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def find_class_names(extract_dir: Path) -> list[str]:
    """
    Cerca i nomi delle classi nell'export CVAT.

    Formati gestiti:
    - data.yaml / dataset.yaml con:
        names:
          0: ball
          1: rim
      oppure:
        names: [ball, rim]

    - obj.names:
        ball
        rim

    - classes.txt:
        ball
        rim
    """

    yaml_candidates = list(extract_dir.rglob("data.yaml")) + list(extract_dir.rglob("dataset.yaml"))

    for yaml_path in yaml_candidates:
        text = read_text(yaml_path)
        names = parse_names_from_yaml_text(text)

        if names:
            return names

    for names_file in list(extract_dir.rglob("obj.names")) + list(extract_dir.rglob("classes.txt")):
        lines = [
            line.strip()
            for line in read_text(names_file).splitlines()
            if line.strip()
        ]

        if lines:
            return lines

    raise RuntimeError(
        f"Impossibile trovare i nomi delle classi in: {extract_dir}. "
        "Controlla che l'export CVAT contenga data.yaml, dataset.yaml, obj.names oppure classes.txt."
    )


def parse_names_from_yaml_text(text: str) -> list[str]:
    """
    Parser semplice per leggere 'names' dai file yaml esportati da CVAT/Ultralytics.
    Evita dipendenze esterne da PyYAML.
    """

    lines = text.splitlines()

    # Caso: names: [ball, rim]
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("names:") and "[" in stripped and "]" in stripped:
            inside = stripped.split("[", 1)[1].split("]", 1)[0]
            names = [
                item.strip().strip("'").strip('"')
                for item in inside.split(",")
                if item.strip()
            ]
            return names

    # Caso:
    # names:
    #   0: ball
    #   1: rim
    names_block_started = False
    indexed_names = {}

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("names:"):
            names_block_started = True
            continue

        if names_block_started:
            if not stripped:
                continue

            # Se inizia una nuova chiave yaml fuori dal blocco names, fermati.
            if not line.startswith(" ") and not line.startswith("\t") and ":" in stripped:
                break

            if ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')

                if key.isdigit() and value:
                    indexed_names[int(key)] = value

    if indexed_names:
        return [indexed_names[i] for i in sorted(indexed_names.keys())]

    return []


def check_class_order(class_names: list[str], export_name: str):
    """
    Controllo rigido:
    classe 0 = ball
    classe 1 = rim

    Non viene fatto nessun mapping o remapping.
    """

    if class_names != EXPECTED_CLASS_NAMES:
        raise RuntimeError(
            f"[{export_name}] Ordine classi non valido.\n"
            f"Atteso: {EXPECTED_CLASS_NAMES}\n"
            f"Trovato: {class_names}\n\n"
            "Correggi le label nel task CVAT oppure riesporta il dataset con ordine classi corretto."
        )

    print(f"[{export_name}] Controllo classi OK: 0=ball, 1=rim")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_possible_label_file(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False

    ignored_names = {
        "classes.txt",
        "obj.names",
        "train.txt",
        "valid.txt",
        "val.txt",
        "test.txt",
    }

    if path.name.lower() in ignored_names:
        return False

    return True


def collect_images(extract_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in extract_dir.rglob("*")
        if path.is_file() and is_image_file(path)
    )


def collect_label_files_by_stem(extract_dir: Path) -> dict[str, Path]:
    label_files = {}

    for path in sorted(extract_dir.rglob("*")):
        if not path.is_file():
            continue

        if not is_possible_label_file(path):
            continue

        stem = path.stem

        if stem in label_files:
            raise RuntimeError(
                f"Trovati più file label con lo stesso nome '{stem}' in {extract_dir}:\n"
                f"- {label_files[stem]}\n"
                f"- {path}"
            )

        label_files[stem] = path

    return label_files


def validate_yolo_label_file(label_path: Path, export_name: str):
    """
    Controlla che ogni riga del file label sia nel formato YOLO:
    class_id x_center y_center width height

    Non modifica class_id.
    Verifica solo che class_id sia 0 o 1 e che le coordinate siano in [0, 1].
    """

    text = read_text(label_path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line_idx, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                f"[{export_name}] Label non valida in {label_path}, riga {line_idx}:\n"
                f"{line}\n"
                "Formato atteso: class_id x_center y_center width height"
            )

        class_id_str = parts[0]

        if not class_id_str.isdigit():
            raise RuntimeError(
                f"[{export_name}] class_id non intero in {label_path}, riga {line_idx}:\n"
                f"{line}"
            )

        class_id = int(class_id_str)

        if class_id not in {0, 1}:
            raise RuntimeError(
                f"[{export_name}] class_id non valido in {label_path}, riga {line_idx}:\n"
                f"{line}\n"
                "Sono ammessi solo 0=ball e 1=rim."
            )

        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            raise RuntimeError(
                f"[{export_name}] Coordinate non numeriche in {label_path}, riga {line_idx}:\n"
                f"{line}"
            )

        for coord in coords:
            if coord < 0.0 or coord > 1.0:
                raise RuntimeError(
                    f"[{export_name}] Coordinata fuori range [0, 1] in {label_path}, riga {line_idx}:\n"
                    f"{line}"
                )


def copy_export_to_dataset(
    export_name: str,
    zip_path: Path,
    split: str,
    temp_root: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    allowed_image_names: set[str] | None = None,
) -> dict:
    extract_dir = temp_root / export_name
    extract_zip(zip_path, extract_dir)

    class_names = find_class_names(extract_dir)
    check_class_order(class_names, export_name)

    images = collect_images(extract_dir)
    labels_by_stem = collect_label_files_by_stem(extract_dir)

    if not images:
        raise RuntimeError(f"[{export_name}] Nessuna immagine trovata nell'export: {zip_path}")

    stats = {
        "export": export_name,
        "zip": str(zip_path),
        "split": split,
        "images": 0,
        "labels_with_objects": 0,
        "empty_labels": 0,
        "boxes_ball": 0,
        "boxes_rim": 0,
        "skipped_images": 0,
    }

    for image_path in images:
        if allowed_image_names is not None and image_path.name not in allowed_image_names:
            stats["skipped_images"] += 1
            continue

        output_image_path = output_images_dir / image_path.name
        output_label_path = output_labels_dir / f"{image_path.stem}.txt"

        if output_image_path.exists():
            raise RuntimeError(
                f"[{export_name}] Immagine duplicata nel dataset finale: {output_image_path.name}"
            )

        if output_label_path.exists():
            raise RuntimeError(
                f"[{export_name}] Label duplicata nel dataset finale: {output_label_path.name}"
            )

        shutil.copy2(image_path, output_image_path)

        source_label_path = labels_by_stem.get(image_path.stem)

        if source_label_path is None:
            # Immagine senza oggetti: mantiene il frame negativo creando label vuota.
            output_label_path.write_text("", encoding="utf-8")
            stats["empty_labels"] += 1
        else:
            validate_yolo_label_file(source_label_path, export_name)

            label_text = read_text(source_label_path)
            output_label_path.write_text(label_text, encoding="utf-8")

            non_empty_lines = [
                line.strip()
                for line in label_text.splitlines()
                if line.strip()
            ]

            if non_empty_lines:
                stats["labels_with_objects"] += 1
            else:
                stats["empty_labels"] += 1

            for line in non_empty_lines:
                class_id = int(line.split()[0])

                if class_id == 0:
                    stats["boxes_ball"] += 1
                elif class_id == 1:
                    stats["boxes_rim"] += 1

        stats["images"] += 1

    print(
        f"[{export_name}] Copiate {stats['images']} immagini in split '{split}' "
        f"({stats['labels_with_objects']} con box, {stats['empty_labels']} senza box, "
        f"{stats['skipped_images']} escluse dal filtro)."
    )

    return stats


def write_data_yaml(output_dir: Path):
    data_yaml = output_dir / "data.yaml"

    content = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 2",
            "",
            "names:",
            "  0: ball",
            "  1: rim",
            "",
        ]
    )

    data_yaml.write_text(content, encoding="utf-8")


def write_summary(output_dir: Path, all_stats: list[dict]):
    summary_path = output_dir / "dataset_summary.txt"

    total_train_images = sum(s["images"] for s in all_stats if s["split"] == "train")
    total_val_images = sum(s["images"] for s in all_stats if s["split"] == "val")
    total_ball = sum(s["boxes_ball"] for s in all_stats)
    total_rim = sum(s["boxes_rim"] for s in all_stats)

    lines = []
    lines.append("Dataset YOLO ball/rim")
    lines.append("=====================")
    lines.append("")
    lines.append(f"Export directory: {EXPORTS_DIR}")
    lines.append(f"Output directory: {OUTPUT_DIR}")
    lines.append("")
    lines.append("Classi:")
    lines.append("  0: ball")
    lines.append("  1: rim")
    lines.append("")
    lines.append("Split:")
    lines.append(f"  train images: {total_train_images}")
    lines.append(f"  val images:   {total_val_images}")
    lines.append("")
    lines.append("Box totali:")
    lines.append(f"  ball: {total_ball}")
    lines.append(f"  rim:  {total_rim}")
    lines.append("")
    lines.append("Dettaglio export:")
    lines.append("")

    for stats in all_stats:
        lines.append(f"- {stats['export']}")
        lines.append(f"  zip: {stats['zip']}")
        lines.append(f"  split: {stats['split']}")
        lines.append(f"  images: {stats['images']}")
        lines.append(f"  labels_with_objects: {stats['labels_with_objects']}")
        lines.append(f"  empty_labels: {stats['empty_labels']}")
        lines.append(f"  skipped_images: {stats.get('skipped_images', 0)}")
        lines.append(f"  boxes_ball: {stats['boxes_ball']}")
        lines.append(f"  boxes_rim: {stats['boxes_rim']}")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_summary(output_dir: Path, all_stats: list[dict]):
    csv_path = output_dir / "dataset_summary.csv"

    fieldnames = [
        "export",
        "zip",
        "split",
        "images",
        "labels_with_objects",
        "empty_labels",
        "boxes_ball",
        "boxes_rim",
        "skipped_images",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)


# ============================================================
# Main
# ============================================================

def main():
    if not EXPORTS_DIR.exists():
        raise FileNotFoundError(
            f"Cartella export CVAT non trovata: {EXPORTS_DIR}"
        )

    images_train_dir = OUTPUT_DIR / "images" / "train"
    images_val_dir = OUTPUT_DIR / "images" / "val"
    labels_train_dir = OUTPUT_DIR / "labels" / "train"
    labels_val_dir = OUTPUT_DIR / "labels" / "val"
    temp_root = OUTPUT_DIR / "_tmp_cvat_exports"

    clean_dir(OUTPUT_DIR)

    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    all_stats = []
    allowed_val_shot_images = load_allowed_val_shot_images(VAL_FRAME_MAPPING)

    for export_name, config in EXPORTS.items():
        zip_path = EXPORTS_DIR / config["zip_name"]
        split = config["split"]

        if not zip_path.exists():
            raise FileNotFoundError(
                f"[{export_name}] Zip non trovato: {zip_path}"
            )

        if split == "train":
            output_images_dir = images_train_dir
            output_labels_dir = labels_train_dir
        elif split == "val":
            output_images_dir = images_val_dir
            output_labels_dir = labels_val_dir
        else:
            raise ValueError(f"Split non valido per {export_name}: {split}")

        print(f"\n[{export_name}] Elaborazione export: {zip_path}")

        allowed_image_names = None
        if config.get("filter") == "shot_labels_from_mapping":
            allowed_image_names = allowed_val_shot_images

        stats = copy_export_to_dataset(
            export_name=export_name,
            zip_path=zip_path,
            split=split,
            temp_root=temp_root,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            allowed_image_names=allowed_image_names,
        )

        all_stats.append(stats)

    shutil.rmtree(temp_root, ignore_errors=True)

    write_data_yaml(OUTPUT_DIR)
    write_summary(OUTPUT_DIR, all_stats)
    write_csv_summary(OUTPUT_DIR, all_stats)

    print("\nDataset YOLO creato correttamente.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"YAML:   {OUTPUT_DIR / 'data.yaml'}")
    print(f"Report: {OUTPUT_DIR / 'dataset_summary.txt'}")
    print(f"CSV:    {OUTPUT_DIR / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()