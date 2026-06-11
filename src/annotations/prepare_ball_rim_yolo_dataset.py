import argparse
import random
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter, defaultdict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_name(text: str) -> str:
    return (
        text.replace("\\", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("__", "_")
    )


def read_class_names(root: Path):
    """
    Prova a leggere i nomi delle classi da export YOLO/CVAT.
    Supporta:
    - obj.names
    - classes.txt
    - *.names
    - data.yaml semplice
    """
    candidates = []

    for name in ["obj.names", "classes.txt"]:
        candidates.extend(root.rglob(name))

    candidates.extend(root.rglob("*.names"))

    for p in candidates:
        lines = [
            line.strip()
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
        if lines:
            return lines

    yaml_files = list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))

    for p in yaml_files:
        text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        names = []

        # Caso:
        # names:
        #   0: ball
        #   1: rim
        in_names = False
        indexed = {}

        for line in text:
            stripped = line.strip()

            if stripped.startswith("names:"):
                in_names = True

                # Caso: names: [ball, rim]
                if "[" in stripped and "]" in stripped:
                    inside = stripped.split("[", 1)[1].split("]", 1)[0]
                    names = [x.strip().strip("'\"") for x in inside.split(",") if x.strip()]
                    return names

                continue

            if in_names:
                if not stripped:
                    continue

                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")

                    if k.isdigit() and v:
                        indexed[int(k)] = v
                    elif not line.startswith(" ") and not line.startswith("-"):
                        break

                elif stripped.startswith("-"):
                    names.append(stripped[1:].strip().strip("'\""))

        if indexed:
            return [indexed[i] for i in sorted(indexed.keys())]

        if names:
            return names

    return None


def is_yolo_label_file(path: Path) -> bool:
    """
    Esclude file txt che non sono label YOLO, tipo train.txt, val.txt, obj.names.
    """
    excluded_names = {
        "train.txt",
        "val.txt",
        "valid.txt",
        "test.txt",
        "obj.names",
        "classes.txt",
    }

    if path.name.lower() in excluded_names:
        return False

    if path.suffix.lower() != ".txt":
        return False

    return True


def clean_yolo_label(src_label: Path | None, dst_label: Path, num_classes: int):
    """
    Copia una label YOLO mantenendo solo righe valide:
    class_id x_center y_center width height

    Se l'immagine non ha label, crea un .txt vuoto.
    """
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if src_label is None or not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        return Counter()

    cleaned_lines = []
    counts = Counter()

    for line_num, line in enumerate(src_label.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            print(f"[WARN] Riga ignorata in {src_label}, riga {line_num}: {line}")
            continue

        # Se per errore ci sono colonne extra, ad esempio confidenza, tengo solo le prime 5.
        parts = parts[:5]

        try:
            class_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:5]]
        except ValueError:
            print(f"[WARN] Riga non valida in {src_label}, riga {line_num}: {line}")
            continue

        if class_id < 0 or class_id >= num_classes:
            print(f"[WARN] Classe fuori range in {src_label}, riga {line_num}: {line}")
            continue

        if not all(0.0 <= c <= 1.0 for c in coords):
            print(f"[WARN] Coordinate non normalizzate in {src_label}, riga {line_num}: {line}")
            continue

        cleaned_lines.append(f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")
        counts[class_id] += 1

    dst_label.write_text("\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""), encoding="utf-8")
    return counts


def collect_samples(extracted_roots):
    """
    Trova immagini e relative label YOLO.
    Funziona sia con export CVAT tipo:
    obj_train_data/img.jpg + obj_train_data/img.txt

    sia con struttura:
    images/.../img.jpg + labels/.../img.txt
    """
    samples = []

    for zip_index, root in enumerate(extracted_roots):
        zip_prefix = f"zip{zip_index + 1}"

        label_files = [
            p for p in root.rglob("*.txt")
            if is_yolo_label_file(p)
        ]

        labels_by_stem = defaultdict(list)
        labels_by_relative_hint = {}

        for label in label_files:
            labels_by_stem[label.stem].append(label)

            # Caso images/train/x.jpg -> labels/train/x.txt
            rel = label.relative_to(root)
            labels_by_relative_hint[str(rel.with_suffix(""))] = label

        image_files = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and "__MACOSX" not in str(p)
        ]

        for image in image_files:
            rel = image.relative_to(root)
            label = None

            # Caso label nella stessa cartella dell'immagine
            same_dir_label = image.with_suffix(".txt")
            if same_dir_label.exists() and is_yolo_label_file(same_dir_label):
                label = same_dir_label

            # Caso images/... -> labels/...
            if label is None:
                rel_str = str(rel.with_suffix(""))
                rel_label_guess = rel_str.replace("images", "labels")
                label = labels_by_relative_hint.get(rel_label_guess)

            # Fallback: label con stesso stem
            if label is None:
                candidates = labels_by_stem.get(image.stem, [])
                if len(candidates) == 1:
                    label = candidates[0]
                elif len(candidates) > 1:
                    # Prendo la più vicina come path testuale
                    candidates = sorted(candidates, key=lambda p: len(str(p)))
                    label = candidates[0]
                    print(f"[WARN] Più label trovate per {image.name}. Uso: {label}")

            unique_stem = safe_name(f"{zip_prefix}_{rel.with_suffix('')}")
            samples.append(
                {
                    "image": image,
                    "label": label,
                    "stem": unique_stem,
                }
            )

    return samples


def write_data_yaml(out_dir: Path, class_names):
    names_block = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)])

    text = f"""path: {out_dir.as_posix()}

train: images/train
val: images/val

names:
{names_block}
"""

    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Unisce export YOLO/CVAT e crea dataset YOLO train/val."
    )

    parser.add_argument(
        "--zips",
        nargs="+",
        required=True,
        help="Path degli zip esportati da CVAT in formato YOLO."
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Cartella output del dataset YOLO."
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Percentuale di immagini da usare come validation interna. Default: 0.10"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per split riproducibile. Default: 42"
    )

    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Nomi classi in ordine. Esempio: --names palla canestro"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive la cartella output se esiste."
    )

    args = parser.parse_args()

    zip_paths = [Path(p) for p in args.zips]
    out_dir = Path(args.out)

    for z in zip_paths:
        if not z.exists():
            raise FileNotFoundError(f"Zip non trovato: {z}")

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"La cartella output esiste già: {out_dir}. "
                f"Usa --overwrite per sovrascriverla."
            )

    out_images_train = out_dir / "images" / "train"
    out_images_val = out_dir / "images" / "val"
    out_labels_train = out_dir / "labels" / "train"
    out_labels_val = out_dir / "labels" / "val"

    for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        extracted_roots = []
        source_class_names = []

        print("[INFO] Estrazione zip...")

        for i, zip_path in enumerate(zip_paths):
            extract_dir = tmp_dir / f"zip_{i + 1}"
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            extracted_roots.append(extract_dir)

            names = read_class_names(extract_dir)
            if names:
                source_class_names.append(names)
                print(f"[INFO] Classi trovate in {zip_path.name}: {names}")
            else:
                print(f"[WARN] Nessun file classi trovato in {zip_path.name}")

        if source_class_names:
            first = source_class_names[0]
            for names in source_class_names[1:]:
                if names != first:
                    raise ValueError(
                        "Gli zip hanno classi diverse o in ordine diverso:\n"
                        f"Primo zip: {first}\n"
                        f"Altro zip: {names}\n"
                        "Correggere l'ordine delle label in CVAT oppure riesportare."
                    )

        if args.names is not None:
            class_names = args.names

            if source_class_names and len(class_names) != len(source_class_names[0]):
                raise ValueError(
                    f"--names contiene {len(class_names)} classi, "
                    f"ma dagli export ne risultano {len(source_class_names[0])}."
                )

            if source_class_names and class_names != source_class_names[0]:
                print(
                    "[WARN] I nomi passati con --names sono diversi da quelli esportati da CVAT.\n"
                    f"       CVAT:   {source_class_names[0]}\n"
                    f"       Output: {class_names}\n"
                    "       Va bene solo se state rinominando le classi mantenendo lo stesso ordine."
                )

        elif source_class_names:
            class_names = source_class_names[0]

        else:
            class_names = ["ball", "rim"]
            print(f"[WARN] Uso classi di default: {class_names}")

        num_classes = len(class_names)

        print("[INFO] Raccolta immagini e label...")
        samples = collect_samples(extracted_roots)

        if not samples:
            raise RuntimeError("Nessuna immagine trovata negli zip.")

        random.seed(args.seed)
        random.shuffle(samples)

        num_val = max(1, round(len(samples) * args.val_ratio))
        val_samples = samples[:num_val]
        train_samples = samples[num_val:]

        print(f"[INFO] Totale immagini: {len(samples)}")
        print(f"[INFO] Train: {len(train_samples)}")
        print(f"[INFO] Val:   {len(val_samples)}")

        summary = {
            "train": Counter(),
            "val": Counter(),
        }

        missing_labels = {
            "train": 0,
            "val": 0,
        }

        def copy_split(split_name, split_samples, out_img_dir, out_lbl_dir):
            for sample in split_samples:
                src_img = sample["image"]
                src_lbl = sample["label"]

                dst_img_name = sample["stem"] + src_img.suffix.lower()
                dst_lbl_name = sample["stem"] + ".txt"

                dst_img = out_img_dir / dst_img_name
                dst_lbl = out_lbl_dir / dst_lbl_name

                shutil.copy2(src_img, dst_img)

                if src_lbl is None:
                    missing_labels[split_name] += 1

                counts = clean_yolo_label(src_lbl, dst_lbl, num_classes)
                summary[split_name].update(counts)

        copy_split("train", train_samples, out_images_train, out_labels_train)
        copy_split("val", val_samples, out_images_val, out_labels_val)

        write_data_yaml(out_dir, class_names)

        print("\n[DONE] Dataset creato in:")
        print(f"       {out_dir}")
        print("\n[INFO] Classi:")
        for i, name in enumerate(class_names):
            print(f"       {i}: {name}")

        print("\n[INFO] Box per split:")
        for split in ["train", "val"]:
            print(f"       {split}:")
            for class_id, class_name in enumerate(class_names):
                print(f"         {class_name}: {summary[split][class_id]}")
            print(f"         immagini senza label trovata: {missing_labels[split]}")

        print("\n[INFO] File YAML:")
        print(f"       {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()