#!/usr/bin/env python3
"""
check_dataset.py

Controllo del dataset originale in v1

Per eseguirlo dalla root del progetto:
    python src/data/check_dataset.py

Lo script controlla:
1. presenza delle colonne fondamentali in manifest.csv;
2. coerenza tra path, label e split;
3. validità della durata delle clip;
4. duplicati di clip_id e path;
5. data leakage tra train, val e test tramite video_id;
6. validità fisica dei file video.

Produce il file:
    outputs/dataset_checks/check_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

print("ok")
# =============================================================================
# 1. Configurazione del progetto
# =============================================================================
# Qui sono definiti i path principali

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "dataset_basket_v1_original"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "dataset_checks" / "check_report.txt"


# =============================================================================
# 2. Costanti del dataset
# =============================================================================

REQUIRED_COLUMNS = [
    "clip_id",
    "path",
    "video_id",
    "start_time",
    "end_time",
    "label",
    "split",
]

EXPECTED_LABELS = {
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
    "idle",
    "non-gioco",
}

EXPECTED_SPLITS = {"train", "val", "test"}


# =============================================================================
# 3. Funzioni di supporto
# =============================================================================

def add_section(lines: list[str], title: str) -> None:
    """Aggiunge una sezione al report testuale."""
    lines.append("")
    lines.append(f"--- {title} ---")


def add_examples(lines: list[str], df: pd.DataFrame, columns: list[str], max_examples: int = 5) -> None:
    """Aggiunge al report alcuni esempi di righe problematiche."""
    if df.empty:
        return

    lines.append("  Esempi:")
    for _, row in df[columns].head(max_examples).iterrows():
        values = ", ".join(f"{col}={row[col]}" for col in columns)
        lines.append(f"  - {values}")


def save_report(lines: list[str]) -> None:
    """Salva il report nel file di output definito nella configurazione."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def check_video_file(video_path: Path) -> tuple[bool, str]:
    """
    Controlla se un video è apribile e ha metadati minimi validi.

    Ritorna:
        (True, "ok") se il video sembra valido;
        (False, motivo) altrimenti.
    """
    if cv2 is None:
        return False, "opencv_not_installed"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False, "cannot_open"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    if frame_count <= 0:
        return False, "zero_frames"
    if fps <= 0:
        return False, "invalid_fps"
    if width <= 0 or height <= 0:
        return False, "invalid_resolution"

    return True, "ok"


# =============================================================================
# 4. Main
# =============================================================================

def main() -> None:
    lines: list[str] = []
    errors = 0

    lines.append("Dataset check report")
    lines.append("====================")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Dataset root: {DATASET_ROOT}")
    lines.append(f"Manifest: {MANIFEST_PATH}")
    lines.append(f"Output report: {OUTPUT_FILE}")

    # -------------------------------------------------------------------------
    # 4.1 Controllo esistenza della root del dataset
    # -------------------------------------------------------------------------
    add_section(lines, "Controllo dataset root")

    if not DATASET_ROOT.exists():
        lines.append(f"[ERROR] Dataset root non trovata: {DATASET_ROOT}")
        save_report(lines)
        print(f"Report salvato in: {OUTPUT_FILE}")
        sys.exit(1)

    lines.append("[OK] Dataset root trovata")

    # -------------------------------------------------------------------------
    # 4.2 Lettura manifest.csv e controllo colonne
    # -------------------------------------------------------------------------
    add_section(lines, "Lettura manifest.csv")

    if not MANIFEST_PATH.exists():
        lines.append(f"[ERROR] manifest.csv non trovato: {MANIFEST_PATH}")
        save_report(lines)
        print(f"Report salvato in: {OUTPUT_FILE}")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    lines.append("[OK] Manifest letto correttamente")
    lines.append(f"[INFO] Numero righe: {len(df)}")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        lines.append(f"[ERROR] Colonne mancanti: {missing_columns}")
        save_report(lines)
        print(f"Report salvato in: {OUTPUT_FILE}")
        sys.exit(1)

    lines.append("[OK] Tutte le colonne fondamentali sono presenti")

    # -------------------------------------------------------------------------
    # 4.3 Normalizzazione minima dei dati
    # -------------------------------------------------------------------------
    # Il file originale non viene modificato. La pulizia avviene solo in memoria.
    for col in ["clip_id", "path", "video_id", "label", "split"]:
        df[col] = df[col].astype(str).str.strip()

    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
    df["duration"] = df["end_time"] - df["start_time"]

    # -------------------------------------------------------------------------
    # 4.4 Controllo path, label e split
    # -------------------------------------------------------------------------
    add_section(lines, "Controllo path, label e split")

    df["abs_path"] = df["path"].apply(lambda p: DATASET_ROOT / Path(p))
    df["file_exists"] = df["abs_path"].apply(lambda p: p.exists())
    df["path_parts"] = df["path"].apply(lambda p: Path(p).parts)
    df["path_is_valid"] = df["path_parts"].apply(lambda parts: len(parts) >= 3)
    df["path_split"] = df["path_parts"].apply(lambda parts: parts[0] if len(parts) >= 1 else None)
    df["path_label"] = df["path_parts"].apply(lambda parts: parts[1] if len(parts) >= 2 else None)

    checks = [
        (
            df["file_exists"],
            "Tutti i file indicati nel manifest esistono",
            "File mancanti",
            ["clip_id", "path"],
        ),
        (
            df["label"].isin(EXPECTED_LABELS),
            "Tutte le label sono valide",
            "Label non valide",
            ["clip_id", "label", "path"],
        ),
        (
            df["split"].isin(EXPECTED_SPLITS),
            "Tutti gli split sono validi",
            "Split non validi",
            ["clip_id", "split", "path"],
        ),
        (
            df["path_is_valid"],
            "Tutti i path hanno struttura split/label/nome_file",
            "Path con struttura inattesa",
            ["clip_id", "path"],
        ),
        (
            df["split"] == df["path_split"],
            "Lo split nel CSV coincide con lo split nel path",
            "Incoerenze split CSV/path",
            ["clip_id", "split", "path_split", "path"],
        ),
        (
            df["label"] == df["path_label"],
            "La label nel CSV coincide con la label nel path",
            "Incoerenze label CSV/path",
            ["clip_id", "label", "path_label", "path"],
        ),
    ]

    for mask, ok_message, error_message, example_columns in checks:
        wrong_rows = df[~mask]
        if wrong_rows.empty:
            lines.append(f"[OK] {ok_message}")
        else:
            errors += 1
            lines.append(f"[ERROR] {error_message}: {len(wrong_rows)}")
            add_examples(lines, wrong_rows, example_columns)

    # -------------------------------------------------------------------------
    # 4.5 Controllo durata delle clip
    # -------------------------------------------------------------------------
    add_section(lines, "Controllo durata clip")

    invalid_times = df["start_time"].isna() | df["end_time"].isna()
    invalid_duration = df["duration"].isna() | (df["duration"] <= 0)

    if invalid_times.any():
        errors += 1
        lines.append(f"[ERROR] start_time/end_time mancanti o non numerici: {int(invalid_times.sum())}")
        add_examples(lines, df[invalid_times], ["clip_id", "path", "start_time", "end_time"])
    else:
        lines.append("[OK] start_time ed end_time sono numerici")

    if invalid_duration.any():
        errors += 1
        lines.append(f"[ERROR] Clip con durata non valida: {int(invalid_duration.sum())}")
        add_examples(lines, df[invalid_duration], ["clip_id", "path", "start_time", "end_time", "duration"])
    else:
        lines.append("[OK] Tutte le clip hanno durata positiva")

    # -------------------------------------------------------------------------
    # 4.6 Controllo duplicati
    # -------------------------------------------------------------------------
    add_section(lines, "Controllo duplicati")

    duplicate_clip_ids = df[df["clip_id"].duplicated(keep=False)]
    duplicate_paths = df[df["path"].duplicated(keep=False)]

    if duplicate_clip_ids.empty:
        lines.append("[OK] Nessun clip_id duplicato")
    else:
        errors += 1
        lines.append(f"[ERROR] clip_id duplicati: {duplicate_clip_ids['clip_id'].nunique()}")
        add_examples(lines, duplicate_clip_ids, ["clip_id", "path"])

    if duplicate_paths.empty:
        lines.append("[OK] Nessun path duplicato")
    else:
        errors += 1
        lines.append(f"[ERROR] path duplicati: {duplicate_paths['path'].nunique()}")
        add_examples(lines, duplicate_paths, ["clip_id", "path"])

    # -------------------------------------------------------------------------
    # 4.7 Controllo data leakage tra train, val e test
    # -------------------------------------------------------------------------
    add_section(lines, "Controllo data leakage")

    leaked_videos = []
    for video_id, group in df.groupby("video_id"):
        splits = sorted(group["split"].unique())
        if len(splits) > 1:
            leaked_videos.append((video_id, splits))

    if not leaked_videos:
        lines.append("[OK] Nessun video_id compare in più split")
    else:
        errors += 1
        lines.append(f"[ERROR] Video presenti in più split: {len(leaked_videos)}")
        for video_id, splits in leaked_videos[:5]:
            lines.append(f"  - video_id={video_id}, split={splits}")

    for split in sorted(EXPECTED_SPLITS):
        n_videos = df[df["split"] == split]["video_id"].nunique()
        lines.append(f"[INFO] video_id in {split}: {n_videos}")

    # -------------------------------------------------------------------------
    # 4.8 Controllo fisico dei file video
    # -------------------------------------------------------------------------
    add_section(lines, "Controllo fisico dei video")

    if cv2 is None:
        errors += 1
        lines.append("[ERROR] OpenCV non è installato. Installa opencv-python.")
    else:
        video_errors: list[tuple[str, str, str]] = []
        existing_files = df[df["file_exists"]]

        for row in existing_files.itertuples(index=False):
            ok, reason = check_video_file(row.abs_path)
            if not ok:
                video_errors.append((row.clip_id, row.path, reason))

        lines.append(f"[INFO] Video controllati: {len(existing_files)}")

        if not video_errors:
            lines.append("[OK] Tutti i video esistenti sono apribili e validi")
        else:
            errors += 1
            lines.append(f"[ERROR] Video non validi: {len(video_errors)}")
            lines.append("  Esempi:")
            for clip_id, path, reason in video_errors[:5]:
                lines.append(f"  - clip_id={clip_id}, path={path}, motivo={reason}")

    # -------------------------------------------------------------------------
    # 4.9 Risultato finale
    # -------------------------------------------------------------------------
    add_section(lines, "Risultato finale")

    if errors == 0:
        lines.append("[OK] Dataset check completato senza errori")
    else:
        lines.append(f"[ERROR] Dataset check completato con {errors} sezioni problematiche")

    save_report(lines)

    print(f"Report salvato in: {OUTPUT_FILE}")
    print(f"Sezioni problematiche: {errors}")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
