from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# =============================================================================
# ESEMPI DI UTILIZZO
# =============================================================================
#
# Validation - exp_long_13:
#
# python -m src.long_video.export_compact_event_report `
#   --input-csv outputs/long_video/exp_long_13/events_postprocessed.csv `
#   --output-csv outputs/long_video/exp_long_13/BasketAR_validation_report_events_exp13.csv `
#   --overwrite
#
#
# Test - exp_long_13:
#
# python -m src.long_video.export_compact_event_report `
#   --input-csv outputs/long_video/test_exp_long_13/events_postprocessed.csv `
#   --output-csv outputs/long_video/test_exp_long_13/BasketAR_test_report_events_exp13.csv `
#   --overwrite
#
#
# Output:
#
# Il file generato contiene solo le colonne principali del report:
#
# label,start_time,end_time,duration_sec,confidence,num_windows
#
# =============================================================================


IMPORTANT_COLUMNS = [
    "label",
    "start_time",
    "end_time",
    "duration_sec",
    "confidence",
    "num_windows",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Esporta un report compatto degli eventi rilevati dalla pipeline "
            "long-video, mantenendo solo le colonne principali."
        )
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path del file events_postprocessed.csv prodotto dal post-processing.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path del CSV compatto da generare.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive il file di output se esiste già.",
    )

    parser.add_argument(
        "--round-digits",
        type=int,
        default=4,
        help="Numero di cifre decimali da mantenere per tempi e confidence.",
    )

    return parser.parse_args()


def ensure_input_exists(input_csv: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"File di input non trovato: {input_csv}")

    if not input_csv.is_file():
        raise FileNotFoundError(f"Il path di input non è un file: {input_csv}")


def prepare_output_path(output_csv: Path, overwrite: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Il file di output esiste già: {output_csv}\n"
            "Usa --overwrite per sovrascriverlo."
        )


def check_required_columns(df: pd.DataFrame, input_csv: Path) -> None:
    missing = [col for col in IMPORTANT_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Il file {input_csv} non contiene tutte le colonne richieste.\n"
            f"Colonne mancanti: {missing}\n"
            f"Colonne disponibili: {list(df.columns)}"
        )


def build_compact_report(df: pd.DataFrame, round_digits: int) -> pd.DataFrame:
    compact = df[IMPORTANT_COLUMNS].copy()

    # Ordinamento temporale degli eventi.
    compact = compact.sort_values(["start_time", "end_time", "label"]).reset_index(drop=True)

    # Arrotondamento delle colonne numeriche principali.
    numeric_columns = [
        "start_time",
        "end_time",
        "duration_sec",
        "confidence",
    ]

    for col in numeric_columns:
        compact[col] = pd.to_numeric(compact[col], errors="coerce").round(round_digits)

    compact["num_windows"] = pd.to_numeric(
        compact["num_windows"],
        errors="coerce",
    ).fillna(0).astype(int)

    return compact


def main() -> None:
    args = parse_args()

    input_csv: Path = args.input_csv
    output_csv: Path = args.output_csv

    ensure_input_exists(input_csv)
    prepare_output_path(output_csv, overwrite=args.overwrite)

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError(f"Il file di input è vuoto: {input_csv}")

    check_required_columns(df, input_csv)

    compact = build_compact_report(
        df=df,
        round_digits=args.round_digits,
    )

    compact.to_csv(output_csv, index=False, encoding="utf-8")

    print("=== Report compatto eventi generato ===")
    print(f"Input:        {input_csv}")
    print(f"Output:       {output_csv}")
    print(f"Num eventi:   {len(compact)}")
    print(f"Colonne:      {', '.join(compact.columns)}")


if __name__ == "__main__":
    main()