# =============================================================================
# Questo script genera una versione compatta del report eventi prodotto dalla
# pipeline long-video del progetto BasketAR. L'input atteso è il file
# events_postprocessed.csv, cioè il CSV ottenuto dopo inferenza e
# post-processing temporale degli eventi rilevati nel video.
#
# Il file non esegue inferenza, non modifica le predizioni e non calcola
# metriche: serve solo a preparare un CSV più pulito e facilmente leggibile
# per report, consegna o confronto qualitativo. In particolare valida i path
# di input/output, controlla che siano presenti le colonne necessarie, mantiene
# solo le informazioni essenziali dell'evento, ordina gli eventi nel tempo,
# arrotonda tempi/confidence e salva il risultato nel path richiesto.
#
# Output finale: un CSV con le sole colonne principali
# label, start_time, end_time, duration_sec, confidence, num_windows.
#
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Colonne che vengono mantenute nel report compatto finale.
# Tutte le altre colonne presenti nel CSV originale vengono scartate perché
# utili alla diagnostica interna, ma non necessarie nel report sintetico.
IMPORTANT_COLUMNS = [
    "label",
    "start_time",
    "end_time",
    "duration_sec",
    "confidence",
    "num_windows",
]


# Definizione dell'interfaccia CLI: permette di scegliere input, output,
# sovrascrittura e numero di decimali senza modificare il codice.
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


# Validazione dell'input: intercetta subito path mancanti o non validi,
# evitando errori meno chiari durante la lettura con pandas.
def ensure_input_exists(input_csv: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"File di input non trovato: {input_csv}")

    if not input_csv.is_file():
        raise FileNotFoundError(f"Il path di input non è un file: {input_csv}")


# Preparazione dell'output: crea la cartella di destinazione e protegge
# da sovrascritture accidentali quando --overwrite non è specificato.
def prepare_output_path(output_csv: Path, overwrite: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Il file di output esiste già: {output_csv}\n"
            "Usa --overwrite per sovrascriverlo."
        )


# Controllo di coerenza sul formato del CSV: il report compatto può essere
# costruito solo se tutte le colonne essenziali sono presenti.
def check_required_columns(df: pd.DataFrame, input_csv: Path) -> None:
    missing = [col for col in IMPORTANT_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Il file {input_csv} non contiene tutte le colonne richieste.\n"
            f"Colonne mancanti: {missing}\n"
            f"Colonne disponibili: {list(df.columns)}"
        )


# Trasformazione centrale dello script: seleziona le colonne importanti,
# ordina gli eventi temporalmente e normalizza il formato numerico.
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


# Flusso principale: legge gli argomenti, valida input/output, costruisce
# il report compatto e lo salva su disco.
def main() -> None:
    args = parse_args()

    input_csv: Path = args.input_csv
    output_csv: Path = args.output_csv

    ensure_input_exists(input_csv)
    prepare_output_path(output_csv, overwrite=args.overwrite)

    # Lettura del report completo prodotto dal post-processing long-video.
    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError(f"Il file di input è vuoto: {input_csv}")

    # Verifica che il CSV abbia il formato minimo atteso prima della selezione.
    check_required_columns(df, input_csv)

    compact = build_compact_report(
        df=df,
        round_digits=args.round_digits,
    )

    # Scrittura del CSV finale, senza indice pandas aggiuntivo.
    compact.to_csv(output_csv, index=False, encoding="utf-8")

    print("=== Report compatto eventi generato ===")
    print(f"Input:        {input_csv}")
    print(f"Output:       {output_csv}")
    print(f"Num eventi:   {len(compact)}")
    print(f"Colonne:      {', '.join(compact.columns)}")


if __name__ == "__main__":
    main()