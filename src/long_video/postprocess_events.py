from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.long_video import defaults
from src.long_video.utils import (
    as_path,
    check_output_files,
    ensure_exists,
    normalize_label,
    overlap_duration,
    round_float,
    safe_float,
)


# =============================================================================
# Configurazione fissa exp_long_13
# =============================================================================

ACTION_LABELS = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
]
SHOT_LABELS = [label for label in ACTION_LABELS if label != "passaggio"]
ALL_LABELS = ACTION_LABELS + ["no-action"]

# exp_long_13 storico: base exp_long_04 + filtro action/no-action con margine 20.0.
# I parametri sono fissati nel codice per mantenere riproducibile la configurazione
# effettivamente usata negli esperimenti long-video.
MIN_CONF_PASSAGGIO = 0.75
MIN_CONF_TIRO = 0.40
MIN_EVENT_DURATION_SEC = 0.70
MERGE_GAP_SEC = 0.20
MAX_WINDOW_SEC_PASSAGGIO = 2.00
MIN_WINDOW_SEC_TIRO = 1.00
MAX_WINDOW_SEC_TIRO = 3.00
MAX_DURATION_PASSAGGIO = 1.50
MAX_DURATION_TIRO = 3.00
NOACTION_MARGIN = 20.0

OUTPUT_COLUMNS = [
    "event_id",
    "label",
    "start_time",
    "end_time",
    "duration_sec",
    "center_time",
    "confidence",
    "confidence_mean",
    "confidence_max",
    "confidence_median",
    "noaction_mean",
    "threshold",
    "num_windows",
    "first_window_id",
    "last_window_id",
    "scales_used",
]


# =============================================================================
# Lettura predizioni raw
# =============================================================================


def score_column_for_label(label: str) -> str:
    """Restituisce il nome colonna prodotto da infer_exp46_from_store.py."""
    if label == "no-action":
        return "score_noaction"
    return f"score_{label}"


def threshold_for_label(label: str) -> float:
    return MIN_CONF_PASSAGGIO if label == "passaggio" else MIN_CONF_TIRO


def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    ensure_exists(predictions_csv, "Predizioni finestre", must_be_file=True)
    df = pd.read_csv(predictions_csv)
    if df.empty:
        raise ValueError(f"Il file predizioni è vuoto: {predictions_csv}")

    required = ["window_id", "start_time", "end_time", "pred_label", "confidence"]
    required += [score_column_for_label(label) for label in ALL_LABELS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Colonne obbligatorie mancanti in {predictions_csv}: {missing}. "
            "Questo post-processing è pensato per il CSV raw prodotto dalla versione exp_long_13 "
            "di infer_exp46_from_store.py."
        )

    out = df.copy()
    out["window_id"] = out["window_id"].astype(str)
    out["pred_label"] = out["pred_label"].apply(normalize_label)

    for col in ["start_time", "end_time", "confidence", *[score_column_for_label(l) for l in ALL_LABELS]]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "center_time" in out.columns:
        out["center_time"] = pd.to_numeric(out["center_time"], errors="coerce")
    else:
        out["center_time"] = (out["start_time"] + out["end_time"]) / 2.0

    if "scale_sec" in out.columns:
        out["scale_sec"] = pd.to_numeric(out["scale_sec"], errors="coerce")
    else:
        out["scale_sec"] = out["end_time"] - out["start_time"]

    numeric_cols = ["start_time", "end_time", "center_time", "scale_sec"]
    if out[numeric_cols].isna().any().any():
        bad_cols = [col for col in numeric_cols if out[col].isna().any()]
        raise ValueError(f"Valori numerici non validi nelle colonne: {bad_cols}")

    out = out[out["end_time"] > out["start_time"]].copy()
    if out.empty:
        raise ValueError("Nessuna finestra valida dopo il filtro end_time > start_time")

    out["noaction_score"] = out[score_column_for_label("no-action")].clip(lower=0.0)

    def action_score(row: pd.Series) -> float:
        label = normalize_label(row["pred_label"])
        if label in ALL_LABELS:
            return safe_float(row[score_column_for_label(label)])
        return safe_float(row["confidence"])

    out["action_score"] = out.apply(action_score, axis=1)
    return out.sort_values(["start_time", "end_time", "scale_sec"]).reset_index(drop=True)


# =============================================================================
# Da finestre candidate a eventi
# =============================================================================


def build_candidate_windows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for _, row in predictions.iterrows():
        label = normalize_label(row["pred_label"])
        if label == "no-action" or label not in ACTION_LABELS:
            continue

        scale_sec = safe_float(row["scale_sec"])
        if label == "passaggio" and scale_sec > MAX_WINDOW_SEC_PASSAGGIO:
            continue
        if label in SHOT_LABELS and not (MIN_WINDOW_SEC_TIRO <= scale_sec <= MAX_WINDOW_SEC_TIRO):
            continue

        score = safe_float(row["action_score"])
        noaction_score = safe_float(row["noaction_score"])
        threshold = threshold_for_label(label)

        if score < threshold:
            continue
        if score < noaction_score * NOACTION_MARGIN:
            continue

        candidates.append(
            {
                "window_id": str(row["window_id"]),
                "label": label,
                "start_time": safe_float(row["start_time"]),
                "end_time": safe_float(row["end_time"]),
                "center_time": safe_float(row["center_time"]),
                "confidence": score,
                "noaction_score": noaction_score,
                "threshold": threshold,
                "scale_sec": scale_sec,
            }
        )

    return sorted(candidates, key=lambda r: (r["label"], r["start_time"], r["end_time"], r["scale_sec"]))


def split_group_by_max_duration(
    rows: list[dict[str, Any]],
    max_duration: float,
) -> list[tuple[list[dict[str, Any]], float | None, float | None]]:
    if not rows:
        return []
    if max_duration <= 0:
        return [(rows, None, None)]

    group_start = min(safe_float(r["start_time"]) for r in rows)
    group_end = max(safe_float(r["end_time"]) for r in rows)
    duration = group_end - group_start
    if duration <= max_duration:
        return [(rows, None, None)]

    chunks: list[tuple[list[dict[str, Any]], float | None, float | None]] = []
    num_chunks = int(math.ceil(duration / max_duration))

    for chunk_idx in range(num_chunks):
        chunk_start = group_start + chunk_idx * max_duration
        chunk_end = min(group_end, chunk_start + max_duration)
        if chunk_end <= chunk_start:
            continue

        chunk_rows = []
        for row in rows:
            center = safe_float(row["center_time"])
            if chunk_start <= center < chunk_end:
                chunk_rows.append(row)
            elif chunk_idx == num_chunks - 1 and math.isclose(center, chunk_end):
                chunk_rows.append(row)

        if chunk_rows:
            chunks.append((chunk_rows, chunk_start, chunk_end))

    return chunks


def aggregate_rows_to_event(
    rows: list[dict[str, Any]],
    event_id: int,
    forced_start: float | None = None,
    forced_end: float | None = None,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: (safe_float(r["start_time"]), safe_float(r["end_time"])))
    start_time = safe_float(forced_start) if forced_start is not None else min(
        safe_float(r["start_time"]) for r in ordered
    )
    end_time = safe_float(forced_end) if forced_end is not None else max(
        safe_float(r["end_time"]) for r in ordered
    )
    if end_time <= start_time:
        end_time = start_time + 1e-6

    scores = np.asarray([safe_float(r["confidence"]) for r in ordered], dtype=np.float64)
    noaction_scores = np.asarray([safe_float(r["noaction_score"]) for r in ordered], dtype=np.float64)
    thresholds = np.asarray([safe_float(r["threshold"]) for r in ordered], dtype=np.float64)
    scales = sorted({round_float(r["scale_sec"], ndigits=3) for r in ordered})

    return {
        "event_id": int(event_id),
        "label": str(ordered[0]["label"]),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration_sec": float(end_time - start_time),
        "center_time": float((start_time + end_time) / 2.0),
        # exp_long_13 usa confidence mode max.
        "confidence": float(np.max(scores)) if scores.size else 0.0,
        "confidence_mean": float(np.mean(scores)) if scores.size else 0.0,
        "confidence_max": float(np.max(scores)) if scores.size else 0.0,
        "confidence_median": float(np.median(scores)) if scores.size else 0.0,
        "noaction_mean": float(np.mean(noaction_scores)) if noaction_scores.size else 0.0,
        "threshold": float(np.min(thresholds)) if thresholds.size else 0.0,
        "num_windows": int(len(ordered)),
        "first_window_id": str(ordered[0]["window_id"]),
        "last_window_id": str(ordered[-1]["window_id"]),
        "scales_used": "mixed" if len(scales) != 1 else f"{scales[0]:.3f}".rstrip("0").rstrip("."),
    }


def append_group_events(
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    next_event_id: int,
) -> int:
    if not rows:
        return next_event_id

    label = str(rows[0]["label"])
    max_duration = MAX_DURATION_PASSAGGIO if label == "passaggio" else MAX_DURATION_TIRO

    for chunk_rows, forced_start, forced_end in split_group_by_max_duration(rows, max_duration=max_duration):
        event = aggregate_rows_to_event(
            chunk_rows,
            event_id=next_event_id,
            forced_start=forced_start,
            forced_end=forced_end,
        )
        if safe_float(event["duration_sec"]) < MIN_EVENT_DURATION_SEC:
            continue
        events.append(event)
        next_event_id += 1

    return next_event_id


def group_candidates_to_events(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    events: list[dict[str, Any]] = []
    next_event_id = 0

    for label in ACTION_LABELS:
        label_rows = [r for r in candidates if r["label"] == label]
        if not label_rows:
            continue

        current: list[dict[str, Any]] = []
        current_end: float | None = None

        for row in sorted(label_rows, key=lambda r: (r["start_time"], r["end_time"])):
            start = safe_float(row["start_time"])
            end = safe_float(row["end_time"])
            if not current:
                current = [row]
                current_end = end
                continue

            assert current_end is not None
            if start - current_end <= MERGE_GAP_SEC:
                current.append(row)
                current_end = max(current_end, end)
            else:
                next_event_id = append_group_events(events, current, next_event_id)
                current = [row]
                current_end = end

        next_event_id = append_group_events(events, current, next_event_id)

    return sorted(events, key=lambda r: (r["start_time"], r["end_time"], r["label"]))


def suppress_overlaps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """exp_long_13 non permette overlap: mantiene l'evento più confidente."""
    order = sorted(
        range(len(events)),
        key=lambda i: (
            -safe_float(events[i].get("confidence", 0.0)),
            -safe_float(events[i].get("confidence_max", events[i].get("confidence", 0.0))),
            -safe_float(events[i].get("duration_sec", 0.0)),
            safe_float(events[i].get("start_time", 0.0)),
            safe_float(events[i].get("end_time", 0.0)),
            str(events[i].get("label", "")),
        ),
    )

    keep: list[int] = []
    for idx in order:
        candidate = events[idx]
        c_start = safe_float(candidate["start_time"])
        c_end = safe_float(candidate["end_time"])
        overlaps_kept = any(
            overlap_duration(
                c_start,
                c_end,
                safe_float(events[kept_idx]["start_time"]),
                safe_float(events[kept_idx]["end_time"]),
            )
            > 1e-9
            for kept_idx in keep
        )
        if not overlaps_kept:
            keep.append(idx)

    final_events = [dict(events[idx]) for idx in keep]
    final_events.sort(key=lambda r: (r["start_time"], r["end_time"], r["label"]))
    for event_id, event in enumerate(final_events):
        event["event_id"] = int(event_id)
    return final_events


# =============================================================================
# Output
# =============================================================================


def events_to_dataframe(events: list[dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(events)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in {"label", "first_window_id", "last_window_id", "scales_used"} else 0

    numeric_cols = [
        "start_time",
        "end_time",
        "duration_sec",
        "center_time",
        "confidence",
        "confidence_mean",
        "confidence_max",
        "confidence_median",
        "noaction_mean",
        "threshold",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(6)

    return df[OUTPUT_COLUMNS]


def count_by_label(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.get("label", ""))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def run_postprocess(predictions_csv: Path, output_dir: Path, overwrite: bool) -> pd.DataFrame:
    check_output_files(
        output_dir,
        output_files=["events_postprocessed.csv"],
        overwrite=overwrite,
    )

    predictions = load_predictions(predictions_csv)
    candidates = build_candidate_windows(predictions)
    raw_events = group_candidates_to_events(candidates)
    final_events = suppress_overlaps(raw_events)

    final_df = events_to_dataframe(final_events)
    final_df.to_csv(output_dir / "events_postprocessed.csv", index=False)

    print("\n=== Summary exp_long_13 ===")
    print(f"finestre totali:      {len(predictions)}")
    print(f"candidate windows:    {len(candidates)} | {count_by_label(candidates)}")
    print(f"eventi raw:           {len(raw_events)} | {count_by_label(raw_events)}")
    print(f"eventi finali:        {len(final_events)} | {count_by_label(final_events)}")
    print(f"\n[OK] events_postprocessed.csv salvato in: {output_dir / 'events_postprocessed.csv'}")

    return final_df


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-processing exp_long_13 delle predizioni raw long-video. "
            "La configurazione è fissa: soglie, margine no-action, merge e overlap "
            "sono quelli dell'esperimento finale."
        )
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=defaults.VAL_OUTPUT_DIR / "window_predictions_raw.csv",
        help="CSV prodotto da infer_exp46_from_store.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.VAL_OUTPUT_DIR,
        help="Cartella in cui salvare events_postprocessed.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.predictions_csv = as_path(args.predictions_csv)
    args.output_dir = as_path(args.output_dir)
    assert args.predictions_csv is not None
    assert args.output_dir is not None

    print("=== Post-processing long-video | exp_long_13 ===")
    print(f"predictions_csv:        {args.predictions_csv}")
    print(f"output_dir:             {args.output_dir}")
    print(f"min_conf_passaggio:     {MIN_CONF_PASSAGGIO:.3f}")
    print(f"min_conf_tiro:          {MIN_CONF_TIRO:.3f}")
    print(f"require action/noaction: True")
    print(f"noaction_margin:        {NOACTION_MARGIN:.3f}")
    print(f"merge_gap_sec:          {MERGE_GAP_SEC:.3f}")
    print(f"min_event_duration:     {MIN_EVENT_DURATION_SEC:.3f}")
    print(f"max_window_passaggio:   {MAX_WINDOW_SEC_PASSAGGIO:.3f}")
    print(f"min_window_tiro:        {MIN_WINDOW_SEC_TIRO:.3f}")
    print(f"max_window_tiro:        {MAX_WINDOW_SEC_TIRO:.3f}")
    print(f"max_duration_passaggio: {MAX_DURATION_PASSAGGIO:.3f}")
    print(f"max_duration_tiro:      {MAX_DURATION_TIRO:.3f}")
    print("confidence_mode:        max")
    print("forbid_overlaps:        True")

    run_postprocess(
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
