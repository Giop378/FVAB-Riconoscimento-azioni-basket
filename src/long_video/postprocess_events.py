from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.long_video import defaults


# =============================================================================
# Label e colonne attese dall'inferenza exp_46
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

ALL_LABELS = ACTION_LABELS + ["no-action"]

SCORE_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "passaggio": ["score_passaggio", "p_passaggio", "p_l1_passaggio"],
    "tiroDaDue0": ["score_tiroDaDue0", "score_tiro_da_due0", "p_tiroDaDue0"],
    "tiroDaDue1": ["score_tiroDaDue1", "score_tiro_da_due1", "p_tiroDaDue1"],
    "tiroDaTre0": ["score_tiroDaTre0", "score_tiro_da_tre0", "p_tiroDaTre0"],
    "tiroDaTre1": ["score_tiroDaTre1", "score_tiro_da_tre1", "p_tiroDaTre1"],
    "tiroLibero0": ["score_tiroLibero0", "score_tiro_libero0", "p_tiroLibero0"],
    "tiroLibero1": ["score_tiroLibero1", "score_tiro_libero1", "p_tiroLibero1"],
    "no-action": ["score_noaction", "score_no-action", "p_noaction", "p_l1_noaction"],
}

RAW_EVENT_COLUMNS = [
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
    "scale_index",
    "scale_sec",
    "first_window_id",
    "last_window_id",
]

FINAL_EVENT_COLUMNS = RAW_EVENT_COLUMNS + [
    "source_event_ids",
    "scales_used",
]


# =============================================================================
# Dataclass
# =============================================================================


@dataclass(frozen=True)
class ArgsSnapshot:
    predictions_csv: str
    output_dir: str
    smooth_window: int
    min_event_duration_sec: float
    merge_gap_sec: float
    temporal_nms_iou: float
    min_conf_passaggio: float
    min_conf_tiro: float
    candidate_mode: str
    confidence_aggregation: str
    require_action_gt_noaction: bool
    noaction_margin: float


# =============================================================================
# Utility generali
# =============================================================================


def as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(value)


def ensure_exists(path: Path, name: str, must_be_file: bool | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} non trovato: {path}")
    if must_be_file is True and not path.is_file():
        raise FileNotFoundError(f"{name} dovrebbe essere un file: {path}")
    if must_be_file is False and not path.is_dir():
        raise NotADirectoryError(f"{name} dovrebbe essere una cartella: {path}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def check_output_files(output_dir: Path, output_files: Iterable[str], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f for f in output_files if (output_dir / f).exists()]
    if existing and not overwrite:
        joined = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "Alcuni file di output esistono già:\n"
            f"{joined}\n"
            "Usa --overwrite per sovrascriverli."
        )
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def threshold_for_label(label: str, min_conf_passaggio: float, min_conf_tiro: float) -> float:
    if label == "passaggio":
        return float(min_conf_passaggio)
    return float(min_conf_tiro)


def normalize_score_col_name(name: str) -> str:
    return str(name).strip()


def find_score_column(df: pd.DataFrame, label: str) -> str:
    columns = {normalize_score_col_name(c): c for c in df.columns}
    columns_lower = {normalize_score_col_name(c).lower(): c for c in df.columns}

    for candidate in SCORE_COLUMN_CANDIDATES[label]:
        if candidate in columns:
            return columns[candidate]
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    # Fallback: infer_exp46_from_store usa score_{label.replace('-', '')}.
    fallback = f"score_{label.replace('-', '')}"
    if fallback in columns:
        return columns[fallback]
    if fallback.lower() in columns_lower:
        return columns_lower[fallback.lower()]

    raise KeyError(
        f"Colonna score per label '{label}' non trovata. "
        f"Candidati: {SCORE_COLUMN_CANDIDATES[label]}"
    )


# =============================================================================
# Lettura e preparazione predizioni
# =============================================================================


def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    ensure_exists(predictions_csv, "Predizioni finestre", must_be_file=True)
    df = pd.read_csv(predictions_csv)
    if df.empty:
        raise ValueError(f"Il file predizioni è vuoto: {predictions_csv}")

    required = ["window_id", "start_time", "end_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colonne obbligatorie mancanti in {predictions_csv}: {missing}")

    if "center_time" not in df.columns:
        df["center_time"] = (pd.to_numeric(df["start_time"]) + pd.to_numeric(df["end_time"])) / 2.0

    if "scale_sec" not in df.columns:
        df["scale_sec"] = pd.to_numeric(df["end_time"]) - pd.to_numeric(df["start_time"])

    if "scale_index" not in df.columns:
        # Codifica stabile per scale diverse.
        unique_scales = sorted(pd.to_numeric(df["scale_sec"]).dropna().unique().tolist())
        scale_to_idx = {float(v): i for i, v in enumerate(unique_scales)}
        df["scale_index"] = pd.to_numeric(df["scale_sec"]).map(lambda x: scale_to_idx.get(float(x), -1))

    numeric_cols = ["start_time", "end_time", "center_time", "scale_sec", "scale_index"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        bad_cols = [c for c in numeric_cols if df[c].isna().any()]
        raise ValueError(f"Valori numerici non validi nelle colonne: {bad_cols}")

    if (df["end_time"] <= df["start_time"]).any():
        raise ValueError("Trovate finestre con end_time <= start_time")

    # Mappa le colonne score a nomi canonici score__{label}.
    for label in ALL_LABELS:
        col = find_score_column(df, label)
        canonical = f"score__{label}"
        df[canonical] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    df = df.sort_values(["scale_index", "start_time", "end_time"]).reset_index(drop=True)
    return df


def add_smoothed_scores(df: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    if smooth_window < 1:
        raise ValueError(f"smooth_window deve essere >= 1, trovato {smooth_window}")

    out = df.copy()
    group_cols = ["scale_index"]

    for label in ALL_LABELS:
        raw_col = f"score__{label}"
        smooth_col = f"score_smooth__{label}"
        if smooth_window == 1:
            out[smooth_col] = out[raw_col].astype(float)
        else:
            out[smooth_col] = (
                out.groupby(group_cols, group_keys=False)[raw_col]
                .apply(
                    lambda s: s.rolling(
                        window=int(smooth_window),
                        min_periods=1,
                        center=True,
                    ).mean()
                )
                .astype(float)
            )
    return out


# =============================================================================
# Candidati finestra -> eventi raw
# =============================================================================


def build_candidate_rows(
    df: pd.DataFrame,
    min_conf_passaggio: float,
    min_conf_tiro: float,
    candidate_mode: str,
    require_action_gt_noaction: bool,
    noaction_margin: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if noaction_margin < 0:
        raise ValueError(f"noaction_margin deve essere >= 0, trovato {noaction_margin}")

    action_score_cols = {label: f"score_smooth__{label}" for label in ACTION_LABELS}
    noaction_col = "score_smooth__no-action"

    if candidate_mode not in {"best_per_window", "threshold_all"}:
        raise ValueError(f"candidate_mode non supportato: {candidate_mode}")

    for _, row in df.iterrows():
        noaction_score = safe_float(row[noaction_col])

        if candidate_mode == "best_per_window":
            scores = {label: safe_float(row[col]) for label, col in action_score_cols.items()}
            label = max(scores, key=scores.get)
            score = float(scores[label])
            raw_score = safe_float(row[f"score__{label}"])
            threshold = threshold_for_label(label, min_conf_passaggio, min_conf_tiro)

            if score < threshold:
                continue
            if require_action_gt_noaction and score < noaction_score * noaction_margin:
                continue

            rows.append(
                make_candidate_dict(
                    row=row,
                    label=label,
                    score=score,
                    raw_score=raw_score,
                    noaction_score=noaction_score,
                    threshold=threshold,
                )
            )

        else:  # threshold_all
            for label, col in action_score_cols.items():
                score = safe_float(row[col])
                raw_score = safe_float(row[f"score__{label}"])
                threshold = threshold_for_label(label, min_conf_passaggio, min_conf_tiro)
                if score < threshold:
                    continue
                if require_action_gt_noaction and score < noaction_score * noaction_margin:
                    continue
                rows.append(
                    make_candidate_dict(
                        row=row,
                        label=label,
                        score=score,
                        raw_score=raw_score,
                        noaction_score=noaction_score,
                        threshold=threshold,
                    )
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "window_id",
                "scale_index",
                "scale_sec",
                "start_time",
                "end_time",
                "center_time",
                "label",
                "score",
                "raw_score",
                "noaction_score",
                "threshold",
            ]
        )

    return pd.DataFrame(rows).sort_values(["scale_index", "label", "start_time"]).reset_index(drop=True)


def make_candidate_dict(
    row: pd.Series,
    label: str,
    score: float,
    raw_score: float,
    noaction_score: float,
    threshold: float,
) -> dict[str, Any]:
    return {
        "window_id": str(row["window_id"]),
        "scale_index": int(row["scale_index"]),
        "scale_sec": float(row["scale_sec"]),
        "start_time": float(row["start_time"]),
        "end_time": float(row["end_time"]),
        "center_time": float(row["center_time"]),
        "label": str(label),
        "score": float(score),
        "raw_score": float(raw_score),
        "noaction_score": float(noaction_score),
        "threshold": float(threshold),
    }


def aggregate_candidate_group(
    group_rows: list[dict[str, Any]],
    event_id: int,
    confidence_aggregation: str,
) -> dict[str, Any]:
    if not group_rows:
        raise ValueError("aggregate_candidate_group chiamata con gruppo vuoto")

    scores = np.array([safe_float(r["score"]) for r in group_rows], dtype=np.float64)
    noaction_scores = np.array([safe_float(r["noaction_score"]) for r in group_rows], dtype=np.float64)
    thresholds = np.array([safe_float(r["threshold"]) for r in group_rows], dtype=np.float64)

    if confidence_aggregation == "max":
        confidence = float(np.max(scores))
    elif confidence_aggregation == "mean":
        confidence = float(np.mean(scores))
    else:
        raise ValueError(f"confidence_aggregation non supportato: {confidence_aggregation}")

    start_time = float(min(r["start_time"] for r in group_rows))
    end_time = float(max(r["end_time"] for r in group_rows))

    return {
        "event_id": int(event_id),
        "label": str(group_rows[0]["label"]),
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": float(end_time - start_time),
        "center_time": float((start_time + end_time) / 2.0),
        "confidence": confidence,
        "confidence_mean": float(np.mean(scores)),
        "confidence_max": float(np.max(scores)),
        "confidence_median": float(np.median(scores)),
        "noaction_mean": float(np.mean(noaction_scores)),
        "threshold": float(np.min(thresholds)),
        "num_windows": int(len(group_rows)),
        "scale_index": int(group_rows[0]["scale_index"]),
        "scale_sec": float(group_rows[0]["scale_sec"]),
        "first_window_id": str(group_rows[0]["window_id"]),
        "last_window_id": str(group_rows[-1]["window_id"]),
    }


def candidates_to_raw_events(
    candidates: pd.DataFrame,
    merge_gap_sec: float,
    confidence_aggregation: str,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=RAW_EVENT_COLUMNS)

    if merge_gap_sec < 0:
        raise ValueError(f"merge_gap_sec deve essere >= 0, trovato {merge_gap_sec}")

    events: list[dict[str, Any]] = []
    event_id = 0

    grouped = candidates.sort_values(["scale_index", "label", "start_time"]).groupby(
        ["scale_index", "label"], sort=True
    )

    for (_, _), group in grouped:
        current: list[dict[str, Any]] = []
        current_end: float | None = None

        for row in group.to_dict("records"):
            start = float(row["start_time"])
            end = float(row["end_time"])
            if not current:
                current = [row]
                current_end = end
                continue

            assert current_end is not None
            gap = start - current_end
            if gap <= merge_gap_sec:
                current.append(row)
                current_end = max(current_end, end)
            else:
                events.append(
                    aggregate_candidate_group(
                        current,
                        event_id=event_id,
                        confidence_aggregation=confidence_aggregation,
                    )
                )
                event_id += 1
                current = [row]
                current_end = end

        if current:
            events.append(
                aggregate_candidate_group(
                    current,
                    event_id=event_id,
                    confidence_aggregation=confidence_aggregation,
                )
            )
            event_id += 1

    if not events:
        return pd.DataFrame(columns=RAW_EVENT_COLUMNS)

    return pd.DataFrame(events, columns=RAW_EVENT_COLUMNS).sort_values(
        ["start_time", "end_time", "label"]
    ).reset_index(drop=True)


# =============================================================================
# Eventi raw -> eventi finali
# =============================================================================


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return float(inter / union)


def merge_event_records(records: list[dict[str, Any]], new_event_id: int) -> dict[str, Any]:
    if not records:
        raise ValueError("merge_event_records chiamata con lista vuota")

    start_time = float(min(r["start_time"] for r in records))
    end_time = float(max(r["end_time"] for r in records))
    num_windows = int(sum(int(r.get("num_windows", 0)) for r in records))
    weights = np.array([max(int(r.get("num_windows", 1)), 1) for r in records], dtype=np.float64)

    conf_mean_values = np.array([safe_float(r.get("confidence_mean", r.get("confidence", 0.0))) for r in records])
    noaction_values = np.array([safe_float(r.get("noaction_mean", 0.0)) for r in records])

    source_ids = []
    scales = []
    for r in records:
        source_ids.append(str(r.get("event_id")))
        scale = r.get("scale_sec", None)
        if scale is not None:
            scales.append(f"{safe_float(scale):.3f}".rstrip("0").rstrip("."))

    # Per first/last window usiamo gli estremi temporali.
    ordered = sorted(records, key=lambda r: (float(r["start_time"]), float(r["end_time"])))

    return {
        "event_id": int(new_event_id),
        "label": str(records[0]["label"]),
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": float(end_time - start_time),
        "center_time": float((start_time + end_time) / 2.0),
        "confidence": float(max(safe_float(r.get("confidence", 0.0)) for r in records)),
        "confidence_mean": float(np.average(conf_mean_values, weights=weights)),
        "confidence_max": float(max(safe_float(r.get("confidence_max", r.get("confidence", 0.0))) for r in records)),
        "confidence_median": float(np.median([safe_float(r.get("confidence_median", r.get("confidence", 0.0))) for r in records])),
        "noaction_mean": float(np.average(noaction_values, weights=weights)),
        "threshold": float(min(safe_float(r.get("threshold", 0.0)) for r in records)),
        "num_windows": num_windows,
        "scale_index": -1 if len(records) > 1 else int(records[0].get("scale_index", -1)),
        "scale_sec": float(records[0].get("scale_sec", 0.0)) if len(records) == 1 else -1.0,
        "first_window_id": str(ordered[0].get("first_window_id", "")),
        "last_window_id": str(ordered[-1].get("last_window_id", "")),
        "source_event_ids": ";".join(source_ids),
        "scales_used": ";".join(sorted(set(scales))),
    }


def filter_short_events(events: pd.DataFrame, min_event_duration_sec: float) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if min_event_duration_sec < 0:
        raise ValueError(
            f"min_event_duration_sec deve essere >= 0, trovato {min_event_duration_sec}"
        )
    out = events.copy()
    out["duration_sec"] = pd.to_numeric(out["end_time"]) - pd.to_numeric(out["start_time"])
    return out[out["duration_sec"] >= float(min_event_duration_sec)].reset_index(drop=True)


def merge_same_label_events(events: pd.DataFrame, merge_gap_sec: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=FINAL_EVENT_COLUMNS)

    merged: list[dict[str, Any]] = []
    new_event_id = 0

    for label, group in events.sort_values(["label", "start_time", "end_time"]).groupby("label"):
        current: list[dict[str, Any]] = []
        current_end: float | None = None

        for row in group.to_dict("records"):
            start = float(row["start_time"])
            end = float(row["end_time"])

            if not current:
                current = [row]
                current_end = end
                continue

            assert current_end is not None
            gap = start - current_end
            if gap <= merge_gap_sec:
                current.append(row)
                current_end = max(current_end, end)
            else:
                merged.append(merge_event_records(current, new_event_id=new_event_id))
                new_event_id += 1
                current = [row]
                current_end = end

        if current:
            merged.append(merge_event_records(current, new_event_id=new_event_id))
            new_event_id += 1

    if not merged:
        return pd.DataFrame(columns=FINAL_EVENT_COLUMNS)

    return pd.DataFrame(merged, columns=FINAL_EVENT_COLUMNS).sort_values(
        ["start_time", "end_time", "label"]
    ).reset_index(drop=True)


def temporal_nms(events: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if iou_threshold < 0:
        raise ValueError(f"temporal_nms_iou deve essere >= 0, trovato {iou_threshold}")
    if iou_threshold <= 0:
        out = events.copy().sort_values(["start_time", "end_time", "label"]).reset_index(drop=True)
        out["event_id"] = np.arange(len(out), dtype=int)
        return out

    records = events.to_dict("records")
    order = sorted(
        range(len(records)),
        key=lambda i: (
            -safe_float(records[i].get("confidence", 0.0)),
            -safe_float(records[i].get("duration_sec", 0.0)),
            safe_float(records[i].get("start_time", 0.0)),
        ),
    )

    keep_indices: list[int] = []
    for idx in order:
        candidate = records[idx]
        c_start = safe_float(candidate["start_time"])
        c_end = safe_float(candidate["end_time"])

        should_keep = True
        for kept_idx in keep_indices:
            kept = records[kept_idx]
            iou = temporal_iou(
                c_start,
                c_end,
                safe_float(kept["start_time"]),
                safe_float(kept["end_time"]),
            )
            if iou > iou_threshold:
                should_keep = False
                break

        if should_keep:
            keep_indices.append(idx)

    kept_records = [records[i] for i in keep_indices]
    out = pd.DataFrame(kept_records, columns=FINAL_EVENT_COLUMNS).sort_values(
        ["start_time", "end_time", "label"]
    ).reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=int)
    return out


def postprocess_events(
    raw_events: pd.DataFrame,
    min_event_duration_sec: float,
    merge_gap_sec: float,
    temporal_nms_iou_value: float,
) -> pd.DataFrame:
    filtered = filter_short_events(raw_events, min_event_duration_sec=min_event_duration_sec)
    merged = merge_same_label_events(filtered, merge_gap_sec=merge_gap_sec)
    final = temporal_nms(merged, iou_threshold=temporal_nms_iou_value)
    if final.empty:
        return pd.DataFrame(columns=FINAL_EVENT_COLUMNS)

    # Ricalcola durata e centro dopo tutte le operazioni.
    final = final.copy()
    final["duration_sec"] = pd.to_numeric(final["end_time"]) - pd.to_numeric(final["start_time"])
    final["center_time"] = (pd.to_numeric(final["start_time"]) + pd.to_numeric(final["end_time"])) / 2.0
    final = final.sort_values(["start_time", "end_time", "label"]).reset_index(drop=True)
    final["event_id"] = np.arange(len(final), dtype=int)
    return final[FINAL_EVENT_COLUMNS]


# =============================================================================
# Export
# =============================================================================


def round_float(value: Any, ndigits: int = 6) -> float:
    return round(safe_float(value), ndigits)


def write_events_csv(path: Path, events: pd.DataFrame, columns: list[str]) -> None:
    out = events.copy()
    for col in [
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
        "scale_sec",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(6)

    # Garantisce colonne stabili anche con DataFrame vuoto.
    for col in columns:
        if col not in out.columns:
            out[col] = []
    out = out[columns]
    out.to_csv(path, index=False)


def events_to_annotations(final_events: pd.DataFrame) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for row in final_events.to_dict("records"):
        annotations.append(
            {
                "event_id": int(row["event_id"]),
                "label": str(row["label"]),
                "start_time": round_float(row["start_time"]),
                "end_time": round_float(row["end_time"]),
                "duration_sec": round_float(row["duration_sec"]),
                "confidence": round_float(row["confidence"], ndigits=8),
                "num_windows": int(row.get("num_windows", 0)),
                "scales_used": str(row.get("scales_used", "")),
            }
        )
    return annotations


def count_by_label(events: pd.DataFrame) -> dict[str, int]:
    if events.empty or "label" not in events.columns:
        return {label: 0 for label in ACTION_LABELS}
    counts = events["label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in ACTION_LABELS}


def write_annotations_json(
    path: Path,
    final_events: pd.DataFrame,
    args: argparse.Namespace,
    predictions_csv: Path,
) -> None:
    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_predictions_csv": str(predictions_csv),
        "num_events": int(len(final_events)),
        "counts_by_label": count_by_label(final_events),
        "events": events_to_annotations(final_events),
        "notes": {
            "time_reference": "Timestamp assoluti del video originale, in secondi.",
            "labels": "Sono esportate solo le 7 azioni reali; no-action è usata come background.",
        },
        "postprocess": asdict(make_args_snapshot(args)),
    }
    write_json(path, data)


def make_args_snapshot(args: argparse.Namespace) -> ArgsSnapshot:
    return ArgsSnapshot(
        predictions_csv=str(args.predictions_csv),
        output_dir=str(args.output_dir),
        smooth_window=int(args.smooth_window),
        min_event_duration_sec=float(args.min_event_duration_sec),
        merge_gap_sec=float(args.merge_gap_sec),
        temporal_nms_iou=float(args.temporal_nms_iou),
        min_conf_passaggio=float(args.min_conf_passaggio),
        min_conf_tiro=float(args.min_conf_tiro),
        candidate_mode=str(args.candidate_mode),
        confidence_aggregation=str(args.confidence_aggregation),
        require_action_gt_noaction=bool(args.require_action_gt_noaction),
        noaction_margin=float(args.noaction_margin),
    )


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    predictions_csv: Path,
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    raw_events: pd.DataFrame,
    final_events: pd.DataFrame,
    started_at: str,
) -> None:
    data = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "source_predictions_csv": str(predictions_csv),
        "args": asdict(make_args_snapshot(args)),
        "summary": {
            "num_windows": int(len(predictions)),
            "num_candidate_windows": int(len(candidates)),
            "num_raw_events": int(len(raw_events)),
            "num_final_events": int(len(final_events)),
            "raw_counts_by_label": count_by_label(raw_events),
            "final_counts_by_label": count_by_label(final_events),
        },
        "outputs": {
            "events_raw_csv": "events_raw.csv",
            "events_postprocessed_csv": "events_postprocessed.csv",
            "annotations_json": "annotations.json",
        },
    }
    write_json(path, data)


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-processing temporale delle predizioni raw su finestre long-video. "
            "Produce eventi raw, eventi post-processati e annotations.json."
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
        help="Cartella in cui salvare events_raw.csv, events_postprocessed.csv e annotations.json.",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Numero di finestre per smoothing rolling per ogni scala. Usa 1 per disattivarlo.",
    )
    parser.add_argument(
        "--min-event-duration-sec",
        type=float,
        default=0.4,
        help="Durata minima degli eventi finali in secondi.",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=0.75,
        help="Gap massimo per unire finestre/eventi adiacenti della stessa classe.",
    )
    parser.add_argument(
        "--temporal-nms-iou",
        type=float,
        default=0.50,
        help="Soglia IoU temporale per sopprimere eventi sovrapposti. Usa 0 per disattivare.",
    )
    parser.add_argument(
        "--min-conf-passaggio",
        type=float,
        default=0.55,
        help="Soglia minima per confermare finestre candidate di passaggio.",
    )
    parser.add_argument(
        "--min-conf-tiro",
        type=float,
        default=0.40,
        help="Soglia minima per confermare finestre candidate di tiro.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["best_per_window", "threshold_all"],
        default="best_per_window",
        help=(
            "best_per_window tiene al massimo una classe azione per finestra; "
            "threshold_all crea candidati per tutte le classi sopra soglia."
        ),
    )
    parser.add_argument(
        "--confidence-aggregation",
        choices=["max", "mean"],
        default="max",
        help="Come calcolare la confidenza principale di un evento da più finestre.",
    )
    parser.add_argument(
        "--require-action-gt-noaction",
        action="store_true",
        help=(
            "Richiede che lo score azione superi lo score no-action moltiplicato per "
            "--noaction-margin. Di default è disattivato per non penalizzare troppo i tiri."
        ),
    )
    parser.add_argument(
        "--noaction-margin",
        type=float,
        default=1.0,
        help="Margine usato con --require-action-gt-noaction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive output esistenti.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.smooth_window < 1:
        raise ValueError("--smooth-window deve essere >= 1")
    if args.min_event_duration_sec < 0:
        raise ValueError("--min-event-duration-sec deve essere >= 0")
    if args.merge_gap_sec < 0:
        raise ValueError("--merge-gap-sec deve essere >= 0")
    if args.temporal_nms_iou < 0:
        raise ValueError("--temporal-nms-iou deve essere >= 0")
    if args.min_conf_passaggio < 0 or args.min_conf_tiro < 0:
        raise ValueError("Le soglie di confidenza devono essere >= 0")
    if args.noaction_margin < 0:
        raise ValueError("--noaction-margin deve essere >= 0")


def main() -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    parser = build_argparser()
    args = parser.parse_args()
    validate_args(args)

    args.predictions_csv = as_path(args.predictions_csv)
    args.output_dir = as_path(args.output_dir)
    assert args.predictions_csv is not None
    assert args.output_dir is not None

    check_output_files(
        args.output_dir,
        output_files=[
            "events_raw.csv",
            "events_postprocessed.csv",
            "annotations.json",
            "postprocess_metadata.json",
        ],
        overwrite=args.overwrite,
    )

    print("=== Post-processing long-video ===")
    print(f"predictions_csv:       {args.predictions_csv}")
    print(f"output_dir:            {args.output_dir}")
    print(f"smooth_window:         {args.smooth_window}")
    print(f"min_conf_passaggio:    {args.min_conf_passaggio:.3f}")
    print(f"min_conf_tiro:         {args.min_conf_tiro:.3f}")
    print(f"min_event_duration:    {args.min_event_duration_sec:.3f}s")
    print(f"merge_gap_sec:         {args.merge_gap_sec:.3f}s")
    print(f"temporal_nms_iou:      {args.temporal_nms_iou:.3f}")
    print(f"candidate_mode:        {args.candidate_mode}")

    predictions = load_predictions(args.predictions_csv)
    predictions = add_smoothed_scores(predictions, smooth_window=int(args.smooth_window))

    candidates = build_candidate_rows(
        predictions,
        min_conf_passaggio=float(args.min_conf_passaggio),
        min_conf_tiro=float(args.min_conf_tiro),
        candidate_mode=str(args.candidate_mode),
        require_action_gt_noaction=bool(args.require_action_gt_noaction),
        noaction_margin=float(args.noaction_margin),
    )

    raw_events = candidates_to_raw_events(
        candidates,
        merge_gap_sec=float(args.merge_gap_sec),
        confidence_aggregation=str(args.confidence_aggregation),
    )

    final_events = postprocess_events(
        raw_events,
        min_event_duration_sec=float(args.min_event_duration_sec),
        merge_gap_sec=float(args.merge_gap_sec),
        temporal_nms_iou_value=float(args.temporal_nms_iou),
    )

    raw_csv = args.output_dir / "events_raw.csv"
    final_csv = args.output_dir / "events_postprocessed.csv"
    annotations_json = args.output_dir / "annotations.json"
    metadata_json = args.output_dir / "postprocess_metadata.json"

    write_events_csv(raw_csv, raw_events, RAW_EVENT_COLUMNS)
    write_events_csv(final_csv, final_events, FINAL_EVENT_COLUMNS)
    write_annotations_json(annotations_json, final_events, args, args.predictions_csv)
    write_metadata(
        metadata_json,
        args=args,
        predictions_csv=args.predictions_csv,
        predictions=predictions,
        candidates=candidates,
        raw_events=raw_events,
        final_events=final_events,
        started_at=started_at,
    )

    print("\n=== Output ===")
    print(f"events_raw.csv:           {raw_csv}")
    print(f"events_postprocessed.csv: {final_csv}")
    print(f"annotations.json:         {annotations_json}")
    print(f"postprocess_metadata.json:{metadata_json}")

    print("\n=== Riepilogo ===")
    print(f"finestre totali:          {len(predictions)}")
    print(f"finestre candidate:       {len(candidates)}")
    print(f"eventi raw:               {len(raw_events)}")
    print(f"eventi finali:            {len(final_events)}")
    print("conteggio eventi finali:")
    for label, count in count_by_label(final_events).items():
        print(f"  {label:12s}: {count}")


if __name__ == "__main__":
    main()
