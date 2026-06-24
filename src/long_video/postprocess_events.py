from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from src.long_video import defaults
except Exception:  # pragma: no cover
    defaults = None


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
SHOT_LABELS = [label for label in ACTION_LABELS if label != "passaggio"]

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

CANDIDATE_COLUMNS = [
    "window_id",
    "label",
    "start_time",
    "end_time",
    "center_time",
    "confidence",
    "noaction_score",
    "threshold",
    "scale_index",
    "scale_sec",
]

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


@dataclass(frozen=True)
class ArgsSnapshot:
    predictions_csv: str
    output_dir: str
    min_conf_passaggio: float
    min_conf_tiro: float
    min_event_duration_sec: float
    merge_gap_sec: float
    max_duration_passaggio: float
    max_duration_tiro: float
    max_window_sec_passaggio: float | None
    min_window_sec_tiro: float | None
    max_window_sec_tiro: float | None
    require_action_gt_noaction: bool
    noaction_margin: float
    noaction_margin_passaggio: float | None
    noaction_margin_tiro: float | None
    event_confidence_mode: str
    prefer_shots_over_passaggi: bool
    prefer_shots_min_confidence: float
    forbid_overlaps: bool


def fallback_val_output_dir() -> Path:
    if defaults is not None:
        return Path(getattr(defaults, "VAL_OUTPUT_DIR", Path("outputs/long_video/validation")))
    return Path("outputs/long_video/validation")


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


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(data), indent=2, ensure_ascii=False), encoding="utf-8")


def check_output_files(output_dir: Path, output_files: Iterable[str], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in output_files if (output_dir / name).exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
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


def round_float(value: Any, ndigits: int = 6) -> float:
    return round(safe_float(value), ndigits)


def threshold_for_label(label: str, min_conf_passaggio: float, min_conf_tiro: float) -> float:
    if label == "passaggio":
        return float(min_conf_passaggio)
    return float(min_conf_tiro)


def noaction_margin_for_label(
    label: str,
    default_margin: float,
    margin_passaggio: float | None,
    margin_tiro: float | None,
) -> float:
    if label == "passaggio" and margin_passaggio is not None:
        return float(margin_passaggio)
    if label in SHOT_LABELS and margin_tiro is not None:
        return float(margin_tiro)
    return float(default_margin)


def normalize_label(label: Any) -> str:
    text = str(label).strip()
    aliases = {
        "noaction": "no-action",
        "no_action": "no-action",
        "idle": "no-action",
        "background": "no-action",
        "non-gioco": "no-action",
        "non_gioco": "no-action",
        "nongioco": "no-action",
        "tiro_da_due_0": "tiroDaDue0",
        "tiro_da_due_1": "tiroDaDue1",
        "tiro_da_tre_0": "tiroDaTre0",
        "tiro_da_tre_1": "tiroDaTre1",
        "tiro_libero_0": "tiroLibero0",
        "tiro_libero_1": "tiroLibero1",
    }
    return aliases.get(text, text)


def normalize_score_col_name(name: str) -> str:
    return str(name).strip()


def find_score_column(df: pd.DataFrame, label: str, required: bool = True) -> str | None:
    columns = {normalize_score_col_name(c): c for c in df.columns}
    columns_lower = {normalize_score_col_name(c).lower(): c for c in df.columns}

    for candidate in SCORE_COLUMN_CANDIDATES[label]:
        if candidate in columns:
            return columns[candidate]
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    fallback = f"score_{label.replace('-', '')}"
    if fallback in columns:
        return columns[fallback]
    if fallback.lower() in columns_lower:
        return columns_lower[fallback.lower()]

    if required:
        raise KeyError(
            f"Colonna score per label '{label}' non trovata. "
            f"Candidati: {SCORE_COLUMN_CANDIDATES[label]}"
        )
    return None


def count_by_label(events: pd.DataFrame) -> dict[str, int]:
    if events.empty or "label" not in events.columns:
        return {}
    return {str(k): int(v) for k, v in events["label"].value_counts().sort_index().items()}


def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    ensure_exists(predictions_csv, "Predizioni finestre", must_be_file=True)
    df = pd.read_csv(predictions_csv)
    if df.empty:
        raise ValueError(f"Il file predizioni è vuoto: {predictions_csv}")

    required = ["window_id", "start_time", "end_time"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Colonne obbligatorie mancanti in {predictions_csv}: {missing}")

    out = df.copy()
    out["window_id"] = out["window_id"].astype(str)
    out["start_time"] = pd.to_numeric(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_numeric(out["end_time"], errors="coerce")

    if "center_time" not in out.columns:
        out["center_time"] = (out["start_time"] + out["end_time"]) / 2.0
    else:
        out["center_time"] = pd.to_numeric(out["center_time"], errors="coerce")

    if "scale_sec" not in out.columns:
        out["scale_sec"] = out["end_time"] - out["start_time"]
    else:
        out["scale_sec"] = pd.to_numeric(out["scale_sec"], errors="coerce")

    if "scale_index" not in out.columns:
        unique_scales = sorted(out["scale_sec"].dropna().astype(float).unique().tolist())
        scale_to_idx = {float(value): idx for idx, value in enumerate(unique_scales)}
        out["scale_index"] = out["scale_sec"].map(lambda x: scale_to_idx.get(float(x), -1))
    else:
        out["scale_index"] = pd.to_numeric(out["scale_index"], errors="coerce").fillna(-1).astype(int)

    numeric_cols = ["start_time", "end_time", "center_time", "scale_sec", "scale_index"]
    if out[numeric_cols].isna().any().any():
        bad_cols = [col for col in numeric_cols if out[col].isna().any()]
        raise ValueError(f"Valori numerici non validi nelle colonne: {bad_cols}")

    out = out[out["end_time"] > out["start_time"]].copy()
    if out.empty:
        raise ValueError("Nessuna finestra valida dopo il filtro end_time > start_time")

    for label in ALL_LABELS:
        col = find_score_column(out, label, required=False)
        if col is None:
            if label == "no-action":
                out[f"score__{label}"] = 0.0
            else:
                raise KeyError(f"Score mancante per label azione '{label}'")
        else:
            out[f"score__{label}"] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "pred_label" in out.columns:
        out["pred_label_resolved"] = out["pred_label"].apply(normalize_label)
    elif "label" in out.columns:
        out["pred_label_resolved"] = out["label"].apply(normalize_label)
    else:
        score_matrix = np.stack(
            [out[f"score__{label}"].to_numpy(dtype=np.float32) for label in ALL_LABELS],
            axis=1,
        )
        pred_idx = score_matrix.argmax(axis=1)
        out["pred_label_resolved"] = [ALL_LABELS[int(idx)] for idx in pred_idx]

    if "confidence" in out.columns:
        out["pred_confidence_resolved"] = (
            pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
    else:
        out["pred_confidence_resolved"] = out.apply(
            lambda row: safe_float(row.get(f"score__{normalize_label(row['pred_label_resolved'])}", 0.0)),
            axis=1,
        )

    def resolved_action_score(row: pd.Series) -> float:
        label = normalize_label(row["pred_label_resolved"])
        if label in ALL_LABELS:
            value = row.get(f"score__{label}", np.nan)
            if pd.notna(value):
                return safe_float(value)
        return safe_float(row.get("pred_confidence_resolved", 0.0))

    out["action_score"] = out.apply(resolved_action_score, axis=1)
    out["noaction_score"] = pd.to_numeric(out["score__no-action"], errors="coerce").fillna(0.0).clip(lower=0.0)

    return out.sort_values(["start_time", "end_time", "scale_sec"]).reset_index(drop=True)


def build_candidate_windows(
    predictions: pd.DataFrame,
    min_conf_passaggio: float,
    min_conf_tiro: float,
    require_action_gt_noaction: bool,
    noaction_margin: float,
    noaction_margin_passaggio: float | None,
    noaction_margin_tiro: float | None,
    max_window_sec_passaggio: float | None,
    min_window_sec_tiro: float | None,
    max_window_sec_tiro: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in predictions.iterrows():
        label = normalize_label(row["pred_label_resolved"])
        if label == "no-action" or label not in ACTION_LABELS:
            continue

        scale_sec = safe_float(row["scale_sec"])

        if label == "passaggio" and max_window_sec_passaggio is not None:
            if scale_sec > float(max_window_sec_passaggio):
                continue

        if label in SHOT_LABELS:
            if min_window_sec_tiro is not None and scale_sec < float(min_window_sec_tiro):
                continue
            if max_window_sec_tiro is not None and scale_sec > float(max_window_sec_tiro):
                continue

        score = safe_float(row["action_score"])
        noaction_score = safe_float(row["noaction_score"])
        threshold = threshold_for_label(label, min_conf_passaggio, min_conf_tiro)

        if score < threshold:
            continue

        label_noaction_margin = noaction_margin_for_label(
            label=label,
            default_margin=float(noaction_margin),
            margin_passaggio=noaction_margin_passaggio,
            margin_tiro=noaction_margin_tiro,
        )
        if require_action_gt_noaction and score < noaction_score * label_noaction_margin:
            continue

        rows.append(
            {
                "window_id": str(row["window_id"]),
                "label": label,
                "start_time": safe_float(row["start_time"]),
                "end_time": safe_float(row["end_time"]),
                "center_time": safe_float(row["center_time"]),
                "confidence": score,
                "noaction_score": noaction_score,
                "threshold": float(threshold),
                "scale_index": int(row["scale_index"]),
                "scale_sec": scale_sec,
            }
        )

    if not rows:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS).sort_values(
        ["label", "start_time", "end_time", "scale_sec"]
    ).reset_index(drop=True)


def aggregate_confidence(scores: np.ndarray, mode: str) -> float:
    if scores.size == 0:
        return 0.0
    if mode == "max":
        return float(np.max(scores))
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "median":
        return float(np.median(scores))
    raise ValueError(f"event_confidence_mode non supportato: {mode}")


def aggregate_rows_to_event(
    rows: list[dict[str, Any]],
    event_id: int,
    event_confidence_mode: str,
    forced_start: float | None = None,
    forced_end: float | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("aggregate_rows_to_event chiamata con rows vuoto")

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
    scales = [safe_float(r["scale_sec"]) for r in ordered]
    scale_indices = {int(r["scale_index"]) for r in ordered}
    unique_scales = sorted({f"{scale:.3f}".rstrip("0").rstrip(".") for scale in scales})

    scale_index = int(ordered[0]["scale_index"]) if len(scale_indices) == 1 else -1
    scale_sec = float(scales[0]) if len(unique_scales) == 1 else -1.0

    return {
        "event_id": int(event_id),
        "label": str(ordered[0]["label"]),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration_sec": float(end_time - start_time),
        "center_time": float((start_time + end_time) / 2.0),
        "confidence": aggregate_confidence(scores, mode=event_confidence_mode),
        "confidence_mean": float(np.mean(scores)) if scores.size else 0.0,
        "confidence_max": float(np.max(scores)) if scores.size else 0.0,
        "confidence_median": float(np.median(scores)) if scores.size else 0.0,
        "noaction_mean": float(np.mean(noaction_scores)) if noaction_scores.size else 0.0,
        "threshold": float(np.min(thresholds)) if thresholds.size else 0.0,
        "num_windows": int(len(ordered)),
        "scale_index": scale_index,
        "scale_sec": scale_sec,
        "first_window_id": str(ordered[0]["window_id"]),
        "last_window_id": str(ordered[-1]["window_id"]),
    }


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
    num_chunks = int(math.ceil(duration / float(max_duration)))

    for chunk_idx in range(num_chunks):
        chunk_start = group_start + chunk_idx * float(max_duration)
        chunk_end = min(group_end, chunk_start + float(max_duration))
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


def append_group_events(
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    event_id: int,
    min_event_duration_sec: float,
    max_duration_passaggio: float,
    max_duration_tiro: float,
    event_confidence_mode: str,
) -> int:
    if not rows:
        return event_id

    label = str(rows[0]["label"])
    max_duration = 0.0
    if label == "passaggio":
        max_duration = float(max_duration_passaggio)
    elif label in SHOT_LABELS:
        max_duration = float(max_duration_tiro)

    for chunk_rows, forced_start, forced_end in split_group_by_max_duration(rows, max_duration=max_duration):
        event = aggregate_rows_to_event(
            chunk_rows,
            event_id=event_id,
            event_confidence_mode=event_confidence_mode,
            forced_start=forced_start,
            forced_end=forced_end,
        )
        if safe_float(event["duration_sec"]) < float(min_event_duration_sec):
            continue
        events.append(event)
        event_id += 1

    return event_id


def group_candidates_to_events(
    candidates: pd.DataFrame,
    merge_gap_sec: float,
    min_event_duration_sec: float,
    max_duration_passaggio: float,
    max_duration_tiro: float,
    event_confidence_mode: str,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=RAW_EVENT_COLUMNS)

    events: list[dict[str, Any]] = []
    event_id = 0

    for _, group in candidates.sort_values(["label", "start_time", "end_time"]).groupby("label", sort=True):
        current: list[dict[str, Any]] = []
        current_end: float | None = None

        for row in group.to_dict("records"):
            start = safe_float(row["start_time"])
            end = safe_float(row["end_time"])
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
                event_id = append_group_events(
                    events=events,
                    rows=current,
                    event_id=event_id,
                    min_event_duration_sec=min_event_duration_sec,
                    max_duration_passaggio=max_duration_passaggio,
                    max_duration_tiro=max_duration_tiro,
                    event_confidence_mode=event_confidence_mode,
                )
                current = [row]
                current_end = end

        if current:
            event_id = append_group_events(
                events=events,
                rows=current,
                event_id=event_id,
                min_event_duration_sec=min_event_duration_sec,
                max_duration_passaggio=max_duration_passaggio,
                max_duration_tiro=max_duration_tiro,
                event_confidence_mode=event_confidence_mode,
            )

    if not events:
        return pd.DataFrame(columns=RAW_EVENT_COLUMNS)

    out = pd.DataFrame(events, columns=RAW_EVENT_COLUMNS).sort_values(
        ["start_time", "end_time", "label"]
    ).reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=int)
    return out


def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return float(max(0.0, min(a_end, b_end) - max(a_start, b_start)))


def merge_final_record(record: dict[str, Any], new_event_id: int) -> dict[str, Any]:
    out = dict(record)
    out["event_id"] = int(new_event_id)
    out["duration_sec"] = safe_float(out["end_time"]) - safe_float(out["start_time"])
    out["center_time"] = (safe_float(out["start_time"]) + safe_float(out["end_time"])) / 2.0
    out["source_event_ids"] = str(record.get("event_id", new_event_id))
    scale = safe_float(record.get("scale_sec", -1.0))
    out["scales_used"] = "mixed" if scale < 0 else f"{scale:.3f}".rstrip("0").rstrip(".")
    return out


def suppress_overlaps(
    events: pd.DataFrame,
    allow_overlaps: bool,
    prefer_shots_over_passaggi: bool = False,
    prefer_shots_min_confidence: float = 0.0,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=FINAL_EVENT_COLUMNS)

    records = events.to_dict("records")
    if allow_overlaps:
        final_records = [merge_final_record(record, i) for i, record in enumerate(records)]
        return pd.DataFrame(final_records, columns=FINAL_EVENT_COLUMNS).sort_values(
            ["start_time", "end_time", "label"]
        ).reset_index(drop=True)

    def priority(record: dict[str, Any]) -> int:
        label = str(record.get("label", ""))
        if not prefer_shots_over_passaggi:
            return 0

        confidence = safe_float(record.get("confidence", 0.0))
        if label in SHOT_LABELS and confidence >= float(prefer_shots_min_confidence):
            return 0
        if label == "passaggio":
            return 1
        if label in SHOT_LABELS:
            return 2
        return 3

    order = sorted(
        range(len(records)),
        key=lambda i: (
            priority(records[i]),
            -safe_float(records[i].get("confidence", 0.0)),
            -safe_float(records[i].get("confidence_max", records[i].get("confidence", 0.0))),
            -safe_float(records[i].get("duration_sec", 0.0)),
            safe_float(records[i].get("start_time", 0.0)),
            safe_float(records[i].get("end_time", 0.0)),
            str(records[i].get("label", "")),
        ),
    )

    keep: list[int] = []
    for idx in order:
        candidate = records[idx]
        c_start = safe_float(candidate["start_time"])
        c_end = safe_float(candidate["end_time"])
        if any(
            overlap_duration(
                c_start,
                c_end,
                safe_float(records[kept_idx]["start_time"]),
                safe_float(records[kept_idx]["end_time"]),
            )
            > 1e-9
            for kept_idx in keep
        ):
            continue
        keep.append(idx)

    final_records = [merge_final_record(records[idx], new_event_id=i) for i, idx in enumerate(keep)]
    out = pd.DataFrame(final_records, columns=FINAL_EVENT_COLUMNS).sort_values(
        ["start_time", "end_time", "label"]
    ).reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=int)
    return out


def round_event_columns(events: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
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
    for col in columns:
        if col not in out.columns:
            out[col] = []
    return out[columns]


def write_events_csv(path: Path, events: pd.DataFrame, columns: list[str]) -> None:
    round_event_columns(events, columns=columns).to_csv(path, index=False)


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


def make_args_snapshot(args: argparse.Namespace) -> ArgsSnapshot:
    return ArgsSnapshot(
        predictions_csv=str(args.predictions_csv),
        output_dir=str(args.output_dir),
        min_conf_passaggio=float(args.min_conf_passaggio),
        min_conf_tiro=float(args.min_conf_tiro),
        min_event_duration_sec=float(args.min_event_duration_sec),
        merge_gap_sec=float(args.merge_gap_sec),
        max_duration_passaggio=float(args.max_duration_passaggio),
        max_duration_tiro=float(args.max_duration_tiro),
        max_window_sec_passaggio=None
        if args.max_window_sec_passaggio is None
        else float(args.max_window_sec_passaggio),
        min_window_sec_tiro=None
        if args.min_window_sec_tiro is None
        else float(args.min_window_sec_tiro),
        max_window_sec_tiro=None
        if args.max_window_sec_tiro is None
        else float(args.max_window_sec_tiro),
        require_action_gt_noaction=bool(args.require_action_gt_noaction),
        noaction_margin=float(args.noaction_margin),
        noaction_margin_passaggio=None
        if args.noaction_margin_passaggio is None
        else float(args.noaction_margin_passaggio),
        noaction_margin_tiro=None
        if args.noaction_margin_tiro is None
        else float(args.noaction_margin_tiro),
        event_confidence_mode=str(args.event_confidence_mode),
        prefer_shots_over_passaggi=bool(args.prefer_shots_over_passaggi),
        prefer_shots_min_confidence=float(args.prefer_shots_min_confidence),
        forbid_overlaps=not bool(args.allow_overlaps),
    )


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    raw_events: pd.DataFrame,
    final_events: pd.DataFrame,
    started_at: str,
) -> None:
    data = {
        "kind": "postprocess_events",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "source_predictions_csv": str(args.predictions_csv),
        "args": asdict(make_args_snapshot(args)),
        "summary": {
            "num_windows": int(len(predictions)),
            "num_candidate_windows": int(len(candidates)),
            "num_raw_events": int(len(raw_events)),
            "num_final_events": int(len(final_events)),
            "candidate_counts_by_label": count_by_label(candidates),
            "raw_counts_by_label": count_by_label(raw_events),
            "final_counts_by_label": count_by_label(final_events),
        },
        "logic": [
            "read window_predictions_raw.csv",
            "use pred_label/confidence for each window",
            "discard no-action",
            "apply passaggio/tiro thresholds",
            "filter passaggio/tiro candidates by window scale when configured",
            "optionally require action score > no-action score * margin, with optional separate margins for passaggio and shots",
            "group windows by label when temporal gap <= merge_gap_sec",
            "split events longer than max_duration_passaggio/max_duration_tiro",
            "discard events shorter than min_event_duration_sec",
            "optionally suppress overlaps greedily by event confidence",
            "when configured, overlapping shots are prioritized over passaggio, optionally only above a confidence threshold",
        ],
        "outputs": {
            "candidate_windows_csv": "candidate_windows.csv",
            "events_raw_csv": "events_raw.csv",
            "events_postprocessed_csv": "events_postprocessed.csv",
            "annotations_json": "annotations.json",
            "postprocess_metadata_json": "postprocess_metadata.json",
        },
    }
    write_json(path, data)


def build_argparser() -> argparse.ArgumentParser:
    val_output = fallback_val_output_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Post-processing delle predizioni raw long-video. "
            "Usa pred_label/confidence, soglie, filtri per scala, merge per gap, durata massima e overlap suppression."
        )
    )

    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=val_output / "window_predictions_raw.csv",
        help="CSV prodotto da infer_exp46_from_store.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=val_output,
        help="Cartella in cui salvare eventi e annotations.json.",
    )
    parser.add_argument("--min-conf-passaggio", type=float, default=0.65)
    parser.add_argument("--min-conf-tiro", type=float, default=0.40)
    parser.add_argument("--min-event-duration-sec", type=float, default=0.25)
    parser.add_argument("--merge-gap-sec", type=float, default=0.20)
    parser.add_argument(
        "--max-duration-passaggio",
        type=float,
        default=1.20,
        help="Durata massima di un evento passaggio. Usa 0 o valore negativo per disattivare.",
    )
    parser.add_argument(
        "--max-duration-tiro",
        type=float,
        default=2.00,
        help="Durata massima di un evento tiro. Usa 0 o valore negativo per disattivare.",
    )
    parser.add_argument(
        "--max-window-sec-passaggio",
        type=float,
        default=None,
        help="Scarta candidati passaggio provenienti da finestre più lunghe di questo valore.",
    )
    parser.add_argument(
        "--min-window-sec-tiro",
        type=float,
        default=None,
        help="Scarta candidati tiro provenienti da finestre più corte di questo valore.",
    )
    parser.add_argument(
        "--max-window-sec-tiro",
        type=float,
        default=None,
        help="Scarta candidati tiro provenienti da finestre più lunghe di questo valore.",
    )
    parser.add_argument(
        "--require-action-gt-noaction",
        action="store_true",
        help="Richiede score azione >= score no-action * noaction-margin.",
    )
    parser.add_argument("--noaction-margin", type=float, default=1.05)
    parser.add_argument(
        "--noaction-margin-passaggio",
        type=float,
        default=None,
        help="Margine action > no-action specifico per passaggio. Se omesso, usa --noaction-margin.",
    )
    parser.add_argument(
        "--noaction-margin-tiro",
        type=float,
        default=None,
        help="Margine action > no-action specifico per i tiri. Se omesso, usa --noaction-margin.",
    )
    parser.add_argument(
        "--event-confidence-mode",
        choices=["max", "mean", "median"],
        default="max",
        help="Come calcolare la confidence finale dell'evento dalle finestre raggruppate.",
    )
    parser.add_argument(
        "--prefer-shots-over-passaggi",
        action="store_true",
        help=(
            "Durante la soppressione degli overlap, dà priorità agli eventi tiro rispetto ai passaggi. "
            "Utile quando i tiri vengono eliminati perché sovrapposti a passaggi più confidenti."
        ),
    )
    parser.add_argument(
        "--prefer-shots-min-confidence",
        type=float,
        default=0.0,
        help=(
            "Quando --prefer-shots-over-passaggi è attivo, dà priorità ai tiri solo se "
            "la confidence del tiro è almeno questo valore. Con 0.0 la priorità è sempre attiva."
        ),
    )
    parser.add_argument(
        "--allow-overlaps",
        action="store_true",
        help="Permette eventi sovrapposti. Di default viene tenuto solo l'evento più confidente.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.min_conf_passaggio < 0 or args.min_conf_tiro < 0:
        raise ValueError("Le soglie di confidenza devono essere >= 0")
    if args.min_event_duration_sec < 0:
        raise ValueError("--min-event-duration-sec deve essere >= 0")
    if args.merge_gap_sec < 0:
        raise ValueError("--merge-gap-sec deve essere >= 0")
    if args.noaction_margin < 0:
        raise ValueError("--noaction-margin deve essere >= 0")
    if args.noaction_margin_passaggio is not None and args.noaction_margin_passaggio < 0:
        raise ValueError("--noaction-margin-passaggio deve essere >= 0")
    if args.noaction_margin_tiro is not None and args.noaction_margin_tiro < 0:
        raise ValueError("--noaction-margin-tiro deve essere >= 0")
    if args.prefer_shots_min_confidence < 0:
        raise ValueError("--prefer-shots-min-confidence deve essere >= 0")
    if args.max_window_sec_passaggio is not None and args.max_window_sec_passaggio <= 0:
        raise ValueError("--max-window-sec-passaggio deve essere > 0, oppure omesso")
    if args.min_window_sec_tiro is not None and args.min_window_sec_tiro <= 0:
        raise ValueError("--min-window-sec-tiro deve essere > 0, oppure omesso")
    if args.max_window_sec_tiro is not None and args.max_window_sec_tiro <= 0:
        raise ValueError("--max-window-sec-tiro deve essere > 0, oppure omesso")
    if (
        args.min_window_sec_tiro is not None
        and args.max_window_sec_tiro is not None
        and args.min_window_sec_tiro > args.max_window_sec_tiro
    ):
        raise ValueError("--min-window-sec-tiro non può essere maggiore di --max-window-sec-tiro")


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
            "candidate_windows.csv",
            "events_raw.csv",
            "events_postprocessed.csv",
            "annotations.json",
            "postprocess_metadata.json",
        ],
        overwrite=args.overwrite,
    )

    print("=== Post-processing long-video ===")
    print(f"predictions_csv:      {args.predictions_csv}")
    print(f"output_dir:           {args.output_dir}")
    print(f"min_conf_passaggio:   {args.min_conf_passaggio:.3f}")
    print(f"min_conf_tiro:        {args.min_conf_tiro:.3f}")
    print(f"merge_gap_sec:        {args.merge_gap_sec:.3f}")
    print(f"min_event_duration:   {args.min_event_duration_sec:.3f}")
    print(f"max_duration_pass:    {args.max_duration_passaggio:.3f}")
    print(f"max_duration_tiro:    {args.max_duration_tiro:.3f}")
    print(f"max_window_passaggio: {args.max_window_sec_passaggio}")
    print(f"min_window_tiro:      {args.min_window_sec_tiro}")
    print(f"max_window_tiro:      {args.max_window_sec_tiro}")
    print(f"action_gt_noaction:   {bool(args.require_action_gt_noaction)}")
    print(f"noaction_margin:      {args.noaction_margin:.3f}")
    print(f"noaction_margin_pass: {args.noaction_margin_passaggio}")
    print(f"noaction_margin_tiro: {args.noaction_margin_tiro}")
    print(f"confidence_mode:      {args.event_confidence_mode}")
    print(f"prefer_shots_pass:    {bool(args.prefer_shots_over_passaggi)}")
    print(f"prefer_shots_minconf: {args.prefer_shots_min_confidence:.3f}")
    print(f"forbid_overlaps:      {not bool(args.allow_overlaps)}")

    predictions = load_predictions(args.predictions_csv)
    candidates = build_candidate_windows(
        predictions=predictions,
        min_conf_passaggio=float(args.min_conf_passaggio),
        min_conf_tiro=float(args.min_conf_tiro),
        require_action_gt_noaction=bool(args.require_action_gt_noaction),
        noaction_margin=float(args.noaction_margin),
        noaction_margin_passaggio=args.noaction_margin_passaggio,
        noaction_margin_tiro=args.noaction_margin_tiro,
        max_window_sec_passaggio=args.max_window_sec_passaggio,
        min_window_sec_tiro=args.min_window_sec_tiro,
        max_window_sec_tiro=args.max_window_sec_tiro,
    )
    raw_events = group_candidates_to_events(
        candidates=candidates,
        merge_gap_sec=float(args.merge_gap_sec),
        min_event_duration_sec=float(args.min_event_duration_sec),
        max_duration_passaggio=float(args.max_duration_passaggio),
        max_duration_tiro=float(args.max_duration_tiro),
        event_confidence_mode=str(args.event_confidence_mode),
    )
    final_events = suppress_overlaps(
        raw_events,
        allow_overlaps=bool(args.allow_overlaps),
        prefer_shots_over_passaggi=bool(args.prefer_shots_over_passaggi),
        prefer_shots_min_confidence=float(args.prefer_shots_min_confidence),
    )

    candidates.to_csv(args.output_dir / "candidate_windows.csv", index=False)
    write_events_csv(args.output_dir / "events_raw.csv", raw_events, columns=RAW_EVENT_COLUMNS)
    write_events_csv(args.output_dir / "events_postprocessed.csv", final_events, columns=FINAL_EVENT_COLUMNS)
    write_json(args.output_dir / "annotations.json", {"annotations": events_to_annotations(final_events)})
    write_metadata(
        args.output_dir / "postprocess_metadata.json",
        args=args,
        predictions=predictions,
        candidates=candidates,
        raw_events=raw_events,
        final_events=final_events,
        started_at=started_at,
    )

    print("\n=== Summary ===")
    print(f"finestre totali:      {len(predictions)}")
    print(f"candidate windows:    {len(candidates)} | {count_by_label(candidates)}")
    print(f"eventi raw:           {len(raw_events)} | {count_by_label(raw_events)}")
    print(f"eventi finali:        {len(final_events)} | {count_by_label(final_events)}")
    print(f"\n[OK] events_postprocessed.csv salvato in: {args.output_dir / 'events_postprocessed.csv'}")


if __name__ == "__main__":
    main()