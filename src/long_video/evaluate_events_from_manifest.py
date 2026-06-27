from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.long_video import defaults
from src.long_video.utils import (
    ensure_exists,
    f1_score,
    nanmean,
    normalize_label,
    prepare_output_dir,
    safe_float,
    temporal_iou,
    write_json,
)


# =============================================================================
# Configurazione exp_long_13
# =============================================================================

# Exp_long_13 viene valutato solo sulle 7 classi azione reali.
# no-action / idle / non-gioco sono ignorate nella valutazione event-level.
ACTION_LABELS_7 = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
]

REQUIRED_MANIFEST_COLUMNS = ["clip_id", "video_id", "start_time", "end_time", "label", "split"]
REQUIRED_PRED_COLUMNS = ["start_time", "end_time", "label"]

# Configurazione usata negli esperimenti exp_long_13 riportati nel tracking.
DEFAULT_IOU_THRESHOLD = 0.20
PRED_TIME_MODE = "absolute"


@dataclass(frozen=True)
class EvalConfig:
    manifest: str
    pred_events_csv: str
    video_id: str
    start_sec: float
    end_sec: float
    output_dir: str
    iou_threshold: float
    pred_time_mode: str
    match_labels: list[str]
    created_at: str


# =============================================================================
# Path di default
# =============================================================================


def fallback_manifest_path() -> Path:
    return Path(getattr(defaults, "MANIFEST_PATH", Path("data/datasets/dataset_basket_v1/manifest.csv")))


def fallback_val_pred_events_path() -> Path:
    return Path(getattr(defaults, "VAL_OUTPUT_DIR", Path("outputs/long_video/primaparte_0215_1215_exp46"))) / "events_postprocessed.csv"


# =============================================================================
# Lettura ground truth e predizioni exp_long_13
# =============================================================================


def load_manifest_events(
    manifest_path: Path,
    video_id: str,
    start_sec: float,
    end_sec: float,
    labels: list[str],
) -> pd.DataFrame:
    """Legge dal manifest gli eventi GT delle 7 azioni nel segmento richiesto."""
    ensure_exists(manifest_path, "manifest.csv", must_be_file=True)
    df = pd.read_csv(manifest_path)

    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Il manifest non contiene le colonne richieste: {missing}. "
            f"Colonne trovate: {list(df.columns)}"
        )

    available_video_ids = sorted(df["video_id"].astype(str).unique().tolist())
    if video_id not in available_video_ids:
        raise ValueError(
            f"video_id='{video_id}' non trovato nel manifest. "
            f"video_id disponibili: {available_video_ids}"
        )

    df = df.copy()
    df["label"] = df["label"].apply(normalize_label)
    df["start_time"] = df["start_time"].astype(float)
    df["end_time"] = df["end_time"].astype(float)

    # Tieni solo eventi del video richiesto che intersecano il segmento.
    df = df[
        (df["video_id"].astype(str) == str(video_id))
        & (df["end_time"] > float(start_sec))
        & (df["start_time"] < float(end_sec))
    ].copy()

    # Valutazione exp_long_13: solo le 7 azioni reali.
    df = df[df["label"].isin(labels)].copy()

    if df.empty:
        print(
            f"[WARN] Nessuna annotazione GT sulle 7 azioni trovata per "
            f"video_id={video_id}, segmento {start_sec}->{end_sec}."
        )

    df = df.sort_values(["start_time", "end_time", "clip_id"]).reset_index(drop=True)
    df.insert(0, "gt_id", np.arange(len(df), dtype=int))
    df["raw_start_time"] = df["start_time"]
    df["raw_end_time"] = df["end_time"]

    # Clipping al segmento valutato.
    df["start_time"] = df["start_time"].clip(lower=float(start_sec), upper=float(end_sec))
    df["end_time"] = df["end_time"].clip(lower=float(start_sec), upper=float(end_sec))
    df["duration_sec"] = df["end_time"] - df["start_time"]
    df = df[df["duration_sec"] > 0].reset_index(drop=True)

    keep_cols = [
        "gt_id",
        "clip_id",
        "path",
        "video_id",
        "split",
        "label",
        "start_time",
        "end_time",
        "duration_sec",
        "raw_start_time",
        "raw_end_time",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    return df[keep_cols].copy()


def load_prediction_events_absolute(
    pred_events_path: Path,
    start_sec: float,
    end_sec: float,
    labels: list[str],
) -> pd.DataFrame:
    """Legge gli eventi predetti assumendo timestamp assoluti, come in exp_long_13."""
    ensure_exists(pred_events_path, "events_postprocessed.csv", must_be_file=True)
    df = pd.read_csv(pred_events_path)

    missing = [col for col in REQUIRED_PRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Il CSV predizioni non contiene le colonne richieste: {missing}. "
            f"Colonne trovate: {list(df.columns)}"
        )

    df = df.copy()
    if "event_id" not in df.columns:
        df.insert(0, "event_id", np.arange(len(df), dtype=int))
    if "confidence" not in df.columns:
        df["confidence"] = np.nan

    df["label"] = df["label"].apply(normalize_label)
    df["start_time"] = df["start_time"].astype(float)
    df["end_time"] = df["end_time"].astype(float)
    df["confidence"] = df["confidence"].apply(lambda x: safe_float(x, float("nan")))

    # Valutazione exp_long_13: ignora no-action/background e classi non azione.
    df = df[df["label"].isin(labels)].copy()
    df = df[df["end_time"] > df["start_time"]].copy()

    df["raw_start_time"] = df["start_time"]
    df["raw_end_time"] = df["end_time"]

    # Timestamp assoluti: tieni solo predizioni che intersecano il segmento.
    df = df[(df["end_time"] > float(start_sec)) & (df["start_time"] < float(end_sec))].copy()
    df["start_time"] = df["start_time"].clip(lower=float(start_sec), upper=float(end_sec))
    df["end_time"] = df["end_time"].clip(lower=float(start_sec), upper=float(end_sec))
    df["duration_sec"] = df["end_time"] - df["start_time"]
    df = df[df["duration_sec"] > 0].copy()

    df = df.sort_values(["start_time", "end_time", "event_id"]).reset_index(drop=True)
    df.insert(0, "pred_id", np.arange(len(df), dtype=int))

    keep_cols = [
        "pred_id",
        "event_id",
        "label",
        "start_time",
        "end_time",
        "duration_sec",
        "confidence",
        "raw_start_time",
        "raw_end_time",
    ]
    extra_cols = [col for col in df.columns if col not in keep_cols and col != "pred_id"]
    return df[keep_cols + extra_cols].copy()


# =============================================================================
# Matching event-level e metriche
# =============================================================================


def compute_candidate_matches(gt: pd.DataFrame, pred: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    gt_l = gt[gt["label"] == label]
    pred_l = pred[pred["label"] == label]

    for _, gt_row in gt_l.iterrows():
        for _, pred_row in pred_l.iterrows():
            iou = temporal_iou(
                safe_float(gt_row["start_time"]),
                safe_float(gt_row["end_time"]),
                safe_float(pred_row["start_time"]),
                safe_float(pred_row["end_time"]),
            )
            if iou <= 0:
                continue
            candidates.append(
                {
                    "label": label,
                    "gt_id": int(gt_row["gt_id"]),
                    "pred_id": int(pred_row["pred_id"]),
                    "iou": float(iou),
                    "pred_confidence": safe_float(pred_row.get("confidence"), float("nan")),
                }
            )

    # Greedy: IoU più alta, poi confidence più alta.
    candidates.sort(
        key=lambda item: (
            item["iou"],
            -1.0 if not np.isfinite(item["pred_confidence"]) else item["pred_confidence"],
        ),
        reverse=True,
    )
    return candidates


def greedy_match_events(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    labels: list[str],
    iou_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict[str, Any]] = []

    for label in labels:
        for candidate in compute_candidate_matches(gt, pred, label=label):
            if candidate["iou"] < iou_threshold:
                continue
            gt_id = int(candidate["gt_id"])
            pred_id = int(candidate["pred_id"])
            if gt_id in matched_gt or pred_id in matched_pred:
                continue
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
            matches.append(candidate)

    match_df = pd.DataFrame(matches)
    if match_df.empty:
        match_df = pd.DataFrame(columns=["label", "gt_id", "pred_id", "iou", "pred_confidence"])
    else:
        match_df = match_df.merge(gt.add_prefix("gt_"), left_on="gt_id", right_on="gt_gt_id", how="left")
        match_df = match_df.merge(pred.add_prefix("pred_"), left_on="pred_id", right_on="pred_pred_id", how="left")
        match_df["start_error_sec"] = match_df["pred_start_time"] - match_df["gt_start_time"]
        match_df["end_error_sec"] = match_df["pred_end_time"] - match_df["gt_end_time"]
        match_df["center_error_sec"] = (
            (match_df["pred_start_time"] + match_df["pred_end_time"]) / 2.0
            - (match_df["gt_start_time"] + match_df["gt_end_time"]) / 2.0
        )
        match_df["duration_error_sec"] = match_df["pred_duration_sec"] - match_df["gt_duration_sec"]

    false_positives = pred[~pred["pred_id"].isin(matched_pred)].copy().reset_index(drop=True)
    false_negatives = gt[~gt["gt_id"].isin(matched_gt)].copy().reset_index(drop=True)
    return match_df, false_positives, false_negatives


def compute_per_class_metrics(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    matches: pd.DataFrame,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in labels:
        gt_count = int((gt["label"] == label).sum())
        pred_count = int((pred["label"] == label).sum())
        tp = int((matches["label"] == label).sum()) if not matches.empty else 0
        fp = int((false_positives["label"] == label).sum()) if not false_positives.empty else 0
        fn = int((false_negatives["label"] == label).sum()) if not false_negatives.empty else 0

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

        rows.append(
            {
                "label": label,
                "gt": gt_count,
                "pred": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1_score(precision, recall),
            }
        )
    return pd.DataFrame(rows)


def compute_global_metrics(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    matches: pd.DataFrame,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    per_class: pd.DataFrame,
) -> dict[str, Any]:
    tp = int(len(matches))
    fp = int(len(false_positives))
    fn = int(len(false_negatives))

    precision_micro = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall_micro = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1_micro = f1_score(precision_micro, recall_micro)

    active_classes = per_class[(per_class["gt"] + per_class["pred"]) > 0]
    gt_classes = per_class[per_class["gt"] > 0]

    if matches.empty:
        iou_values = np.asarray([], dtype=np.float64)
        start_abs = np.asarray([], dtype=np.float64)
        end_abs = np.asarray([], dtype=np.float64)
        center_abs = np.asarray([], dtype=np.float64)
        duration_abs = np.asarray([], dtype=np.float64)
    else:
        iou_values = matches["iou"].astype(float).to_numpy()
        start_abs = matches["start_error_sec"].abs().astype(float).to_numpy()
        end_abs = matches["end_error_sec"].abs().astype(float).to_numpy()
        center_abs = matches["center_error_sec"].abs().astype(float).to_numpy()
        duration_abs = matches["duration_error_sec"].abs().astype(float).to_numpy()

    return {
        "num_gt_events": int(len(gt)),
        "num_pred_events": int(len(pred)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "macro_precision_active_classes": nanmean(active_classes["precision"].astype(float).tolist()),
        "macro_recall_active_classes": nanmean(active_classes["recall"].astype(float).tolist()),
        "macro_f1_active_classes": nanmean(active_classes["f1"].astype(float).tolist()),
        "macro_precision_gt_classes": nanmean(gt_classes["precision"].astype(float).tolist()),
        "macro_recall_gt_classes": nanmean(gt_classes["recall"].astype(float).tolist()),
        "macro_f1_gt_classes": nanmean(gt_classes["f1"].astype(float).tolist()),
        "matched_mean_iou": float(iou_values.mean()) if iou_values.size else float("nan"),
        "matched_median_iou": float(np.median(iou_values)) if iou_values.size else float("nan"),
        "matched_min_iou": float(iou_values.min()) if iou_values.size else float("nan"),
        "start_mae_sec": float(start_abs.mean()) if start_abs.size else float("nan"),
        "end_mae_sec": float(end_abs.mean()) if end_abs.size else float("nan"),
        "center_mae_sec": float(center_abs.mean()) if center_abs.size else float("nan"),
        "duration_mae_sec": float(duration_abs.mean()) if duration_abs.size else float("nan"),
        "gt_by_label": {str(k): int(v) for k, v in gt["label"].value_counts().to_dict().items()},
        "pred_by_label": {str(k): int(v) for k, v in pred["label"].value_counts().to_dict().items()},
    }


# =============================================================================
# Report
# =============================================================================


def format_metric(value: Any, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "n/d" if value is None else str(value)
    if not np.isfinite(value_f):
        return "n/d"
    return f"{value_f:.{digits}f}"


def build_text_report(config: EvalConfig, global_metrics: dict[str, Any], per_class: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Event-level evaluation long-video - exp_long_13")
    lines.append("")
    lines.append("## Configurazione")
    lines.append("")
    lines.append(f"- Manifest: `{config.manifest}`")
    lines.append(f"- Predizioni: `{config.pred_events_csv}`")
    lines.append(f"- Video ID: `{config.video_id}`")
    lines.append(f"- Segmento: `{config.start_sec:.3f}s -> {config.end_sec:.3f}s`")
    lines.append(f"- Label valutate: `{', '.join(config.match_labels)}`")
    lines.append(f"- IoU threshold: `{config.iou_threshold:.3f}`")
    lines.append(f"- Pred time mode: `{config.pred_time_mode}`")
    lines.append("")
    lines.append("## Metriche globali")
    lines.append("")
    lines.append("| Metrica | Valore |")
    lines.append("|---|---:|")

    for key in [
        "num_gt_events",
        "num_pred_events",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "macro_precision_active_classes",
        "macro_recall_active_classes",
        "macro_f1_active_classes",
        "macro_precision_gt_classes",
        "macro_recall_gt_classes",
        "macro_f1_gt_classes",
        "matched_mean_iou",
        "matched_median_iou",
        "start_mae_sec",
        "end_mae_sec",
        "center_mae_sec",
        "duration_mae_sec",
    ]:
        lines.append(f"| `{key}` | {format_metric(global_metrics.get(key))} |")

    lines.append("")
    lines.append("## Metriche per classe")
    lines.append("")
    lines.append("| Classe | GT | Pred | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in per_class.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['gt'])} | {int(row['pred'])} | "
            f"{int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} | "
            f"{format_metric(row['precision'])} | {format_metric(row['recall'])} | {format_metric(row['f1'])} |"
        )

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Valutazione exp_long_13
# =============================================================================


def evaluate(
    manifest: Path,
    pred_events_csv: Path,
    video_id: str,
    start_sec: float,
    end_sec: float,
    output_dir: Path,
    iou_threshold: float,
    overwrite: bool,
) -> dict[str, Any]:
    if end_sec <= start_sec:
        raise ValueError(f"end_sec deve essere > start_sec, trovato {start_sec} -> {end_sec}")
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold deve stare in [0, 1], trovato {iou_threshold}")

    prepare_output_dir(output_dir, overwrite=overwrite, clear_if_exists=True)

    gt = load_manifest_events(
        manifest_path=manifest,
        video_id=video_id,
        start_sec=start_sec,
        end_sec=end_sec,
        labels=ACTION_LABELS_7,
    )
    pred = load_prediction_events_absolute(
        pred_events_path=pred_events_csv,
        start_sec=start_sec,
        end_sec=end_sec,
        labels=ACTION_LABELS_7,
    )

    matches, false_positives, false_negatives = greedy_match_events(
        gt=gt,
        pred=pred,
        labels=ACTION_LABELS_7,
        iou_threshold=iou_threshold,
    )

    per_class = compute_per_class_metrics(
        gt=gt,
        pred=pred,
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        labels=ACTION_LABELS_7,
    )
    global_metrics = compute_global_metrics(
        gt=gt,
        pred=pred,
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        per_class=per_class,
    )

    config = EvalConfig(
        manifest=str(manifest),
        pred_events_csv=str(pred_events_csv),
        video_id=str(video_id),
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        output_dir=str(output_dir),
        iou_threshold=float(iou_threshold),
        pred_time_mode=PRED_TIME_MODE,
        match_labels=ACTION_LABELS_7,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    # Output principali più file di diagnostica utili per l'analisi di exp_long_13.
    gt.to_csv(output_dir / "gt_events.csv", index=False)
    pred.to_csv(output_dir / "pred_events_filtered.csv", index=False)
    matches.to_csv(output_dir / "matched_events.csv", index=False)
    false_positives.to_csv(output_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)

    results = {
        "config": asdict(config),
        "global_metrics": global_metrics,
        "per_class_metrics": per_class.to_dict(orient="records"),
    }
    write_json(output_dir / "event_metrics.json", results)

    report = build_text_report(config, global_metrics, per_class)
    (output_dir / "event_metrics.md").write_text(report, encoding="utf-8")

    return results


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Valuta gli eventi exp_long_13 contro manifest.csv usando solo le 7 azioni reali. "
            "I timestamp delle predizioni sono considerati assoluti, come negli esperimenti finali."
        )
    )
    parser.add_argument("--manifest", type=Path, default=fallback_manifest_path())
    parser.add_argument("--pred-events-csv", type=Path, default=fallback_val_pred_events_path())
    parser.add_argument(
        "--video-id",
        type=str,
        required=True,
        help="video_id nel manifest.csv, ad esempio prima_parte oppure psa_converted.",
    )
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"Soglia Temporal IoU per considerare corretta una predizione. Default exp_long_13: {DEFAULT_IOU_THRESHOLD:.2f}.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    results = evaluate(
        manifest=args.manifest,
        pred_events_csv=args.pred_events_csv,
        video_id=args.video_id,
        start_sec=float(args.start_sec),
        end_sec=float(args.end_sec),
        output_dir=args.output_dir,
        iou_threshold=float(args.iou_threshold),
        overwrite=bool(args.overwrite),
    )

    gm = results["global_metrics"]
    print("\n=== Valutazione event-level exp_long_13 completata ===")
    print(f"GT events 7 azioni:      {gm['num_gt_events']}")
    print(f"Pred events 7 azioni:    {gm['num_pred_events']}")
    print(f"TP / FP / FN:            {gm['true_positives']} / {gm['false_positives']} / {gm['false_negatives']}")
    print(f"Precision micro:         {format_metric(gm['precision_micro'])}")
    print(f"Recall micro:            {format_metric(gm['recall_micro'])}")
    print(f"F1 micro:                {format_metric(gm['f1_micro'])}")
    print(f"Macro F1 classi attive:  {format_metric(gm['macro_f1_active_classes'])}")
    print(f"Mean IoU match:          {format_metric(gm['matched_mean_iou'])}")
    print(f"Center MAE:              {format_metric(gm['center_mae_sec'])} s")
    print(f"\nOutput dir: {args.output_dir}")
    print("File principali:")
    print("- event_metrics.md")
    print("- event_metrics.json")
    print("- per_class_metrics.csv")
    print("- matched_events.csv")
    print("- false_positives.csv")
    print("- false_negatives.csv")


if __name__ == "__main__":
    main()
