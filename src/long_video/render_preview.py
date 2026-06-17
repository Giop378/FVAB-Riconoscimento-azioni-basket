from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.long_video import defaults


# =============================================================================
# Costanti
# =============================================================================

REQUIRED_EVENT_COLUMNS = ["label", "start_time", "end_time"]

DEFAULT_LABEL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "passaggio": (64, 200, 255),
    "tiroDaDue0": (80, 120, 255),
    "tiroDaDue1": (80, 220, 120),
    "tiroDaTre0": (180, 110, 255),
    "tiroDaTre1": (80, 255, 80),
    "tiroLibero0": (255, 160, 80),
    "tiroLibero1": (120, 255, 180),
    "no-action": (180, 180, 180),
}

ACTION_LABELS = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
]


# =============================================================================
# Dataclass e utility generali
# =============================================================================


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    num_frames: int
    width: int
    height: int
    duration_sec: float


@dataclass(frozen=True)
class RenderStats:
    input_video: str
    events_csv: str
    output_video: str
    start_sec: float
    end_sec: float
    duration_sec: float
    source_fps: float
    output_fps: float
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    num_events_total: int
    num_events_in_segment: int
    num_frames_written: int
    events_time_mode: str
    created_at: str


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
        raise FileNotErrorCompat(f"{name} dovrebbe essere un file: {path}")
    if must_be_file is False and not path.is_dir():
        raise NotADirectoryError(f"{name} dovrebbe essere una cartella: {path}")


class FileNotErrorCompat(FileNotFoundError):
    """Compatibilità: mantiene messaggi espliciti per file mancanti/non-file."""


def prepare_output_file(output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Il video di output esiste già: {output_path}\n"
                "Usa --overwrite per sostituirlo."
            )
        output_path.unlink()

    metadata_path = output_path.with_suffix(".metadata.json")
    if metadata_path.exists() and overwrite:
        metadata_path.unlink()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"FPS non valido per il video {video_path}: {fps}")
    if num_frames <= 0:
        raise RuntimeError(f"Numero frame non valido per il video {video_path}: {num_frames}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Risoluzione non valida per il video {video_path}: {width}x{height}")

    return VideoInfo(
        fps=fps,
        num_frames=num_frames,
        width=width,
        height=height,
        duration_sec=float(num_frames / fps),
    )


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


def format_time(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


# =============================================================================
# Lettura eventi
# =============================================================================


def load_events(events_csv: Path) -> pd.DataFrame:
    ensure_exists(events_csv, "CSV eventi", must_be_file=True)
    df = pd.read_csv(events_csv)

    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Il file eventi {events_csv} non contiene le colonne richieste: {missing}. "
            f"Colonne trovate: {list(df.columns)}"
        )

    if "event_id" not in df.columns:
        df.insert(0, "event_id", np.arange(len(df), dtype=int))
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    if "duration_sec" not in df.columns:
        df["duration_sec"] = df["end_time"].astype(float) - df["start_time"].astype(float)

    df = df.copy()
    df["event_id"] = df["event_id"].apply(lambda x: safe_int(x, 0))
    df["label"] = df["label"].astype(str)
    df["start_time"] = df["start_time"].astype(float)
    df["end_time"] = df["end_time"].astype(float)
    df["confidence"] = df["confidence"].apply(lambda x: safe_float(x, np.nan))
    df["duration_sec"] = df["duration_sec"].apply(lambda x: safe_float(x, 0.0))

    invalid = df[df["end_time"] <= df["start_time"]]
    if not invalid.empty:
        print(f"[WARN] Scarto {len(invalid)} eventi con end_time <= start_time.")
        df = df[df["end_time"] > df["start_time"]].copy()

    return df.sort_values(["start_time", "end_time", "event_id"]).reset_index(drop=True)


def infer_events_time_mode(events: pd.DataFrame, start_sec: float, end_sec: float) -> str:
    if events.empty:
        return "absolute"

    min_start = float(events["start_time"].min())
    max_end = float(events["end_time"].max())
    segment_duration = float(end_sec - start_sec)

    # Caso più probabile della pipeline: tempi assoluti del video originale.
    absolute_overlap = max(0.0, min(max_end, end_sec) - max(min_start, start_sec))
    relative_overlap = max(0.0, min(max_end, segment_duration) - max(min_start, 0.0))

    if absolute_overlap > 0 and absolute_overlap >= relative_overlap:
        return "absolute"
    if relative_overlap > 0:
        return "relative"

    # Fallback: se gli eventi iniziano chiaramente dopo start_sec sono assoluti,
    # altrimenti assumiamo relativi.
    if min_start >= start_sec * 0.75:
        return "absolute"
    return "relative"


def normalize_event_times(
    events: pd.DataFrame,
    start_sec: float,
    end_sec: float,
    time_mode: str,
) -> tuple[pd.DataFrame, str]:
    if time_mode not in {"auto", "absolute", "relative"}:
        raise ValueError(f"events_time_mode non supportato: {time_mode}")

    resolved_mode = infer_events_time_mode(events, start_sec, end_sec) if time_mode == "auto" else time_mode

    out = events.copy()
    if resolved_mode == "relative":
        out["start_time"] = out["start_time"] + float(start_sec)
        out["end_time"] = out["end_time"] + float(start_sec)

    # Teniamo solo gli eventi che intersecano il segmento renderizzato.
    out = out[(out["end_time"] >= start_sec) & (out["start_time"] <= end_sec)].copy()
    out["start_time_clipped"] = out["start_time"].clip(lower=start_sec, upper=end_sec)
    out["end_time_clipped"] = out["end_time"].clip(lower=start_sec, upper=end_sec)
    out = out[out["end_time_clipped"] > out["start_time_clipped"]].copy()

    return out.sort_values(["start_time", "end_time", "event_id"]).reset_index(drop=True), resolved_mode


# =============================================================================
# Disegno overlay
# =============================================================================


def get_label_color(label: str) -> tuple[int, int, int]:
    if label in DEFAULT_LABEL_COLORS_BGR:
        return DEFAULT_LABEL_COLORS_BGR[label]

    # Colore deterministico per eventuali label non previste.
    seed = abs(hash(label)) % (256**3)
    b = 80 + (seed & 0x7F)
    g = 80 + ((seed >> 8) & 0x7F)
    r = 80 + ((seed >> 16) & 0x7F)
    return int(b), int(g), int(r)


def blend_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        overlay[y1:y2, x1:x2], alpha, frame[y1:y2, x1:x2], 1.0 - alpha, 0
    )


def put_text_with_shadow(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    shadow_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = org
    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        shadow_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def confidence_to_text(confidence: float) -> str:
    if np.isnan(confidence):
        return "conf n/d"
    return f"conf {confidence:.2f}"


def resize_frame_if_needed(frame: np.ndarray, max_width: int | None) -> np.ndarray:
    if max_width is None or max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = float(max_width) / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def active_events_at_time(
    events: pd.DataFrame,
    current_time: float,
    max_active_events: int,
) -> pd.DataFrame:
    if events.empty:
        return events
    active = events[(events["start_time"] <= current_time) & (events["end_time"] >= current_time)].copy()
    if active.empty:
        return active
    active["_conf_sort"] = active["confidence"].fillna(-1.0)
    active = active.sort_values(["_conf_sort", "start_time"], ascending=[False, True])
    return active.head(max_active_events).drop(columns=["_conf_sort"])


def draw_top_overlay(
    frame: np.ndarray,
    current_time: float,
    start_sec: float,
    end_sec: float,
    active: pd.DataFrame,
    max_active_events: int,
) -> None:
    h, w = frame.shape[:2]
    overlay_h = 104 + max(0, min(max_active_events, max(len(active), 1)) - 1) * 34
    overlay_h = min(overlay_h, max(90, h // 3))

    blend_rect(frame, 0, 0, w, overlay_h, (0, 0, 0), alpha=0.58)

    rel_time = current_time - start_sec
    total = end_sec - start_sec
    header = (
        f"Tempo video: {format_time(current_time)}  |  "
        f"segmento: {format_time(rel_time)} / {format_time(total)}"
    )
    put_text_with_shadow(frame, header, (18, 34), 0.72, (255, 255, 255), 2)

    if active.empty:
        put_text_with_shadow(frame, "Azione attiva: nessuna", (18, 76), 0.78, (215, 215, 215), 2)
        return

    y = 76
    for idx, (_, event) in enumerate(active.iterrows()):
        label = str(event["label"])
        color = get_label_color(label)
        confidence = safe_float(event.get("confidence"), np.nan)
        event_id = safe_int(event.get("event_id"), idx)
        start = safe_float(event.get("start_time"), current_time)
        end = safe_float(event.get("end_time"), current_time)
        duration = max(end - start, 1e-6)
        progress = max(0.0, min(1.0, (current_time - start) / duration))

        tag_text = f"#{event_id} {label}  {confidence_to_text(confidence)}  {format_time(start)}-{format_time(end)}"

        # Tag colore + testo.
        cv2.rectangle(frame, (18, y - 23), (30, y - 5), color, thickness=-1)
        put_text_with_shadow(frame, tag_text, (38, y), 0.74 if idx == 0 else 0.62, color, 2)

        # Barra progresso evento.
        bar_x1 = 38
        bar_y1 = y + 9
        bar_x2 = min(w - 18, bar_x1 + 360)
        bar_y2 = bar_y1 + 8
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (70, 70, 70), thickness=-1)
        cv2.rectangle(
            frame,
            (bar_x1, bar_y1),
            (int(bar_x1 + (bar_x2 - bar_x1) * progress), bar_y2),
            color,
            thickness=-1,
        )

        y += 34


def draw_timeline(
    frame: np.ndarray,
    events: pd.DataFrame,
    current_time: float,
    start_sec: float,
    end_sec: float,
    height: int,
) -> None:
    if height <= 0:
        return

    h, w = frame.shape[:2]
    y1 = h - height
    y2 = h
    pad_x = 18
    track_y1 = y1 + 26
    track_y2 = y2 - 20

    blend_rect(frame, 0, y1, w, y2, (0, 0, 0), alpha=0.58)
    put_text_with_shadow(frame, "Timeline eventi", (pad_x, y1 + 20), 0.5, (230, 230, 230), 1)

    duration = max(end_sec - start_sec, 1e-6)
    x_left = pad_x
    x_right = w - pad_x
    track_w = x_right - x_left

    cv2.rectangle(frame, (x_left, track_y1), (x_right, track_y2), (80, 80, 80), thickness=1)

    for _, event in events.iterrows():
        ev_start = max(safe_float(event["start_time"]), start_sec)
        ev_end = min(safe_float(event["end_time"]), end_sec)
        if ev_end <= ev_start:
            continue
        x1 = int(x_left + ((ev_start - start_sec) / duration) * track_w)
        x2 = int(x_left + ((ev_end - start_sec) / duration) * track_w)
        x2 = max(x2, x1 + 2)
        color = get_label_color(str(event["label"]))
        cv2.rectangle(frame, (x1, track_y1 + 2), (x2, track_y2 - 2), color, thickness=-1)

    # Indicatore tempo corrente.
    x_cur = int(x_left + ((current_time - start_sec) / duration) * track_w)
    cv2.line(frame, (x_cur, track_y1 - 8), (x_cur, track_y2 + 8), (255, 255, 255), thickness=2)

    put_text_with_shadow(frame, format_time(start_sec), (x_left, y2 - 4), 0.42, (230, 230, 230), 1)
    end_text = format_time(end_sec)
    text_size, _ = cv2.getTextSize(end_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    put_text_with_shadow(frame, end_text, (x_right - text_size[0], y2 - 4), 0.42, (230, 230, 230), 1)


def draw_legend(frame: np.ndarray, labels: list[str], max_rows: int = 8) -> None:
    if not labels:
        return

    h, w = frame.shape[:2]
    labels = [label for label in ACTION_LABELS if label in set(labels)] + [
        label for label in labels if label not in ACTION_LABELS
    ]
    labels = labels[:max_rows]

    box_w = 230
    row_h = 24
    box_h = 18 + row_h * len(labels)
    x1 = max(0, w - box_w - 16)
    y1 = 118
    x2 = w - 16
    y2 = min(h - 90, y1 + box_h)

    if y2 <= y1 + 20:
        return

    blend_rect(frame, x1, y1, x2, y2, (0, 0, 0), alpha=0.42)
    put_text_with_shadow(frame, "Legenda", (x1 + 10, y1 + 18), 0.48, (230, 230, 230), 1)

    y = y1 + 40
    for label in labels:
        if y + 8 >= y2:
            break
        color = get_label_color(label)
        cv2.rectangle(frame, (x1 + 10, y - 12), (x1 + 24, y + 2), color, thickness=-1)
        put_text_with_shadow(frame, label, (x1 + 32, y), 0.45, (235, 235, 235), 1)
        y += row_h


# =============================================================================
# Rendering
# =============================================================================


def get_fourcc(codec: str) -> int:
    codec = str(codec)
    if len(codec) != 4:
        raise ValueError(f"Il codec deve avere 4 caratteri, trovato: {codec}")
    return cv2.VideoWriter_fourcc(*codec)


def compute_output_size(video_info: VideoInfo, max_width: int | None) -> tuple[int, int]:
    width, height = video_info.width, video_info.height
    if max_width is None or max_width <= 0 or width <= max_width:
        return width, height
    scale = float(max_width) / float(width)
    return int(round(width * scale)), int(round(height * scale))


def render_preview(
    input_video: Path,
    events_csv: Path,
    output_video: Path,
    start_sec: float,
    end_sec: float,
    events_time_mode: str,
    max_width: int | None,
    output_fps: float | None,
    codec: str,
    max_active_events: int,
    draw_timeline_flag: bool,
    timeline_height: int,
    draw_legend_flag: bool,
    overwrite: bool,
) -> RenderStats:
    ensure_exists(input_video, "Video input", must_be_file=True)
    ensure_exists(events_csv, "CSV eventi", must_be_file=True)
    prepare_output_file(output_video, overwrite=overwrite)

    video_info = get_video_info(input_video)
    if start_sec < 0:
        raise ValueError(f"start_sec deve essere >= 0, trovato {start_sec}")
    if end_sec <= start_sec:
        raise ValueError(f"end_sec deve essere > start_sec, trovati {start_sec} -> {end_sec}")
    if end_sec > video_info.duration_sec + (1.0 / video_info.fps):
        raise ValueError(
            f"Il segmento richiesto termina a {end_sec:.2f}s, ma il video dura "
            f"{video_info.duration_sec:.2f}s: {input_video}"
        )

    events_all = load_events(events_csv)
    events_segment, resolved_time_mode = normalize_event_times(
        events_all,
        start_sec=start_sec,
        end_sec=end_sec,
        time_mode=events_time_mode,
    )

    if events_all.empty:
        print("[WARN] Il CSV eventi è vuoto: il preview mostrerà sempre 'nessuna azione'.")
    elif events_segment.empty:
        print(
            "[WARN] Nessun evento interseca il segmento renderizzato. "
            "Controlla se gli eventi sono assoluti o relativi al segmento."
        )

    out_w, out_h = compute_output_size(video_info, max_width=max_width)
    fps_out = float(output_fps) if output_fps is not None and output_fps > 0 else float(video_info.fps)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {input_video}")

    writer = cv2.VideoWriter(str(output_video), get_fourcc(codec), fps_out, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Impossibile creare VideoWriter per {output_video}. "
            f"Prova un codec diverso, ad esempio --codec mp4v."
        )

    start_frame = max(0, int(round(start_sec * video_info.fps)))
    end_frame = min(video_info.num_frames, int(round(end_sec * video_info.fps)))
    total_frames = max(0, end_frame - start_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    labels_in_segment = sorted(set(events_segment["label"].astype(str).tolist())) if not events_segment.empty else []

    frames_written = 0
    with tqdm(total=total_frames, desc="Rendering preview", unit="frame") as pbar:
        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                print(f"[WARN] Lettura interrotta al frame {frame_idx}.")
                break

            current_time = frame_idx / video_info.fps
            frame = resize_frame_if_needed(frame, max_width=max_width)

            active = active_events_at_time(
                events_segment,
                current_time=current_time,
                max_active_events=max_active_events,
            )

            draw_top_overlay(
                frame,
                current_time=current_time,
                start_sec=start_sec,
                end_sec=end_sec,
                active=active,
                max_active_events=max_active_events,
            )

            if draw_legend_flag:
                draw_legend(frame, labels_in_segment)

            if draw_timeline_flag:
                draw_timeline(
                    frame,
                    events=events_segment,
                    current_time=current_time,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    height=timeline_height,
                )

            writer.write(frame)
            frames_written += 1
            pbar.update(1)

    cap.release()
    writer.release()

    stats = RenderStats(
        input_video=str(input_video),
        events_csv=str(events_csv),
        output_video=str(output_video),
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        duration_sec=float(end_sec - start_sec),
        source_fps=float(video_info.fps),
        output_fps=float(fps_out),
        source_width=int(video_info.width),
        source_height=int(video_info.height),
        output_width=int(out_w),
        output_height=int(out_h),
        num_events_total=int(len(events_all)),
        num_events_in_segment=int(len(events_segment)),
        num_frames_written=int(frames_written),
        events_time_mode=resolved_time_mode,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    metadata_path = output_video.with_suffix(".metadata.json")
    write_json(metadata_path, asdict(stats))

    return stats


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Renderizza un video preview annotato a partire da events_postprocessed.csv "
            "della pipeline long-video."
        )
    )

    parser.add_argument(
        "--input-video",
        type=Path,
        default=defaults.VAL_VIDEO_PATH,
        help=f"Video sorgente. Default: {defaults.VAL_VIDEO_PATH}",
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=defaults.VAL_OUTPUT_DIR / "events_postprocessed.csv",
        help="CSV eventi post-processati.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=defaults.VAL_OUTPUT_DIR / "preview_annotated.mp4",
        help="Path del video annotato da creare.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=defaults.VAL_START_SEC,
        help=f"Inizio segmento da renderizzare in secondi. Default: {defaults.VAL_START_SEC}",
    )
    parser.add_argument(
        "--end-sec",
        type=parse_optional_float,
        default=defaults.VAL_END_SEC,
        help=f"Fine segmento da renderizzare in secondi. Default: {defaults.VAL_END_SEC}",
    )

    parser.add_argument(
        "--events-time-mode",
        choices=["auto", "absolute", "relative"],
        default="auto",
        help=(
            "Indica se start_time/end_time degli eventi sono tempi assoluti del video o "
            "tempi relativi al segmento. Con auto viene inferito. Default: auto."
        ),
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Ridimensiona il preview a questa larghezza massima. 0 mantiene la risoluzione originale.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help="FPS del video di output. 0 usa gli FPS del video sorgente.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Codec fourcc OpenCV. Default: mp4v.",
    )
    parser.add_argument(
        "--max-active-events",
        type=int,
        default=2,
        help="Numero massimo di eventi attivi mostrati in alto. Default: 2.",
    )
    parser.add_argument(
        "--timeline-height",
        type=int,
        default=72,
        help="Altezza della timeline in basso. Usa 0 insieme a --no-timeline per disattivarla.",
    )
    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="Disattiva la timeline degli eventi in basso.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disattiva la legenda delle classi a destra.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive il video di output se esiste già.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.end_sec is None:
        video_info = get_video_info(args.input_video)
        end_sec = float(video_info.duration_sec)
    else:
        end_sec = float(args.end_sec)

    max_width = None if args.max_width is None or args.max_width <= 0 else int(args.max_width)
    output_fps = None if args.output_fps is None or args.output_fps <= 0 else float(args.output_fps)

    stats = render_preview(
        input_video=args.input_video,
        events_csv=args.events_csv,
        output_video=args.output_video,
        start_sec=float(args.start_sec),
        end_sec=end_sec,
        events_time_mode=args.events_time_mode,
        max_width=max_width,
        output_fps=output_fps,
        codec=args.codec,
        max_active_events=max(1, int(args.max_active_events)),
        draw_timeline_flag=not args.no_timeline,
        timeline_height=max(0, int(args.timeline_height)),
        draw_legend_flag=not args.no_legend,
        overwrite=bool(args.overwrite),
    )

    print("\n=== Preview renderizzato ===")
    print(f"Video input:        {stats.input_video}")
    print(f"Eventi:             {stats.events_csv}")
    print(f"Video output:       {stats.output_video}")
    print(f"Metadata:           {Path(stats.output_video).with_suffix('.metadata.json')}")
    print(f"Segmento:           {stats.start_sec:.2f}s -> {stats.end_sec:.2f}s")
    print(f"Eventi nel segmento:{stats.num_events_in_segment}/{stats.num_events_total}")
    print(f"Frame scritti:      {stats.num_frames_written}")
    print(f"Risoluzione output: {stats.output_width}x{stats.output_height}")
    print(f"Time mode eventi:   {stats.events_time_mode}")


if __name__ == "__main__":
    main()
