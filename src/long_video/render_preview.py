# =============================================================================
# Questo script genera una preview video annotata a partire:
# - dal video originale;
# - dal CSV degli eventi post-processati prodotto dalla pipeline long-video;
# - da un intervallo temporale da renderizzare.
#
# Gli eventi sono interpretati con timestamp assoluti del video, coerentemente
# con exp_long_13. Per ogni frame del segmento richiesto lo script controlla
# se esiste un evento attivo in quell'istante e disegna un overlay informativo:
# tempo corrente, azione riconosciuta, confidence e barra di avanzamento.
#
# Il file non modifica le predizioni e non ricalcola eventi: serve solo per
# visualizzare qualitativamente il risultato finale della pipeline, producendo
# un video .mp4 più leggibile da usare per controllo, demo o presentazione.
#
# Flusso principale:
# 1. legge e valida il CSV eventi;
# 2. mantiene solo le 7 classi azione reali;
# 3. filtra gli eventi che intersecano il segmento richiesto;
# 4. apre il video sorgente e crea il VideoWriter di output;
# 5. scorre i frame del segmento, disegna l'overlay e salva la preview.
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.long_video import defaults
from src.long_video.utils import (
    ensure_exists,
    format_time,
    get_video_info,
    normalize_label,
    parse_optional_float,
    prepare_output_file,
    safe_float,
    safe_int,
)


# =============================================================================
# Configurazione exp_long_13
# =============================================================================

# Parametri globali della preview: indicano che gli eventi sono in tempo
# assoluto rispetto al video originale, definiscono il codec di output e
# fissano le classi/colonne minime attese nel CSV degli eventi.

# La pipeline finale exp_long_13 salva eventi con tempi assoluti del video.
EVENTS_TIME_MODE = "absolute"
DEFAULT_CODEC = "mp4v"

REQUIRED_EVENT_COLUMNS = ["label", "start_time", "end_time"]

# Le classi mostrate nella preview sono solo le 7 azioni reali valutate nel
# progetto; eventuali classi di background/no-action vengono ignorate.

ACTION_LABELS = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
]

# Colori OpenCV in formato BGR associati alle etichette: servono solo per
# rendere più leggibile l'overlay e distinguere rapidamente i tipi di azione.
LABEL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "passaggio": (64, 200, 255),
    "tiroDaDue0": (80, 120, 255),
    "tiroDaDue1": (80, 220, 120),
    "tiroDaTre0": (180, 110, 255),
    "tiroDaTre1": (80, 255, 80),
    "tiroLibero0": (255, 160, 80),
    "tiroLibero1": (120, 255, 180),
}


# =============================================================================
# Lettura eventi
# =============================================================================


# Carica il CSV degli eventi, controlla che abbia le colonne minime richieste,
# normalizza label e tipi numerici, rimuove eventi non validi e mantiene solo
# le azioni reali da visualizzare.
def load_events(events_csv: Path) -> pd.DataFrame:
    ensure_exists(events_csv, "CSV eventi", must_be_file=True)
    df = pd.read_csv(events_csv)

    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Il file eventi {events_csv} non contiene le colonne richieste: {missing}. "
            f"Colonne trovate: {list(df.columns)}"
        )

    out = df.copy()
    if "event_id" not in out.columns:
        out.insert(0, "event_id", np.arange(len(out), dtype=int))
    if "confidence" not in out.columns:
        out["confidence"] = np.nan

    out["event_id"] = out["event_id"].apply(lambda x: safe_int(x, 0))
    out["label"] = out["label"].apply(normalize_label)
    out["start_time"] = pd.to_numeric(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_numeric(out["end_time"], errors="coerce")
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")

    out = out[out["label"].isin(ACTION_LABELS)].copy()
    out = out.dropna(subset=["start_time", "end_time"])
    out = out[out["end_time"] > out["start_time"]].copy()

    if out.empty:
        return pd.DataFrame(columns=["event_id", "label", "start_time", "end_time", "confidence"])

    return out.sort_values(["start_time", "end_time", "event_id"]).reset_index(drop=True)


# Seleziona gli eventi che intersecano il segmento renderizzato e crea anche
# i tempi clippati al segmento, utili per evitare eventi fuori dai limiti della preview.
def filter_events_to_segment(events: pd.DataFrame, start_sec: float, end_sec: float) -> pd.DataFrame:
    """Filtra eventi exp_long_13 assumendo tempi assoluti del video."""
    if events.empty:
        return events.copy()

    out = events[(events["end_time"] >= start_sec) & (events["start_time"] <= end_sec)].copy()
    out["start_time_clipped"] = out["start_time"].clip(lower=start_sec, upper=end_sec)
    out["end_time_clipped"] = out["end_time"].clip(lower=start_sec, upper=end_sec)
    out = out[out["end_time_clipped"] > out["start_time_clipped"]].copy()
    return out.sort_values(["start_time", "end_time", "event_id"]).reset_index(drop=True)


# =============================================================================
# Disegno overlay
# =============================================================================


# Funzioni di supporto grafico: recuperano il colore dell'etichetta, disegnano
# rettangoli semitrasparenti e scritte con ombra per mantenere leggibile
# l'informazione anche su frame con sfondo variabile.
def get_label_color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS_BGR.get(str(label), (220, 220, 220))


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
) -> None:
    x, y = org
    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
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


# Converte la confidence in testo compatto; se il valore non è disponibile
# mostra una stringa neutra invece di interrompere il rendering.
def confidence_to_text(confidence: Any) -> str:
    value = safe_float(confidence, default=float("nan"))
    if not np.isfinite(value):
        return "conf n/d"
    return f"conf {value:.2f}"


# Riduce opzionalmente la risoluzione del frame mantenendo le proporzioni:
# utile per ottenere preview più leggere senza cambiare la logica degli eventi.
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


# Trova l'evento attivo nel frame corrente. Se più eventi si sovrappongono,
# viene mostrato quello con confidence più alta.
def active_event_at_time(events: pd.DataFrame, current_time: float) -> pd.Series | None:
    if events.empty:
        return None
    active = events[(events["start_time"] <= current_time) & (events["end_time"] >= current_time)].copy()
    if active.empty:
        return None
    active["_confidence_sort"] = active["confidence"].fillna(-1.0)
    active = active.sort_values(["_confidence_sort", "start_time"], ascending=[False, True])
    return active.iloc[0]


# Disegna l'overlay principale sul frame: tempo video, tempo relativo al
# segmento, azione attiva, confidence e barra di avanzamento dell'evento.
def draw_overlay(
    frame: np.ndarray,
    current_time: float,
    start_sec: float,
    end_sec: float,
    active_event: pd.Series | None,
) -> None:
    h, w = frame.shape[:2]
    top_h = min(110, max(90, h // 4))
    blend_rect(frame, 0, 0, w, top_h, (0, 0, 0), alpha=0.58)

    segment_time = current_time - start_sec
    segment_duration = end_sec - start_sec
    header = (
        f"Tempo video: {format_time(current_time)}  |  "
        f"segmento: {format_time(segment_time)} / {format_time(segment_duration)}"
    )
    put_text_with_shadow(frame, header, (18, 34), 0.72, (255, 255, 255), 2)

    if active_event is None:
        put_text_with_shadow(frame, "Azione attiva: nessuna", (18, 78), 0.78, (215, 215, 215), 2)
    else:
        label = str(active_event["label"])
        color = get_label_color(label)
        event_id = safe_int(active_event.get("event_id"), 0)
        ev_start = safe_float(active_event.get("start_time"), current_time)
        ev_end = safe_float(active_event.get("end_time"), current_time)
        confidence = active_event.get("confidence", np.nan)

        text = (
            f"Azione attiva: #{event_id} {label}  "
            f"{confidence_to_text(confidence)}  "
            f"{format_time(ev_start)}-{format_time(ev_end)}"
        )
        cv2.rectangle(frame, (18, 57), (32, 80), color, thickness=-1)
        put_text_with_shadow(frame, text, (42, 78), 0.76, color, 2)

        duration = max(ev_end - ev_start, 1e-6)
        progress = max(0.0, min(1.0, (current_time - ev_start) / duration))
        bar_x1, bar_y1 = 42, 90
        bar_x2, bar_y2 = min(w - 18, bar_x1 + 420), 99
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (70, 70, 70), thickness=-1)
        cv2.rectangle(
            frame,
            (bar_x1, bar_y1),
            (int(bar_x1 + (bar_x2 - bar_x1) * progress), bar_y2),
            color,
            thickness=-1,
        )

    draw_segment_progress_bar(frame, current_time=current_time, start_sec=start_sec, end_sec=end_sec)


# Aggiunge in basso una barra di avanzamento dell'intero segmento renderizzato.
def draw_segment_progress_bar(frame: np.ndarray, current_time: float, start_sec: float, end_sec: float) -> None:
    h, w = frame.shape[:2]
    duration = max(end_sec - start_sec, 1e-6)
    progress = max(0.0, min(1.0, (current_time - start_sec) / duration))

    x1, x2 = 18, w - 18
    y1, y2 = h - 22, h - 12
    blend_rect(frame, 0, h - 42, w, h, (0, 0, 0), alpha=0.40)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), thickness=-1)
    cv2.rectangle(frame, (x1, y1), (int(x1 + (x2 - x1) * progress), y2), (255, 255, 255), thickness=-1)


# =============================================================================
# Rendering
# =============================================================================


# Funzioni di supporto al rendering: validano il codec e calcolano la dimensione
# finale del video in base all'eventuale limite di larghezza.
def get_fourcc(codec: str) -> int:
    if len(str(codec)) != 4:
        raise ValueError(f"Il codec deve avere 4 caratteri, trovato: {codec}")
    return cv2.VideoWriter_fourcc(*str(codec))


def compute_output_size(video_width: int, video_height: int, max_width: int | None) -> tuple[int, int]:
    if max_width is None or max_width <= 0 or video_width <= max_width:
        return int(video_width), int(video_height)
    scale = float(max_width) / float(video_width)
    return int(round(video_width * scale)), int(round(video_height * scale))


# Funzione principale di rendering: apre video ed eventi, valida il segmento,
# inizializza il writer e scorre i frame disegnando l'overlay prima di salvarli.
def render_preview(
    input_video: Path,
    events_csv: Path,
    output_video: Path,
    start_sec: float,
    end_sec: float,
    max_width: int | None,
    overwrite: bool,
) -> dict[str, Any]:
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

    # Lettura eventi e filtro sul segmento richiesto: da qui in poi il rendering
    # lavora solo sugli eventi temporalmente rilevanti per la preview.
    events_all = load_events(events_csv)
    events_segment = filter_events_to_segment(events_all, start_sec=start_sec, end_sec=end_sec)

    if events_all.empty:
        print("[WARN] Il CSV eventi è vuoto: il preview mostrerà sempre 'nessuna azione'.")
    elif events_segment.empty:
        print("[WARN] Nessun evento interseca il segmento renderizzato.")

    out_w, out_h = compute_output_size(video_info.width, video_info.height, max_width=max_width)

    # Apertura del video sorgente e del writer di output. Il codec è fisso per
    # mantenere semplice e riproducibile la generazione del file .mp4.
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {input_video}")

    writer = cv2.VideoWriter(
        str(output_video),
        get_fourcc(DEFAULT_CODEC),
        float(video_info.fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossibile creare VideoWriter per {output_video} con codec {DEFAULT_CODEC}.")

    # Conversione dell'intervallo temporale in indici frame: il ciclo seguente
    # renderizza solo il segmento richiesto.
    start_frame = max(0, int(round(start_sec * video_info.fps)))
    end_frame = min(video_info.num_frames, int(round(end_sec * video_info.fps)))
    total_frames = max(0, end_frame - start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Ciclo sui frame: per ogni istante cerca l'evento attivo, disegna le
    # informazioni grafiche e scrive il frame nel video di output.
    frames_written = 0
    with tqdm(total=total_frames, desc="Rendering preview", unit="frame") as pbar:
        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                print(f"[WARN] Lettura interrotta al frame {frame_idx}.")
                break

            current_time = frame_idx / video_info.fps
            frame = resize_frame_if_needed(frame, max_width=max_width)
            active = active_event_at_time(events_segment, current_time=current_time)
            draw_overlay(
                frame,
                current_time=current_time,
                start_sec=start_sec,
                end_sec=end_sec,
                active_event=active,
            )

            writer.write(frame)
            frames_written += 1
            pbar.update(1)

    cap.release()
    writer.release()

    # Statistiche finali restituite al chiamante e stampate dalla CLI.
    return {
        "input_video": str(input_video),
        "events_csv": str(events_csv),
        "output_video": str(output_video),
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "source_fps": float(video_info.fps),
        "output_width": int(out_w),
        "output_height": int(out_h),
        "num_events_total": int(len(events_all)),
        "num_events_in_segment": int(len(events_segment)),
        "num_frames_written": int(frames_written),
        "events_time_mode": EVENTS_TIME_MODE,
    }


# =============================================================================
# CLI
# =============================================================================


# Definisce l'interfaccia da riga di comando con path di input/output,
# segmento temporale, ridimensionamento opzionale e gestione overwrite.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Renderizza la preview annotata per la pipeline finale exp_long_13."
    )
    parser.add_argument("--input-video", type=Path, default=defaults.VAL_VIDEO_PATH)
    parser.add_argument("--events-csv", type=Path, default=defaults.VAL_OUTPUT_DIR / "events_postprocessed.csv")
    parser.add_argument("--output-video", type=Path, default=defaults.VAL_OUTPUT_DIR / "preview_annotated.mp4")
    parser.add_argument("--start-sec", type=float, default=defaults.VAL_START_SEC)
    parser.add_argument("--end-sec", type=parse_optional_float, default=defaults.VAL_END_SEC)
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Larghezza massima del preview. 0 mantiene la risoluzione originale.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


# Entry point CLI: risolve end_sec quando omesso, prepara max_width e invoca
# il rendering vero e proprio, stampando poi un riepilogo dell'output generato.
def main() -> None:
    args = build_parser().parse_args()

    if args.end_sec is None:
        video_info = get_video_info(args.input_video)
        end_sec = float(video_info.duration_sec)
    else:
        end_sec = float(args.end_sec)

    max_width = None if args.max_width is None or args.max_width <= 0 else int(args.max_width)

    stats = render_preview(
        input_video=args.input_video,
        events_csv=args.events_csv,
        output_video=args.output_video,
        start_sec=float(args.start_sec),
        end_sec=end_sec,
        max_width=max_width,
        overwrite=bool(args.overwrite),
    )

    print("\n=== Preview renderizzato ===")
    print(f"Video input:         {stats['input_video']}")
    print(f"Eventi:              {stats['events_csv']}")
    print(f"Video output:        {stats['output_video']}")
    print(f"Segmento:            {stats['start_sec']:.2f}s -> {stats['end_sec']:.2f}s")
    print(f"Eventi nel segmento: {stats['num_events_in_segment']}/{stats['num_events_total']}")
    print(f"Frame scritti:       {stats['num_frames_written']}")
    print(f"Risoluzione output:  {stats['output_width']}x{stats['output_height']}")
    print(f"Time mode eventi:    {stats['events_time_mode']}")


if __name__ == "__main__":
    main()
