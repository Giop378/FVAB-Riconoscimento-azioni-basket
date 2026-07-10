# =============================================================================
# Scopo del modulo
# =============================================================================
# Implementa la lettura completa di una clip video con OpenCV e la conversione
# dei frame da BGR a immagini PIL RGB. È richiamato da
# features/extract_features.py prima del preprocessing DINOv3 definito in
# features/dinov3_extractor.py. La durata variabile viene preservata: non sono
# applicati campionamento, padding o troncamento in questa fase.
# =============================================================================

from pathlib import Path

import cv2
from PIL import Image


def read_video_frames(video_path: str | Path) -> list[Image.Image]:
    """
    Legge tutti i frame reali di una clip video.

    Parametri:
        video_path: percorso (stringa o path) del file video da leggere. 

    Restituisce:
        Una lista di immagini PIL in formato RGB, una per ogni frame del video.

    La funzione mantiene tutti i frame della clip, quindi non forza una
    lunghezza fissa. Questo è utile per clip di durata variabile.
    """
    
    video_path = str(video_path)

    # OpenCV gestisce la decodifica; l’apertura viene verificata prima del ciclo.
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    # Lista che conterrà tutti i frame convertiti in immagini PIL.
    frames = []

    # La lettura sequenziale evita salti temporali e conserva tutti i frame reali.
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # DINOv3 riceve immagini RGB, mentre OpenCV restituisce frame in BGR.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Nessun frame letto dal video: {video_path}")

    # Restituisce la lista completa dei frame della clip.
    return frames