from pathlib import Path

import cv2
import torch  # Attualmente non usato in questo file; può essere rimosso se non serve altrove.
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

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    # Lista che conterrà tutti i frame convertiti in immagini PIL.
    frames = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Nessun frame letto dal video: {video_path}")

    # Restituisce la lista completa dei frame della clip.
    return frames