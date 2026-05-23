from pathlib import Path

import cv2
import torch
from PIL import Image


def read_video_frames(video_path: str | Path) -> list[Image.Image]:
    """
    Legge tutti i frame reali di una clip video.
    Restituisce una lista di PIL Image in RGB.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

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

    return frames