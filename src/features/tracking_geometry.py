"""
Funzioni geometriche e temporali per il tracking palla/canestro.

Contiene la definizione dei set temporali temp29 e temp43, le utility per
convertire le detection YOLO in feature normalizzate e il calcolo della sequenza
per-frame usata dai modelli gerarchici.
"""
# Collegamenti con la pipeline:
# - extract_ball_rim_tracking_features.py passa qui gli output YOLO per ottenere
#   detection normalizzate, relazioni palla-canestro e derivate temporali;
# - tracking_io.py serializza le sequenze secondo l’ordine dei nomi definito qui;
# - TrackingSequenceFeatureStore e i checkpoint assumono che tale ordine resti stabile.


import math

import numpy as np


# L’ordine delle feature è parte del formato dati: modificarlo renderebbe le
# sequenze incompatibili con normalizzazioni e checkpoint già salvati.
TEMPORAL_TRACKING_FEATURE_NAMES = [
    # Detection quality / visibility.
    "t_rel",
    "ball_detected",
    "rim_detected",
    "both_detected",
    "ball_conf",
    "rim_conf",
    "ball_xc",
    "ball_yc",
    "ball_w",
    "ball_h",
    "ball_area",
    "rim_xc",
    "rim_yc",
    "rim_w",
    "rim_h",
    "rim_area",

    # Ball-rim geometry.
    "dx",
    "dy",
    "ball_rim_dist",
    "ball_near_rim",
    "ball_above_rim",
    "ball_below_rim",
    "ball_center_inside_rim",
    "ball_center_inside_expanded_rim",
    "ball_rim_iou",
    "ball_passes_close_to_rim",

    # Absolute ball motion.
    "ball_vx",
    "ball_vy",
    "ball_speed",
    "ball_ax",
    "ball_ay",
    "ball_acceleration",
    "ball_motion_horizontal_ratio",
    "ball_motion_vertical_ratio",

    # Relative ball-rim motion and rim crossing events.
    "ball_rim_dist_delta",
    "ball_relative_vx",
    "ball_relative_vy",
    "ball_relative_speed",
    "ball_rim_approach_speed",
    "ball_rim_departure_speed",
    "ball_crosses_rim_y_frame",
    "ball_crosses_rim_y_downward_frame",
    "ball_crosses_rim_y_upward_frame",
]


# Set storico a 29 feature usato dai checkpoint temporali "temp29".
# L'ordine deve restare identico a quello salvato nei checkpoint.
TEMPORAL_TRACKING_FEATURE_NAMES_TEMP29 = [
    "t_rel",
    "ball_detected",
    "rim_detected",
    "both_detected",
    "ball_conf",
    "rim_conf",
    "ball_xc",
    "ball_yc",
    "ball_w",
    "ball_h",
    "ball_area",
    "rim_xc",
    "rim_yc",
    "rim_w",
    "rim_h",
    "rim_area",
    "dx",
    "dy",
    "ball_rim_dist",
    "ball_near_rim",
    "ball_above_rim",
    "ball_below_rim",
    "ball_vx",
    "ball_vy",
    "ball_speed",
    "ball_ax",
    "ball_ay",
    "ball_acceleration",
    "ball_rim_dist_delta",
]

# Set esteso a 43 feature usato dai checkpoint "temp43".
TEMPORAL_TRACKING_FEATURE_NAMES_TEMP43 = TEMPORAL_TRACKING_FEATURE_NAMES

TEMPORAL_TRACKING_FEATURE_SETS = {
    "temp29": TEMPORAL_TRACKING_FEATURE_NAMES_TEMP29,
    "temp43": TEMPORAL_TRACKING_FEATURE_NAMES_TEMP43,
}



def safe_float(value, default=0.0):
    """Converte un valore numerico in float sostituendo NaN/inf con un default."""
    if value is None:
        return default
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    return float(value)


# Converte coordinate pixel YOLO in centro, dimensioni e area normalizzati in [0, 1].
def detection_to_dict(box_xyxy, conf, frame_width, frame_height):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    x1 = max(0.0, min(x1, frame_width - 1))
    x2 = max(0.0, min(x2, frame_width - 1))
    y1 = max(0.0, min(y1, frame_height - 1))
    y2 = max(0.0, min(y2, frame_height - 1))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return {
        "detected": 1,
        "conf": float(conf),
        "xc": xc / frame_width if frame_width > 0 else 0.0,
        "yc": yc / frame_height if frame_height > 0 else 0.0,
        "w": bw / frame_width if frame_width > 0 else 0.0,
        "h": bh / frame_height if frame_height > 0 else 0.0,
        "area": (bw * bh) / (frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0,
    }


def empty_detection():
    return {
        "detected": 0,
        "conf": 0.0,
        "xc": 0.0,
        "yc": 0.0,
        "w": 0.0,
        "h": 0.0,
        "area": 0.0,
    }


def detection_bbox(det, margin: float = 0.0):
    """
    Restituisce il bbox normalizzato [x1, y1, x2, y2].

    margin espande il bbox di margin * w e margin * h per lato. Questo è utile
    perché il bbox del rim non coincide perfettamente con il cilindro del ferro,
    quindi una piccola espansione rende il proxy di ingresso palla-canestro meno
    fragile rispetto a detection leggermente rumorose.
    """
    xc = float(det.get("xc", 0.0))
    yc = float(det.get("yc", 0.0))
    w = float(det.get("w", 0.0))
    h = float(det.get("h", 0.0))

    half_w = w * (0.5 + margin)
    half_h = h * (0.5 + margin)

    x1 = max(0.0, xc - half_w)
    y1 = max(0.0, yc - half_h)
    x2 = min(1.0, xc + half_w)
    y2 = min(1.0, yc + half_h)

    return x1, y1, x2, y2


def point_inside_bbox(x: float, y: float, bbox) -> int:
    x1, y1, x2, y2 = bbox
    return int(x1 <= float(x) <= x2 and y1 <= float(y) <= y2)


def detection_iou(det_a, det_b) -> float:
    ax1, ay1, ax2, ay2 = detection_bbox(det_a, margin=0.0)
    bx1, by1, bx2, by2 = detection_bbox(det_b, margin=0.0)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 1e-12:
        return 0.0

    return float(inter_area / union)


# Per ogni frame conserva al massimo la detection più confidente di palla e rim;
# in assenza di una classe restituisce un record esplicito di non rilevamento.
def parse_yolo_result(result, ball_class_id, rim_class_id, frame_width, frame_height):
    best = {
        "ball": None,
        "rim": None,
    }

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return empty_detection(), empty_detection()

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)

    for box, conf, class_id in zip(xyxy, confs, classes):
        if class_id == ball_class_id:
            current = best["ball"]
            if current is None or conf > current["conf"]:
                best["ball"] = detection_to_dict(box, conf, frame_width, frame_height)

        elif class_id == rim_class_id:
            current = best["rim"]
            if current is None or conf > current["conf"]:
                best["rim"] = detection_to_dict(box, conf, frame_width, frame_height)

    ball = best["ball"] if best["ball"] is not None else empty_detection()
    rim = best["rim"] if best["rim"] is not None else empty_detection()

    return ball, rim


# Deriva feature geometriche soltanto quando entrambi gli oggetti sono visibili;
# altrimenti usa valori neutri coerenti con il formato della sequenza.
def compute_pair_features(ball, rim, near_threshold, rim_inside_margin=0.15):
    both_detected = int(ball["detected"] == 1 and rim["detected"] == 1)

    if not both_detected:
        return {
            "both_detected": 0,
            "dx": 0.0,
            "dy": 0.0,
            "dist": 1.0,
            "ball_above_rim": 0,
            "ball_below_rim": 0,
            "ball_near_rim": 0,
            "ball_center_inside_rim": 0,
            "ball_center_inside_expanded_rim": 0,
            "ball_rim_iou": 0.0,
            "ball_passes_close_to_rim": 0,
        }

    dx = ball["xc"] - rim["xc"]
    dy = ball["yc"] - rim["yc"]
    dist = math.sqrt(dx * dx + dy * dy)

    rim_bbox = detection_bbox(rim, margin=0.0)
    expanded_rim_bbox = detection_bbox(rim, margin=rim_inside_margin)

    center_inside_rim = point_inside_bbox(ball["xc"], ball["yc"], rim_bbox)
    center_inside_expanded_rim = point_inside_bbox(ball["xc"], ball["yc"], expanded_rim_bbox)
    ball_rim_iou = detection_iou(ball, rim)

    # Proxy leggero per frame in cui la palla passa davvero nella zona del ferro:
    # vicino al rim e orizzontalmente compatibile con la larghezza del bbox del rim.
    horizontal_gate = max(float(rim.get("w", 0.0)), near_threshold * 0.50, 1e-6)
    ball_passes_close_to_rim = int(dist <= near_threshold and abs(dx) <= horizontal_gate)

    return {
        "both_detected": 1,
        "dx": dx,
        "dy": dy,
        "dist": dist,
        "ball_above_rim": int(ball["yc"] < rim["yc"]),
        "ball_below_rim": int(ball["yc"] > rim["yc"]),
        "ball_near_rim": int(dist <= near_threshold),
        "ball_center_inside_rim": center_inside_rim,
        "ball_center_inside_expanded_rim": center_inside_expanded_rim,
        "ball_rim_iou": ball_rim_iou,
        "ball_passes_close_to_rim": ball_passes_close_to_rim,
    }



# Calcola differenze finite rispetto al frame campionato precedente e combina
# visibilità, geometria, moto assoluto e moto relativo nel vettore per-frame.
def compute_temporal_sequence_features(frame_rows, fps, temporal_feature_names=None):
    """
    Converte le detection per-frame in una sequenza [S, K] di feature temporali.

    La rappresentazione conserva l'ordine temporale. Include feature di visibilità, geometria
    palla-canestro, velocità assoluta, velocità relativa palla-rim e proxy
    dell'ingresso della palla nel ferro.
    """
    if temporal_feature_names is None:
        temporal_feature_names = TEMPORAL_TRACKING_FEATURE_NAMES_TEMP43

    rows = sorted(frame_rows, key=lambda r: int(r["frame_order"]))

    if not rows:
        return np.zeros((0, len(temporal_feature_names)), dtype=np.float32)

    sequence_rows = []
    prev_row = None
    prev_vx = None
    prev_vy = None
    prev_speed = None

    for row in rows:
        ball_detected = int(row["ball_detected"]) == 1
        rim_detected = int(row["rim_detected"]) == 1
        both_detected = int(row["both_detected"]) == 1

        ball_vx = 0.0
        ball_vy = 0.0
        ball_speed = 0.0
        ball_ax = 0.0
        ball_ay = 0.0
        ball_acceleration = 0.0
        dist_delta = 0.0
        rel_vx = 0.0
        rel_vy = 0.0
        rel_speed = 0.0
        approach_speed = 0.0
        departure_speed = 0.0
        horizontal_ratio = 0.0
        vertical_ratio = 0.0
        crosses_rim_y_frame = 0.0
        crosses_rim_y_downward_frame = 0.0
        crosses_rim_y_upward_frame = 0.0
        current_velocity_valid = False

        # Le derivate sono valide solo se il tempo trascorso è positivo e le
        # detection necessarie sono presenti in entrambi i frame confrontati.
        if prev_row is not None:
            delta_frames = int(row["frame_idx"]) - int(prev_row["frame_idx"])
            dt = delta_frames / fps if fps > 0 and delta_frames > 0 else 0.0

            prev_ball_detected = int(prev_row["ball_detected"]) == 1
            prev_both_detected = int(prev_row["both_detected"]) == 1

            if dt > 0 and ball_detected and prev_ball_detected:
                ball_vx = (float(row["ball_xc"]) - float(prev_row["ball_xc"])) / dt
                ball_vy = (float(row["ball_yc"]) - float(prev_row["ball_yc"])) / dt
                ball_speed = math.sqrt(ball_vx * ball_vx + ball_vy * ball_vy)
                den = abs(ball_vx) + abs(ball_vy)
                horizontal_ratio = abs(ball_vx) / den if den > 1e-12 else 0.0
                vertical_ratio = abs(ball_vy) / den if den > 1e-12 else 0.0
                current_velocity_valid = True

                if prev_vx is not None and prev_vy is not None and prev_speed is not None:
                    ball_ax = (ball_vx - prev_vx) / dt
                    ball_ay = (ball_vy - prev_vy) / dt
                    ball_acceleration = (ball_speed - prev_speed) / dt

            if dt > 0 and both_detected and prev_both_detected:
                dist_delta = (float(row["ball_rim_dist"]) - float(prev_row["ball_rim_dist"])) / dt
                rel_vx = (float(row["dx"]) - float(prev_row["dx"])) / dt
                rel_vy = (float(row["dy"]) - float(prev_row["dy"])) / dt
                rel_speed = math.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)
                approach_speed = max(0.0, -dist_delta)
                departure_speed = max(0.0, dist_delta)

                prev_dy = float(prev_row["dy"])
                curr_dy = float(row["dy"])
                if abs(prev_dy) > 1e-6 and abs(curr_dy) > 1e-6 and np.sign(prev_dy) != np.sign(curr_dy):
                    crosses_rim_y_frame = 1.0
                    if prev_dy < 0 and curr_dy > 0:
                        crosses_rim_y_downward_frame = 1.0
                    elif prev_dy > 0 and curr_dy < 0:
                        crosses_rim_y_upward_frame = 1.0

        feature_values = {
            "t_rel": float(row["t_rel"]),
            "ball_detected": float(row["ball_detected"]),
            "rim_detected": float(row["rim_detected"]),
            "both_detected": float(row["both_detected"]),
            "ball_conf": float(row["ball_conf"]),
            "rim_conf": float(row["rim_conf"]),
            "ball_xc": float(row["ball_xc"]),
            "ball_yc": float(row["ball_yc"]),
            "ball_w": float(row["ball_w"]),
            "ball_h": float(row["ball_h"]),
            "ball_area": float(row["ball_area"]),
            "rim_xc": float(row["rim_xc"]),
            "rim_yc": float(row["rim_yc"]),
            "rim_w": float(row["rim_w"]),
            "rim_h": float(row["rim_h"]),
            "rim_area": float(row["rim_area"]),
            "dx": float(row["dx"]),
            "dy": float(row["dy"]),
            "ball_rim_dist": float(row["ball_rim_dist"]),
            "ball_near_rim": float(row["ball_near_rim"]),
            "ball_above_rim": float(row["ball_above_rim"]),
            "ball_below_rim": float(row["ball_below_rim"]),
            "ball_center_inside_rim": float(row.get("ball_center_inside_rim", 0.0)),
            "ball_center_inside_expanded_rim": float(row.get("ball_center_inside_expanded_rim", 0.0)),
            "ball_rim_iou": float(row.get("ball_rim_iou", 0.0)),
            "ball_passes_close_to_rim": float(row.get("ball_passes_close_to_rim", 0.0)),
            "ball_vx": ball_vx,
            "ball_vy": ball_vy,
            "ball_speed": ball_speed,
            "ball_ax": ball_ax,
            "ball_ay": ball_ay,
            "ball_acceleration": ball_acceleration,
            "ball_motion_horizontal_ratio": horizontal_ratio,
            "ball_motion_vertical_ratio": vertical_ratio,
            "ball_rim_dist_delta": dist_delta,
            "ball_relative_vx": rel_vx,
            "ball_relative_vy": rel_vy,
            "ball_relative_speed": rel_speed,
            "ball_rim_approach_speed": approach_speed,
            "ball_rim_departure_speed": departure_speed,
            "ball_crosses_rim_y_frame": crosses_rim_y_frame,
            "ball_crosses_rim_y_downward_frame": crosses_rim_y_downward_frame,
            "ball_crosses_rim_y_upward_frame": crosses_rim_y_upward_frame,
        }

        # La selezione per nome garantisce esattamente l’ordine del set temp29/temp43.
        sequence_rows.append([
            safe_float(feature_values[name], default=0.0)
            for name in temporal_feature_names
        ])

        if current_velocity_valid:
            prev_vx = ball_vx
            prev_vy = ball_vy
            prev_speed = ball_speed

        prev_row = row

    return np.asarray(sequence_rows, dtype=np.float32)
