from pathlib import Path


# =============================================================================
# Dataset / video lunghi
# =============================================================================

DATASET_ROOT = Path("data/datasets/dataset_basket_v1")
VIDEOS_ROOT = DATASET_ROOT / "videos"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"


# =============================================================================
# Video di validation per sviluppo pipeline long-video
# =============================================================================

# Video di validation usato per sviluppare e tarare la pipeline.
# Segmento completo scelto: 2:15 -> 12:15, cioè 135s -> 735s.
VAL_VIDEO_ID = "primaparte_0215_1215"
VAL_VIDEO_PATH = VIDEOS_ROOT / "PrimaParte.mp4"
VAL_START_SEC = 135.0
VAL_END_SEC = 735.0

# Segmento breve di debug preso sempre dal video di validation.
# Serve solo per testare velocemente gli script.
VAL_DEBUG_VIDEO_ID = "primaparte_0215_0245_exp46_debug"
VAL_DEBUG_VIDEO_PATH = VAL_VIDEO_PATH
VAL_DEBUG_START_SEC = 135.0
VAL_DEBUG_END_SEC = 165.0


# =============================================================================
# Video di test finale
# =============================================================================

# Video di test finale.
# Segmento scelto: 0:10 -> 10:10, cioè 10s -> 610s.
TEST_VIDEO_ID = "psa_converted_0010_1010"
TEST_VIDEO_PATH = VIDEOS_ROOT / "PSA_converted.mp4"
TEST_START_SEC = 10.0
TEST_END_SEC = 610.0


# =============================================================================
# Feature extractor video
# =============================================================================

DINOV3_REPO = Path("third_party/dinov3")
DINOV3_SOURCE = "local"
DINOV3_MODEL_NAME = "dinov3_vitl16"
DINOV3_INPUT_SIZE = 336
DINOV3_FEATURE_DIM = 1024
DINOV3_OUTPUT_TOKEN = "x_norm_clstoken"

# Pesi DINOv3 da usare sia per le clip sia per i video lunghi.
DINOV3_WEIGHTS = Path(
    "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)

# Feature DINOv3 già usate negli esperimenti clip-level.
# Per la pipeline long-video verranno create nuove feature store in data/features_long.
DINOV3_CLIP_FEATURES_ROOT = Path("data/features/dinov3_vitl16_336")


# =============================================================================
# Detector palla/canestro
# =============================================================================

YOLO_V1_WEIGHTS = Path(
    "runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt"
)

YOLO_V2_WEIGHTS = Path(
    "runs/detect/outputs/ball_rim_detector/yolo11m_1280_v2/weights/best.pt"
)


# =============================================================================
# Checkpoint exp_46
# =============================================================================

EXP46_L1_CHECKPOINT = Path(
    "outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt"
)

EXP46_L2_CHECKPOINT = Path(
    "outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt"
)

EXP46_L3_CHECKPOINT = Path(
    "outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt"
)


# =============================================================================
# Feature tracking clip-level usate da exp_46
# =============================================================================

# Queste sono le feature usate negli esperimenti su clip.
# Nella pipeline long-video non verranno lette direttamente per il video lungo,
# però sono utili come riferimento per sapere quale configurazione replica exp_46.

EXP46_L1_TRACKING_ROOT = Path(
    "data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2_v4"
)

EXP46_L2_TRACKING_ROOT = Path(
    "data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2"
)

EXP46_L3_TRACKING_ROOT = Path(
    "data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1_v4"
)


# =============================================================================
# Configurazione tracking attesa da exp_46
# =============================================================================

L1_TRACKING_SOURCE = "yolo_v2"
L1_TRACKING_TYPE = "temp43"

L2_TRACKING_SOURCE = "yolo_v2"
L2_TRACKING_TYPE = "temp29"

L3_TRACKING_SOURCE = "yolo_v1"
L3_TRACKING_TYPE = "temp43"


# =============================================================================
# Root standard per feature store e output long-video
# =============================================================================

LONG_VIDEO_FEATURES_ROOT = Path("data/features_long")
LONG_VIDEO_OUTPUTS_ROOT = Path("outputs/long_video")


# Validation debug, 30 secondi
VAL_DEBUG_FEATURE_STORE_DIR = LONG_VIDEO_FEATURES_ROOT / VAL_DEBUG_VIDEO_ID
VAL_DEBUG_OUTPUT_DIR = LONG_VIDEO_OUTPUTS_ROOT / VAL_DEBUG_VIDEO_ID


# Validation completa, 10 minuti
VAL_FEATURE_STORE_DIR = LONG_VIDEO_FEATURES_ROOT / f"{VAL_VIDEO_ID}_exp46"
VAL_OUTPUT_DIR = LONG_VIDEO_OUTPUTS_ROOT / f"{VAL_VIDEO_ID}_exp46"


# Test finale, 10 minuti
TEST_FEATURE_STORE_DIR = LONG_VIDEO_FEATURES_ROOT / f"{TEST_VIDEO_ID}_exp46"
TEST_OUTPUT_DIR = LONG_VIDEO_OUTPUTS_ROOT / f"{TEST_VIDEO_ID}_exp46"


# =============================================================================
# Nomi file standard della pipeline long-video
# =============================================================================

METADATA_FILENAME = "metadata.json"
TIMESTAMPS_FILENAME = "timestamps.npy"
FRAME_INDICES_FILENAME = "frame_indices.npy"
DINOV3_FEATURES_FILENAME = "dinov3_features.npy"

YOLO_V1_DETECTIONS_FILENAME = "yolo_v1_detections.csv"
YOLO_V1_PRIMITIVES_FILENAME = "yolo_v1_primitives.npz"

YOLO_V2_DETECTIONS_FILENAME = "yolo_v2_detections.csv"
YOLO_V2_PRIMITIVES_FILENAME = "yolo_v2_primitives.npz"

WINDOWS_MANIFEST_FILENAME = "windows_manifest.csv"
WINDOWS_METADATA_FILENAME = "windows_metadata.json"

WINDOW_FEATURES_DIRNAME = "window_features_exp46"
WINDOW_FEATURES_L1_FILENAME = "window_features_l1.npz"
WINDOW_FEATURES_L2_FILENAME = "window_features_l2.npz"
WINDOW_FEATURES_L3_FILENAME = "window_features_l3.npz"
WINDOW_FEATURES_INDEX_FILENAME = "window_features_index.json"
WINDOW_FEATURES_METADATA_FILENAME = "window_features_metadata.json"

WINDOW_PREDICTIONS_FILENAME = "window_predictions_raw.csv"
INFERENCE_METADATA_FILENAME = "inference_metadata.json"

EVENTS_RAW_FILENAME = "events_raw.csv"
EVENTS_POSTPROCESSED_FILENAME = "events_postprocessed.csv"
ANNOTATIONS_FILENAME = "annotations.json"
POSTPROCESS_METADATA_FILENAME = "postprocess_metadata.json"

PREVIEW_VIDEO_FILENAME = "preview_annotated.mp4"
PREVIEW_METADATA_FILENAME = "preview_annotated.metadata.json"