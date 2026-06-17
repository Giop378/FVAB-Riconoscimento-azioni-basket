from pathlib import Path


# =============================================================================
# Dataset / video lunghi
# =============================================================================

DATASET_ROOT = Path("data/datasets/dataset_basket_v1")
VIDEOS_ROOT = DATASET_ROOT / "videos"

# Video di validation usato per sviluppare e tarare la pipeline long-video.
# Segmento scelto: 2:15 -> 12:15, cioè 135s -> 735s.
VAL_VIDEO_ID = "primaparte_0215_1215"
VAL_VIDEO_PATH = VIDEOS_ROOT / "PrimaParte.mp4"
VAL_START_SEC = 135.0
VAL_END_SEC = 735.0

# Video di test finale.
# Per ora salvo solo il path: il segmento temporale del test va deciso in seguito
# oppure passato da riga di comando.
TEST_VIDEO_ID = "psa_converted"
TEST_VIDEO_PATH = VIDEOS_ROOT / "PSA_converted.mp4"
TEST_START_SEC = None
TEST_END_SEC = None


# =============================================================================
# Feature extractor video
# =============================================================================

DINOV3_REPO = Path("third_party/dinov3")
DINOV3_INPUT_SIZE = 336
DINOV3_FEATURE_DIM = 1024


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
# Root standard per feature store e output long-video
# =============================================================================

LONG_VIDEO_FEATURES_ROOT = Path("data/features_long")
LONG_VIDEO_OUTPUTS_ROOT = Path("outputs/long_video")

VAL_FEATURE_STORE_DIR = LONG_VIDEO_FEATURES_ROOT / f"{VAL_VIDEO_ID}_exp46"
VAL_OUTPUT_DIR = LONG_VIDEO_OUTPUTS_ROOT / f"{VAL_VIDEO_ID}_exp46"

TEST_FEATURE_STORE_DIR = LONG_VIDEO_FEATURES_ROOT / f"{TEST_VIDEO_ID}_exp46"
TEST_OUTPUT_DIR = LONG_VIDEO_OUTPUTS_ROOT / f"{TEST_VIDEO_ID}_exp46"


# =============================================================================
# Feature tracking attese da exp_46
# =============================================================================

L1_TRACKING_SOURCE = "yolo_v2"
L1_TRACKING_TYPE = "temp43"

L2_TRACKING_SOURCE = "yolo_v2"
L2_TRACKING_TYPE = "temp29"

L3_TRACKING_SOURCE = "yolo_v1"
L3_TRACKING_TYPE = "temp43"