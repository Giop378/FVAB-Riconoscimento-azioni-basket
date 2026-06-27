# Comandi PowerShell per eseguire `exp_long_13` con il codice aggiornato

La configurazione finale è:

```text
Modello: exp_46 gerarchico
Post-processing: exp_long_13
Timestamp eventi: absolute
IoU threshold valutazione: 0.20
Finestre: 1.0 1.5 2.0 2.5 3.0
Stride: 0.25s
```


# 1. Preparazione ambiente

Esegui sempre i comandi dalla root della repository.

```powershell
conda activate fvab-basket
```

Controlla che i `video_id` del manifest siano quelli attesi:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/datasets/dataset_basket_v1/manifest.csv'); print(sorted(df['video_id'].astype(str).unique()))"
```

Nei comandi sotto si assume:

```text
Validation video_id: prima_parte
Test video_id: psa_converted
```

---

# 2. Validation completa - `PrimaParte.mp4`, 135s → 735s

## 2.1 Variabili PowerShell

```powershell
$VAL_VIDEO = "data/datasets/dataset_basket_v1/videos/PrimaParte.mp4"
$VAL_FEATURE_DIR = "data/features_long/primaparte_0215_1215_exp46"
$VAL_OUTPUT_DIR = "outputs/long_video/primaparte_0215_1215_exp_long_13"
$VAL_START = 135
$VAL_END = 735
$VAL_VIDEO_ID_MANIFEST = "prima_parte"
$MANIFEST = "data/datasets/dataset_basket_v1/manifest.csv"
```

---

## 2.2 Feature extraction validation

Genera:

```text
metadata.json
timestamps.npy
frame_indices.npy
dinov3_features.npy
yolo_v1_primitives.npz
yolo_v2_primitives.npz
```

Comando essenziale, usando i path DINOv3/YOLO definiti in `defaults.py`:

```powershell
python -m src.long_video.extract_feature_store `
  --input-video $VAL_VIDEO `
  --start-sec $VAL_START `
  --end-sec $VAL_END `
  --output-dir $VAL_FEATURE_DIR `
  --device 0 `
  --batch-size-decode 128 `
  --batch-size-dino 32 `
  --batch-size-yolo 16 `
  --overwrite
```

Versione esplicita, se vuoi indicare manualmente DINOv3 e parametri YOLO:

```powershell
python -m src.long_video.extract_feature_store `
  --input-video $VAL_VIDEO `
  --start-sec $VAL_START `
  --end-sec $VAL_END `
  --output-dir $VAL_FEATURE_DIR `
  --dino-repo third_party/dinov3 `
  --dino-weights checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth `
  --dino-source local `
  --dino-model-name dinov3_vitl16 `
  --dino-input-size 336 `
  --dino-feature-dim 1024 `
  --yolo-v1-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt `
  --yolo-v2-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v2/weights/best.pt `
  --imgsz 1280 `
  --conf 0.10 `
  --iou 0.50 `
  --device 0 `
  --batch-size-decode 128 `
  --batch-size-dino 32 `
  --batch-size-yolo 16 `
  --overwrite
```

---

## 2.3 Costruzione finestre validation

Per `exp_long_13` usare sempre queste finestre:

```text
window-sizes = 1.0 1.5 2.0 2.5 3.0
stride-sec = 0.25
```

```powershell
python -m src.long_video.build_windows_from_store `
  --feature-store-dir $VAL_FEATURE_DIR `
  --output-dir $VAL_OUTPUT_DIR `
  --start-sec $VAL_START `
  --end-sec $VAL_END `
  --window-sizes 1.0 1.5 2.0 2.5 3.0 `
  --stride-sec 0.25 `
  --overwrite
```

Output:

```text
windows_manifest.csv
windows_metadata.json
```

---

## 2.4 Inferenza validation con modello gerarchico exp_46

I checkpoint L1/L2/L3 vengono letti direttamente da `defaults.py`.

```powershell
python -m src.long_video.infer_exp46_from_store `
  --feature-store-dir $VAL_FEATURE_DIR `
  --windows-csv "$VAL_OUTPUT_DIR/windows_manifest.csv" `
  --output-dir $VAL_OUTPUT_DIR `
  --device 0 `
  --batch-size 128 `
  --overwrite
```

Output:

```text
window_predictions_raw.csv
inference_metadata.json
```

---

## 2.5 Post-processing validation fisso `exp_long_13`

Il nuovo `postprocess_events.py` usa internamente la configurazione finale `exp_long_13`.

```powershell
python -m src.long_video.postprocess_events `
  --predictions-csv "$VAL_OUTPUT_DIR/window_predictions_raw.csv" `
  --output-dir $VAL_OUTPUT_DIR `
  --overwrite
```

Output:

```text
events_postprocessed.csv
```

---

## 2.6 Valutazione event-level validation

```powershell
python -m src.long_video.evaluate_events_from_manifest `
  --manifest $MANIFEST `
  --pred-events-csv "$VAL_OUTPUT_DIR/events_postprocessed.csv" `
  --video-id $VAL_VIDEO_ID_MANIFEST `
  --start-sec $VAL_START `
  --end-sec $VAL_END `
  --output-dir "$VAL_OUTPUT_DIR/eval_events" `
  --iou-threshold 0.20 `
  --overwrite
```

Output principali:

```text
event_metrics.md
event_metrics.json
per_class_metrics.csv
matched_events.csv
false_positives.csv
false_negatives.csv
```

---

## 2.7 Preview validation

Preview a risoluzione originale:

```powershell
python -m src.long_video.render_preview `
  --input-video $VAL_VIDEO `
  --events-csv "$VAL_OUTPUT_DIR/events_postprocessed.csv" `
  --output-video "$VAL_OUTPUT_DIR/preview_annotated.mp4" `
  --start-sec $VAL_START `
  --end-sec $VAL_END `
  --overwrite
```

Preview più leggera a larghezza massima 1280:

```powershell
python -m src.long_video.render_preview `
  --input-video $VAL_VIDEO `
  --events-csv "$VAL_OUTPUT_DIR/events_postprocessed.csv" `
  --output-video "$VAL_OUTPUT_DIR/preview_annotated_1280.mp4" `
  --start-sec $VAL_START `
  --end-sec $VAL_END `
  --max-width 1280 `
  --overwrite
```

---

## 2.8 Report compatto validation

```powershell
python -m src.long_video.export_compact_event_report `
  --input-csv "$VAL_OUTPUT_DIR/events_postprocessed.csv" `
  --output-csv "$VAL_OUTPUT_DIR/BasketAR_validation_report_events_exp13.csv" `
  --overwrite
```

---

# 3. Test finale - `PSA_converted.mp4`, 10s → 610s

## 3.1 Variabili PowerShell

```powershell
$TEST_VIDEO = "data/datasets/dataset_basket_v1/videos/PSA_converted.mp4"
$TEST_FEATURE_DIR = "data/features_long/psa_converted_0010_1010_exp46"
$TEST_OUTPUT_DIR = "outputs/long_video/psa_converted_0010_1010_exp_long_13"
$TEST_START = 10
$TEST_END = 610
$TEST_VIDEO_ID_MANIFEST = "psa_converted"
$MANIFEST = "data/datasets/dataset_basket_v1/manifest.csv"
```

---

## 3.2 Feature extraction test

Comando essenziale:

```powershell
python -m src.long_video.extract_feature_store `
  --input-video $TEST_VIDEO `
  --start-sec $TEST_START `
  --end-sec $TEST_END `
  --output-dir $TEST_FEATURE_DIR `
  --device 0 `
  --batch-size-decode 128 `
  --batch-size-dino 32 `
  --batch-size-yolo 16 `
  --overwrite
```

Versione esplicita:

```powershell
python -m src.long_video.extract_feature_store `
  --input-video $TEST_VIDEO `
  --start-sec $TEST_START `
  --end-sec $TEST_END `
  --output-dir $TEST_FEATURE_DIR `
  --dino-repo third_party/dinov3 `
  --dino-weights checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth `
  --dino-source local `
  --dino-model-name dinov3_vitl16 `
  --dino-input-size 336 `
  --dino-feature-dim 1024 `
  --yolo-v1-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt `
  --yolo-v2-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v2/weights/best.pt `
  --imgsz 1280 `
  --conf 0.10 `
  --iou 0.50 `
  --device 0 `
  --batch-size-decode 128 `
  --batch-size-dino 32 `
  --batch-size-yolo 16 `
  --overwrite
```

---

## 3.3 Costruzione finestre test

```powershell
python -m src.long_video.build_windows_from_store `
  --feature-store-dir $TEST_FEATURE_DIR `
  --output-dir $TEST_OUTPUT_DIR `
  --start-sec $TEST_START `
  --end-sec $TEST_END `
  --window-sizes 1.0 1.5 2.0 2.5 3.0 `
  --stride-sec 0.25 `
  --overwrite
```

---

## 3.4 Inferenza test

```powershell
python -m src.long_video.infer_exp46_from_store `
  --feature-store-dir $TEST_FEATURE_DIR `
  --windows-csv "$TEST_OUTPUT_DIR/windows_manifest.csv" `
  --output-dir $TEST_OUTPUT_DIR `
  --device 0 `
  --batch-size 128 `
  --overwrite
```

---

## 3.5 Post-processing test fisso `exp_long_13`

```powershell
python -m src.long_video.postprocess_events `
  --predictions-csv "$TEST_OUTPUT_DIR/window_predictions_raw.csv" `
  --output-dir $TEST_OUTPUT_DIR `
  --overwrite
```

---

## 3.6 Valutazione event-level test

Usa questo comando solo se il manifest contiene le annotazioni del test.

```powershell
python -m src.long_video.evaluate_events_from_manifest `
  --manifest $MANIFEST `
  --pred-events-csv "$TEST_OUTPUT_DIR/events_postprocessed.csv" `
  --video-id $TEST_VIDEO_ID_MANIFEST `
  --start-sec $TEST_START `
  --end-sec $TEST_END `
  --output-dir "$TEST_OUTPUT_DIR/eval_events" `
  --iou-threshold 0.20 `
  --overwrite
```

---

## 3.7 Preview test

Preview a risoluzione originale:

```powershell
python -m src.long_video.render_preview `
  --input-video $TEST_VIDEO `
  --events-csv "$TEST_OUTPUT_DIR/events_postprocessed.csv" `
  --output-video "$TEST_OUTPUT_DIR/preview_annotated.mp4" `
  --start-sec $TEST_START `
  --end-sec $TEST_END `
  --overwrite
```

Preview più leggera:

```powershell
python -m src.long_video.render_preview `
  --input-video $TEST_VIDEO `
  --events-csv "$TEST_OUTPUT_DIR/events_postprocessed.csv" `
  --output-video "$TEST_OUTPUT_DIR/preview_annotated_1280.mp4" `
  --start-sec $TEST_START `
  --end-sec $TEST_END `
  --max-width 1280 `
  --overwrite
```

---

## 3.8 Report compatto test

```powershell
python -m src.long_video.export_compact_event_report `
  --input-csv "$TEST_OUTPUT_DIR/events_postprocessed.csv" `
  --output-csv "$TEST_OUTPUT_DIR/BasketAR_test_report_events_exp13.csv" `
  --overwrite
```

---

# 4. Annotazione automatica video esterno - `video2.mp4`

## 4.1 Variabili PowerShell

```powershell
$VIDEO2 = "data/datasets/dataset_basket_v1/videos/video2.mp4"
$VIDEO2_FEATURE_DIR = "data/features_long/video2_full_exp46"
$VIDEO2_OUTPUT_DIR = "outputs/long_video/video2_exp_long_13"
$VIDEO2_START = 0
```

---

## 4.2 Calcolo robusto durata video2

```powershell
$VIDEO2_END = python -c "import cv2; p=r'$VIDEO2'; cap=cv2.VideoCapture(p); opened=cap.isOpened(); fps=cap.get(cv2.CAP_PROP_FPS); n=cap.get(cv2.CAP_PROP_FRAME_COUNT); cap.release(); assert opened and fps>0 and n>0, f'Video non leggibile o FPS/frame non validi: {p}, opened={opened}, fps={fps}, frames={n}'; print(n/fps)"

Write-Host "Durata video2.mp4:" $VIDEO2_END "secondi"
```

Se questo comando fallisce con FPS pari a zero, il problema non è nella pipeline ma nella lettura del video da OpenCV: controlla path, codec del file o installazione OpenCV/FFmpeg.

---

## 4.3 Feature extraction video2

Comando essenziale:

```powershell
python -m src.long_video.extract_feature_store `
  --input-video $VIDEO2 `
  --start-sec $VIDEO2_START `
  --end-sec $VIDEO2_END `
  --output-dir $VIDEO2_FEATURE_DIR `
  --device 0 `
  --batch-size-decode 128 `
  --batch-size-dino 32 `
  --batch-size-yolo 16 `
  --overwrite
```

Versione esplicita:

```powershell
python -m src.long_video.extract_feature_store `
  --input-video $VIDEO2 `
  --start-sec $VIDEO2_START `
  --end-sec $VIDEO2_END `
  --output-dir $VIDEO2_FEATURE_DIR `
  --dino-repo third_party/dinov3 `
  --dino-weights checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth `
  --dino-source local `
  --dino-model-name dinov3_vitl16 `
  --dino-input-size 336 `
  --dino-feature-dim 1024 `
  --yolo-v1-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt `
  --yolo-v2-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v2/weights/best.pt `
  --imgsz 1280 `
  --conf 0.10 `
  --iou 0.50 `
  --device 0 `
  --batch-size-decode 128 `
  --batch-size-dino 32 `
  --batch-size-yolo 16 `
  --overwrite
```

---

## 4.4 Costruzione finestre video2

```powershell
python -m src.long_video.build_windows_from_store `
  --feature-store-dir $VIDEO2_FEATURE_DIR `
  --output-dir $VIDEO2_OUTPUT_DIR `
  --start-sec $VIDEO2_START `
  --end-sec $VIDEO2_END `
  --window-sizes 1.0 1.5 2.0 2.5 3.0 `
  --stride-sec 0.25 `
  --overwrite
```

---

## 4.5 Inferenza video2

```powershell
python -m src.long_video.infer_exp46_from_store `
  --feature-store-dir $VIDEO2_FEATURE_DIR `
  --windows-csv "$VIDEO2_OUTPUT_DIR/windows_manifest.csv" `
  --output-dir $VIDEO2_OUTPUT_DIR `
  --device 0 `
  --batch-size 128 `
  --overwrite
```

---

## 4.6 Post-processing video2 fisso `exp_long_13`

```powershell
python -m src.long_video.postprocess_events `
  --predictions-csv "$VIDEO2_OUTPUT_DIR/window_predictions_raw.csv" `
  --output-dir $VIDEO2_OUTPUT_DIR `
  --overwrite
```

---

## 4.7 Report compatto video2

```powershell
python -m src.long_video.export_compact_event_report `
  --input-csv "$VIDEO2_OUTPUT_DIR/events_postprocessed.csv" `
  --output-csv "$VIDEO2_OUTPUT_DIR/BasketAR_video2_report_events_exp13.csv" `
  --overwrite
```

---

## 4.8 Preview video2

Preview a risoluzione originale:

```powershell
python -m src.long_video.render_preview `
  --input-video $VIDEO2 `
  --events-csv "$VIDEO2_OUTPUT_DIR/events_postprocessed.csv" `
  --output-video "$VIDEO2_OUTPUT_DIR/BasketAR_video2_preview_exp13.mp4" `
  --start-sec $VIDEO2_START `
  --end-sec $VIDEO2_END `
  --overwrite
```

Preview più leggera:

```powershell
python -m src.long_video.render_preview `
  --input-video $VIDEO2 `
  --events-csv "$VIDEO2_OUTPUT_DIR/events_postprocessed.csv" `
  --output-video "$VIDEO2_OUTPUT_DIR/BasketAR_video2_preview_exp13_1280.mp4" `
  --start-sec $VIDEO2_START `
  --end-sec $VIDEO2_END `
  --max-width 1280 `
  --overwrite
```
