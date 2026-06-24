# Annotazione automatica video2.mp4 con configurazione exp_long_13

Questo file contiene i comandi PowerShell per applicare la pipeline long-video finale `exp_long_13` al video esterno fornito dal professore.

Video di input:

```text
 data/datasets/dataset_basket_v1/videos/video2.mp4
```

Il video non è presente nel manifest e non ha annotazioni ground truth. Per questo motivo non va lanciata la valutazione quantitativa con `evaluate_events_from_manifest`: il risultato va considerato come annotazione automatica / valutazione qualitativa su video esterno al dataset.

Output principali finali:

```text
outputs/long_video/video2_exp_long_13/BasketAR_video2_report_events_exp13.csv
outputs/long_video/video2_exp_long_13/BasketAR_video2_preview_exp13.mp4
```

---

## 0. Ricavare la durata del video

Prima di lanciare la pipeline, calcoliamo automaticamente la durata di `video2.mp4` e la salviamo nella variabile PowerShell `$END_SEC`.

```powershell
$END_SEC = python -c "import cv2; p='data/datasets/dataset_basket_v1/videos/video2.mp4'; cap=cv2.VideoCapture(p); fps=cap.get(cv2.CAP_PROP_FPS); n=cap.get(cv2.CAP_PROP_FRAME_COUNT); cap.release(); print(n/fps)"

Write-Host "Durata video2.mp4:" $END_SEC "secondi"
```

---

## 1. Feature extraction

Questa fase legge il video frame per frame e costruisce la feature store con:

- feature visuali DINOv3;
- primitive YOLO palla/canestro;
- timestamp dei frame reali del video.

```powershell
python -m src.long_video.extract_feature_store `
  --input-video data/datasets/dataset_basket_v1/videos/video2.mp4 `
  --start-sec 0 `
  --end-sec $END_SEC `
  --output-dir data/features_long/video2_full_exp46 `
  --dino-repo third_party/dinov3 `
  --dino-weights checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth `
  --dino-model-name dinov3_vitl16 `
  --dino-input-size 336 `
  --dino-feature-dim 1024 `
  --dino-resize-mode stretch `
  --device 0 `
  --batch-size-decode 64 `
  --batch-size-dino 16 `
  --batch-size-yolo 8 `
  --imgsz 1280 `
  --conf 0.10 `
  --iou 0.50 `
  --no-amp `
  --overwrite
```

---

## 2. Costruzione finestre multi-scala

Questa fase costruisce le finestre temporali virtuali sulla feature store. È la stessa base usata dalla configurazione finale `exp_long_13`.

```powershell
python -m src.long_video.build_windows_from_store `
  --feature-store-dir data/features_long/video2_full_exp46 `
  --output-dir outputs/long_video/video2_exp_long_04_base `
  --start-sec 0 `
  --end-sec $END_SEC `
  --window-sizes 1.0 1.5 2.0 2.5 3.0 `
  --stride-sec 0.25 `
  --overwrite
```

---

## 3. Inferenza sulle finestre

Questa fase applica il modello gerarchico exp_46 alle finestre temporali generate al passo precedente.

```powershell
python -m src.long_video.infer_exp46_from_store `
  --feature-store-dir data/features_long/video2_full_exp46 `
  --windows-csv outputs/long_video/video2_exp_long_04_base/windows_manifest.csv `
  --output-dir outputs/long_video/video2_exp_long_04_base `
  --device 0 `
  --batch-size 128 `
  --overwrite
```

Output principale generato da questa fase:

```text
outputs/long_video/video2_exp_long_04_base/window_predictions_raw.csv
```

---

## 4. Post-processing con configurazione exp_long_13

Questa fase trasforma le predizioni sulle finestre in eventi finali. Usa la configurazione finale `exp_long_13`.

```powershell
python -m src.long_video.postprocess_events `
  --predictions-csv outputs/long_video/video2_exp_long_04_base/window_predictions_raw.csv `
  --output-dir outputs/long_video/video2_exp_long_13 `
  --min-conf-passaggio 0.75 `
  --min-conf-tiro 0.40 `
  --min-event-duration-sec 0.70 `
  --merge-gap-sec 0.20 `
  --max-duration-passaggio 1.50 `
  --max-duration-tiro 3.00 `
  --max-window-sec-passaggio 2.00 `
  --min-window-sec-tiro 1.00 `
  --max-window-sec-tiro 3.00 `
  --require-action-gt-noaction `
  --noaction-margin 20.0 `
  --event-confidence-mode max `
  --overwrite
```

Output principale generato da questa fase:

```text
outputs/long_video/video2_exp_long_13/events_postprocessed.csv
```

---

## 5. Generazione report CSV compatto

Questa fase prende `events_postprocessed.csv` e genera un CSV più pulito mantenendo solo le colonne principali:

```text
label,start_time,end_time,duration_sec,confidence,num_windows
```

```powershell
python -m src.long_video.export_compact_event_report `
  --input-csv outputs/long_video/video2_exp_long_13/events_postprocessed.csv `
  --output-csv outputs/long_video/video2_exp_long_13/BasketAR_video2_report_events_exp13.csv `
  --overwrite
```

Output finale:

```text
outputs/long_video/video2_exp_long_13/BasketAR_video2_report_events_exp13.csv
```

---

## 6. Generazione preview annotata

Questa fase genera il video annotato con le predizioni finali sovrapposte.

```powershell
python -m src.long_video.render_preview `
  --input-video data/datasets/dataset_basket_v1/videos/video2.mp4 `
  --events-csv outputs/long_video/video2_exp_long_13/events_postprocessed.csv `
  --output-video outputs/long_video/video2_exp_long_13/BasketAR_video2_preview_exp13.mp4 `
  --start-sec 0 `
  --end-sec $END_SEC `
  --events-time-mode auto `
  --max-width 1280 `
  --overwrite
```

Output finale:

```text
outputs/long_video/video2_exp_long_13/BasketAR_video2_preview_exp13.mp4
```