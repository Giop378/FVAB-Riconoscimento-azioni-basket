# Comandi finali long-video: exp_long_13 e exp_long_18

Questo file contiene i comandi PowerShell per rieseguire la pipeline long-video completa, dalla feature extraction fino a post-processing, valutazione e preview.

Gli esperimenti finali considerati sono:

```text
exp_long_13 = miglior compromesso quantitativo globale
exp_long_18 = variante orientata alla demo, con maggiore attenzione ai tiri
```

Entrambi gli esperimenti usano la stessa base di finestre e inferenza. Cambia solo il post-processing finale.

---

# 1. Validation: PrimaParte.mp4, segmento 135s → 735s

## 1.1 Feature extraction validation

```powershell
python -m src.long_video.extract_feature_store `
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 `
  --start-sec 135 `
  --end-sec 735 `
  --output-dir data/features_long/primaparte_0215_1215_exp46 `
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

## 1.2 Costruzione finestre validation

Questa è la base usata da `exp_long_13` e `exp_long_18`.

```powershell
python -m src.long_video.build_windows_from_store `
  --feature-store-dir data/features_long/primaparte_0215_1215_exp46 `
  --output-dir outputs/long_video/exp_long_04 `
  --start-sec 135 `
  --end-sec 735 `
  --window-sizes 1.0 1.5 2.0 2.5 3.0 `
  --stride-sec 0.25 `
  --overwrite
```

---

## 1.3 Inferenza validation

```powershell
python -m src.long_video.infer_exp46_from_store `
  --feature-store-dir data/features_long/primaparte_0215_1215_exp46 `
  --windows-csv outputs/long_video/exp_long_04/windows_manifest.csv `
  --output-dir outputs/long_video/exp_long_04 `
  --device 0 `
  --batch-size 128 `
  --overwrite
```

L'output principale usato dai due post-processing è:

```text
outputs/long_video/exp_long_04/window_predictions_raw.csv
```

---

# 2. Validation: exp_long_13

## 2.1 Post-processing exp_long_13

```powershell
python -m src.long_video.postprocess_events `
  --predictions-csv outputs/long_video/exp_long_04/window_predictions_raw.csv `
  --output-dir outputs/long_video/exp_long_13 `
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

## 2.2 Valutazione exp_long_13

```powershell
python -m src.long_video.evaluate_events_from_manifest `
  --manifest data/datasets/dataset_basket_v1/manifest.csv `
  --pred-events-csv outputs/long_video/exp_long_13/events_postprocessed.csv `
  --video-id prima_parte `
  --start-sec 135 `
  --end-sec 735 `
  --output-dir outputs/long_video/exp_long_13/eval_iou020 `
  --iou-threshold 0.20 `
  --pred-time-mode auto `
  --overwrite
```

## 2.3 Preview exp_long_13

```powershell
python -m src.long_video.render_preview `
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 `
  --events-csv outputs/long_video/exp_long_13/events_postprocessed.csv `
  --output-video outputs/long_video/exp_long_13/preview_annotated.mp4 `
  --start-sec 135 `
  --end-sec 735 `
  --events-time-mode auto `
  --max-width 1280 `
  --overwrite
```

---

# 3. Validation: exp_long_18

## 3.1 Post-processing exp_long_18

```powershell
python -m src.long_video.postprocess_events `
  --predictions-csv outputs/long_video/exp_long_04/window_predictions_raw.csv `
  --output-dir outputs/long_video/exp_long_18 `
  --min-conf-passaggio 0.75 `
  --min-conf-tiro 0.50 `
  --min-event-duration-sec 0.70 `
  --merge-gap-sec 0.20 `
  --max-duration-passaggio 1.50 `
  --max-duration-tiro 3.00 `
  --max-window-sec-passaggio 2.00 `
  --min-window-sec-tiro 1.00 `
  --max-window-sec-tiro 3.00 `
  --require-action-gt-noaction `
  --noaction-margin 20.0 `
  --noaction-margin-passaggio 20.0 `
  --noaction-margin-tiro 5.0 `
  --event-confidence-mode max `
  --prefer-shots-over-passaggi `
  --prefer-shots-min-confidence 0.55 `
  --overwrite
```

## 3.2 Valutazione exp_long_18

```powershell
python -m src.long_video.evaluate_events_from_manifest `
  --manifest data/datasets/dataset_basket_v1/manifest.csv `
  --pred-events-csv outputs/long_video/exp_long_18/events_postprocessed.csv `
  --video-id prima_parte `
  --start-sec 135 `
  --end-sec 735 `
  --output-dir outputs/long_video/exp_long_18/eval_iou020 `
  --iou-threshold 0.20 `
  --pred-time-mode auto `
  --overwrite
```

## 3.3 Preview exp_long_18

```powershell
python -m src.long_video.render_preview `
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 `
  --events-csv outputs/long_video/exp_long_18/events_postprocessed.csv `
  --output-video outputs/long_video/exp_long_18/preview_annotated.mp4 `
  --start-sec 135 `
  --end-sec 735 `
  --events-time-mode auto `
  --max-width 1280 `
  --overwrite
```

---

# 4. Test: PSA_converted.mp4, segmento 10s → 610s

Prima di lanciare la valutazione sul test, controlla il nome esatto del `video_id` nel manifest:

```powershell
@'
import pandas as pd

manifest = "data/datasets/dataset_basket_v1/manifest.csv"
df = pd.read_csv(manifest)
print(sorted(df["video_id"].astype(str).unique()))
'@ | python
```

Nei comandi sotto viene usato:

```text
--video-id psa_converted
```

Se il comando precedente mostra un nome diverso, sostituiscilo nei comandi di valutazione.

---

## 4.1 Feature extraction test

```powershell
python -m src.long_video.extract_feature_store `
  --input-video data/datasets/dataset_basket_v1/videos/PSA_converted.mp4 `
  --start-sec 10 `
  --end-sec 610 `
  --output-dir data/features_long/psa_converted_0010_1010_exp46 `
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

## 4.2 Costruzione finestre test

Questa è la base comune per testare sia la configurazione `exp_long_13` sia `exp_long_18` sul video di test.

```powershell
python -m src.long_video.build_windows_from_store `
  --feature-store-dir data/features_long/psa_converted_0010_1010_exp46 `
  --output-dir outputs/long_video/test_exp_long_04_base `
  --start-sec 10 `
  --end-sec 610 `
  --window-sizes 1.0 1.5 2.0 2.5 3.0 `
  --stride-sec 0.25 `
  --overwrite
```

---

## 4.3 Inferenza test

```powershell
python -m src.long_video.infer_exp46_from_store `
  --feature-store-dir data/features_long/psa_converted_0010_1010_exp46 `
  --windows-csv outputs/long_video/test_exp_long_04_base/windows_manifest.csv `
  --output-dir outputs/long_video/test_exp_long_04_base `
  --device 0 `
  --batch-size 128 `
  --overwrite
```

L'output principale usato dai due post-processing sul test è:

```text
outputs/long_video/test_exp_long_04_base/window_predictions_raw.csv
```

---

# 5. Test: configurazione exp_long_13

## 5.1 Post-processing test con configurazione exp_long_13

```powershell
python -m src.long_video.postprocess_events `
  --predictions-csv outputs/long_video/test_exp_long_04_base/window_predictions_raw.csv `
  --output-dir outputs/long_video/test_exp_long_13 `
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

## 5.2 Valutazione test con configurazione exp_long_13

```powershell
python -m src.long_video.evaluate_events_from_manifest `
  --manifest data/datasets/dataset_basket_v1/manifest.csv `
  --pred-events-csv outputs/long_video/test_exp_long_13/events_postprocessed.csv `
  --video-id psa_converted `
  --start-sec 10 `
  --end-sec 610 `
  --output-dir outputs/long_video/test_exp_long_13/eval_iou020 `
  --iou-threshold 0.20 `
  --pred-time-mode auto `
  --overwrite
```

## 5.3 Preview test con configurazione exp_long_13

```powershell
python -m src.long_video.render_preview `
  --input-video data/datasets/dataset_basket_v1/videos/PSA_converted.mp4 `
  --events-csv outputs/long_video/test_exp_long_13/events_postprocessed.csv `
  --output-video outputs/long_video/test_exp_long_13/preview_annotated.mp4 `
  --start-sec 10 `
  --end-sec 610 `
  --events-time-mode auto `
  --max-width 1280 `
  --overwrite
```

---

# 6. Test: configurazione exp_long_18

## 6.1 Post-processing test con configurazione exp_long_18

```powershell
python -m src.long_video.postprocess_events `
  --predictions-csv outputs/long_video/test_exp_long_04_base/window_predictions_raw.csv `
  --output-dir outputs/long_video/test_exp_long_18 `
  --min-conf-passaggio 0.75 `
  --min-conf-tiro 0.50 `
  --min-event-duration-sec 0.70 `
  --merge-gap-sec 0.20 `
  --max-duration-passaggio 1.50 `
  --max-duration-tiro 3.00 `
  --max-window-sec-passaggio 2.00 `
  --min-window-sec-tiro 1.00 `
  --max-window-sec-tiro 3.00 `
  --require-action-gt-noaction `
  --noaction-margin 20.0 `
  --noaction-margin-passaggio 20.0 `
  --noaction-margin-tiro 5.0 `
  --event-confidence-mode max `
  --prefer-shots-over-passaggi `
  --prefer-shots-min-confidence 0.55 `
  --overwrite
```

## 6.2 Valutazione test con configurazione exp_long_18

```powershell
python -m src.long_video.evaluate_events_from_manifest `
  --manifest data/datasets/dataset_basket_v1/manifest.csv `
  --pred-events-csv outputs/long_video/test_exp_long_18/events_postprocessed.csv `
  --video-id psa_converted `
  --start-sec 10 `
  --end-sec 610 `
  --output-dir outputs/long_video/test_exp_long_18/eval_iou020 `
  --iou-threshold 0.20 `
  --pred-time-mode auto `
  --overwrite
```

## 6.3 Preview test con configurazione exp_long_18

```powershell
python -m src.long_video.render_preview `
  --input-video data/datasets/dataset_basket_v1/videos/PSA_converted.mp4 `
  --events-csv outputs/long_video/test_exp_long_18/events_postprocessed.csv `
  --output-video outputs/long_video/test_exp_long_18/preview_annotated.mp4 `
  --start-sec 10 `
  --end-sec 610 `
  --events-time-mode auto `
  --max-width 1280 `
  --overwrite
```

---

# 7. Output finali attesi

## Validation

```text
outputs/long_video/exp_long_13/events_postprocessed.csv
outputs/long_video/exp_long_13/eval_iou020/event_metrics.md
outputs/long_video/exp_long_13/preview_annotated.mp4

outputs/long_video/exp_long_18/events_postprocessed.csv
outputs/long_video/exp_long_18/eval_iou020/event_metrics.md
outputs/long_video/exp_long_18/preview_annotated.mp4
```

## Test

```text
outputs/long_video/test_exp_long_13/events_postprocessed.csv
outputs/long_video/test_exp_long_13/eval_iou020/event_metrics.md
outputs/long_video/test_exp_long_13/preview_annotated.mp4

outputs/long_video/test_exp_long_18/events_postprocessed.csv
outputs/long_video/test_exp_long_18/eval_iou020/event_metrics.md
outputs/long_video/test_exp_long_18/preview_annotated.mp4
```

---

