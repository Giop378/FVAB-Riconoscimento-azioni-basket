# Comandi pipeline long-video `exp_46`

Questo file raccoglie i comandi da usare per eseguire la pipeline separata che applica il modello `exp_46` a video lunghi.

La pipeline è divisa in fasi:

```text
1. check_setup.py
2. extract_feature_store.py
3. build_windows_from_store.py
4. infer_exp46_from_store.py
5. postprocess_events.py
6. render_preview.py
7. evaluate_events_from_manifest.py
```

La logica generale è:

```text
video lungo
  -> feature store temporale
  -> finestre virtuali
  -> inferenza exp_46
  -> post-processing eventi
  -> preview video annotata
  -> valutazione automatica event-level su manifest.csv
```

---

## 0. Segmenti usati

### Validation debug

Segmento corto usato solo per verificare velocemente che il codice funzioni.

```text
video: data/datasets/dataset_basket_v1/videos/PrimaParte.mp4
start: 135s = 2:15
end:   165s = 2:45
durata: 30 secondi
```

Cartelle:

```text
feature store: data/features_long/primaparte_0215_0245_exp46_debug
output:        outputs/long_video/primaparte_0215_0245_exp46_debug
```

### Validation completa

Segmento usato per sviluppare e tarare la pipeline long-video.

```text
video: data/datasets/dataset_basket_v1/videos/PrimaParte.mp4
start: 135s = 2:15
end:   735s = 12:15
durata: 10 minuti
```

Cartelle:

```text
feature store: data/features_long/primaparte_0215_1215_exp46
output:        outputs/long_video/primaparte_0215_1215_exp46
```

### Test completo

Segmento del video di test da usare solo alla fine, quando la pipeline è stabile.

```text
video: data/datasets/dataset_basket_v1/videos/PSA_converted.mp4
start: 10s  = 0:10
end:   610s = 10:10
durata: 10 minuti
```

Cartelle:

```text
feature store: data/features_long/psa_converted_0010_1010_exp46
output:        outputs/long_video/psa_converted_0010_1010_exp46
```

---

## 1. Controllo setup

Da eseguire prima di tutto.

```bash
python -m src.long_video.check_setup --check-video-duration --print-features
```

Controllare che stampi una configurazione coerente con `exp_46`:

```text
L1 -> 43 feature tracking
L2 -> 29 feature tracking
L3 -> 43 feature tracking
```

Se questo controllo fallisce, non proseguire con la pipeline.

---

## 1B. Valutazione automatica su `manifest.csv`

Dopo aver prodotto `events_postprocessed.csv`, la pipeline può essere valutata automaticamente confrontando gli eventi predetti con le annotazioni presenti in:

```text
data/datasets/dataset_basket_v1/manifest.csv
```

La valutazione long-video considera **solo le 7 azioni reali**:

```text
passaggio
tiroDaDue0
tiroDaDue1
tiroDaTre0
tiroDaTre1
tiroLibero0
tiroLibero1
```

Le classi `idle`, `non-gioco` e `no-action` non vengono valutate come eventi finali.

Una predizione viene considerata corretta se:

```text
label predetta = label ground truth
+
temporal IoU >= soglia
```

Soglia iniziale consigliata:

```text
temporal IoU >= 0.30
```

Output della valutazione:

```text
evaluation/
├── gt_events.csv
├── pred_events_filtered.csv
├── matched_events.csv
├── false_positives.csv
├── false_negatives.csv
├── per_class_metrics.csv
├── event_metrics.json
├── event_metrics.txt
└── event_metrics.md
```

Metriche principali da usare per confrontare gli esperimenti:

```text
macro_f1_active_classes
f1_micro
precision_micro
recall_micro
mean_iou
center_mae_sec
TP / FP / FN
```

---

# Pipeline validation debug

Usare questa pipeline corta prima di lanciare i 10 minuti completi.

---

## 2A. Estrazione feature store validation debug

```bash
python -m src.long_video.extract_feature_store \
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 \
  --start-sec 135 \
  --end-sec 165 \
  --output-dir data/features_long/primaparte_0215_0245_exp46_debug \
  --feature-fps 24 \
  --device 0 \
  --batch-size-decode 128 \
  --batch-size-dino 32 \
  --batch-size-yolo 16 \
  --overwrite
```

Output atteso:

```text
data/features_long/primaparte_0215_0245_exp46_debug/
├── metadata.json
├── timestamps.npy
├── frame_indices.npy
├── dinov3_features.npy
├── yolo_v1_detections.csv
├── yolo_v1_primitives.npz
├── yolo_v2_detections.csv
└── yolo_v2_primitives.npz
```

---

## 3A. Creazione finestre validation debug

```bash
python -m src.long_video.build_windows_from_store \
  --feature-store-dir data/features_long/primaparte_0215_0245_exp46_debug \
  --output-dir outputs/long_video/primaparte_0215_0245_exp46_debug \
  --window-sizes 1.0 1.5 2.0 3.0 \
  --stride-sec 0.25 \
  --overwrite
```

Output atteso:

```text
outputs/long_video/primaparte_0215_0245_exp46_debug/
├── windows_manifest.csv
└── windows_metadata.json
```

---

## 4A. Inferenza `exp_46` validation debug

```bash
python -m src.long_video.infer_exp46_from_store \
  --feature-store-dir data/features_long/primaparte_0215_0245_exp46_debug \
  --windows-csv outputs/long_video/primaparte_0215_0245_exp46_debug/windows_manifest.csv \
  --output-dir outputs/long_video/primaparte_0215_0245_exp46_debug \
  --device 0 \
  --batch-size 128 \
  --num-frames 48 \
  --overwrite
```

Output atteso:

```text
outputs/long_video/primaparte_0215_0245_exp46_debug/
├── window_predictions_raw.csv
└── inference_metadata.json
```

---

## 5A. Post-processing validation debug

```bash
python -m src.long_video.postprocess_events \
  --predictions-csv outputs/long_video/primaparte_0215_0245_exp46_debug/window_predictions_raw.csv \
  --output-dir outputs/long_video/primaparte_0215_0245_exp46_debug \
  --smooth-window 3 \
  --min-event-duration-sec 0.4 \
  --merge-gap-sec 0.75 \
  --temporal-nms-iou 0.50 \
  --min-conf-passaggio 0.55 \
  --min-conf-tiro 0.40 \
  --overwrite
```

Output atteso:

```text
outputs/long_video/primaparte_0215_0245_exp46_debug/
├── events_raw.csv
├── events_postprocessed.csv
├── annotations.json
└── postprocess_metadata.json
```

---

## 6A. Preview video validation debug

```bash
python -m src.long_video.render_preview \
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 \
  --events-csv outputs/long_video/primaparte_0215_0245_exp46_debug/events_postprocessed.csv \
  --output-video outputs/long_video/primaparte_0215_0245_exp46_debug/preview_annotated.mp4 \
  --start-sec 135 \
  --end-sec 165 \
  --overwrite
```

Output atteso:

```text
outputs/long_video/primaparte_0215_0245_exp46_debug/
├── preview_annotated.mp4
└── preview_annotated.metadata.json
```

---

## 7A. Valutazione automatica validation debug

```bash
python -m src.long_video.evaluate_events_from_manifest \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --pred-events-csv outputs/long_video/primaparte_0215_0245_exp46_debug/events_postprocessed.csv \
  --video-id prima_parte \
  --start-sec 135 \
  --end-sec 165 \
  --output-dir outputs/long_video/primaparte_0215_0245_exp46_debug/evaluation \
  --iou-threshold 0.30 \
  --overwrite
```

Output atteso:

```text
outputs/long_video/primaparte_0215_0245_exp46_debug/evaluation/
├── gt_events.csv
├── pred_events_filtered.csv
├── matched_events.csv
├── false_positives.csv
├── false_negatives.csv
├── per_class_metrics.csv
├── event_metrics.json
├── event_metrics.txt
└── event_metrics.md
```

---

# Pipeline validation completa

Lanciare questa pipeline solo dopo che la validation debug funziona.

---

## 2B. Estrazione feature store validation completa

```bash
python -m src.long_video.extract_feature_store \
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 \
  --start-sec 135 \
  --end-sec 735 \
  --output-dir data/features_long/primaparte_0215_1215_exp46 \
  --feature-fps 24 \
  --device 0 \
  --batch-size-decode 128 \
  --batch-size-dino 32 \
  --batch-size-yolo 16 \
  --overwrite
```

---

## 3B. Creazione finestre validation completa

```bash
python -m src.long_video.build_windows_from_store \
  --feature-store-dir data/features_long/primaparte_0215_1215_exp46 \
  --output-dir outputs/long_video/primaparte_0215_1215_exp46 \
  --window-sizes 1.0 1.5 2.0 3.0 \
  --stride-sec 0.25 \
  --overwrite
```

---

## 4B. Inferenza `exp_46` validation completa

```bash
python -m src.long_video.infer_exp46_from_store \
  --feature-store-dir data/features_long/primaparte_0215_1215_exp46 \
  --windows-csv outputs/long_video/primaparte_0215_1215_exp46/windows_manifest.csv \
  --output-dir outputs/long_video/primaparte_0215_1215_exp46 \
  --device 0 \
  --batch-size 128 \
  --num-frames 48 \
  --overwrite
```

---

## 5B. Post-processing validation completa

```bash
python -m src.long_video.postprocess_events \
  --predictions-csv outputs/long_video/primaparte_0215_1215_exp46/window_predictions_raw.csv \
  --output-dir outputs/long_video/primaparte_0215_1215_exp46 \
  --smooth-window 3 \
  --min-event-duration-sec 0.4 \
  --merge-gap-sec 0.75 \
  --temporal-nms-iou 0.50 \
  --min-conf-passaggio 0.55 \
  --min-conf-tiro 0.40 \
  --overwrite
```

---

## 6B. Preview video validation completa

```bash
python -m src.long_video.render_preview \
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 \
  --events-csv outputs/long_video/primaparte_0215_1215_exp46/events_postprocessed.csv \
  --output-video outputs/long_video/primaparte_0215_1215_exp46/preview_annotated.mp4 \
  --start-sec 135 \
  --end-sec 735 \
  --overwrite
```

---

## 7B. Valutazione automatica validation completa

```bash
python -m src.long_video.evaluate_events_from_manifest \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --pred-events-csv outputs/long_video/primaparte_0215_1215_exp46/events_postprocessed.csv \
  --video-id prima_parte \
  --start-sec 135 \
  --end-sec 735 \
  --output-dir outputs/long_video/primaparte_0215_1215_exp46/evaluation \
  --iou-threshold 0.30 \
  --overwrite
```

Output principale da leggere dopo ogni esperimento:

```text
outputs/long_video/primaparte_0215_1215_exp46/evaluation/event_metrics.md
```

Questa è la valutazione da usare per scegliere soglie, finestre e post-processing.

---

# Pipeline test completo

Da usare solo alla fine, dopo aver scelto definitivamente soglie, finestre e post-processing sulla validation completa.

---

## 2C. Estrazione feature store test completo

```bash
python -m src.long_video.extract_feature_store \
  --input-video data/datasets/dataset_basket_v1/videos/PSA_converted.mp4 \
  --start-sec 10 \
  --end-sec 610 \
  --output-dir data/features_long/psa_converted_0010_1010_exp46 \
  --feature-fps 24 \
  --device 0 \
  --batch-size-decode 128 \
  --batch-size-dino 32 \
  --batch-size-yolo 16 \
  --overwrite
```

---

## 3C. Creazione finestre test completo

```bash
python -m src.long_video.build_windows_from_store \
  --feature-store-dir data/features_long/psa_converted_0010_1010_exp46 \
  --output-dir outputs/long_video/psa_converted_0010_1010_exp46 \
  --window-sizes 1.0 1.5 2.0 3.0 \
  --stride-sec 0.25 \
  --overwrite
```

---

## 4C. Inferenza `exp_46` test completo

```bash
python -m src.long_video.infer_exp46_from_store \
  --feature-store-dir data/features_long/psa_converted_0010_1010_exp46 \
  --windows-csv outputs/long_video/psa_converted_0010_1010_exp46/windows_manifest.csv \
  --output-dir outputs/long_video/psa_converted_0010_1010_exp46 \
  --device 0 \
  --batch-size 128 \
  --num-frames 48 \
  --overwrite
```

---

## 5C. Post-processing test completo

```bash
python -m src.long_video.postprocess_events \
  --predictions-csv outputs/long_video/psa_converted_0010_1010_exp46/window_predictions_raw.csv \
  --output-dir outputs/long_video/psa_converted_0010_1010_exp46 \
  --smooth-window 3 \
  --min-event-duration-sec 0.4 \
  --merge-gap-sec 0.75 \
  --temporal-nms-iou 0.50 \
  --min-conf-passaggio 0.55 \
  --min-conf-tiro 0.40 \
  --overwrite
```

---

## 6C. Preview video test completo

```bash
python -m src.long_video.render_preview \
  --input-video data/datasets/dataset_basket_v1/videos/PSA_converted.mp4 \
  --events-csv outputs/long_video/psa_converted_0010_1010_exp46/events_postprocessed.csv \
  --output-video outputs/long_video/psa_converted_0010_1010_exp46/preview_annotated.mp4 \
  --start-sec 10 \
  --end-sec 610 \
  --overwrite
```

---

## 7C. Valutazione automatica test completo

```bash
python -m src.long_video.evaluate_events_from_manifest \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --pred-events-csv outputs/long_video/psa_converted_0010_1010_exp46/events_postprocessed.csv \
  --video-id psa_converted \
  --start-sec 10 \
  --end-sec 610 \
  --output-dir outputs/long_video/psa_converted_0010_1010_exp46/evaluation \
  --iou-threshold 0.30 \
  --overwrite
```

Output principale:

```text
outputs/long_video/psa_converted_0010_1010_exp46/evaluation/event_metrics.md
```

Questo comando va usato solo alla fine, dopo aver scelto la configurazione migliore sulla validation completa.

---

# Comandi rapidi per rilanciare solo alcune fasi

## Se cambi solo soglie o parametri di post-processing

Non rieseguire feature extraction, windows e inferenza. Rilancia solo:

```bash
python -m src.long_video.postprocess_events \
  --predictions-csv outputs/long_video/primaparte_0215_1215_exp46/window_predictions_raw.csv \
  --output-dir outputs/long_video/primaparte_0215_1215_exp46 \
  --smooth-window 3 \
  --min-event-duration-sec 0.4 \
  --merge-gap-sec 0.75 \
  --temporal-nms-iou 0.50 \
  --min-conf-passaggio 0.55 \
  --min-conf-tiro 0.40 \
  --overwrite
```

```bash
python -m src.long_video.render_preview \
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 \
  --events-csv outputs/long_video/primaparte_0215_1215_exp46/events_postprocessed.csv \
  --output-video outputs/long_video/primaparte_0215_1215_exp46/preview_annotated.mp4 \
  --start-sec 135 \
  --end-sec 735 \
  --overwrite
```

```bash
python -m src.long_video.evaluate_events_from_manifest \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --pred-events-csv outputs/long_video/primaparte_0215_1215_exp46/events_postprocessed.csv \
  --video-id prima_parte \
  --start-sec 135 \
  --end-sec 735 \
  --output-dir outputs/long_video/primaparte_0215_1215_exp46/evaluation \
  --iou-threshold 0.30 \
  --overwrite
```

## Se cambi `window-sizes` o `stride-sec`

Rilancia da `build_windows_from_store.py` in poi:

```text
build_windows_from_store.py
infer_exp46_from_store.py
postprocess_events.py
render_preview.py
evaluate_events_from_manifest.py
```

## Se cambi `feature-fps`, segmento video, DINOv3 o YOLO

Rilancia tutta la pipeline da:

```text
extract_feature_store.py
```

---

# Come confrontare gli esperimenti

Per stabilire se una modifica migliora la pipeline, usare come riferimento il primo run completo stabile su validation completa.

Metriche principali da riportare in `EXPERIMENTS_LONG_VIDEO.md`:

```text
macro_f1_active_classes
f1_micro
precision_micro
recall_micro
mean_iou
center_mae_sec
true_positive
false_positive
false_negative
```

Criterio consigliato:

```text
Una modifica migliora se aumenta macro_f1_active_classes sulla validation completa
e non peggiora in modo evidente recall dei tiri, precision dei passaggi e qualità temporale degli eventi.
```

Per leggere rapidamente il risultato di un run:

```bash
cat outputs/long_video/primaparte_0215_1215_exp46/evaluation/event_metrics.md
```

Per controllare gli errori:

```bash
python - <<'PY'
import pandas as pd
base = 'outputs/long_video/primaparte_0215_1215_exp46/evaluation'
print('False positives:')
print(pd.read_csv(f'{base}/false_positives.csv').head(20))
print('False negatives:')
print(pd.read_csv(f'{base}/false_negatives.csv').head(20))
PY
```

---

# Regole operative

1. Usare sempre prima la validation debug.
2. Passare alla validation completa solo quando la debug non dà errori.
3. Usare il test completo solo alla fine.
4. Non rilanciare `extract_feature_store.py` inutilmente: è la fase più pesante.
5. Per provare soglie diverse basta rilanciare `postprocess_events.py`, `render_preview.py` ed `evaluate_events_from_manifest.py`.
6. Per confrontare gli esperimenti usare principalmente `event_metrics.md` prodotto sulla validation completa.
7. Il test completo va valutato solo una volta scelta la configurazione finale sulla validation completa.
8. Non modificare il codice di training di `exp_46`: questa pipeline deve restare separata.