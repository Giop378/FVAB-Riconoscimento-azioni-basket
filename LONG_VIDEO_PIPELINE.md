# Pipeline Long-Video con `exp_46`

Questo file documenta la pipeline applicativa usata per applicare il modello clip-level `exp_46` a segmenti video lunghi, con l'obiettivo di produrre automaticamente una lista di azioni rilevate con timestamp.

La pipeline è separata dal codice di training/valutazione del modello su clip: `exp_46` viene considerato modello congelato e viene usato solo in inferenza.

---

## 1. Obiettivo

Dato un video lungo di una partita di basket, la pipeline deve:

1. leggere un segmento temporale del video;
2. estrarre una feature store temporale una sola volta;
3. costruire finestre temporali virtuali sopra la feature store;
4. applicare il modello gerarchico `exp_46` a ogni finestra;
5. filtrare e unire le predizioni finestra-per-finestra;
6. produrre una lista finale di eventi con `label`, `start_time`, `end_time` e `confidence`;
7. creare un video preview annotato per la verifica qualitativa.

Schema generale:

```text
video lungo
  -> feature store temporale
  -> finestre virtuali
  -> inferenza exp_46
  -> post-processing temporale
  -> eventi finali + preview annotata
```

---

## 2. Modello congelato di riferimento

Il modello usato dalla pipeline è `exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43`.

Configurazione:

| Livello | Task | Checkpoint | Tracking | Tipo feature |
|---|---|---|---|---|
| L1 | `passaggio / tiro / no-action` | `outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt` | YOLO v2 | `temp43` |
| L2 | `tiroDaDue / tiroDaTre / tiroLibero` | `outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt` | YOLO v2 | `temp29` |
| L3 | `tiro0 / tiro1` | `outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt` | YOLO v1 | `temp43` |

Classi finali della pipeline:

```text
passaggio
tiroDaDue0
tiroDaDue1
tiroDaTre0
tiroDaTre1
tiroLibero0
tiroLibero1
no-action
```

Nel report finale delle azioni, `no-action` viene usata come background/scarto e non viene inserita nella lista delle azioni rilevate.

---

## 3. Segmenti video usati

| ID segmento | Split/Uso | Video | Start | End | Durata | Output feature store | Output pipeline |
|---|---|---|---:|---:|---:|---|---|
| `primaparte_0215_0245_exp46_debug` | validation debug | `data/datasets/dataset_basket_v1/videos/PrimaParte.mp4` | 135 | 165 | 30s | `data/features_long/primaparte_0215_0245_exp46_debug` | `outputs/long_video/primaparte_0215_0245_exp46_debug` |
| `primaparte_0215_1215_exp46` | validation completa | `data/datasets/dataset_basket_v1/videos/PrimaParte.mp4` | 135 | 735 | 600s | `data/features_long/primaparte_0215_1215_exp46` | `outputs/long_video/primaparte_0215_1215_exp46` |
| `psa_converted_0010_1010_exp46` | test finale | `data/datasets/dataset_basket_v1/videos/PSA_converted.mp4` | 10 | 610 | 600s | `data/features_long/psa_converted_0010_1010_exp46` | `outputs/long_video/psa_converted_0010_1010_exp46` |

Nota:

- `validation debug` serve solo per controllare velocemente che gli script funzionino;
- `validation completa` serve per sviluppare e tarare la pipeline;
- `test finale` deve essere eseguito solo dopo aver scelto i parametri finali su validation.

---

## 4. File della pipeline

I file principali sono in:

```text
src/long_video/
```

| File | Scopo |
|---|---|
| `defaults.py` | Path e segmenti temporali stabili della pipeline. |
| `check_setup.py` | Controlla video, checkpoint, repo DINOv3, pesi YOLO e configurazioni tracking. |
| `extract_feature_store.py` | Estrae feature DINOv3 e primitive YOLO v1/v2 una sola volta sul segmento video. |
| `build_windows_from_store.py` | Crea il manifest delle finestre virtuali da classificare. |
| `infer_exp46_from_store.py` | Applica `exp_46` alle finestre virtuali e salva le predizioni raw. |
| `postprocess_events.py` | Trasforma le predizioni per finestra in eventi finali. |
| `render_preview.py` | Crea un video annotato per controllare visivamente gli eventi. |

---

## 5. Output principali

### 5.1 Feature store

Creata da `extract_feature_store.py`.

Esempio:

```text
data/features_long/primaparte_0215_1215_exp46/
├── metadata.json
├── timestamps.npy
├── frame_indices.npy
├── dinov3_features.npy
├── yolo_v1_detections.csv
├── yolo_v1_primitives.npz
├── yolo_v2_detections.csv
└── yolo_v2_primitives.npz
```

Significato:

| File | Descrizione |
|---|---|
| `metadata.json` | Informazioni su video, segmento, fps, modello DINO/YOLO e parametri di estrazione. |
| `timestamps.npy` | Timestamp assoluti dei frame campionati. |
| `frame_indices.npy` | Indici dei frame nel video originale. |
| `dinov3_features.npy` | Feature DINOv3, shape `[N, 1024]`. |
| `yolo_v1_primitives.npz` | Primitive palla/canestro estratte con YOLO v1, usate da L3. |
| `yolo_v2_primitives.npz` | Primitive palla/canestro estratte con YOLO v2, usate da L1 e L2. |

---

### 5.2 Manifest finestre

Creato da `build_windows_from_store.py`.

```text
outputs/long_video/<segment_id>/windows_manifest.csv
outputs/long_video/<segment_id>/windows_metadata.json
```

Il manifest contiene finestre virtuali, non clip video fisiche.

Colonne principali:

```text
window_id
scale_sec
start_time
end_time
center_time
store_start_index
store_end_index
num_store_samples
```

---

### 5.3 Predizioni raw

Create da `infer_exp46_from_store.py`.

```text
outputs/long_video/<segment_id>/window_predictions_raw.csv
outputs/long_video/<segment_id>/inference_metadata.json
```

Colonne principali:

```text
p_l1_passaggio
p_l1_tiro
p_l1_noaction
p_l2_tiroDaDue
p_l2_tiroDaTre
p_l2_tiroLibero
p_l3_0
p_l3_1
score_passaggio
score_tiroDaDue0
score_tiroDaDue1
score_tiroDaTre0
score_tiroDaTre1
score_tiroLibero0
score_tiroLibero1
score_noaction
pred_label
confidence
```

---

### 5.4 Eventi finali

Creati da `postprocess_events.py`.

```text
outputs/long_video/<segment_id>/events_raw.csv
outputs/long_video/<segment_id>/events_postprocessed.csv
outputs/long_video/<segment_id>/annotations.json
outputs/long_video/<segment_id>/postprocess_metadata.json
```

File più importante:

```text
events_postprocessed.csv
```

Formato atteso:

```text
event_id,label,start_time,end_time,duration_sec,confidence,num_windows,scale_sec
```

---

### 5.5 Video preview

Creato da `render_preview.py`.

```text
outputs/long_video/<segment_id>/preview_annotated.mp4
outputs/long_video/<segment_id>/preview_annotated.metadata.json
```

Il video preview mostra:

- tempo assoluto del video;
- tempo relativo al segmento;
- azione attiva;
- confidenza;
- barra di progresso dell'evento;
- timeline degli eventi;
- legenda classi.

---

## 6. Ordine di esecuzione

### 6.1 Controllo iniziale

```bash
python -m src.long_video.check_setup --check-video-duration --print-features
```

Verifiche attese:

```text
L1 -> 43 feature tracking
L2 -> 29 feature tracking
L3 -> 43 feature tracking
```

---

### 6.2 Pipeline validation debug, 30 secondi

#### 1. Feature store

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

#### 2. Finestre virtuali

```bash
python -m src.long_video.build_windows_from_store \
  --feature-store-dir data/features_long/primaparte_0215_0245_exp46_debug \
  --output-dir outputs/long_video/primaparte_0215_0245_exp46_debug \
  --window-sizes 1.0 1.5 2.0 3.0 \
  --stride-sec 0.25 \
  --overwrite
```

#### 3. Inferenza exp_46

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

#### 4. Post-processing eventi

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

#### 5. Preview

```bash
python -m src.long_video.render_preview \
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 \
  --events-csv outputs/long_video/primaparte_0215_0245_exp46_debug/events_postprocessed.csv \
  --output-video outputs/long_video/primaparte_0215_0245_exp46_debug/preview_annotated.mp4 \
  --start-sec 135 \
  --end-sec 165 \
  --overwrite
```

---

### 6.3 Pipeline validation completa, 10 minuti

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

```bash
python -m src.long_video.build_windows_from_store \
  --feature-store-dir data/features_long/primaparte_0215_1215_exp46 \
  --output-dir outputs/long_video/primaparte_0215_1215_exp46 \
  --window-sizes 1.0 1.5 2.0 3.0 \
  --stride-sec 0.25 \
  --overwrite
```

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

---

### 6.4 Pipeline test finale, 10 minuti

Da eseguire solo dopo aver scelto i parametri finali su validation.

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

```bash
python -m src.long_video.build_windows_from_store \
  --feature-store-dir data/features_long/psa_converted_0010_1010_exp46 \
  --output-dir outputs/long_video/psa_converted_0010_1010_exp46 \
  --window-sizes 1.0 1.5 2.0 3.0 \
  --stride-sec 0.25 \
  --overwrite
```

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

## 7. Quando rilanciare cosa

| Cosa cambi | Da quale script rilanciare |
|---|---|
| Solo soglie, smoothing, merge, NMS | `postprocess_events.py` + `render_preview.py` |
| Window sizes o stride | `build_windows_from_store.py` + successivi |
| `num_frames`, batch size inferenza o mapping input modello | `infer_exp46_from_store.py` + successivi |
| Feature FPS, segmento video, DINOv3, YOLO | `extract_feature_store.py` + tutti i successivi |
| Checkpoint modello | `infer_exp46_from_store.py` + successivi |

---

## 8. Note metodologiche

La pipeline long-video non valuta direttamente Accuracy/Macro F1 sulle clip, perché lavora su un segmento video continuo. Fino a quando non viene costruito un ground truth event-level per il segmento, la valutazione della pipeline è:

- quantitativa descrittiva: numero eventi, distribuzione classi, durata media eventi, numero finestre;
- qualitativa: controllo del video `preview_annotated.mp4`;
- comparativa: confronto tra varianti di post-processing sulla validation completa.

Il test finale deve essere eseguito una sola volta con la configurazione scelta sulla validation completa.
