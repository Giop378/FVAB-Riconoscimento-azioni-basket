# Esperimenti Pipeline Long-Video

Questo file tiene traccia degli esperimenti effettuati sulla pipeline che applica il modello clip-level `exp_46` a video lunghi.

La pipeline non modifica il modello `exp_46`: il modello resta congelato. Gli esperimenti riguardano invece:

- estrazione della feature store temporale;
- fps di campionamento delle feature;
- dimensioni delle finestre temporali;
- stride;
- inferenza multi-scala;
- soglie di confidenza;
- smoothing;
- merge temporale;
- NMS temporale;
- qualità del video preview annotato.

---

## 1. Modello congelato

Modello di riferimento: `exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43`.

| Livello | Checkpoint | Tracking | Feature |
|---|---|---|---|
| L1 | `outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt` | YOLO v2 | `temp43` |
| L2 | `outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt` | YOLO v2 | `temp29` |
| L3 | `outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt` | YOLO v1 | `temp43` |

Classi finali:

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

`no-action` viene usata come background e viene esclusa dal report finale delle azioni rilevate.

---

## 2. Segmenti video

| ID segmento | Split/Uso | Video | Start | End | Durata | Feature store | Output dir |
|---|---|---|---:|---:|---:|---|---|
| `primaparte_0215_0245_exp46_debug` | validation debug | `data/datasets/dataset_basket_v1/videos/PrimaParte.mp4` | 135 | 165 | 30s | `data/features_long/primaparte_0215_0245_exp46_debug` | `outputs/long_video/primaparte_0215_0245_exp46_debug` |
| `primaparte_0215_1215_exp46` | validation completa | `data/datasets/dataset_basket_v1/videos/PrimaParte.mp4` | 135 | 735 | 600s | `data/features_long/primaparte_0215_1215_exp46` | `outputs/long_video/primaparte_0215_1215_exp46` |
| `psa_converted_0010_1010_exp46` | test finale | `data/datasets/dataset_basket_v1/videos/PSA_converted.mp4` | 10 | 610 | 600s | `data/features_long/psa_converted_0010_1010_exp46` | `outputs/long_video/psa_converted_0010_1010_exp46` |

---

## 3. Tabella riassuntiva esperimenti

| ID | Segmento | Feature FPS | Window sizes | Stride | Post-processing | Eventi finali | Preview | Decisione |
|---|---|---:|---|---:|---|---:|---|---|
| `lv_00_setup` | - | - | - | - | - | - | - | TBD |
| `lv_01_debug_base` | val debug | 24 | 1.0, 1.5, 2.0, 3.0 | 0.25 | base | TBD | TBD | TBD |
| `lv_02_val_base` | val completa | 24 | 1.0, 1.5, 2.0, 3.0 | 0.25 | base | TBD | TBD | TBD |
| `lv_03_val_tuned` | val completa | TBD | TBD | TBD | tuned | TBD | TBD | TBD |
| `lv_04_test_final` | test finale | come migliore validation | come migliore validation | come migliore validation | come migliore validation | TBD | TBD | TBD |

---

## 4. Parametri base della pipeline

Questi sono i parametri iniziali usati come baseline.

| Parametro | Valore |
|---|---:|
| `feature_fps` | 24 |
| `window_sizes` | `1.0 1.5 2.0 3.0` |
| `stride_sec` | 0.25 |
| `num_frames` inferenza | 48 |
| `batch_size` inferenza | 128 |
| `smooth_window` | 3 |
| `min_event_duration_sec` | 0.4 |
| `merge_gap_sec` | 0.75 |
| `temporal_nms_iou` | 0.50 |
| `min_conf_passaggio` | 0.55 |
| `min_conf_tiro` | 0.40 |

---

# 5. Dettaglio esperimenti

---

## `lv_00_setup` - Controllo setup pipeline

### Obiettivo

Verificare che tutti i path e i checkpoint necessari per la pipeline long-video siano disponibili e coerenti.

### Comando

```bash
python -m src.long_video.check_setup --check-video-duration --print-features
```

### Controlli attesi

| Controllo | Esito |
|---|---|
| `PrimaParte.mp4` esiste | TBD |
| `PSA_converted.mp4` esiste | TBD |
| DINOv3 repo esiste | TBD |
| YOLO v1 weights esistono | TBD |
| YOLO v2 weights esistono | TBD |
| Checkpoint L1 esiste | TBD |
| Checkpoint L2 esiste | TBD |
| Checkpoint L3 esiste | TBD |
| L1 ha 43 feature tracking | TBD |
| L2 ha 29 feature tracking | TBD |
| L3 ha 43 feature tracking | TBD |

### Output rilevante

```text
TBD
```

### Problemi osservati

- TBD

### Decisione

- TBD

---

## `lv_01_debug_base` - Primo test end-to-end su validation debug

### Obiettivo

Verificare che tutti gli script della pipeline funzionino sul segmento breve di validation.

### Segmento

| Campo | Valore |
|---|---|
| Video | `data/datasets/dataset_basket_v1/videos/PrimaParte.mp4` |
| Start | 135s |
| End | 165s |
| Durata | 30s |
| Feature store | `data/features_long/primaparte_0215_0245_exp46_debug` |
| Output dir | `outputs/long_video/primaparte_0215_0245_exp46_debug` |

### Parametri

| Parametro | Valore |
|---|---:|
| `feature_fps` | 24 |
| `window_sizes` | `1.0 1.5 2.0 3.0` |
| `stride_sec` | 0.25 |
| `num_frames` inferenza | 48 |
| `batch_size` inferenza | 128 |
| `smooth_window` | 3 |
| `min_event_duration_sec` | 0.4 |
| `merge_gap_sec` | 0.75 |
| `temporal_nms_iou` | 0.50 |
| `min_conf_passaggio` | 0.55 |
| `min_conf_tiro` | 0.40 |

### Comandi

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

```bash
python -m src.long_video.build_windows_from_store \
  --feature-store-dir data/features_long/primaparte_0215_0245_exp46_debug \
  --output-dir outputs/long_video/primaparte_0215_0245_exp46_debug \
  --window-sizes 1.0 1.5 2.0 3.0 \
  --stride-sec 0.25 \
  --overwrite
```

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

```bash
python -m src.long_video.render_preview \
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 \
  --events-csv outputs/long_video/primaparte_0215_0245_exp46_debug/events_postprocessed.csv \
  --output-video outputs/long_video/primaparte_0215_0245_exp46_debug/preview_annotated.mp4 \
  --start-sec 135 \
  --end-sec 165 \
  --overwrite
```

### Output prodotti

| File | Stato |
|---|---|
| `metadata.json` | TBD |
| `timestamps.npy` | TBD |
| `frame_indices.npy` | TBD |
| `dinov3_features.npy` | TBD |
| `yolo_v1_primitives.npz` | TBD |
| `yolo_v2_primitives.npz` | TBD |
| `windows_manifest.csv` | TBD |
| `window_predictions_raw.csv` | TBD |
| `events_raw.csv` | TBD |
| `events_postprocessed.csv` | TBD |
| `annotations.json` | TBD |
| `preview_annotated.mp4` | TBD |

### Risultati quantitativi

| Metrica | Valore |
|---|---:|
| Numero timestamp feature store | TBD |
| Numero finestre | TBD |
| Eventi raw | TBD |
| Eventi post-processati | TBD |
| Passaggi rilevati | TBD |
| Tiri da 2 rilevati | TBD |
| Tiri da 3 rilevati | TBD |
| Tiri liberi rilevati | TBD |
| Durata media eventi | TBD |
| Tempo estrazione feature | TBD |
| Tempo inferenza | TBD |

### Valutazione qualitativa preview

- TBD

### Problemi osservati

- TBD

### Decisione

- TBD

---

## `lv_02_val_base` - Baseline su validation completa 10 minuti

### Obiettivo

Eseguire la pipeline base sul segmento completo di validation per valutare qualitativamente e quantitativamente il comportamento su 10 minuti ricchi di azioni.

### Segmento

| Campo | Valore |
|---|---|
| Video | `data/datasets/dataset_basket_v1/videos/PrimaParte.mp4` |
| Start | 135s |
| End | 735s |
| Durata | 600s |
| Feature store | `data/features_long/primaparte_0215_1215_exp46` |
| Output dir | `outputs/long_video/primaparte_0215_1215_exp46` |

### Parametri

| Parametro | Valore |
|---|---:|
| `feature_fps` | 24 |
| `window_sizes` | `1.0 1.5 2.0 3.0` |
| `stride_sec` | 0.25 |
| `num_frames` inferenza | 48 |
| `batch_size` inferenza | 128 |
| `smooth_window` | 3 |
| `min_event_duration_sec` | 0.4 |
| `merge_gap_sec` | 0.75 |
| `temporal_nms_iou` | 0.50 |
| `min_conf_passaggio` | 0.55 |
| `min_conf_tiro` | 0.40 |

### Comandi

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

### Output prodotti

| File | Stato |
|---|---|
| `metadata.json` | TBD |
| `dinov3_features.npy` | TBD |
| `yolo_v1_primitives.npz` | TBD |
| `yolo_v2_primitives.npz` | TBD |
| `windows_manifest.csv` | TBD |
| `window_predictions_raw.csv` | TBD |
| `events_postprocessed.csv` | TBD |
| `annotations.json` | TBD |
| `preview_annotated.mp4` | TBD |

### Risultati quantitativi

| Metrica | Valore |
|---|---:|
| Numero timestamp feature store | TBD |
| Numero finestre | TBD |
| Eventi raw | TBD |
| Eventi post-processati | TBD |
| Passaggi rilevati | TBD |
| Tiri da 2 rilevati | TBD |
| Tiri da 3 rilevati | TBD |
| Tiri liberi rilevati | TBD |
| Durata media eventi | TBD |
| Tempo estrazione feature | TBD |
| Tempo inferenza | TBD |

### Valutazione qualitativa preview

- TBD

### Problemi osservati

- TBD

### Decisione

- TBD

---

## `lv_03_val_tuned` - Tuning post-processing su validation completa

### Obiettivo

Provare varianti di post-processing sulla validation completa senza rieseguire feature extraction e, se possibile, senza rieseguire inferenza.

Varianti possibili:

- soglia `min_conf_passaggio`;
- soglia `min_conf_tiro`;
- `smooth_window`;
- `merge_gap_sec`;
- `temporal_nms_iou`;
- opzione `--require-action-gt-noaction`.

### Segmento

| Campo | Valore |
|---|---|
| Video | `data/datasets/dataset_basket_v1/videos/PrimaParte.mp4` |
| Start | 135s |
| End | 735s |
| Durata | 600s |
| Feature store | `data/features_long/primaparte_0215_1215_exp46` |
| Output dir | `outputs/long_video/primaparte_0215_1215_exp46` oppure nuova cartella dedicata |

### Variante testata

| Parametro | Valore |
|---|---:|
| `smooth_window` | TBD |
| `min_event_duration_sec` | TBD |
| `merge_gap_sec` | TBD |
| `temporal_nms_iou` | TBD |
| `min_conf_passaggio` | TBD |
| `min_conf_tiro` | TBD |
| `require_action_gt_noaction` | TBD |

### Comando post-processing

```bash
python -m src.long_video.postprocess_events \
  --predictions-csv outputs/long_video/primaparte_0215_1215_exp46/window_predictions_raw.csv \
  --output-dir outputs/long_video/primaparte_0215_1215_exp46 \
  --smooth-window TBD \
  --min-event-duration-sec TBD \
  --merge-gap-sec TBD \
  --temporal-nms-iou TBD \
  --min-conf-passaggio TBD \
  --min-conf-tiro TBD \
  --overwrite
```

### Comando preview

```bash
python -m src.long_video.render_preview \
  --input-video data/datasets/dataset_basket_v1/videos/PrimaParte.mp4 \
  --events-csv outputs/long_video/primaparte_0215_1215_exp46/events_postprocessed.csv \
  --output-video outputs/long_video/primaparte_0215_1215_exp46/preview_annotated.mp4 \
  --start-sec 135 \
  --end-sec 735 \
  --overwrite
```

### Risultati quantitativi

| Metrica | Valore |
|---|---:|
| Eventi raw | TBD |
| Eventi post-processati | TBD |
| Passaggi rilevati | TBD |
| Tiri da 2 rilevati | TBD |
| Tiri da 3 rilevati | TBD |
| Tiri liberi rilevati | TBD |
| Durata media eventi | TBD |
| Falsi positivi evidenti in preview | TBD |
| Azioni mancanti evidenti in preview | TBD |

### Valutazione qualitativa preview

- TBD

### Problemi osservati

- TBD

### Decisione

- TBD

---

## `lv_04_test_final` - Run finale su test 10 minuti

### Obiettivo

Applicare al segmento test la configurazione scelta sulla validation completa, senza ulteriore tuning.

### Segmento

| Campo | Valore |
|---|---|
| Video | `data/datasets/dataset_basket_v1/videos/PSA_converted.mp4` |
| Start | 10s |
| End | 610s |
| Durata | 600s |
| Feature store | `data/features_long/psa_converted_0010_1010_exp46` |
| Output dir | `outputs/long_video/psa_converted_0010_1010_exp46` |

### Parametri finali scelti su validation

| Parametro | Valore |
|---|---:|
| `feature_fps` | TBD |
| `window_sizes` | TBD |
| `stride_sec` | TBD |
| `num_frames` inferenza | TBD |
| `batch_size` inferenza | TBD |
| `smooth_window` | TBD |
| `min_event_duration_sec` | TBD |
| `merge_gap_sec` | TBD |
| `temporal_nms_iou` | TBD |
| `min_conf_passaggio` | TBD |
| `min_conf_tiro` | TBD |

### Comandi

```bash
python -m src.long_video.extract_feature_store \
  --input-video data/datasets/dataset_basket_v1/videos/PSA_converted.mp4 \
  --start-sec 10 \
  --end-sec 610 \
  --output-dir data/features_long/psa_converted_0010_1010_exp46 \
  --feature-fps TBD \
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
  --window-sizes TBD \
  --stride-sec TBD \
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
  --smooth-window TBD \
  --min-event-duration-sec TBD \
  --merge-gap-sec TBD \
  --temporal-nms-iou TBD \
  --min-conf-passaggio TBD \
  --min-conf-tiro TBD \
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

### Output prodotti

| File | Stato |
|---|---|
| `metadata.json` | TBD |
| `dinov3_features.npy` | TBD |
| `yolo_v1_primitives.npz` | TBD |
| `yolo_v2_primitives.npz` | TBD |
| `windows_manifest.csv` | TBD |
| `window_predictions_raw.csv` | TBD |
| `events_postprocessed.csv` | TBD |
| `annotations.json` | TBD |
| `preview_annotated.mp4` | TBD |

### Risultati quantitativi

| Metrica | Valore |
|---|---:|
| Numero timestamp feature store | TBD |
| Numero finestre | TBD |
| Eventi raw | TBD |
| Eventi post-processati | TBD |
| Passaggi rilevati | TBD |
| Tiri da 2 rilevati | TBD |
| Tiri da 3 rilevati | TBD |
| Tiri liberi rilevati | TBD |
| Durata media eventi | TBD |
| Tempo estrazione feature | TBD |
| Tempo inferenza | TBD |

### Valutazione qualitativa preview

- TBD

### Problemi osservati

- TBD

### Decisione finale

- TBD

---

# 6. Come aggiornare questo file

Per ogni nuovo esperimento:

1. aggiungere una riga nella tabella riassuntiva;
2. copiare il template di dettaglio;
3. incollare i comandi effettivamente eseguiti;
4. riportare i path degli output;
5. riportare conteggi e distribuzione degli eventi;
6. aggiungere note qualitative dopo aver visto il preview;
7. indicare una decisione finale.

---

# 7. Metriche/indicatori da riportare

Fino a quando non viene creato un ground truth event-level per i 10 minuti, riportare almeno:

| Indicatore | Perché serve |
|---|---|
| Numero finestre | Verifica della copertura temporale. |
| Eventi raw | Misura quanto è rumorosa la classificazione prima del merge. |
| Eventi post-processati | Misura il risultato finale della pipeline. |
| Distribuzione eventi per classe | Individua sbilanciamenti o classi mai rilevate. |
| Durata media eventi | Evidenzia eventi troppo corti/lunghi. |
| Preview annotata | Controllo qualitativo principale. |
| Falsi positivi evidenti | Serve per tarare soglie e merge. |
| Azioni mancanti evidenti | Serve per capire se le soglie sono troppo severe. |

---

# 8. Regola per il test finale

Il segmento test `PSA_converted.mp4` da 10s a 610s deve essere usato solo dopo aver scelto la configurazione finale su validation.

Dopo il test finale non bisogna modificare soglie o parametri sulla base del risultato del test, altrimenti si rischia di trasformare il test in validation.
