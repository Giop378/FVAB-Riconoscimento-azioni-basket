# Pipeline pulita exp_46

Questo file raccoglie i comandi per rieseguire la versione pulita della pipeline `exp_46`

La configurazione finale rimane:

```text
L1: action_noaction   -> YOLO v2 + temp43
L2: shot_type_only    -> YOLO v2 + temp29
L3: shot_outcome_only -> YOLO v1 + temp43
```

La pipeline usa solo feature tracking temporali:

```text
tracking_sequences.npz
tracking_sequence_index.json
tracking_sequence_feature_names.json
```

Non vengono più usati:

```text
tracking_features.csv
tracking_feature_names.json
--tracking-features-csv
--l1-tracking-features-csv
--l2-tracking-features-csv
--l3-tracking-features-csv
```

I comandi sono scritti per PowerShell.

---

## 0. Variabili comuni

```powershell
$DATASET_ROOT = "data/datasets/dataset_basket_v1"
$MANIFEST = "data/datasets/dataset_basket_v1/manifest.csv"

$DINO_REPO = "third_party/dinov3"
$DINO_WEIGHTS = "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
$FEATURES_ROOT = "data/features/dinov3_vitl16_336"

$YOLO_V1 = "runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt"
$YOLO_V2 = "runs/detect/outputs/ball_rim_detector/yolo11m_1280_v2/weights/best.pt"

$TRACK_L1_V2_TEMP43 = "data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2_v4"
$TRACK_L2_V2_TEMP29 = "data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2"
$TRACK_L3_V1_TEMP43 = "data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1_v4"

$OUT_L1 = "outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean"
$OUT_L2 = "outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean"
$OUT_L3 = "outputs/exp_l3_yolo_v1_temp43_shots_d256_mean"
$OUT_EVAL_VAL = "outputs/exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43"
$OUT_EVAL_TEST = "outputs/exp_46_test"
```

---

## 1. Estrazione feature DINOv3 clip-level

Esegui questo comando solo se la cartella `$FEATURES_ROOT` non è già stata generata.

```powershell
python -m src.features.extract_features `
  --dataset-root $DATASET_ROOT `
  --manifest $MANIFEST `
  --output-dir $FEATURES_ROOT `
  --model-name dinov3_vitl16 `
  --weights $DINO_WEIGHTS `
  --repo-or-dir $DINO_REPO `
  --source local `
  --image-size 336 `
  --chunk-size 128 `
  --device 0 `
  --overwrite
```

Nota: nella pipeline pulita non vengono usate augmentation nella feature extraction DINOv3.

---

## 2. Estrazione tracking temporale YOLO v2 temp43 per L1

Questa estrazione serve per L1, cioè `passaggio / tiro / no-action`.

```powershell
python -m src.features.extract_ball_rim_tracking_features `
  --dataset-root $DATASET_ROOT `
  --manifest $MANIFEST `
  --yolo-weights $YOLO_V2 `
  --output-dir $TRACK_L1_V2_TEMP43 `
  --splits train val test `
  --num-frames 48 `
  --sample-mode uniform `
  --imgsz 1280 `
  --conf 0.10 `
  --iou 0.50 `
  --device 0 `
  --batch-size 16 `
  --temporal-feature-set temp43 `
  --rim-inside-margin 0.15 `
  --overwrite
```

Output principali attesi:

```text
$TRACK_L1_V2_TEMP43/tracking_sequences.npz
$TRACK_L1_V2_TEMP43/tracking_sequence_index.json
$TRACK_L1_V2_TEMP43/tracking_sequence_feature_names.json
$TRACK_L1_V2_TEMP43/processed_clips.csv
$TRACK_L1_V2_TEMP43/extract_tracking_results.txt
```

---

## 3. Estrazione tracking temporale YOLO v2 temp29 per L2

Questa estrazione serve per L2, cioè `tiroDaDue / tiroDaTre / tiroLibero`.

```powershell
python -m src.features.extract_ball_rim_tracking_features `
  --dataset-root $DATASET_ROOT `
  --manifest $MANIFEST `
  --yolo-weights $YOLO_V2 `
  --output-dir $TRACK_L2_V2_TEMP29 `
  --splits train val test `
  --num-frames 48 `
  --sample-mode uniform `
  --imgsz 1280 `
  --conf 0.10 `
  --iou 0.50 `
  --device 0 `
  --batch-size 16 `
  --temporal-feature-set temp29 `
  --rim-inside-margin 0.15 `
  --overwrite
```

Output principali attesi:

```text
$TRACK_L2_V2_TEMP29/tracking_sequences.npz
$TRACK_L2_V2_TEMP29/tracking_sequence_index.json
$TRACK_L2_V2_TEMP29/tracking_sequence_feature_names.json
$TRACK_L2_V2_TEMP29/processed_clips.csv
$TRACK_L2_V2_TEMP29/extract_tracking_results.txt
```

---

## 4. Estrazione tracking temporale YOLO v1 temp43 per L3

Questa estrazione serve per L3, cioè `tiro0 / tiro1`.

```powershell
python -m src.features.extract_ball_rim_tracking_features `
  --dataset-root $DATASET_ROOT `
  --manifest $MANIFEST `
  --yolo-weights $YOLO_V1 `
  --output-dir $TRACK_L3_V1_TEMP43 `
  --splits train val test `
  --num-frames 48 `
  --sample-mode uniform `
  --imgsz 1280 `
  --conf 0.10 `
  --iou 0.50 `
  --device 0 `
  --batch-size 16 `
  --temporal-feature-set temp43 `
  --rim-inside-margin 0.15 `
  --overwrite
```

Output principali attesi:

```text
$TRACK_L3_V1_TEMP43/tracking_sequences.npz
$TRACK_L3_V1_TEMP43/tracking_sequence_index.json
$TRACK_L3_V1_TEMP43/tracking_sequence_feature_names.json
$TRACK_L3_V1_TEMP43/processed_clips.csv
$TRACK_L3_V1_TEMP43/extract_tracking_results.txt
```

---

## 5. Controllo rapido degli output tracking

```powershell
Test-Path "$TRACK_L1_V2_TEMP43/tracking_sequences.npz"
Test-Path "$TRACK_L1_V2_TEMP43/tracking_sequence_index.json"

Test-Path "$TRACK_L2_V2_TEMP29/tracking_sequences.npz"
Test-Path "$TRACK_L2_V2_TEMP29/tracking_sequence_index.json"

Test-Path "$TRACK_L3_V1_TEMP43/tracking_sequences.npz"
Test-Path "$TRACK_L3_V1_TEMP43/tracking_sequence_index.json"
```

Tutti i comandi dovrebbero restituire `True`.

---

## 6. Training L1 - action_noaction con YOLO v2 temp43

```powershell
python -m src.training.train `
  --features-root $FEATURES_ROOT `
  --output-dir $OUT_L1 `
  --label-mode action_noaction `
  --tracking-sequences-npz "$TRACK_L1_V2_TEMP43/tracking_sequences.npz" `
  --tracking-sequence-index "$TRACK_L1_V2_TEMP43/tracking_sequence_index.json" `
  --epochs 100 `
  --batch-size 64 `
  --lr 5e-5 `
  --input-dim 1024 `
  --d-model 256 `
  --num-layers 2 `
  --num-heads 4 `
  --ff-dim 768 `
  --dropout 0.45 `
  --weight-decay 5e-3 `
  --pooling mean `
  --class-weight-power 0.5 `
  --sampler-power 0.5 `
  --tracking-missing-policy error `
  --num-workers 0 `
  --seed 42
```

Output principale:

```text
$OUT_L1/best_model.pt
```

---

## 7. Training L2 - shot_type_only con YOLO v2 temp29

```powershell
python -m src.training.train `
  --features-root $FEATURES_ROOT `
  --output-dir $OUT_L2 `
  --label-mode shot_type_only `
  --tracking-sequences-npz "$TRACK_L2_V2_TEMP29/tracking_sequences.npz" `
  --tracking-sequence-index "$TRACK_L2_V2_TEMP29/tracking_sequence_index.json" `
  --epochs 100 `
  --batch-size 64 `
  --lr 5e-5 `
  --input-dim 1024 `
  --d-model 256 `
  --num-layers 2 `
  --num-heads 4 `
  --ff-dim 768 `
  --dropout 0.45 `
  --weight-decay 5e-3 `
  --pooling mean `
  --class-weight-power 0.5 `
  --sampler-power 0.5 `
  --tracking-missing-policy error `
  --num-workers 0 `
  --seed 42
```

Output principale:

```text
$OUT_L2/best_model.pt
```

---

## 8. Training L3 - shot_outcome_only con YOLO v1 temp43

```powershell
python -m src.training.train `
  --features-root $FEATURES_ROOT `
  --output-dir $OUT_L3 `
  --label-mode shot_outcome_only `
  --tracking-sequences-npz "$TRACK_L3_V1_TEMP43/tracking_sequences.npz" `
  --tracking-sequence-index "$TRACK_L3_V1_TEMP43/tracking_sequence_index.json" `
  --epochs 100 `
  --batch-size 64 `
  --lr 5e-5 `
  --input-dim 1024 `
  --d-model 256 `
  --num-layers 2 `
  --num-heads 4 `
  --ff-dim 768 `
  --dropout 0.45 `
  --weight-decay 5e-3 `
  --pooling mean `
  --class-weight-power 0.5 `
  --sampler-power 0.5 `
  --tracking-missing-policy error `
  --num-workers 0 `
  --seed 42
```

Output principale:

```text
$OUT_L3/best_model.pt
```

---

## 9. Valutazione gerarchica exp_46 su validation

```powershell
python -m src.evaluation.evaluate_hierarchical `
  --features-root $FEATURES_ROOT `
  --split val `
  --batch-size 64 `
  --num-workers 2 `
  --output-dir $OUT_EVAL_VAL `
  --l1-checkpoint "$OUT_L1/best_model.pt" `
  --l2-checkpoint "$OUT_L2/best_model.pt" `
  --l3-checkpoint "$OUT_L3/best_model.pt" `
  --l1-tracking-sequences-npz "$TRACK_L1_V2_TEMP43/tracking_sequences.npz" `
  --l1-tracking-sequence-index "$TRACK_L1_V2_TEMP43/tracking_sequence_index.json" `
  --l2-tracking-sequences-npz "$TRACK_L2_V2_TEMP29/tracking_sequences.npz" `
  --l2-tracking-sequence-index "$TRACK_L2_V2_TEMP29/tracking_sequence_index.json" `
  --l3-tracking-sequences-npz "$TRACK_L3_V1_TEMP43/tracking_sequences.npz" `
  --l3-tracking-sequence-index "$TRACK_L3_V1_TEMP43/tracking_sequence_index.json" `
  --tracking-missing-policy error
```

Output principali:

```text
$OUT_EVAL_VAL/results.txt
$OUT_EVAL_VAL/predictions.csv
```

---

## 10. Valutazione gerarchica exp_46 su test clip-level

Esegui questo comando se vuoi valutare la gerarchia anche sullo split `test` del dataset clip-level.

```powershell
python -m src.evaluation.evaluate_hierarchical `
  --features-root $FEATURES_ROOT `
  --split test `
  --batch-size 64 `
  --num-workers 2 `
  --output-dir $OUT_EVAL_TEST `
  --l1-checkpoint "$OUT_L1/best_model.pt" `
  --l2-checkpoint "$OUT_L2/best_model.pt" `
  --l3-checkpoint "$OUT_L3/best_model.pt" `
  --l1-tracking-sequences-npz "$TRACK_L1_V2_TEMP43/tracking_sequences.npz" `
  --l1-tracking-sequence-index "$TRACK_L1_V2_TEMP43/tracking_sequence_index.json" `
  --l2-tracking-sequences-npz "$TRACK_L2_V2_TEMP29/tracking_sequences.npz" `
  --l2-tracking-sequence-index "$TRACK_L2_V2_TEMP29/tracking_sequence_index.json" `
  --l3-tracking-sequences-npz "$TRACK_L3_V1_TEMP43/tracking_sequences.npz" `
  --l3-tracking-sequence-index "$TRACK_L3_V1_TEMP43/tracking_sequence_index.json" `
  --tracking-missing-policy error
```

Output principali:

```text
$OUT_EVAL_TEST/results.txt
$OUT_EVAL_TEST/predictions.csv
```

---

## 11. Valutazione usando i checkpoint storici di exp_46

Se non vuoi riaddestrare i tre livelli e vuoi solo rieseguire la valutazione con i checkpoint già ottenuti in passato, usa questo comando. Le cartelle tracking indicate sono quelle storiche.

```powershell
python -m src.evaluation.evaluate_hierarchical `
  --features-root data/features/dinov3_vitl16_336 `
  --split val `
  --batch-size 64 `
  --num-workers 2 `
  --output-dir outputs/exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43 `
  --l1-checkpoint outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt `
  --l2-checkpoint outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt `
  --l3-checkpoint outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt `
  --l1-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2_v4/tracking_sequences.npz `
  --l1-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2_v4/tracking_sequence_index.json `
  --l2-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequences.npz `
  --l2-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequence_index.json `
  --l3-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1_v4/tracking_sequences.npz `
  --l3-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1_v4/tracking_sequence_index.json `
  --tracking-missing-policy error
```

---

## Note finali

- I comandi storici degli esperimenti precedenti possono rimanere invariati nei vecchi markdown.
- Questo file descrive solo la pipeline pulita, cioè senza feature aggregate CSV.
- L'argomento `--save-temporal-sequences` non è più necessario: le sequenze temporali vengono sempre salvate da `extract_ball_rim_tracking_features.py`.
- Usare `--overwrite` nelle estrazioni tracking rigenera gli output temporali e rimuove eventuali vecchi file aggregate rimasti nella stessa cartella.
