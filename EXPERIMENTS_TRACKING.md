# Esperimenti Tracking Palla/Canestro - Stadio 3, Gerarchia e Modello Non Gerarchico

Questo file tiene traccia sintetica degli esperimenti effettuati introducendo feature di **tracking palla/canestro** nella pipeline di action recognition per il basket, sia nella versione gerarchica sia nel successivo modello non gerarchico.

L'obiettivo principale di questi esperimenti è verificare se informazioni esplicite sulla posizione della **palla** e del **canestro** possano migliorare la distinzione tra tiro sbagliato e tiro segnato e, più in generale, aumentare la qualità della classificazione finale delle azioni.

La gerarchia di riferimento resta strutturata in tre stadi:

```text
Stadio 1:
passaggio / tiro / no-action

Stadio 2, solo se Stadio 1 = tiro:
tiroDaDue / tiroDaTre / tiroLibero

Stadio 3, solo se Stadio 1 = tiro:
tiro0 / tiro1
```

In una prima fase gli Stadi 1 e 2 restano invariati, mentre viene modificato solo lo Stadio 3 aggiungendo feature di tracking estratte con un detector YOLO addestrato sulle classi `ball` e `rim`. In una fase successiva viene testato anche un modello non gerarchico a 8 classi, con `idle` e `non-gioco` uniti in `no-action`, usando le stesse informazioni di tracking tramite late fusion.

## Pipeline tracking palla/canestro

La pipeline introdotta è composta da tre passaggi:

1. Addestramento di un detector YOLO per localizzare `ball` e `rim` nei frame annotati manualmente.
2. Estrazione di feature numeriche di tracking per ogni clip di tiro del dataset, usando il detector YOLO addestrato.
3. Uso delle feature tracking insieme alle feature video DINOv3-L/16: inizialmente mediante concatenazione nello Stadio 3 della gerarchia, poi tramite late fusion nel modello non gerarchico.

Le feature tracking vengono salvate in due versioni:

```text
data/features/ball_rim_tracking_features_v1/
├── tracking_features.csv
├── tracking_feature_names.json
├── per_frame_detections.csv
└── extract_tracking_results.txt
```

Questa prima versione contiene feature tracking estratte sulle clip di tiro ed è stata usata nello Stadio 3 della gerarchia.

```text
data/features/ball_rim_tracking_all_train_val/
├── tracking_features.csv
├── tracking_feature_names.json
└── errors.csv, se presenti errori di estrazione
```

Questa seconda versione contiene feature tracking estratte su tutte le clip di training e validation, incluse `passaggio`, `idle` e `non-gioco`, ed è stata usata nel modello non gerarchico `exp_39`.

```text
data/features/ball_rim_tracking_temporal_v1/
├── tracking_sequences.npz
├── tracking_sequence_index.json
├── tracking_sequence_feature_names.json
└── extract_tracking_results.txt
```

Questa terza versione contiene sequenze temporali di tracking palla/canestro estratte sulle clip di tiro. A differenza delle feature aggregate, ogni clip è rappresentata da una sequenza per-frame allineabile alle feature video DINOv3 ed è stata usata nel nuovo Stadio 3 temporale `exp_l3_tracking_temporal_v1`.

Nel primo esperimento di tracking sono state usate **39 feature** palla/canestro, normalizzate sul training set e concatenate a ogni timestep delle feature video. L'input dello Stadio 3 passa quindi da `1024` a `1063` dimensioni:

```text
Input dim feature video: 1024
Input dim feature tracking: 39
Input dim totale modello: 1063
```

Nel successivo esperimento di tracking temporale vengono invece usate **29 feature per frame**, normalizzate sul training set e concatenate alle feature DINOv3 timestep per timestep. In questo caso l'input dello Stadio 3 passa da `1024` a `1053` dimensioni:

```text
Input dim feature video: 1024
Input dim feature tracking temporali: 29
Input dim totale modello: 1053
```

Nel modello non gerarchico `exp_39`, invece, le feature DINOv3 restano a `1024` dimensioni e le 39 feature tracking vengono elaborate da una piccola MLP separata da 64 dimensioni. La fusione avviene dopo il pooling temporale del Transformer:

```text
Embedding video: 256
Embedding tracking: 64
Input classificatore finale: 320
```

## Detector YOLO palla/canestro

Il detector usato per estrarre le feature tracking è stato addestrato con YOLO su frame annotati manualmente in CVAT con due classi:

```text
0: ball
1: rim
```

Il modello usato per le feature tracking è:

```text
runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt
```

Risultato finale del detector su validation interna YOLO:

|Classe|Precision|Recall|mAP50|mAP50-95|
|-|-:|-:|-:|-:|
|all|0.951|0.939|0.972|0.528|
|ball|0.937|0.920|0.961|0.540|
|rim|0.964|0.957|0.982|0.515|

Il risultato è adeguato per usare il detector come generatore di feature: il recall della palla è alto (`0.920`) e il mAP50 complessivo è molto elevato (`0.972`). Il valore più basso di mAP50-95 è atteso, perché la palla è piccola e piccoli spostamenti del bounding box incidono molto sulle soglie IoU più severe.

## Tabella riassuntiva

|ID|Modello|Feature extractor / tracking|Label mode / valutazione|Classi|Epoche|Batch size|LR|d_model|Layers|Heads|FF dim|Dropout|Weight decay|Pooling|Class weight power|Sampler|Val Loss|Val Accuracy|Val Macro F1|Val Weighted F1|Output dir|
|-|-|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-|-:|-|-:|-:|-:|-:|-|
|exp_l3_tracking_v1|Temporal Transformer d256|DINOv3-L/16 frozen + YOLO ball/rim tracking|shot_outcome_only|2|50|64|5e-5|256|2|4|768|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.5|0.4656|0.8154|0.8132|0.8179|outputs/exp_l3_tracking_v1|
|exp_l3_tracking_v2_d384_sampler04|Temporal Transformer d384|DINOv3-L/16 frozen + YOLO ball/rim tracking|shot_outcome_only|2|50|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.4559|0.7846|0.7842|0.7865|outputs/exp_l3_tracking_v2_d384_sampler04|
|exp_l3_tracking_temporal_v1|Temporal Transformer d256|DINOv3-L/16 frozen + YOLO ball/rim tracking temporale|shot_outcome_only|2|100|64|5e-5|256|2|4|768|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.5|0.4576|0.8769|0.8733|0.8782|outputs/exp_l3_tracking_temporal_v1|
|exp_38|Gerarchia L1+L2+L3 tracking|DINOv3-L/16 frozen + YOLO ball/rim tracking nello Stadio 3|hierarchical end-to-end|8|-|64|-|misto|misto|misto|misto|misto|misto|Mean|-|-|-|0.8759|0.6996|0.8766|outputs/exp_38_dinov3_vitl16_hierarchical_tracking_l3|
|exp_40|Gerarchia L1+L2+L3 tracking temporale|DINOv3-L/16 frozen + YOLO ball/rim tracking temporale nello Stadio 3|hierarchical end-to-end|8|-|64|-|misto|misto|misto|misto|misto|misto|Mean|-|-|-|0.8815|0.7223|0.8808|outputs/exp_40_dinov3_hierarchical_tracking_temporal_l3|
|exp_39|Temporal Transformer d256 + late fusion tracking|DINOv3-L/16 frozen + YOLO ball/rim tracking su tutte le clip train/val|non-hierarchical, idle/non-gioco -> no-action|8|40|64|5e-5|256|2|4|768|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.5|0.5429|0.8426|0.5620|0.8447|outputs/exp_39_nonhierarchical_dinov3_tracking_noaction|

## Risultati aggregati su validation

|ID|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|
|exp_l3_tracking_v1|0.8154|0.82|0.83|0.8132|0.84|0.82|0.8179|
|exp_l3_tracking_v2_d384_sampler04|0.7846|0.81|0.82|0.7842|0.84|0.78|0.7865|
|exp_l3_tracking_temporal_v1|0.8769|0.87|0.89|0.8733|0.89|0.88|0.8782|
|exp_38|0.8759|0.74|0.72|0.6996|0.89|0.88|0.8766|
|exp_40|0.8815|0.74|0.73|0.7223|0.89|0.88|0.8808|
|exp_39|0.8426|0.61|0.57|0.5620|0.86|0.84|0.8447|

## Risultati aggregati sulle classi rilevanti

Questa tabella riporta le metriche calcolate sulle classi più rilevanti per ciascun esperimento:

- **exp_l3_tracking_v1**: solo `tiro0` e `tiro1`, con feature tracking palla/canestro concatenate alle feature DINOv3.
- **exp_l3_tracking_v2_d384_sampler04**: solo `tiro0` e `tiro1`, con feature tracking palla/canestro e configurazione più grande ispirata a `exp_31` / `exp_30` (`d_model=384`, `num_heads=6`, `ff_dim=1024`, `sampler_power=0.4`).
- **exp_l3_tracking_temporal_v1**: solo `tiro0` e `tiro1`, con sequenze temporali palla/canestro concatenate alle feature DINOv3 frame per frame.
- **exp_38**: solo le 7 azioni finali prodotte dalla nuova gerarchia end-to-end con Stadio 3 tracking.
- **exp_38 collassato**: valutazione della nuova gerarchia collassando l'esito del tiro, cioè considerando solo il tipo di tiro.
- **exp_40**: solo le 7 azioni finali prodotte dalla gerarchia end-to-end con Stadio 3 basato su tracking temporale.
- **exp_40 collassato**: valutazione della gerarchia con tracking temporale collassando l'esito del tiro, cioè considerando solo il tipo di tiro.
- **exp_39**: solo le 7 azioni reali del modello non gerarchico a 8 classi con `idle` e `non-gioco` uniti in `no-action`.

|ID|Classi considerate|Micro Precision|Micro Recall|Micro F1|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|exp_l3_tracking_v1|tiro0, tiro1|0.82|0.82|0.82|0.82|0.83|0.81|0.84|0.82|0.82|
|exp_l3_tracking_v2_d384_sampler04|tiro0, tiro1|0.78|0.78|0.78|0.81|0.82|0.78|0.84|0.78|0.79|
|exp_l3_tracking_temporal_v1|tiro0, tiro1|0.88|0.88|0.88|0.87|0.89|0.87|0.89|0.88|0.88|
|exp_38|7 azioni finali|0.82|0.89|0.85|0.71|0.70|0.67|0.84|0.89|0.85|
|exp_38 collassato|tipo azione senza esito|0.89|0.89|0.89|0.84|0.81|0.82|0.90|0.89|0.89|
|exp_40|7 azioni finali|0.83|0.90|0.86|0.71|0.71|0.70|0.83|0.90|0.86|
|exp_40 collassato|tipo azione senza esito|0.89|0.89|0.89|0.84|0.81|0.82|0.90|0.89|0.89|
|exp_39|7 azioni finali|0.76|0.86|0.80|0.56|0.54|0.52|0.77|0.86|0.80|

## Risultati per classe su validation

### exp_l3_tracking_v1 - DINOv3-L/16 frozen + YOLO ball/rim tracking, shot_outcome_only

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.94|0.75|0.83|40|
|tiro1|0.70|0.92|0.79|25|

Confusion matrix:

```text
[[30 10]
 [ 2 23]]
```

### exp_l3_tracking_v2_d384_sampler04 - DINOv3-L/16 frozen + YOLO ball/rim tracking, shot_outcome_only

Questo esperimento mantiene il tracking palla/canestro sullo Stadio 3, ma usa una configurazione più grande e più vicina agli esperimenti `exp_30` / `exp_31`: `d_model=384`, `num_heads=6`, `ff_dim=1024` e `sampler_power=0.4`. Il miglior checkpoint viene salvato alla quinta epoca.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.96|0.68|0.79|40|
|tiro1|0.65|0.96|0.77|25|

Confusion matrix:

```text
[[27 13]
 [ 1 24]]
```

Rispetto a `exp_l3_tracking_v1`, questa configurazione peggiora leggermente la Macro F1 (`0.7842` contro `0.8132`) e aumenta i falsi positivi su `tiro1`: i `tiro0` predetti come `tiro1` passano da 10 a 13. Il modello più grande recupera quasi tutti i tiri segnati, ma diventa meno conservativo sugli sbagliati.

### exp_l3_tracking_temporal_v1 - DINOv3-L/16 frozen + YOLO ball/rim tracking temporale, shot_outcome_only

Questo esperimento sostituisce le 39 feature aggregate con sequenze temporali di 29 feature per frame, concatenate alle feature DINOv3 a ogni timestep. Il modello resta un Temporal Transformer d256, ma l'input dello Stadio 3 passa da `1024` a `1053` dimensioni. Il miglior checkpoint viene salvato all'epoca 48.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.94|0.85|0.89|40|
|tiro1|0.79|0.92|0.85|25|

Confusion matrix:

```text
[[34  6]
 [ 2 23]]
```

Rispetto a `exp_l3_tracking_v1`, il tracking temporale migliora sia la Macro F1 (`0.8733` contro `0.8132`) sia l'equilibrio tra le due classi. In particolare, i falsi `tiro1` diminuiscono da 10 a 6, mantenendo invariati i falsi `tiro0` su `tiro1` reali.

### Estrazione sequenze temporali tracking palla/canestro

```bash
python -m src.features.extract_ball_rim_tracking_features \
  --dataset-root data/datasets/dataset_basket_v1 \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --yolo-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt \
  --output-dir data/features/ball_rim_tracking_temporal_v1 \
  --splits train val \
  --num-frames 48 \
  --sample-mode uniform \
  --imgsz 1280 \
  --conf 0.10 \
  --batch-size 16 \
  --device 0 \
  --save-temporal-sequences \
  --overwrite
```

### exp_l3_tracking_v2_d384_sampler04 - Training Stadio 3 con tracking e parametri d384

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --output-dir outputs/exp_l3_tracking_v2_d384_sampler04 \
  --label-mode shot_outcome_only \
  --tracking-features-csv data/features/ball_rim_tracking_features_v1/tracking_features.csv \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 384 \
  --num-layers 2 \
  --num-heads 6 \
  --ff-dim 1024 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --num-workers 2 \
  --seed 42
```

### exp_l3_tracking_temporal_v1 - Training Stadio 3 con tracking temporale

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --output-dir outputs/exp_l3_tracking_temporal_v1 \
  --label-mode shot_outcome_only \
  --tracking-sequences-npz data/features/ball_rim_tracking_temporal_v1/tracking_sequences.npz \
  --tracking-sequence-index data/features/ball_rim_tracking_temporal_v1/tracking_sequence_index.json \
  --epochs 100 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.5 \
  --num-workers 0 \
  --seed 42
```

### exp_38 - Gerarchia end-to-end con Stadio 3 tracking, 8 classi finali con no-action

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.92|0.52|0.67|21|
|tiroDaDue1|0.43|0.82|0.56|11|
|tiroDaTre0|0.75|0.75|0.75|12|
|tiroDaTre1|0.33|0.67|0.44|3|
|tiroLibero0|0.83|0.71|0.77|7|
|tiroLibero1|0.83|0.45|0.59|11|
|no-action|0.95|0.86|0.90|263|

Confusion matrix:

```text
[[205   0   1   0   0   0   0   6]
 [  0  11   6   2   1   0   0   1]
 [  0   0   9   1   1   0   0   0]
 [  0   1   0   9   1   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   5   1   1]
 [  0   0   1   0   1   0   5   4]
 [ 32   0   3   0   0   1   0 227]]
```

### exp_38 - solo 7 azioni finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.92|0.52|0.67|21|
|tiroDaDue1|0.43|0.82|0.56|11|
|tiroDaTre0|0.75|0.75|0.75|12|
|tiroDaTre1|0.33|0.67|0.44|3|
|tiroLibero0|0.83|0.71|0.77|7|
|tiroLibero1|0.83|0.45|0.59|11|

Metriche aggregate sulle sole 7 azioni:

|Metrica|Valore|
|-|-:|
|Micro F1|0.85|
|Macro F1|0.67|
|Weighted F1|0.85|

Confusion matrix:

```text
[[205   0   1   0   0   0   0]
 [  0  11   6   2   1   0   0]
 [  0   0   9   1   1   0   0]
 [  0   1   0   9   1   0   0]
 [  0   0   1   0   2   0   0]
 [  0   0   0   0   0   5   1]
 [  0   0   1   0   1   0   5]]
```

### exp_38 - valutazione collassata senza esito del tiro

In questa valutazione le classi finali dei tiri vengono ricondotte al solo tipo di azione, ignorando l'esito:

```text
tiroDaDue0, tiroDaDue1 -> tiroDaDue
tiroDaTre0, tiroDaTre1 -> tiroDaTre
tiroLibero0, tiroLibero1 -> tiroLibero
```

|Metrica|Valore|
|-|-:|
|Accuracy tipo azione|0.8907|
|Macro F1 tipo azione|0.8153|
|Weighted F1 tipo azione|0.8901|

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue|0.79|0.81|0.80|32|
|tiroDaTre|0.67|0.80|0.73|15|
|tiroLibero|0.92|0.61|0.73|18|
|no-action|0.95|0.86|0.90|263|

Confusion matrix:

```text
[[205   1   0   0   6]
 [  0  26   5   0   1]
 [  0   2  12   0   1]
 [  0   1   1  11   5]
 [ 32   3   0   1 227]]
```


### exp_40 - Gerarchia end-to-end con Stadio 3 tracking temporale, 8 classi finali con no-action

In questo esperimento vengono mantenuti invariati Stadio 1 e Stadio 2, mentre lo Stadio 3 usa il checkpoint `exp_l3_tracking_temporal_v1` basato su sequenze temporali palla/canestro.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.88|0.67|0.76|21|
|tiroDaDue1|0.53|0.82|0.64|11|
|tiroDaTre0|0.71|0.83|0.77|12|
|tiroDaTre1|0.50|0.67|0.57|3|
|tiroLibero0|0.80|0.57|0.67|7|
|tiroLibero1|0.71|0.45|0.56|11|
|no-action|0.95|0.86|0.90|263|

Confusion matrix:

```text
[[205   0   1   0   0   0   0   6]
 [  0  14   3   3   0   0   0   1]
 [  0   0   9   1   1   0   0   0]
 [  0   1   0  10   0   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   4   2   1]
 [  0   0   1   0   1   0   5   4]
 [ 32   1   2   0   0   1   0 227]]
```

### exp_40 - solo 7 azioni finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.88|0.67|0.76|21|
|tiroDaDue1|0.53|0.82|0.64|11|
|tiroDaTre0|0.71|0.83|0.77|12|
|tiroDaTre1|0.50|0.67|0.57|3|
|tiroLibero0|0.80|0.57|0.67|7|
|tiroLibero1|0.71|0.45|0.56|11|

Metriche aggregate sulle sole 7 azioni:

|Metrica|Valore|
|-|-:|
|Micro F1|0.86|
|Macro F1|0.70|
|Weighted F1|0.86|

Confusion matrix:

```text
[[205   0   1   0   0   0   0]
 [  0  14   3   3   0   0   0]
 [  0   0   9   1   1   0   0]
 [  0   1   0  10   0   0   0]
 [  0   0   1   0   2   0   0]
 [  0   0   0   0   0   4   2]
 [  0   0   1   0   1   0   5]]
```

### exp_40 - valutazione collassata senza esito del tiro

In questa valutazione le classi finali dei tiri vengono ricondotte al solo tipo di azione, ignorando l'esito:

```text
tiroDaDue0, tiroDaDue1 -> tiroDaDue
tiroDaTre0, tiroDaTre1 -> tiroDaTre
tiroLibero0, tiroLibero1 -> tiroLibero
```

|Metrica|Valore|
|-|-:|
|Accuracy tipo azione|0.8907|
|Macro F1 tipo azione|0.8153|
|Weighted F1 tipo azione|0.8901|

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue|0.79|0.81|0.80|32|
|tiroDaTre|0.67|0.80|0.73|15|
|tiroLibero|0.92|0.61|0.73|18|
|no-action|0.95|0.86|0.90|263|

Confusion matrix:

```text
[[205   1   0   0   6]
 [  0  26   5   0   1]
 [  0   2  12   0   1]
 [  0   1   1  11   5]
 [ 32   3   0   1 227]]
```


### exp_39 - Modello non gerarchico con late fusion tracking, 8 classi con no-action

In questo esperimento viene abbandonata la gerarchia e viene addestrato un unico modello non gerarchico sulle 8 classi finali. Le classi `idle` e `non-gioco` vengono accorpate in `no-action`.

Le feature DINOv3-L/16 restano separate dalle feature palla/canestro: il Transformer produce un embedding video da 256 dimensioni, mentre le 39 feature tracking vengono elaborate da una MLP con embedding da 64 dimensioni. I due embedding vengono concatenati prima del classificatore finale.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.84|0.96|0.90|212|
|tiroDaDue0|0.52|0.52|0.52|21|
|tiroDaDue1|0.38|0.45|0.42|11|
|tiroDaTre0|0.60|0.75|0.67|12|
|tiroDaTre1|0.25|0.33|0.29|3|
|tiroLibero0|1.00|0.29|0.44|7|
|tiroLibero1|0.31|0.45|0.37|11|
|no-action|0.96|0.83|0.89|263|

Confusion matrix:

```text
[[204   3   0   0   0   0   0   5]
 [  0  11   6   1   0   0   3   0]
 [  0   3   5   1   1   0   1   0]
 [  0   1   1   9   1   0   0   0]
 [  0   2   0   0   1   0   0   0]
 [  0   0   0   1   0   2   3   1]
 [  0   0   0   2   1   0   5   3]
 [ 38   1   1   1   0   0   4 218]]
```

### exp_39 - solo 7 azioni finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.84|0.96|0.90|212|
|tiroDaDue0|0.52|0.52|0.52|21|
|tiroDaDue1|0.38|0.45|0.42|11|
|tiroDaTre0|0.60|0.75|0.67|12|
|tiroDaTre1|0.25|0.33|0.29|3|
|tiroLibero0|1.00|0.29|0.44|7|
|tiroLibero1|0.31|0.45|0.37|11|

Metriche aggregate sulle sole 7 azioni:

|Metrica|Valore|
|-|-:|
|Micro F1|0.80|
|Macro F1|0.52|
|Weighted F1|0.80|

Confusion matrix:

```text
[[204   3   0   0   0   0   0]
 [  0  11   6   1   0   0   3]
 [  0   3   5   1   1   0   1]
 [  0   1   1   9   1   0   0]
 [  0   2   0   0   1   0   0]
 [  0   0   0   1   0   2   3]
 [  0   0   0   2   1   0   5]]
```


## Comandi utilizzati

### Estrazione feature tracking palla/canestro

```bash
python -m src.features.extract_ball_rim_tracking_features \
  --dataset-root data/datasets/dataset_basket_v1 \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --yolo-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt \
  --output-dir data/features/ball_rim_tracking_features_v1 \
  --splits train val \
  --num-frames 48 \
  --sample-mode uniform \
  --imgsz 1280 \
  --conf 0.10 \
  --batch-size 16 \
  --device 0 \
  --save-per-frame \
  --overwrite
```

### Estrazione feature tracking palla/canestro su tutte le clip train/val

```bash
python -m src.features.extract_ball_rim_tracking_features \
  --dataset-root data/datasets/dataset_basket_v1 \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --yolo-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt \
  --output-dir data/features/ball_rim_tracking_all_train_val \
  --num-frames 48 \
  --imgsz 1280 \
  --conf 0.10 \
  --iou 0.50 \
  --device 0 \
  --batch-size 16 \
  --overwrite
```

### exp_l3_tracking_v1 - Training Stadio 3 con tracking

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --output-dir outputs/exp_l3_tracking_v1 \
  --label-mode shot_outcome_only \
  --tracking-features-csv data/features/ball_rim_tracking_features_v1/tracking_features.csv \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.5 \
  --num-workers 2 \
  --seed 42
```

### Estrazione sequenze temporali tracking palla/canestro

```bash
python -m src.features.extract_ball_rim_tracking_features \
  --dataset-root data/datasets/dataset_basket_v1 \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --yolo-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt \
  --output-dir data/features/ball_rim_tracking_temporal_v1 \
  --splits train val \
  --num-frames 48 \
  --sample-mode uniform \
  --imgsz 1280 \
  --conf 0.10 \
  --batch-size 16 \
  --device 0 \
  --save-temporal-sequences \
  --overwrite
```

### exp_l3_tracking_v2_d384_sampler04 - Training Stadio 3 con tracking e parametri d384

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --output-dir outputs/exp_l3_tracking_v2_d384_sampler04 \
  --label-mode shot_outcome_only \
  --tracking-features-csv data/features/ball_rim_tracking_features_v1/tracking_features.csv \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 384 \
  --num-layers 2 \
  --num-heads 6 \
  --ff-dim 1024 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --num-workers 2 \
  --seed 42
```

### exp_l3_tracking_temporal_v1 - Training Stadio 3 con tracking temporale

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --output-dir outputs/exp_l3_tracking_temporal_v1 \
  --label-mode shot_outcome_only \
  --tracking-sequences-npz data/features/ball_rim_tracking_temporal_v1/tracking_sequences.npz \
  --tracking-sequence-index data/features/ball_rim_tracking_temporal_v1/tracking_sequence_index.json \
  --epochs 100 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.5 \
  --num-workers 0 \
  --seed 42
```

### exp_38 - Gerarchia end-to-end con Stadio 3 tracking

```bash
python -m src.evaluation.evaluate_hierarchical \
  --features-root data/features/dinov3_vitl16_336 \
  --split val \
  --batch-size 64 \
  --num-workers 2 \
  --l1-checkpoint outputs/exp_28_dinov3_vitl16_transformer_mean_action_noaction/best_model.pt \
  --l2-checkpoint outputs/exp_29_dinov3_vitl16_transformer_mean_shot_type_only/best_model.pt \
  --l3-checkpoint outputs/exp_l3_tracking_v1/best_model.pt \
  --tracking-features-csv data/features/ball_rim_tracking_features_v1/tracking_features.csv \
  --output-dir outputs/exp_38_dinov3_vitl16_hierarchical_tracking_l3
```


### exp_40 - Gerarchia end-to-end con Stadio 3 tracking temporale

```bash
python -m src.evaluation.evaluate_hierarchical \
  --features-root data/features/dinov3_vitl16_336 \
  --split val \
  --batch-size 64 \
  --num-workers 2 \
  --l1-checkpoint outputs/exp_28_dinov3_vitl16_transformer_mean_action_noaction/best_model.pt \
  --l2-checkpoint outputs/exp_29_dinov3_vitl16_transformer_mean_shot_type_only/best_model.pt \
  --l3-checkpoint outputs/exp_l3_tracking_temporal_v1/best_model.pt \
  --tracking-sequences-npz data/features/ball_rim_tracking_temporal_v1/tracking_sequences.npz \
  --tracking-sequence-index data/features/ball_rim_tracking_temporal_v1/tracking_sequence_index.json \
  --output-dir outputs/exp_40_dinov3_hierarchical_tracking_temporal_l3
```


### exp_39 - Training non gerarchico a 8 classi con late fusion tracking

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --tracking-features-csv data/features/ball_rim_tracking_all_train_val/tracking_features.csv \
  --output-dir outputs/exp_39_nonhierarchical_dinov3_tracking_noaction \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.5
```

## Nota finale

L’introduzione delle feature di tracking palla/canestro nello Stadio 3 ha prodotto un miglioramento netto nella distinzione tra tiro segnato e tiro sbagliato. Il modello `exp_l3_tracking_v1`, basato su feature DINOv3-L/16 concatenate con 39 feature geometriche e temporali derivate dal detector YOLO, raggiunge Accuracy = 0.8154 e Macro F1 = 0.8132 sul task binario `tiro0` / `tiro1`.

Rispetto al precedente riferimento per lo Stadio 3, `exp_30`, che otteneva Accuracy = 0.6615 e Macro F1 = 0.6425, il tracking aggregato porta un incremento di +0.1539 in Accuracy e +0.1707 in Macro F1. Il miglioramento è superiore anche a quello ottenuto con le successive variazioni su pooling, dropout, sampler e feature VideoMAE.

Il tentativo `exp_l3_tracking_v2_d384_sampler04` prova ad aumentare la capacità dello Stadio 3 usando la configurazione più grande degli esperimenti `exp_30` / `exp_31` (`d_model=384`, `num_heads=6`, `ff_dim=1024`) e riportando il sampler a `power=0.4`. Il risultato non migliora `exp_l3_tracking_v1`: la Macro F1 scende da 0.8132 a 0.7842. La matrice di confusione mostra che il modello diventa più sbilanciato verso `tiro1`, con 13 `tiro0` classificati come `tiro1` e un solo `tiro1` classificato come `tiro0`.

L'esperimento `exp_l3_tracking_temporal_v1` introduce invece sequenze temporali di tracking palla/canestro, usando 29 feature per frame concatenate alle feature DINOv3. Questa modifica porta il miglior risultato sullo Stadio 3: Accuracy = 0.8769, Macro F1 = 0.8733 e Weighted F1 = 0.8782. Rispetto a `exp_l3_tracking_v1`, i falsi positivi su `tiro1` diminuiscono da 10 a 6, mentre i `tiro1` classificati come `tiro0` restano 2. Questo indica che la dinamica temporale della relazione palla-canestro è più informativa del solo riassunto globale della clip.

Il beneficio si riflette anche nella gerarchia end-to-end `exp_40`, in cui vengono mantenuti invariati Stadio 1 e Stadio 2 e viene sostituito solo lo Stadio 3 con il checkpoint temporale. Rispetto a `exp_38`, la Macro F1 sulle 8 classi passa da 0.6996 a 0.7223 e la Macro F1 sulle sole 7 azioni finali passa da 0.67 a 0.70. La valutazione collassata senza esito resta identica ad `exp_38` (Accuracy = 0.8907, Macro F1 = 0.8153), confermando che il miglioramento deriva dalla migliore classificazione dell'esito del tiro, non da cambiamenti nel riconoscimento del tipo di azione.

Il successivo esperimento `exp_39` riporta il tracking anche nel modello non gerarchico a 8 classi, con `idle` e `non-gioco` uniti in `no-action`. Questa soluzione è più semplice dal punto di vista del codice e dell'inferenza, perché non richiede il routing tra stadi. Tuttavia, sui risultati di validation ottiene Accuracy = 0.8426 e Macro F1 = 0.5620 sulle 8 classi, mentre sulle sole 7 azioni finali raggiunge Macro F1 = 0.52. Il risultato è quindi inferiore sia alla gerarchia con tracking aggregato `exp_38` sia alla nuova gerarchia con tracking temporale `exp_40`.

Nel complesso, questi risultati mostrano che le informazioni esplicite sulla relazione spaziale e temporale tra palla e canestro riducono in modo significativo il principale collo di bottiglia della gerarchia precedente. Il nuovo riferimento migliore diventa `exp_l3_tracking_temporal_v1` per lo Stadio 3 e `exp_40` per la gerarchia end-to-end. Eventuali miglioramenti successivi dovrebbero concentrarsi su analisi degli errori residui, tuning della soglia decisionale di `tiro1` o regolarizzazione/early stopping del modello temporale, dato che dopo circa 50 epoche il training continua a migliorare ma la validation si stabilizza.
