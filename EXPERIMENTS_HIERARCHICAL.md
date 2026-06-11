# Esperimenti Label Mode e Gerarchia - Temporal Transformer

Questo file tiene traccia sintetica degli esperimenti effettuati mantenendo il classificatore basato su **Temporal Transformer Encoder**, ma modificando la modalità di etichettatura delle classi tramite `--label-mode` e valutando poi una pipeline **gerarchica end-to-end**.

Gli esperimenti usano feature già estratte con **DINOv3-L/16 frozen** e mantengono la configurazione migliore degli esperimenti precedenti con **mean pooling**, confrontando diverse rimappature delle classi per capire se il modello è più confuso dall'esito del tiro, dal tipo di tiro o dalla separazione tra azione e non-azione.

Gli esperimenti più recenti introducono una gerarchia in tre stadi:

```text
Stadio 1:
passaggio / tiro / no-action

Stadio 2, solo se Stadio 1 = tiro:
tiroDaDue / tiroDaTre / tiroLibero

Stadio 3, solo se Stadio 1 = tiro:
tiro0 / tiro1
```

## Tabella riassuntiva

|ID|Modello|Feature extractor|Label mode / valutazione|Classi|Epoche|Batch size|LR|d_model|Layers|Heads|FF dim|Dropout|Weight decay|Pooling|Class weight power|Sampler|Val Loss|Val Accuracy|Val Macro F1|Val Weighted F1|Output dir|
|-|-|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-|-:|-|-:|-:|-:|-:|-|
|exp_25|Temporal Transformer d384|DINOv3-L/16 frozen|shot_outcome|5|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.5103|0.8352|0.7080|0.8255|outputs/exp_25_dinov3_vitl16_transformer_mean_shot_outcome|
|exp_26|Temporal Transformer d384|DINOv3-L/16 frozen|shot_type|6|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.4477|0.8611|0.7768|0.8515|outputs/exp_26_dinov3_vitl16_transformer_mean_shot_type|
|exp_27|Temporal Transformer d384|DINOv3-L/16 frozen|action_group|4|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.4959|0.8685|0.8367|0.8622|outputs/exp_27_dinov3_vitl16_transformer_mean_action_group|
|exp_28|Temporal Transformer d384|DINOv3-L/16 frozen|action_noaction|3|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.3650|0.9074|0.9073|0.9072|outputs/exp_28_dinov3_vitl16_transformer_mean_action_noaction|
|exp_29|Temporal Transformer d384|DINOv3-L/16 frozen|shot_type_only|3|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.4804|0.8462|0.8410|0.8520|outputs/exp_29_dinov3_vitl16_transformer_mean_shot_type_only|
|exp_30|Temporal Transformer d384|DINOv3-L/16 frozen|shot_outcome_only|2|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.7302|0.6615|0.6425|0.6615|outputs/exp_30_dinov3_vitl16_transformer_mean_shot_outcome_only|
|exp_31|Gerarchia L1+L2+L3|DINOv3-L/16 frozen|hierarchical end-to-end|8|-|64|-|384|2|6|1024|0.45|5e-3|Mean|-|-|-|0.8648|0.5981|0.8629|outputs/exp_31_dinov3_vitl16_hierarchical_end_to_end|
|exp_32|Temporal Transformer d384|DINOv3-L/16 frozen|shot_outcome_only|2|40|64|5e-5|384|2|6|1024|0.45|5e-3|CLS|0.5|WeightedRandomSampler power 0.4|0.6593|0.6462|0.6289|0.6474|outputs/exp_32_dinov3_vitl16_transformer_cls_shot_outcome_only|
|exp_33|Temporal Transformer d384|DINOv3-L/16 frozen|shot_outcome_only|2|40|64|5e-5|384|2|6|1024|0.45|5e-3|LastMean 0.30|0.5|WeightedRandomSampler power 0.4|0.7207|0.6462|0.6289|0.6474|outputs/exp_33_dinov3_vitl16_transformer_lastmean030_shot_outcome_only|
|exp_34|Temporal Transformer d384|DINOv3-L/16 frozen|shot_outcome_only|2|40|64|5e-5|384|2|6|1024|0.30|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.7056|0.6615|0.6299|0.6549|outputs/exp_34_dinov3_vitl16_transformer_mean_dropout030_shot_outcome_only|
|exp_35|Temporal Transformer d384|DINOv3-L/16 frozen|shot_outcome_only|2|40|64|5e-5|384|2|6|1024|0.30|5e-3|Mean|0.5|No weighted sampler|0.6960|0.6154|0.6061|0.6200|outputs/exp_35_dinov3_vitl16_transformer_mean_dropout030_no_sampler_shot_outcome_only|
|exp_36|Temporal Transformer d256|VideoMAE-B/16 Kinetics frozen|shot_outcome_only|2|40|64|5e-5|256|2|4|768|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|-|0.6769|0.6431|0.6684|outputs/exp_36_videomae_kinetics16_temporal_stretch_shot_outcome_only|
|exp_37|Temporal Transformer d256|VideoMAE-B/16 Kinetics frozen|shot_outcome_only|2|40|64|5e-5|256|2|4|768|0.30|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|-|0.7077|0.6411|0.6768|outputs/exp_37_videomae_kinetics16_temporal_stretch_dropout030_shot_outcome_only|

## Risultati aggregati su validation - classi rimappate

|ID|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|
|exp_25|0.8352|0.73|0.70|0.71|0.83|0.84|0.83|
|exp_26|0.8611|0.80|0.77|0.78|0.86|0.86|0.85|
|exp_27|0.8685|0.85|0.83|0.84|0.86|0.87|0.86|
|exp_28|0.9074|0.91|0.91|0.91|0.91|0.91|0.91|
|exp_29|0.8462|0.85|0.85|0.84|0.87|0.85|0.85|
|exp_30|0.6615|0.64|0.64|0.64|0.66|0.66|0.66|
|exp_31|0.8648|0.63|0.61|0.60|0.87|0.86|0.86|
|exp_32|0.6462|0.63|0.63|0.63|0.65|0.65|0.65|
|exp_33|0.6462|0.63|0.63|0.63|0.65|0.65|0.65|
|exp_34|0.6615|0.64|0.63|0.63|0.65|0.66|0.65|
|exp_35|0.6154|0.61|0.61|0.61|0.63|0.62|0.62|
|exp_36|0.6769|0.66|0.64|0.64|0.67|0.68|0.67|
|exp_37|0.7077|0.72|0.64|0.64|0.72|0.71|0.68|

## Risultati aggregati sulle classi rilevanti

Questa tabella riporta le metriche calcolate solo sulle classi più rilevanti per ciascuna rimappatura:

- **exp_25**: solo `tiro0` e `tiro1`.
- **exp_26**: solo `tiroDaDue`, `tiroDaTre` e `tiroLibero`.
- **exp_27**: solo `passaggio` e `tiro`, cioè le classi che verrebbero mantenute come azioni nel report finale.
- **exp_28**: solo `passaggio` e `tiro`, ignorando `no-action` come classe di report finale.
- **exp_29**: solo `tiroDaDue`, `tiroDaTre` e `tiroLibero`, addestrate filtrando il dataset sui soli tiri.
- **exp_30**: solo `tiro0` e `tiro1`, addestrate filtrando il dataset sui soli tiri.
- **exp_31**: solo le 7 azioni finali prodotte dalla gerarchia end-to-end, quindi senza `no-action`.
- **exp_31 collassato**: valutazione della stessa gerarchia collassando l'esito del tiro, cioè `tiroDaDue0/1 -> tiroDaDue`, `tiroDaTre0/1 -> tiroDaTre` e `tiroLibero0/1 -> tiroLibero`.
- **exp_32**: solo `tiro0` e `tiro1`, come exp_30, ma con pooling `cls`.
- **exp_33**: solo `tiro0` e `tiro1`, come exp_30, ma con pooling `last_mean` sull'ultimo 30% della clip.
- **exp_34**: solo `tiro0` e `tiro1`, come exp_30, ma con dropout ridotto a 0.30.
- **exp_35**: solo `tiro0` e `tiro1`, come exp_34, ma senza `WeightedRandomSampler`.
- **exp_36**: solo `tiro0` e `tiro1`, usando feature VideoMAE temporali a 16 frame, senza center crop e con checkpoint Kinetics.
- **exp_37**: solo `tiro0` e `tiro1`, come exp_36, ma con dropout ridotto a 0.30.

|ID|Classi considerate|Micro Precision|Micro Recall|Micro F1|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|exp_25|tiro0, tiro1|0.60|0.58|0.59|0.56|0.54|0.54|0.58|0.58|0.58|
|exp_26|tiroDaDue, tiroDaTre, tiroLibero|0.74|0.77|0.75|0.73|0.74|0.73|0.74|0.77|0.75|
|exp_27|passaggio, tiro|0.87|0.92|0.90|0.87|0.92|0.89|0.88|0.92|0.90|
|exp_28|passaggio, tiro|0.88|0.95|0.91|0.89|0.93|0.91|0.88|0.95|0.91|
|exp_29|tiroDaDue, tiroDaTre, tiroLibero|0.85|0.85|0.85|0.85|0.85|0.84|0.87|0.85|0.85|
|exp_30|tiro0, tiro1|0.66|0.66|0.66|0.64|0.64|0.64|0.66|0.66|0.66|
|exp_31|7 azioni finali|0.80|0.87|0.83|0.58|0.57|0.55|0.80|0.87|0.83|
|exp_31 collassato|tipo azione senza esito|0.89|0.89|0.89|0.84|0.81|0.82|0.90|0.89|0.89|
|exp_32|tiro0, tiro1|0.65|0.65|0.65|0.63|0.63|0.63|0.65|0.65|0.65|
|exp_33|tiro0, tiro1|0.65|0.65|0.65|0.63|0.63|0.63|0.65|0.65|0.65|
|exp_34|tiro0, tiro1|0.66|0.66|0.66|0.64|0.63|0.63|0.65|0.66|0.65|
|exp_35|tiro0, tiro1|0.62|0.62|0.62|0.61|0.61|0.61|0.63|0.62|0.62|
|exp_36|tiro0, tiro1|0.68|0.68|0.68|0.66|0.64|0.64|0.67|0.68|0.67|
|exp_37|tiro0, tiro1|0.71|0.71|0.71|0.72|0.64|0.64|0.72|0.71|0.68|

## Risultati per classe su validation

### exp_25 - DINOv3-L/16 frozen, shot_outcome, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.87|0.95|0.91|212|
|tiro0|0.67|0.75|0.71|40|
|tiro1|0.44|0.32|0.37|25|
|idle|0.76|0.54|0.63|93|
|non-gioco|0.90|0.95|0.93|170|

### exp_26 - DINOv3-L/16 frozen, shot_type, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.87|0.96|0.91|212|
|tiroDaDue|0.76|0.88|0.81|32|
|tiroDaTre|0.62|0.67|0.65|15|
|tiroLibero|0.80|0.67|0.73|18|
|idle|0.82|0.51|0.63|93|
|non-gioco|0.91|0.97|0.94|170|

### exp_27 - DINOv3-L/16 frozen, action_group, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.88|0.93|0.90|212|
|tiro|0.86|0.91|0.88|65|
|idle|0.75|0.55|0.63|93|
|non-gioco|0.91|0.95|0.93|170|

### exp_28 - DINOv3-L/16 frozen, action_noaction, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiro|0.92|0.89|0.91|65|
|no-action|0.95|0.86|0.90|263|

Confusion matrix:

```text
[[205   1   6]
 [  0  58   7]
 [ 32   4 227]]
```

### exp_29 - DINOv3-L/16 frozen, shot_type_only, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiroDaDue|0.90|0.84|0.87|32|
|tiroDaTre|0.65|0.87|0.74|15|
|tiroLibero|1.00|0.83|0.91|18|

Confusion matrix:

```text
[[27  5  0]
 [ 2 13  0]
 [ 1  2 15]]
```

### exp_30 - DINOv3-L/16 frozen, shot_outcome_only, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.72|0.72|0.72|40|
|tiro1|0.56|0.56|0.56|25|

Confusion matrix:

```text
[[29 11]
 [11 14]]
```

### exp_31 - Gerarchia end-to-end, 8 classi finali con no-action

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.62|0.38|0.47|21|
|tiroDaDue1|0.35|0.64|0.45|11|
|tiroDaTre0|0.59|0.83|0.69|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.83|0.71|0.77|7|
|tiroLibero1|0.83|0.45|0.59|11|
|no-action|0.95|0.86|0.90|263|

Confusion matrix:

```text
[[205   0   1   0   0   0   0   6]
 [  0   8   9   3   0   0   0   1]
 [  0   2   7   2   0   0   0   0]
 [  0   1   0  10   0   0   0   1]
 [  0   1   0   2   0   0   0   0]
 [  0   0   0   0   0   5   1   1]
 [  0   1   0   0   1   0   5   4]
 [ 32   0   3   0   0   1   0 227]]
```

#### Valutazione collassata senza esito del tiro

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

Questa valutazione conferma che, quando si ignora l'esito del tiro, la gerarchia riconosce in modo abbastanza solido il tipo di azione. Il risultato collassato raggiunge Accuracy = 0.8907 e Macro F1 = 0.8153, nettamente superiori alla Macro F1 = 0.5981 della valutazione completa a 8 classi. Il degrado principale della gerarchia end-to-end è quindi dovuto soprattutto alla distinzione tra tiro segnato e tiro sbagliato, più che al riconoscimento del tipo di tiro.

### exp_32 - DINOv3-L/16 frozen, shot_outcome_only, CLS pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.72|0.70|0.71|40|
|tiro1|0.54|0.56|0.55|25|

Confusion matrix:

```text
[[28 12]
 [11 14]]
```

### exp_33 - DINOv3-L/16 frozen, shot_outcome_only, last_mean 0.30, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.72|0.70|0.71|40|
|tiro1|0.54|0.56|0.55|25|

Confusion matrix:

```text
[[28 12]
 [11 14]]
```

### exp_34 - DINOv3-L/16 frozen, shot_outcome_only, mean pooling, dropout 0.30, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.70|0.78|0.74|40|
|tiro1|0.57|0.48|0.52|25|

Confusion matrix:

```text
[[31  9]
 [13 12]]
```

### exp_35 - DINOv3-L/16 frozen, shot_outcome_only, mean pooling, dropout 0.30, no weighted sampler

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.71|0.62|0.67|40|
|tiro1|0.50|0.60|0.55|25|

Confusion matrix:

```text
[[25 15]
 [10 15]]
```

### exp_36 - VideoMAE-B/16 Kinetics frozen, shot_outcome_only, feature temporali 16 frame, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.71|0.80|0.75|40|
|tiro1|0.60|0.48|0.53|25|

Confusion matrix:

```text
[[32  8]
 [13 12]]
```

### exp_37 - VideoMAE-B/16 Kinetics frozen, shot_outcome_only, feature temporali 16 frame, dropout 0.30, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.70|0.93|0.80|40|
|tiro1|0.75|0.36|0.49|25|

Confusion matrix:

```text
[[37  3]
 [16  9]]
```

## Comandi utilizzati

### exp_25

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
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
  --label-mode shot_outcome \
  --output-dir outputs/exp_25_dinov3_vitl16_transformer_mean_shot_outcome
```

### exp_26

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
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
  --label-mode shot_type \
  --output-dir outputs/exp_26_dinov3_vitl16_transformer_mean_shot_type
```

### exp_27

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
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
  --label-mode action_group \
  --output-dir outputs/exp_27_dinov3_vitl16_transformer_mean_action_group
```

### exp_28

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
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
  --label-mode action_noaction \
  --output-dir outputs/exp_28_dinov3_vitl16_transformer_mean_action_noaction
```

### exp_29

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
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
  --label-mode shot_type_only \
  --output-dir outputs/exp_29_dinov3_vitl16_transformer_mean_shot_type_only
```

### exp_30

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
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
  --label-mode shot_outcome_only \
  --output-dir outputs/exp_30_dinov3_vitl16_transformer_mean_shot_outcome_only
```

### exp_31

```bash
python -m src.evaluation.evaluate_hierarchical \
  --features-root data/features/dinov3_vitl16_336 \
  --split val \
  --batch-size 64 \
  --num-workers 2 \
  --l1-checkpoint outputs/exp_28_dinov3_vitl16_transformer_mean_action_noaction/best_model.pt \
  --l2-checkpoint outputs/exp_29_dinov3_vitl16_transformer_mean_shot_type_only/best_model.pt \
  --l3-checkpoint outputs/exp_30_dinov3_vitl16_transformer_mean_shot_outcome_only/best_model.pt \
  --output-dir outputs/exp_31_dinov3_vitl16_hierarchical_end_to_end
```


### exp_32

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 384 \
  --num-layers 2 \
  --num-heads 6 \
  --ff-dim 1024 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling cls \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --label-mode shot_outcome_only \
  --output-dir outputs/exp_32_dinov3_vitl16_transformer_cls_shot_outcome_only
```

### exp_33

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 384 \
  --num-layers 2 \
  --num-heads 6 \
  --ff-dim 1024 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling last_mean \
  --last-mean-ratio 0.30 \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --label-mode shot_outcome_only \
  --output-dir outputs/exp_33_dinov3_vitl16_transformer_lastmean030_shot_outcome_only
```

### exp_34

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 384 \
  --num-layers 2 \
  --num-heads 6 \
  --ff-dim 1024 \
  --dropout 0.30 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --label-mode shot_outcome_only \
  --output-dir outputs/exp_34_dinov3_vitl16_transformer_mean_dropout030_shot_outcome_only
```

### exp_35

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 384 \
  --num-layers 2 \
  --num-heads 6 \
  --ff-dim 1024 \
  --dropout 0.30 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --no-weighted-sampler \
  --scheduler-patience 5 \
  --label-mode shot_outcome_only \
  --output-dir outputs/exp_35_dinov3_vitl16_transformer_mean_dropout030_no_sampler_shot_outcome_only
```

### exp_36

Estrazione feature VideoMAE temporali:

```bash
python -m src.features.extract_videomae_features \
  --dataset-root data/datasets/dataset_basket_v1 \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --output-dir data/features/videomae_kinetics16_temporal_stretch \
  --model-name MCG-NJU/videomae-base-finetuned-kinetics \
  --num-frames 16 \
  --image-size 224 \
  --shot-only \
  --fp16 \
  --overwrite
```

Training Stadio 3:

```bash
python -m src.training.train \
  --features-root data/features/videomae_kinetics16_temporal_stretch \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 768 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --label-mode shot_outcome_only \
  --output-dir outputs/exp_36_videomae_kinetics16_temporal_stretch_shot_outcome_only
```

### exp_37

```bash
python -m src.training.train \
  --features-root data/features/videomae_kinetics16_temporal_stretch \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 768 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.30 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --label-mode shot_outcome_only \
  --output-dir outputs/exp_37_videomae_kinetics16_temporal_stretch_dropout030_shot_outcome_only
```

## Nota finale

Gli esperimenti 25, 26 e 27 sono stati progettati come test diagnostici sulle classi, mantenendo invariata la configurazione del modello usata negli esperimenti migliori precedenti:

```text
DINOv3-L/16 frozen + Temporal Transformer
input_dim = 1024
d_model = 384
num_layers = 2
num_heads = 6
ff_dim = 1024
dropout = 0.45
weight_decay = 5e-3
pooling = Mean
class_weight_power = 0.5
sampler_power = 0.4
```

Il confronto tra gli esperimenti suggerisce che:

- **exp_25** riduce le classi dei tiri a `tiro0` e `tiro1`, cioè tiro sbagliato e tiro segnato. Il modello riconosce abbastanza bene `tiro0` con F1 = 0.71, ma fatica molto su `tiro1`, che raggiunge F1 = 0.37. Questo indica che la distinzione tra tiro segnato e tiro sbagliato resta difficile anche quando si elimina la distinzione tra tiro da due, tiro da tre e tiro libero.
- **exp_26** riduce le classi dei tiri al solo tipo di tiro: `tiroDaDue`, `tiroDaTre` e `tiroLibero`. Questo esperimento ottiene risultati migliori sulle classi di tiro rispetto alla distinzione dell'esito, con F1 = 0.81 per `tiroDaDue`, F1 = 0.65 per `tiroDaTre` e F1 = 0.73 per `tiroLibero`.
- **exp_27** riduce il problema a quattro gruppi: `passaggio`, `tiro`, `idle` e `non-gioco`. Ottiene Val Macro F1 = 0.8367 e Val Accuracy = 0.8685, ma `idle` resta la classe più problematica, con F1 = 0.63.
- **exp_28** unisce `idle` e `non-gioco` nella classe `no-action`. Questo migliora nettamente la stabilità del primo stadio, con Val Macro F1 = 0.9073 e F1 simili su `passaggio`, `tiro` e `no-action`.
- **exp_29** addestra il tipo di tiro solo sulle clip di tiro. Il risultato è buono: Val Macro F1 = 0.8410, con F1 = 0.87 su `tiroDaDue`, F1 = 0.74 su `tiroDaTre` e F1 = 0.91 su `tiroLibero`.
- **exp_30** addestra l'esito solo sulle clip di tiro. Il risultato resta debole: Val Macro F1 = 0.6425, con F1 = 0.72 su `tiro0` e F1 = 0.56 su `tiro1`. Questo conferma che la distinzione segnato/sbagliato è il collo di bottiglia principale.
- **exp_31** valuta la gerarchia completa end-to-end. La pipeline a 8 classi ottiene Accuracy = 0.8648 e Weighted F1 = 0.8629, ma la Macro F1 è solo 0.5981. Sulle sole 7 azioni finali, la Macro F1 scende a 0.55. La valutazione collassata senza esito del tiro raggiunge invece Accuracy = 0.8907 e Macro F1 = 0.8153, mostrando che la gerarchia riconosce abbastanza bene il tipo di azione quando non deve distinguere tra tiro segnato e tiro sbagliato.
- **exp_32** prova a sostituire il mean pooling con `cls` sul solo Stadio 3. Il risultato non migliora exp_30: Val Macro F1 = 0.6289, con F1 = 0.71 su `tiro0` e F1 = 0.55 su `tiro1`.
- **exp_33** prova un pooling finale più focalizzato sugli ultimi frame (`last_mean` con ratio 0.30). Il comportamento resta identico a exp_32 e leggermente sotto exp_30, indicando che usare solo la parte finale della clip non è sufficiente per separare meglio `tiro0` e `tiro1`.
- **exp_34** riduce il dropout da 0.45 a 0.30. L'accuracy resta 0.6615, ma la Macro F1 scende a 0.6299: il modello migliora `tiro0` con F1 = 0.74, ma peggiora `tiro1` con F1 = 0.52.
- **exp_35** mantiene dropout 0.30 ma disattiva il `WeightedRandomSampler`. Il recall di `tiro1` sale a 0.60, ma peggiora `tiro0` e la Macro F1 scende a 0.6061. La rimozione del sampler sposta quindi il tipo di errore, senza produrre un miglioramento complessivo.
- **exp_36** introduce VideoMAE-B/16 fine-tuned su Kinetics come feature extractor video frozen per il solo Stadio 3 (`shot_outcome_only`). L’obiettivo era verificare se un modello preaddestrato direttamente su video potesse catturare meglio la dinamica temporale del tiro rispetto alle feature frame-level di DINOv3. Le clip vengono campionate a 16 frame e preprocessate senza center crop, usando un resize stretched a 224×224 per preservare l’intero contenuto del frame, inclusi elementi laterali potenzialmente importanti come palla o canestro. Inoltre, invece di salvare una singola feature globale per clip, vengono mantenute feature temporali: i token spaziali prodotti da VideoMAE vengono mediati per ciascun blocco temporale, ottenendo una sequenza di feature video elaborabile dal Temporal Transformer. Il risultato è competitivo con exp_30, con Accuracy = 0.6769 e Macro F1 = 0.6431, ma non rappresenta un miglioramento netto: `tiro0` migliora leggermente, mentre `tiro1` resta debole con F1 = 0.53.
- **exp_37** mantiene le feature VideoMAE temporali di exp_36 ma riduce il dropout a 0.30. L'accuracy sale a 0.7077 grazie a un forte miglioramento su `tiro0` (F1 = 0.80), ma `tiro1` peggiora a F1 = 0.49. Anche questo esperimento non risolve quindi il collo di bottiglia sull'esito positivo del tiro.

La conclusione principale è che il modello distingue bene **azione/non-azione** e riconosce in modo abbastanza convincente il **tipo di tiro**, mentre la distinzione **segnato/sbagliato** rimane il punto più critico. La valutazione collassata di exp_31 rafforza questa interpretazione: rimuovendo l'esito del tiro, la gerarchia sale a Accuracy = 0.8907 e Macro F1 = 0.8153 sul tipo di azione. Gli esperimenti 32-35 mostrano che piccoli cambiamenti conservativi su pooling finale, dropout e sampler non producono un miglioramento rilevante dello Stadio 3. Gli esperimenti 36-37 mostrano inoltre che VideoMAE frozen, anche in una configurazione con feature temporali a 16 frame e preprocessing senza center crop, è competitivo ma non superiore a DINOv3 sul riconoscimento dell'esito. Il miglior riferimento per `tiro1` resta quindi exp_30.

La gerarchia considerata rimane quindi:

```text
Stadio 1:
passaggio / tiro / no-action

Stadio 2, solo se tiro:
tiroDaDue / tiroDaTre / tiroLibero

Stadio 3, solo se tiro:
tiro0 / tiro1
```

Nel complesso, questi esperimenti confermano che la gerarchia è una formulazione utile e interpretabile del problema, ma l'esito del tiro resta la componente più difficile e limita le prestazioni finali sulle classi complete.
