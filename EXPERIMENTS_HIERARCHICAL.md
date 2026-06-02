# Esperimenti Label Mode - Temporal Transformer

Questo file tiene traccia sintetica degli esperimenti effettuati mantenendo il classificatore basato su **Temporal Transformer Encoder**, ma modificando la modalità di etichettatura delle classi tramite `--label-mode`.

Gli esperimenti usano feature già estratte con **DINOv3-L/16 frozen** e mantengono la configurazione migliore degli esperimenti precedenti con **mean pooling**, confrontando diverse rimappature delle classi per capire se il modello è più confuso dall'esito del tiro, dal tipo di tiro o dalla separazione tra azione e non-azione.

## Tabella riassuntiva

|ID|Modello|Feature extractor|Label mode|Classi|Epoche|Batch size|LR|d_model|Layers|Heads|FF dim|Dropout|Weight decay|Pooling|Class weight power|Sampler|Val Loss|Val Accuracy|Val Macro F1|Val Weighted F1|Output dir|
|-|-|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-|-:|-|-:|-:|-:|-:|-|
|exp_25|Temporal Transformer d384|DINOv3-L/16 frozen|shot_outcome|5|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.5103|0.8352|0.7080|0.8255|outputs/exp_25_dinov3_vitl16_transformer_mean_shot_outcome|
|exp_26|Temporal Transformer d384|DINOv3-L/16 frozen|shot_type|6|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.4477|0.8611|0.7768|0.8515|outputs/exp_26_dinov3_vitl16_transformer_mean_shot_type|
|exp_27|Temporal Transformer d384|DINOv3-L/16 frozen|action_group|4|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.4959|0.8685|0.8367|0.8622|outputs/exp_27_dinov3_vitl16_transformer_mean_action_group|

## Risultati aggregati su validation - classi rimappate

|ID|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|
|exp_25|0.8352|0.73|0.70|0.71|0.83|0.84|0.83|
|exp_26|0.8611|0.80|0.77|0.78|0.86|0.86|0.85|
|exp_27|0.8685|0.85|0.83|0.84|0.86|0.87|0.86|

## Risultati aggregati sulle classi rilevanti

Questa tabella riporta le metriche calcolate solo sulle classi più rilevanti per ciascuna rimappatura:

- **exp_25**: solo `tiro0` e `tiro1`.
- **exp_26**: solo `tiroDaDue`, `tiroDaTre` e `tiroLibero`.
- **exp_27**: solo `passaggio` e `tiro`, cioè le classi che verrebbero mantenute come azioni nel report finale.

|ID|Classi considerate|Micro Precision|Micro Recall|Micro F1|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|exp_25|tiro0, tiro1|0.60|0.58|0.59|0.56|0.54|0.54|0.58|0.58|0.58|
|exp_26|tiroDaDue, tiroDaTre, tiroLibero|0.74|0.77|0.75|0.73|0.74|0.73|0.74|0.77|0.75|
|exp_27|passaggio, tiro|0.87|0.92|0.90|0.87|0.92|0.89|0.88|0.92|0.90|

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
- **exp_26** riduce le classi dei tiri al solo tipo di tiro: `tiroDaDue`, `tiroDaTre` e `tiroLibero`. Questo esperimento ottiene risultati nettamente migliori sulle classi di tiro, con F1 = 0.81 per `tiroDaDue`, F1 = 0.65 per `tiroDaTre` e F1 = 0.73 per `tiroLibero`. Il modello sembra quindi riconoscere meglio il tipo di tiro rispetto all'esito del tiro.
- **exp_27** riduce il problema a quattro gruppi: `passaggio`, `tiro`, `idle` e `non-gioco`. Questo esperimento ottiene le metriche globali migliori tra i tre, con Val Macro F1 = 0.8367 e Val Accuracy = 0.8685. Lo stadio `passaggio`/`tiro`/`idle`/`non-gioco` è quindi promettente come primo livello di un modello gerarchico.

Il miglior esperimento complessivo tra questi tre è **exp_27**, con:

```text
Val Macro F1 = 0.8367
Val Accuracy = 0.8685
Val Weighted F1 = 0.8622
```

Tuttavia, `idle` resta la classe più problematica anche nelle rimappature semplificate. In exp_27 ottiene:

```text
Precision = 0.75
Recall = 0.55
F1-score = 0.63
```

La confusion matrix di exp_27 mostra che molte clip `idle` vengono confuse soprattutto con `passaggio` e in parte con `non-gioco`:

```text
idle -> passaggio: 28
idle -> non-gioco: 12
idle -> tiro: 2
```

Questa osservazione è coerente con la natura della classe `idle`: rappresenta gioco attivo senza azione rilevante, ma può contenere movimento dei giocatori, palla in movimento e situazioni vicine temporalmente a passaggi o tiri.

La conclusione principale è che il modello distingue bene **azione di tiro** e **tipo di tiro**, mentre la distinzione **segnato/sbagliato** rimane il punto più critico. Per questo motivo, l'approccio gerarchico più sensato è:

```text
Stadio 1:
passaggio / tiro / idle / non-gioco

Stadio 2, solo se tiro:
tiroDaDue / tiroDaTre / tiroLibero

Stadio 3, solo se tiro:
tiro0 / tiro1
```

Il prossimo passo consigliato è analizzare manualmente gli errori di `idle`, soprattutto i casi `idle -> passaggio`, e aggiungere un ulteriore esperimento binario solo sulle clip di tiro per verificare se l'esito `tiro0`/`tiro1` migliora quando il modello non deve più distinguere anche `passaggio`, `idle` e `non-gioco`.
