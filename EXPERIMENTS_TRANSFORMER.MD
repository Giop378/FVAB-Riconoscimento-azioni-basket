# Esperimenti Temporal Transformer

Questo file tiene traccia sintetica degli esperimenti effettuati sostituendo il classificatore GRU/BiGRU con un classificatore basato su **Temporal Transformer Encoder**.

Gli esperimenti usano le feature già estratte con **ConvNeXt-Tiny frozen** e confrontano diverse configurazioni del classificatore temporale, del pooling e delle strategie per gestire lo sbilanciamento tra classi.

## Tabella riassuntiva

| ID | Modello | Feature extractor | Epoche | Batch size | LR | d_model | Layers | Heads | FF dim | Dropout | Weight decay | Pooling | Class weight power | Sampler | Val Loss | Val Accuracy | Val Macro F1 | Val Weighted F1 | Output dir |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| exp_10 | Temporal Transformer | ConvNeXt-Tiny frozen | 80 | 64 | 1e-4 | 384 | 3 | 6 | 1024 | 0.35 | 2e-3 | CLS | 0.5 | WeightedRandomSampler power 0.7 | 1.0768 | 0.6852 | 0.4289 | 0.6932 | outputs/exp_10_convnext_temporal_transformer |
| exp_11 | Temporal Transformer regularized | ConvNeXt-Tiny frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | CLS | 0.5 | No sampler | 0.7902 | 0.7389 | 0.4046 | 0.7087 | outputs/exp_11_transformer_no_sampler |
| exp_12 | Temporal Transformer regularized | ConvNeXt-Tiny frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | CLS | 0.5 | WeightedRandomSampler power 0.4 | 0.9510 | 0.6889 | 0.4353 | 0.6987 | outputs/exp_12_transformer_soft_sampler |
| exp_13 | Temporal Transformer regularized | ConvNeXt-Tiny frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | Mean | 0.5 | WeightedRandomSampler power 0.4 | 0.9363 | 0.7167 | 0.4557 | 0.7187 | outputs/exp_13_transformer_soft_sampler_mean |
| exp_14 | Temporal Transformer regularized | ConvNeXt-Tiny frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | Mean | 0.5 | WeightedRandomSampler power 0.5 | 0.9263 | 0.6889 | 0.4381 | 0.7048 | outputs/exp_14_transformer_mean_sampler_05 |

## Risultati aggregati su validation - 9 classi

|ID|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|
|exp_10|0.6852|0.43|0.45|0.43|0.71|0.69|0.69|
|exp_11|0.7389|0.49|0.42|0.40|0.72|0.74|0.71|
|exp_12|0.6889|0.42|0.48|0.44|0.72|0.69|0.70|
|exp_13|0.7167|0.45|0.48|0.46|0.72|0.72|0.72|
|exp_14|0.6889|0.43|0.48|0.44|0.73|0.69|0.70|

## Risultati aggregati sulle sole 7 azioni reali

|ID|Micro Precision|Micro Recall|Micro F1|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|exp_10|0.66|0.74|0.70|0.37|0.42|0.38|0.70|0.74|0.71|
|exp_11|0.71|0.81|0.76|0.43|0.38|0.35|0.71|0.81|0.73|
|exp_12|0.68|0.71|0.70|0.35|0.44|0.38|0.71|0.71|0.71|
|exp_13|0.71|0.74|0.72|0.39|0.43|0.40|0.72|0.74|0.73|
|exp_14|0.68|0.69|0.68|0.36|0.43|0.38|0.74|0.69|0.70|

## Risultati per classe su validation

### exp_10 - Temporal Transformer, CLS pooling, sampler power 0.7

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.80|0.84|0.82|212|
|tiroDaDue0|0.50|0.52|0.51|21|
|tiroDaDue1|0.33|0.18|0.24|11|
|tiroDaTre0|0.50|0.58|0.54|12|
|tiroDaTre1|0.17|0.33|0.22|3|
|tiroLibero0|0.08|0.29|0.12|7|
|tiroLibero1|0.20|0.18|0.19|11|
|idle|0.42|0.38|0.40|93|
|non-gioco|0.89|0.77|0.82|170|

### exp_11 - Temporal Transformer regolarizzato, CLS pooling, senza WeightedRandomSampler

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.78|0.93|0.85|212|
|tiroDaDue0|0.44|0.67|0.53|21|
|tiroDaDue1|1.00|0.09|0.17|11|
|tiroDaTre0|0.42|0.42|0.42|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.00|0.00|0.00|7|
|tiroLibero1|0.40|0.55|0.46|11|
|idle|0.54|0.27|0.36|93|
|non-gioco|0.83|0.89|0.86|170|

### exp_12 - Temporal Transformer regolarizzato, CLS pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.84|0.82|0.83|212|
|tiroDaDue0|0.41|0.33|0.37|21|
|tiroDaDue1|0.18|0.18|0.18|11|
|tiroDaTre0|0.33|0.75|0.46|12|
|tiroDaTre1|0.17|0.33|0.22|3|
|tiroLibero0|0.33|0.29|0.31|7|
|tiroLibero1|0.21|0.36|0.27|11|
|idle|0.47|0.49|0.48|93|
|non-gioco|0.85|0.75|0.80|170|

### exp_13 - Temporal Transformer regolarizzato, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.82|0.83|0.83|212|
|tiroDaDue0|0.45|0.48|0.47|21|
|tiroDaDue1|0.25|0.18|0.21|11|
|tiroDaTre0|0.53|0.67|0.59|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.31|0.57|0.40|7|
|tiroLibero1|0.33|0.27|0.30|11|
|idle|0.45|0.45|0.45|93|
|non-gioco|0.88|0.83|0.85|170|

### exp_14 - Temporal Transformer regolarizzato, mean pooling, sampler power 0.5

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.85|0.77|0.81|212|
|tiroDaDue0|0.57|0.38|0.46|21|
|tiroDaDue1|0.27|0.27|0.27|11|
|tiroDaTre0|0.41|0.75|0.53|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.22|0.57|0.32|7|
|tiroLibero1|0.21|0.27|0.24|11|
|idle|0.45|0.55|0.50|93|
|non-gioco|0.88|0.77|0.82|170|

## Comandi utilizzati

### exp_10

```bash
python -m src.training.train \
  --features-root data/features/convnext_tiny \
  --epochs 80 \
  --batch-size 64 \
  --lr 1e-4 \
  --d-model 384 \
  --num-layers 3 \
  --num-heads 6 \
  --ff-dim 1024 \
  --dropout 0.35 \
  --weight-decay 2e-3 \
  --pooling cls \
  --class-weight-power 0.5 \
  --sampler-power 0.7 \
  --scheduler-patience 8 \
  --output-dir outputs/exp_10_convnext_temporal_transformer
```

### exp_11

```bash
python -m src.training.train \
  --features-root data/features/convnext_tiny \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling cls \
  --class-weight-power 0.5 \
  --no-weighted-sampler \
  --scheduler-patience 5 \
  --output-dir outputs/exp_11_transformer_no_sampler
```

### exp_12

```bash
python -m src.training.train \
  --features-root data/features/convnext_tiny \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling cls \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --output-dir outputs/exp_12_transformer_soft_sampler
```

### exp_13

```bash
python -m src.training.train \
  --features-root data/features/convnext_tiny \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
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
  --output-dir outputs/exp_13_transformer_soft_sampler_mean
```

### exp_14

```bash
python -m src.training.train \
  --features-root data/features/convnext_tiny \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --d-model 256 \
  --num-layers 2 \
  --num-heads 4 \
  --ff-dim 768 \
  --dropout 0.45 \
  --weight-decay 5e-3 \
  --pooling mean \
  --class-weight-power 0.5 \
  --sampler-power 0.5 \
  --scheduler-patience 5 \
  --output-dir outputs/exp_14_transformer_mean_sampler_05
```

## Nota finale

Il miglior modello tra gli esperimenti Temporal Transformer è **exp_13**, con:

```text
Val Macro F1 = 0.4557
Val Accuracy = 0.7167
Val Weighted F1 = 0.7187
Macro F1 sulle sole 7 azioni reali = 0.40
```

Il confronto tra gli esperimenti suggerisce che:

- **exp_10** usa un Transformer più grande e un `WeightedRandomSampler` più aggressivo (`sampler-power = 0.7`). Il modello raggiunge rapidamente ottime metriche sul training, ma mostra overfitting e non supera exp_13 in validation.
- **exp_11** rimuove il `WeightedRandomSampler`. Questo aumenta accuracy e weighted F1 rispetto ad alcune configurazioni, ma peggiora la Macro F1 perché il modello tende a privilegiare le classi più frequenti e perde completamente alcune classi rare.
- **exp_12** reintroduce un sampler più moderato (`sampler-power = 0.4`), migliorando la Macro F1 rispetto a exp_11.
- **exp_13** sostituisce il CLS pooling con mean pooling e ottiene il miglior compromesso complessivo.
- **exp_14** aumenta il `sampler-power` da 0.4 a 0.5, ma non migliora rispetto a exp_13.

Il miglior compromesso attuale è quindi:

```text
ConvNeXt-Tiny frozen + Temporal Transformer
d_model = 256
num_layers = 2
num_heads = 4
ff_dim = 768
dropout = 0.45
weight_decay = 5e-3
pooling = mean
class_weight_power = 0.5
sampler_power = 0.4
```

Rispetto alla baseline BiGRU, il Temporal Transformer non produce un miglioramento netto e stabile sulla Macro F1. Il risultato migliore, exp_13, è comparabile con exp_09 della baseline, ma non supera chiaramente le prestazioni precedenti. Questo suggerisce che il collo di bottiglia potrebbe non essere più il classificatore temporale, ma la qualità delle feature visuali estratte da ConvNeXt-Tiny frozen.

Le classi più problematiche restano:

```text
tiroDaTre1
tiroDaDue1
tiroLibero1
```

In particolare `tiroDaTre1` ha solo 3 esempi in validation, quindi le sue metriche sono molto instabili. Il prossimo passo consigliato è mantenere la configurazione di exp_13 e cambiare feature extractor, ad esempio provando **DINOv2 frozen + Temporal Transformer mean pooling**, oppure effettuare un controllo mirato del dataset sulle classi rare e sulle clip in cui l'esito del tiro non è chiaramente visibile.
