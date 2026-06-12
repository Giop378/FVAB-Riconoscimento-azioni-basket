# Esperimenti Temporal Transformer

Questo file tiene traccia sintetica degli esperimenti effettuati sostituendo il classificatore GRU/BiGRU con un classificatore basato su **Temporal Transformer Encoder**.

Gli esperimenti usano feature già estratte con **ConvNeXt-Tiny frozen**, **DINOv2-B/14 frozen**, **DINOv2-L/14 frozen** e **DINOv3-L/16 frozen**, confrontando diverse configurazioni del classificatore temporale, del pooling e delle strategie per gestire lo sbilanciamento tra classi.

## Tabella riassuntiva

|ID|Modello|Feature extractor|Epoche|Batch size|LR|d\_model|Layers|Heads|FF dim|Dropout|Weight decay|Pooling|Class weight power|Sampler|Val Loss|Val Accuracy|Val Macro F1|Val Weighted F1|Output dir|
|-|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-|-:|-|-:|-:|-:|-:|-|
|exp\_10|Temporal Transformer|ConvNeXt-Tiny frozen|80|64|1e-4|384|3|6|1024|0.35|2e-3|CLS|0.5|WeightedRandomSampler power 0.7|1.0768|0.6852|0.4289|0.6932|outputs/exp\_10\_convnext\_temporal\_transformer|
|exp\_11|Temporal Transformer regularized|ConvNeXt-Tiny frozen|40|64|5e-5|256|2|4|768|0.45|5e-3|CLS|0.5|No sampler|0.7902|0.7389|0.4046|0.7087|outputs/exp\_11\_transformer\_no\_sampler|
|exp\_12|Temporal Transformer regularized|ConvNeXt-Tiny frozen|40|64|5e-5|256|2|4|768|0.45|5e-3|CLS|0.5|WeightedRandomSampler power 0.4|0.9510|0.6889|0.4353|0.6987|outputs/exp\_12\_transformer\_soft\_sampler|
|exp\_13|Temporal Transformer regularized|ConvNeXt-Tiny frozen|40|64|5e-5|256|2|4|768|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.9363|0.7167|0.4557|0.7187|outputs/exp\_13\_transformer\_soft\_sampler\_mean|
|exp\_14|Temporal Transformer regularized|ConvNeXt-Tiny frozen|40|64|5e-5|256|2|4|768|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.5|0.9263|0.6889|0.4381|0.7048|outputs/exp\_14\_transformer\_mean\_sampler\_05|
|exp\_15|Temporal Transformer regularized|DINOv2-B/14 frozen|40|64|5e-5|256|2|4|768|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.7226|0.7796|0.5833|0.7782|outputs/exp\_15\_dinov2\_vitb14\_transformer\_mean|
|exp\_16|Temporal Transformer regularized|DINOv2-B/14 frozen|40|64|5e-5|256|2|4|768|0.45|5e-3|CLS|0.5|WeightedRandomSampler power 0.4|0.7135|0.7852|0.5865|0.7817|outputs/exp\_16\_dinov2\_vitb14\_transformer\_cls|
|exp\_17|Temporal Transformer regularized|DINOv2-L/14 frozen|40|64|5e-5|256|2|4|768|0.45|5e-3|CLS|0.5|WeightedRandomSampler power 0.4|0.6047|0.7981|0.5749|0.7979|outputs/exp\_17\_dinov2\_vitl14\_transformer\_cls|
|exp\_18|Temporal Transformer regularized|DINOv2-L/14 frozen|40|64|5e-5|256|2|4|768|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.6160|0.7944|0.5762|0.7840|outputs/exp\_18\_dinov2\_vitl14\_transformer\_mean|
|exp\_19|Temporal Transformer d384|DINOv2-L/14 frozen|40|64|5e-5|384|2|6|1024|0.45|5e-3|CLS|0.5|WeightedRandomSampler power 0.4|0.5823|0.8167|0.5672|0.8079|outputs/exp\_19\_dinov2\_vitl14\_transformer\_cls\_d384|
|exp\_20|Temporal Transformer d384 balanced|DINOv2-L/14 frozen|40|64|5e-5|384|2|6|1024|0.50|7e-3|CLS|0.6|WeightedRandomSampler power 0.5|0.6246|0.7889|0.5455|0.7894|outputs/exp\_20\_dinov2\_vitl14\_transformer\_cls\_d384\_balanced|
|exp\_21|Temporal Transformer d384|DINOv3-L/16 frozen|40|64|5e-5|384|2|6|1024|0.45|5e-3|CLS|0.5|WeightedRandomSampler power 0.4|0.5752|0.8148|0.6256|0.8095|outputs/exp\_21\_dinov3\_vitl16\_transformer\_cls\_d384|
|exp\_22|Temporal Transformer d384|DINOv3-L/16 frozen|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.5707|0.8296|0.6146|0.8232|outputs/exp\_22\_dinov3\_vitl16\_transformer\_mean\_d384|
|exp\_23|Temporal Transformer d384 + augmentation|DINOv3-L/16 frozen|60|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.5|WeightedRandomSampler power 0.4|0.6953|0.8185|0.6025|0.8081|outputs/exp\_23\_dinov3\_vitl16\_transformer\_mean\_d384\_aug\_shots|
|exp\_24|Temporal Transformer d384 + augmentation soft balance|DINOv3-L/16 frozen|40|64|5e-5|384|2|6|1024|0.45|5e-3|Mean|0.3|WeightedRandomSampler power 0.2|0.6852|0.8241|0.5941|0.8102|outputs/exp\_24\_dinov3\_vitl16\_transformer\_mean\_d384\_aug\_shots\_soft\_balance|

## Risultati aggregati su validation - 9 classi

|ID|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|
|exp\_10|0.6852|0.43|0.45|0.43|0.71|0.69|0.69|
|exp\_11|0.7389|0.49|0.42|0.40|0.72|0.74|0.71|
|exp\_12|0.6889|0.42|0.48|0.44|0.72|0.69|0.70|
|exp\_13|0.7167|0.45|0.48|0.46|0.72|0.72|0.72|
|exp\_14|0.6889|0.43|0.48|0.44|0.73|0.69|0.70|
|exp\_15|0.7796|0.63|0.58|0.58|0.79|0.78|0.78|
|exp\_16|0.7852|0.66|0.58|0.59|0.79|0.79|0.78|
|exp\_17|0.7981|0.57|0.59|0.57|0.80|0.80|0.80|
|exp\_18|0.7944|0.64|0.58|0.58|0.81|0.79|0.78|
|exp\_19|0.8167|0.59|0.57|0.57|0.81|0.82|0.81|
|exp\_20|0.7889|0.54|0.56|0.55|0.79|0.79|0.79|
|exp\_21|0.8148|0.66|0.62|0.63|0.81|0.81|0.81|
|exp\_22|0.8296|0.65|0.61|0.61|0.83|0.83|0.82|
|exp\_23|0.8185|0.64|0.59|0.60|0.81|0.82|0.81|
|exp\_24|0.8241|0.65|0.58|0.59|0.82|0.82|0.81|

## Risultati aggregati sulle sole 7 azioni reali

|ID|Micro Precision|Micro Recall|Micro F1|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|exp\_10|0.66|0.74|0.70|0.37|0.42|0.38|0.70|0.74|0.71|
|exp\_11|0.71|0.81|0.76|0.43|0.38|0.35|0.71|0.81|0.73|
|exp\_12|0.68|0.71|0.70|0.35|0.44|0.38|0.71|0.71|0.71|
|exp\_13|0.71|0.74|0.72|0.39|0.43|0.40|0.72|0.74|0.73|
|exp\_14|0.68|0.69|0.68|0.36|0.43|0.38|0.74|0.69|0.70|
|exp\_15|0.77|0.81|0.79|0.59|0.55|0.54|0.79|0.81|0.79|
|exp\_16|0.77|0.77|0.77|0.63|0.53|0.54|0.79|0.77|0.77|
|exp\_17|0.77|0.81|0.79|0.51|0.54|0.52|0.78|0.81|0.79|
|exp\_18|0.73|0.86|0.79|0.58|0.55|0.54|0.74|0.86|0.79|
|exp\_19|0.80|0.83|0.81|0.53|0.52|0.51|0.79|0.83|0.80|
|exp\_20|0.78|0.80|0.79|0.48|0.51|0.49|0.78|0.80|0.79|
|exp\_21|0.81|0.81|0.81|0.63|0.58|0.58|0.82|0.81|0.81|
|exp\_22|0.80|0.86|0.83|0.59|0.56|0.57|0.80|0.86|0.82|
|exp\_23|0.79|0.86|0.82|0.59|0.56|0.56|0.79|0.86|0.82|
|exp\_24|0.78|0.86|0.82|0.59|0.54|0.55|0.78|0.86|0.81|

## Risultati per classe su validation

### exp\_10 - Temporal Transformer, CLS pooling, sampler power 0.7

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

### exp\_11 - Temporal Transformer regolarizzato, CLS pooling, senza WeightedRandomSampler

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

### exp\_12 - Temporal Transformer regolarizzato, CLS pooling, sampler power 0.4

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

### exp\_13 - Temporal Transformer regolarizzato, mean pooling, sampler power 0.4

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

### exp\_14 - Temporal Transformer regolarizzato, mean pooling, sampler power 0.5

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

### exp\_15 - DINOv2-B/14 frozen, Temporal Transformer regolarizzato, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.90|0.88|212|
|tiroDaDue0|0.62|0.38|0.47|21|
|tiroDaDue1|0.83|0.45|0.59|11|
|tiroDaTre0|0.43|0.83|0.57|12|
|tiroDaTre1|0.50|0.33|0.40|3|
|tiroLibero0|0.67|0.57|0.62|7|
|tiroLibero1|0.22|0.36|0.28|11|
|idle|0.62|0.54|0.57|93|
|non-gioco|0.88|0.87|0.88|170|

### exp\_16 - DINOv2-B/14 frozen, Temporal Transformer regolarizzato, CLS pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.87|0.85|0.86|212|
|tiroDaDue0|0.58|0.33|0.42|21|
|tiroDaDue1|0.30|0.55|0.39|11|
|tiroDaTre0|0.50|0.83|0.62|12|
|tiroDaTre1|1.00|0.33|0.50|3|
|tiroLibero0|0.80|0.57|0.67|7|
|tiroLibero1|0.38|0.27|0.32|11|
|idle|0.63|0.56|0.59|93|
|non-gioco|0.87|0.94|0.91|170|

### exp\_17 - DINOv2-L/14 frozen, Temporal Transformer regolarizzato, CLS pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.87|0.90|0.88|212|
|tiroDaDue0|0.50|0.48|0.49|21|
|tiroDaDue1|0.50|0.64|0.56|11|
|tiroDaTre0|0.58|0.58|0.58|12|
|tiroDaTre1|0.20|0.33|0.25|3|
|tiroLibero0|0.50|0.43|0.46|7|
|tiroLibero1|0.42|0.45|0.43|11|
|idle|0.62|0.57|0.60|93|
|non-gioco|0.93|0.91|0.92|170|

### exp\_18 - DINOv2-L/14 frozen, Temporal Transformer regolarizzato, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.83|0.95|0.89|212|
|tiroDaDue0|0.38|0.71|0.50|21|
|tiroDaDue1|0.30|0.27|0.29|11|
|tiroDaTre0|0.58|0.58|0.58|12|
|tiroDaTre1|1.00|0.33|0.50|3|
|tiroLibero0|0.57|0.57|0.57|7|
|tiroLibero1|0.38|0.45|0.42|11|
|idle|0.82|0.39|0.53|93|
|non-gioco|0.91|0.92|0.91|170|

### exp\_19 - DINOv2-L/14 frozen, Temporal Transformer d384, CLS pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.92|0.89|212|
|tiroDaDue0|0.52|0.62|0.57|21|
|tiroDaDue1|0.50|0.55|0.52|11|
|tiroDaTre0|0.78|0.58|0.67|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.45|0.71|0.56|7|
|tiroLibero1|0.60|0.27|0.38|11|
|idle|0.68|0.54|0.60|93|
|non-gioco|0.90|0.95|0.93|170|

### exp\_20 - DINOv2-L/14 frozen, Temporal Transformer d384 bilanciato, CLS pooling, sampler power 0.5

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.88|0.89|0.88|212|
|tiroDaDue0|0.45|0.48|0.47|21|
|tiroDaDue1|0.44|0.64|0.52|11|
|tiroDaTre0|0.57|0.67|0.62|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.60|0.43|0.50|7|
|tiroLibero1|0.42|0.45|0.43|11|
|idle|0.60|0.59|0.59|93|
|non-gioco|0.92|0.88|0.90|170|

### exp\_21 - DINOv3-L/16 frozen, Temporal Transformer d384, CLS pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.88|0.90|0.89|212|
|tiroDaDue0|0.62|0.38|0.47|21|
|tiroDaDue1|0.37|0.64|0.47|11|
|tiroDaTre0|0.60|0.75|0.67|12|
|tiroDaTre1|0.33|0.33|0.33|3|
|tiroLibero0|0.83|0.71|0.77|7|
|tiroLibero1|0.80|0.36|0.50|11|
|idle|0.66|0.57|0.61|93|
|non-gioco|0.89|0.96|0.92|170|

### exp\_22 - DINOv3-L/16 frozen, Temporal Transformer d384, mean pooling, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.87|0.95|0.91|212|
|tiroDaDue0|0.62|0.62|0.62|21|
|tiroDaDue1|0.38|0.27|0.32|11|
|tiroDaTre0|0.64|0.75|0.69|12|
|tiroDaTre1|0.25|0.33|0.29|3|
|tiroLibero0|1.00|0.57|0.73|7|
|tiroLibero1|0.38|0.45|0.42|11|
|idle|0.76|0.56|0.65|93|
|non-gioco|0.90|0.94|0.92|170|

### exp\_23 - DINOv3-L/16 frozen, Temporal Transformer d384, mean pooling, augmentation sui tiri, sampler power 0.4

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.96|0.91|212|
|tiroDaDue0|0.56|0.48|0.51|21|
|tiroDaDue1|0.27|0.27|0.27|11|
|tiroDaTre0|0.69|0.92|0.79|12|
|tiroDaTre1|0.25|0.33|0.29|3|
|tiroLibero0|1.00|0.57|0.73|7|
|tiroLibero1|0.50|0.36|0.42|11|
|idle|0.75|0.51|0.60|93|
|non-gioco|0.89|0.93|0.91|170|

### exp\_24 - DINOv3-L/16 frozen, Temporal Transformer d384, mean pooling, augmentation sui tiri, bilanciamento più leggero

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.96|0.91|212|
|tiroDaDue0|0.52|0.62|0.57|21|
|tiroDaDue1|0.29|0.18|0.22|11|
|tiroDaTre0|0.47|0.75|0.58|12|
|tiroDaTre1|0.50|0.33|0.40|3|
|tiroLibero0|1.00|0.57|0.73|7|
|tiroLibero1|0.50|0.36|0.42|11|
|idle|0.84|0.46|0.60|93|
|non-gioco|0.88|0.97|0.92|170|

## Comandi utilizzati

### exp\_10

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 80 \\
  --batch-size 64 \\
  --lr 1e-4 \\
  --d-model 384 \\
  --num-layers 3 \\
  --num-heads 6 \\
  --ff-dim 1024 \\
  --dropout 0.35 \\
  --weight-decay 2e-3 \\
  --pooling cls \\
  --class-weight-power 0.5 \\
  --sampler-power 0.7 \\
  --scheduler-patience 8 \\
  --output-dir outputs/exp\_10\_convnext\_temporal\_transformer
```

### exp\_11

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --d-model 256 \\
  --num-layers 2 \\
  --num-heads 4 \\
  --ff-dim 768 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling cls \\
  --class-weight-power 0.5 \\
  --no-weighted-sampler \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_11\_transformer\_no\_sampler
```

### exp\_12

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --d-model 256 \\
  --num-layers 2 \\
  --num-heads 4 \\
  --ff-dim 768 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling cls \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_12\_transformer\_soft\_sampler
```

### exp\_13

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --d-model 256 \\
  --num-layers 2 \\
  --num-heads 4 \\
  --ff-dim 768 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling mean \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_13\_transformer\_soft\_sampler\_mean
```

### exp\_14

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --d-model 256 \\
  --num-layers 2 \\
  --num-heads 4 \\
  --ff-dim 768 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling mean \\
  --class-weight-power 0.5 \\
  --sampler-power 0.5 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_14\_transformer\_mean\_sampler\_05
```

### exp\_15

```bash
python -m src.training.train \\
  --features-root data/features/dinov2\_vitb14\_336 \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 768 \\
  --d-model 256 \\
  --num-layers 2 \\
  --num-heads 4 \\
  --ff-dim 768 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling mean \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_15\_dinov2\_vitb14\_transformer\_mean
```

### exp\_16

```bash
python -m src.training.train \\
  --features-root data/features/dinov2\_vitb14\_336 \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 768 \\
  --d-model 256 \\
  --num-layers 2 \\
  --num-heads 4 \\
  --ff-dim 768 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling cls \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_16\_dinov2\_vitb14\_transformer\_cls
```

### exp\_17

```bash
python -m src.training.train \\
  --features-root data/features/dinov2\_vitl14\_336 \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 1024 \\
  --d-model 256 \\
  --num-layers 2 \\
  --num-heads 4 \\
  --ff-dim 768 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling cls \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_17\_dinov2\_vitl14\_transformer\_cls
```

### exp\_18

```bash
python -m src.training.train \\
  --features-root data/features/dinov2\_vitl14\_336 \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 1024 \\
  --d-model 256 \\
  --num-layers 2 \\
  --num-heads 4 \\
  --ff-dim 768 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling mean \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_18\_dinov2\_vitl14\_transformer\_mean
```

### exp\_19

```bash
python -m src.training.train \\
  --features-root data/features/dinov2\_vitl14\_336 \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 1024 \\
  --d-model 384 \\
  --num-layers 2 \\
  --num-heads 6 \\
  --ff-dim 1024 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling cls \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_19\_dinov2\_vitl14\_transformer\_cls\_d384
```

### exp\_20

```bash
python -m src.training.train \\
  --features-root data/features/dinov2\_vitl14\_336 \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 1024 \\
  --d-model 384 \\
  --num-layers 2 \\
  --num-heads 6 \\
  --ff-dim 1024 \\
  --dropout 0.50 \\
  --weight-decay 7e-3 \\
  --pooling cls \\
  --class-weight-power 0.6 \\
  --sampler-power 0.5 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_20\_dinov2\_vitl14\_transformer\_cls\_d384\_balanced
```

### exp\_21

```bash
python -m src.training.train \\
  --features-root data/features/dinov3\_vitl16\_336 \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 1024 \\
  --d-model 384 \\
  --num-layers 2 \\
  --num-heads 6 \\
  --ff-dim 1024 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling cls \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_21\_dinov3\_vitl16\_transformer\_cls\_d384
```

### exp\_22

```bash
python -m src.training.train \\
  --features-root data/features/dinov3\_vitl16\_336 \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 1024 \\
  --d-model 384 \\
  --num-layers 2 \\
  --num-heads 6 \\
  --ff-dim 1024 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling mean \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_22\_dinov3\_vitl16\_transformer\_mean\_d384
```

### exp\_23

```bash
python -m src.training.train \\
  --features-root data/features/dinov3\_vitl16\_336\_aug\_shots \\
  --epochs 60 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 1024 \\
  --d-model 384 \\
  --num-layers 2 \\
  --num-heads 6 \\
  --ff-dim 1024 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling mean \\
  --class-weight-power 0.5 \\
  --sampler-power 0.4 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_23\_dinov3\_vitl16\_transformer\_mean\_d384\_aug\_shots
```

### exp\_24

```bash
python -m src.training.train \\
  --features-root data/features/dinov3\_vitl16\_336\_aug\_shots \\
  --epochs 40 \\
  --batch-size 64 \\
  --lr 5e-5 \\
  --input-dim 1024 \\
  --d-model 384 \\
  --num-layers 2 \\
  --num-heads 6 \\
  --ff-dim 1024 \\
  --dropout 0.45 \\
  --weight-decay 5e-3 \\
  --pooling mean \\
  --class-weight-power 0.3 \\
  --sampler-power 0.2 \\
  --scheduler-patience 5 \\
  --output-dir outputs/exp\_24\_dinov3\_vitl16\_transformer\_mean\_d384\_aug\_shots\_soft\_balance
```

## Nota finale

Il miglior modello in termini di **Val Macro F1 sulle 9 classi** è **exp\_21**, con:

```text
Val Macro F1 = 0.6256
Val Accuracy = 0.8148
Val Weighted F1 = 0.8095
Macro F1 sulle sole 7 azioni reali = 0.58
```

Il miglior compromesso operativo complessivo resta invece **exp\_22**, con:

```text
Val Macro F1 = 0.6146
Val Accuracy = 0.8296
Val Weighted F1 = 0.8232
Micro F1 sulle sole 7 azioni reali = 0.83
Weighted F1 sulle sole 7 azioni reali = 0.82
```

Il confronto tra gli esperimenti suggerisce che:

* **exp\_10** usa un Transformer più grande e un `WeightedRandomSampler` più aggressivo (`sampler-power = 0.7`). Il modello raggiunge rapidamente ottime metriche sul training, ma mostra overfitting e non supera le configurazioni successive in validation.
* **exp\_11** rimuove il `WeightedRandomSampler`. Questo aumenta accuracy e weighted F1 rispetto ad alcune configurazioni, ma peggiora la Macro F1 perché il modello tende a privilegiare le classi più frequenti e perde completamente alcune classi rare.
* **exp\_12** reintroduce un sampler più moderato (`sampler-power = 0.4`), migliorando la Macro F1 rispetto a exp\_11.
* **exp\_13** sostituisce il CLS pooling con mean pooling e ottiene il miglior compromesso complessivo tra gli esperimenti basati su ConvNeXt-Tiny.
* **exp\_14** aumenta il `sampler-power` da 0.4 a 0.5, ma non migliora rispetto a exp\_13.
* **exp\_15** sostituisce ConvNeXt-Tiny con **DINOv2-B/14 frozen** e usa mean pooling. Il miglioramento è netto rispetto agli esperimenti precedenti: la Val Macro F1 passa da 0.4557 di exp\_13 a 0.5833.
* **exp\_16** usa le stesse feature DINOv2-B/14 di exp\_15, ma cambia il pooling da mean a CLS. Ottiene il miglior risultato tra gli esperimenti DINOv2-B/14, con Val Macro F1 = 0.5865.
* **exp\_17** sostituisce DINOv2-B/14 con **DINOv2-L/14 frozen**, mantenendo CLS pooling e `d\_model = 256`. Migliora accuracy e weighted F1 rispetto a exp\_16, ma non migliora la Macro F1 complessiva.
* **exp\_18** usa DINOv2-L/14 con mean pooling. Ottiene una Macro F1 molto vicina a exp\_17 e una buona recall sulle azioni reali, ma tende a generare più falsi positivi sulle azioni, in particolare confondendo diverse clip `idle` con `passaggio`.
* **exp\_19** aumenta la capacità del classificatore temporale usando `d\_model = 384`, `num\_heads = 6` e `ff\_dim = 1024`. Ottiene la migliore accuracy e il miglior weighted F1 tra gli esperimenti DINOv2, oltre alla migliore micro F1 sulle 7 azioni reali fino a quel punto.
* **exp\_20** parte da exp\_19 e aumenta regolarizzazione e bilanciamento (`dropout = 0.50`, `weight\_decay = 7e-3`, `class\_weight\_power = 0.6`, `sampler\_power = 0.5`). L'esperimento non migliora exp\_19 e peggiora sia accuracy sia Macro F1, suggerendo che un bilanciamento più aggressivo non è vantaggioso in questa configurazione.
* **exp\_21** sostituisce DINOv2-L/14 con **DINOv3-L/16 frozen**, mantenendo la configurazione di exp\_19 e CLS pooling. Ottiene la migliore Val Macro F1 complessiva, migliorando soprattutto il bilanciamento tra le classi e recuperando parzialmente classi rare come `tiroDaTre1`, `tiroLibero0` e `tiroLibero1`.
* **exp\_22** usa DINOv3-L/16 con mean pooling. Ottiene la migliore accuracy, il miglior weighted F1 sulle 9 classi e la migliore micro F1 sulle 7 azioni reali. La Macro F1 è leggermente inferiore rispetto a exp\_21, ma il comportamento globale risulta più solido.
* **exp\_23** introduce data augmentation leggera sulle classi di tiro mantenendo il bilanciamento standard (`class\_weight\_power = 0.5`, `sampler\_power = 0.4`). L'augmentation migliora alcune classi, in particolare `tiroDaTre0`, ma non supera exp\_22 sulle metriche globali.
* **exp\_24** usa le stesse feature augmentate di exp\_23 ma riduce il bilanciamento (`class\_weight\_power = 0.3`, `sampler\_power = 0.2`). Migliora leggermente accuracy e weighted F1 rispetto a exp\_23, ma resta sotto exp\_22 e non migliora la Macro F1.

Il miglior compromesso attuale è quindi:

```text
DINOv3-L/16 frozen + Temporal Transformer
input\_dim = 1024
d\_model = 384
num\_layers = 2
num\_heads = 6
ff\_dim = 1024
dropout = 0.45
weight\_decay = 5e-3
pooling = Mean
class\_weight\_power = 0.5
sampler\_power = 0.4
```

Rispetto agli esperimenti basati su ConvNeXt-Tiny, il passaggio a DINOv2-B/14 produce un miglioramento netto e stabile. Il passaggio successivo a DINOv2-L/14 migliora soprattutto le metriche globali quando il classificatore temporale viene reso più capiente in exp\_19. Il passaggio a DINOv3-L/16 produce invece il miglioramento più importante sulle metriche complessive e sulle classi rare, portando la Macro F1 massima da 0.5865 di exp\_16 a 0.6256 di exp\_21.

Gli esperimenti con augmentation, exp\_23 ed exp\_24, non migliorano il miglior modello senza augmentation. Questo suggerisce che il collo di bottiglia non sia solo la quantità di esempi delle classi rare, ma anche la struttura stessa delle classi di tiro e la visibilità dell'esito nella finestra temporale della clip.

Le classi più problematiche restano:

```text
tiroDaDue1
tiroDaTre1
tiroLibero1
```

`tiroDaTre1` ha solo 3 esempi in validation, quindi la sua metrica è molto instabile. Per questo motivo exp\_21 resta importante come miglior modello in termini di Macro F1, mentre exp\_22 è più adatto come modello operativo complessivo.

Il prossimo passo consigliato non è aumentare ulteriormente la dimensione del modello né continuare con augmentation generica, ma seguire un'analisi più mirata sulle classi di tiro. In particolare, conviene provare una rimappatura delle classi di tiro in `tiro0` e `tiro1`, cioè tiro sbagliato e tiro segnato, per capire se il modello non riesce a riconoscere l'esito del tiro oppure se è confuso dalla distinzione tra tiro da due, tiro da tre e tiro libero.

