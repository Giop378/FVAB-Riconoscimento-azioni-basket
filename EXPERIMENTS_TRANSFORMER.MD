# Esperimenti Temporal Transformer

Questo file tiene traccia sintetica degli esperimenti effettuati sostituendo il classificatore GRU/BiGRU con un classificatore basato su **Temporal Transformer Encoder**.

Gli esperimenti usano feature già estratte con **ConvNeXt-Tiny frozen**, **DINOv2-B/14 frozen**, **DINOv2-L/14 frozen** e **DINOv3-L/16 frozen**, confrontando diverse configurazioni del classificatore temporale, del pooling e delle strategie per gestire lo sbilanciamento tra classi.

## Tabella riassuntiva

| ID | Modello | Feature extractor | Epoche | Batch size | LR | d_model | Layers | Heads | FF dim | Dropout | Weight decay | Pooling | Class weight power | Sampler | Val Loss | Val Accuracy | Val Macro F1 | Val Weighted F1 | Output dir |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| exp_10 | Temporal Transformer | ConvNeXt-Tiny frozen | 80 | 64 | 1e-4 | 384 | 3 | 6 | 1024 | 0.35 | 2e-3 | CLS | 0.5 | WeightedRandomSampler power 0.7 | 1.0768 | 0.6852 | 0.4289 | 0.6932 | outputs/exp_10_convnext_temporal_transformer |
| exp_11 | Temporal Transformer regularized | ConvNeXt-Tiny frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | CLS | 0.5 | No sampler | 0.7902 | 0.7389 | 0.4046 | 0.7087 | outputs/exp_11_transformer_no_sampler |
| exp_12 | Temporal Transformer regularized | ConvNeXt-Tiny frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | CLS | 0.5 | WeightedRandomSampler power 0.4 | 0.9510 | 0.6889 | 0.4353 | 0.6987 | outputs/exp_12_transformer_soft_sampler |
| exp_13 | Temporal Transformer regularized | ConvNeXt-Tiny frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | Mean | 0.5 | WeightedRandomSampler power 0.4 | 0.9363 | 0.7167 | 0.4557 | 0.7187 | outputs/exp_13_transformer_soft_sampler_mean |
| exp_14 | Temporal Transformer regularized | ConvNeXt-Tiny frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | Mean | 0.5 | WeightedRandomSampler power 0.5 | 0.9263 | 0.6889 | 0.4381 | 0.7048 | outputs/exp_14_transformer_mean_sampler_05 |
| exp_15 | Temporal Transformer regularized | DINOv2-B/14 frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | Mean | 0.5 | WeightedRandomSampler power 0.4 | 0.7226 | 0.7796 | 0.5833 | 0.7782 | outputs/exp_15_dinov2_vitb14_transformer_mean |
| exp_16 | Temporal Transformer regularized | DINOv2-B/14 frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | CLS | 0.5 | WeightedRandomSampler power 0.4 | 0.7135 | 0.7852 | 0.5865 | 0.7817 | outputs/exp_16_dinov2_vitb14_transformer_cls |
| exp_17 | Temporal Transformer regularized | DINOv2-L/14 frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | CLS | 0.5 | WeightedRandomSampler power 0.4 | 0.6047 | 0.7981 | 0.5749 | 0.7979 | outputs/exp_17_dinov2_vitl14_transformer_cls |
| exp_18 | Temporal Transformer regularized | DINOv2-L/14 frozen | 40 | 64 | 5e-5 | 256 | 2 | 4 | 768 | 0.45 | 5e-3 | Mean | 0.5 | WeightedRandomSampler power 0.4 | 0.6160 | 0.7944 | 0.5762 | 0.7840 | outputs/exp_18_dinov2_vitl14_transformer_mean |
| exp_19 | Temporal Transformer d384 | DINOv2-L/14 frozen | 40 | 64 | 5e-5 | 384 | 2 | 6 | 1024 | 0.45 | 5e-3 | CLS | 0.5 | WeightedRandomSampler power 0.4 | 0.5823 | 0.8167 | 0.5672 | 0.8079 | outputs/exp_19_dinov2_vitl14_transformer_cls_d384 |
| exp_20 | Temporal Transformer d384 balanced | DINOv2-L/14 frozen | 40 | 64 | 5e-5 | 384 | 2 | 6 | 1024 | 0.50 | 7e-3 | CLS | 0.6 | WeightedRandomSampler power 0.5 | 0.6246 | 0.7889 | 0.5455 | 0.7894 | outputs/exp_20_dinov2_vitl14_transformer_cls_d384_balanced |
| exp_21 | Temporal Transformer d384 | DINOv3-L/16 frozen | 40 | 64 | 5e-5 | 384 | 2 | 6 | 1024 | 0.45 | 5e-3 | CLS | 0.5 | WeightedRandomSampler power 0.4 | 0.5752 | 0.8148 | 0.6256 | 0.8095 | outputs/exp_21_dinov3_vitl16_transformer_cls_d384 |
| exp_22 | Temporal Transformer d384 | DINOv3-L/16 frozen | 40 | 64 | 5e-5 | 384 | 2 | 6 | 1024 | 0.45 | 5e-3 | Mean | 0.5 | WeightedRandomSampler power 0.4 | 0.5707 | 0.8296 | 0.6146 | 0.8232 | outputs/exp_22_dinov3_vitl16_transformer_mean_d384 |

## Risultati aggregati su validation - 9 classi

|ID|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|
|exp_10|0.6852|0.43|0.45|0.43|0.71|0.69|0.69|
|exp_11|0.7389|0.49|0.42|0.40|0.72|0.74|0.71|
|exp_12|0.6889|0.42|0.48|0.44|0.72|0.69|0.70|
|exp_13|0.7167|0.45|0.48|0.46|0.72|0.72|0.72|
|exp_14|0.6889|0.43|0.48|0.44|0.73|0.69|0.70|
|exp_15|0.7796|0.63|0.58|0.58|0.79|0.78|0.78|
|exp_16|0.7852|0.66|0.58|0.59|0.79|0.79|0.78|
|exp_17|0.7981|0.57|0.59|0.57|0.80|0.80|0.80|
|exp_18|0.7944|0.64|0.58|0.58|0.81|0.79|0.78|
|exp_19|0.8167|0.59|0.57|0.57|0.81|0.82|0.81|
|exp_20|0.7889|0.54|0.56|0.55|0.79|0.79|0.79|
|exp_21|0.8148|0.66|0.62|0.63|0.81|0.81|0.81|
|exp_22|0.8296|0.65|0.61|0.61|0.83|0.83|0.82|

## Risultati aggregati sulle sole 7 azioni reali

|ID|Micro Precision|Micro Recall|Micro F1|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|exp_10|0.66|0.74|0.70|0.37|0.42|0.38|0.70|0.74|0.71|
|exp_11|0.71|0.81|0.76|0.43|0.38|0.35|0.71|0.81|0.73|
|exp_12|0.68|0.71|0.70|0.35|0.44|0.38|0.71|0.71|0.71|
|exp_13|0.71|0.74|0.72|0.39|0.43|0.40|0.72|0.74|0.73|
|exp_14|0.68|0.69|0.68|0.36|0.43|0.38|0.74|0.69|0.70|
|exp_15|0.77|0.81|0.79|0.59|0.55|0.54|0.79|0.81|0.79|
|exp_16|0.77|0.77|0.77|0.63|0.53|0.54|0.79|0.77|0.77|
|exp_17|0.77|0.81|0.79|0.51|0.54|0.52|0.78|0.81|0.79|
|exp_18|0.73|0.86|0.79|0.58|0.55|0.54|0.74|0.86|0.79|
|exp_19|0.80|0.83|0.81|0.53|0.52|0.51|0.79|0.83|0.80|
|exp_20|0.78|0.80|0.79|0.48|0.51|0.49|0.78|0.80|0.79|
|exp_21|0.81|0.81|0.81|0.63|0.58|0.58|0.82|0.81|0.81|
|exp_22|0.80|0.86|0.83|0.59|0.56|0.57|0.80|0.86|0.82|

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

### exp_15 - DINOv2-B/14 frozen, Temporal Transformer regolarizzato, mean pooling, sampler power 0.4

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

### exp_16 - DINOv2-B/14 frozen, Temporal Transformer regolarizzato, CLS pooling, sampler power 0.4

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

### exp_17 - DINOv2-L/14 frozen, Temporal Transformer regolarizzato, CLS pooling, sampler power 0.4

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

### exp_18 - DINOv2-L/14 frozen, Temporal Transformer regolarizzato, mean pooling, sampler power 0.4

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

### exp_19 - DINOv2-L/14 frozen, Temporal Transformer d384, CLS pooling, sampler power 0.4

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

### exp_20 - DINOv2-L/14 frozen, Temporal Transformer d384 bilanciato, CLS pooling, sampler power 0.5

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

### exp_21 - DINOv3-L/16 frozen, Temporal Transformer d384, CLS pooling, sampler power 0.4

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

### exp_22 - DINOv3-L/16 frozen, Temporal Transformer d384, mean pooling, sampler power 0.4

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

### exp_15

```bash
python -m src.training.train \
  --features-root data/features/dinov2_vitb14_336 \
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
  --output-dir outputs/exp_15_dinov2_vitb14_transformer_mean
```

### exp_16

```bash
python -m src.training.train \
  --features-root data/features/dinov2_vitb14_336 \
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
  --pooling cls \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --output-dir outputs/exp_16_dinov2_vitb14_transformer_cls
```

### exp_17

```bash
python -m src.training.train \
  --features-root data/features/dinov2_vitl14_336 \
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
  --pooling cls \
  --class-weight-power 0.5 \
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --output-dir outputs/exp_17_dinov2_vitl14_transformer_cls
```

### exp_18

```bash
python -m src.training.train \
  --features-root data/features/dinov2_vitl14_336 \
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
  --sampler-power 0.4 \
  --scheduler-patience 5 \
  --output-dir outputs/exp_18_dinov2_vitl14_transformer_mean
```

### exp_19

```bash
python -m src.training.train \
  --features-root data/features/dinov2_vitl14_336 \
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
  --output-dir outputs/exp_19_dinov2_vitl14_transformer_cls_d384
```

### exp_20

```bash
python -m src.training.train \
  --features-root data/features/dinov2_vitl14_336 \
  --epochs 40 \
  --batch-size 64 \
  --lr 5e-5 \
  --input-dim 1024 \
  --d-model 384 \
  --num-layers 2 \
  --num-heads 6 \
  --ff-dim 1024 \
  --dropout 0.50 \
  --weight-decay 7e-3 \
  --pooling cls \
  --class-weight-power 0.6 \
  --sampler-power 0.5 \
  --scheduler-patience 5 \
  --output-dir outputs/exp_20_dinov2_vitl14_transformer_cls_d384_balanced
```

### exp_21

```bash
python -m src.training.train   --features-root data/features/dinov3_vitl16_336   --epochs 40   --batch-size 64   --lr 5e-5   --input-dim 1024   --d-model 384   --num-layers 2   --num-heads 6   --ff-dim 1024   --dropout 0.45   --weight-decay 5e-3   --pooling cls   --class-weight-power 0.5   --sampler-power 0.4   --scheduler-patience 5   --output-dir outputs/exp_21_dinov3_vitl16_transformer_cls_d384
```

### exp_22

```bash
python -m src.training.train   --features-root data/features/dinov3_vitl16_336   --epochs 40   --batch-size 64   --lr 5e-5   --input-dim 1024   --d-model 384   --num-layers 2   --num-heads 6   --ff-dim 1024   --dropout 0.45   --weight-decay 5e-3   --pooling mean   --class-weight-power 0.5   --sampler-power 0.4   --scheduler-patience 5   --output-dir outputs/exp_22_dinov3_vitl16_transformer_mean_d384
```

## Nota finale

Il miglior modello in termini di **Val Macro F1 sulle 9 classi** è **exp_21**, con:

```text
Val Macro F1 = 0.6256
Val Accuracy = 0.8148
Val Weighted F1 = 0.8095
Macro F1 sulle sole 7 azioni reali = 0.58
```

Il miglior compromesso operativo complessivo è invece **exp_22**, con:

```text
Val Macro F1 = 0.6146
Val Accuracy = 0.8296
Val Weighted F1 = 0.8232
Micro F1 sulle sole 7 azioni reali = 0.83
Weighted F1 sulle sole 7 azioni reali = 0.82
```

Il confronto tra gli esperimenti suggerisce che:

- **exp_10** usa un Transformer più grande e un `WeightedRandomSampler` più aggressivo (`sampler-power = 0.7`). Il modello raggiunge rapidamente ottime metriche sul training, ma mostra overfitting e non supera le configurazioni successive in validation.
- **exp_11** rimuove il `WeightedRandomSampler`. Questo aumenta accuracy e weighted F1 rispetto ad alcune configurazioni, ma peggiora la Macro F1 perché il modello tende a privilegiare le classi più frequenti e perde completamente alcune classi rare.
- **exp_12** reintroduce un sampler più moderato (`sampler-power = 0.4`), migliorando la Macro F1 rispetto a exp_11.
- **exp_13** sostituisce il CLS pooling con mean pooling e ottiene il miglior compromesso complessivo tra gli esperimenti basati su ConvNeXt-Tiny.
- **exp_14** aumenta il `sampler-power` da 0.4 a 0.5, ma non migliora rispetto a exp_13.
- **exp_15** sostituisce ConvNeXt-Tiny con **DINOv2-B/14 frozen** e usa mean pooling. Il miglioramento è netto rispetto agli esperimenti precedenti: la Val Macro F1 passa da 0.4557 di exp_13 a 0.5833.
- **exp_16** usa le stesse feature DINOv2-B/14 di exp_15, ma cambia il pooling da mean a CLS. Ottiene il miglior risultato tra gli esperimenti DINOv2-B/14, con Val Macro F1 = 0.5865.
- **exp_17** sostituisce DINOv2-B/14 con **DINOv2-L/14 frozen**, mantenendo CLS pooling e `d_model = 256`. Migliora accuracy e weighted F1 rispetto a exp_16, ma non migliora la Macro F1 complessiva.
- **exp_18** usa DINOv2-L/14 con mean pooling. Ottiene una Macro F1 molto vicina a exp_17 e una buona recall sulle azioni reali, ma tende a generare più falsi positivi sulle azioni, in particolare confondendo diverse clip `idle` con `passaggio`.
- **exp_19** aumenta la capacità del classificatore temporale usando `d_model = 384`, `num_heads = 6` e `ff_dim = 1024`. Ottiene la migliore accuracy e il miglior weighted F1 tra gli esperimenti DINOv2, oltre alla migliore micro F1 sulle 7 azioni reali fino a quel punto.
- **exp_20** parte da exp_19 e aumenta regolarizzazione e bilanciamento (`dropout = 0.50`, `weight_decay = 7e-3`, `class_weight_power = 0.6`, `sampler_power = 0.5`). L'esperimento non migliora exp_19 e peggiora sia accuracy sia Macro F1, suggerendo che un bilanciamento più aggressivo non è vantaggioso in questa configurazione.
- **exp_21** sostituisce DINOv2-L/14 con **DINOv3-L/16 frozen**, mantenendo la configurazione di exp_19 e CLS pooling. Ottiene la migliore Val Macro F1 complessiva, migliorando soprattutto il bilanciamento tra le classi e recuperando parzialmente classi rare come `tiroDaTre1`, `tiroLibero0` e `tiroLibero1`.
- **exp_22** usa DINOv3-L/16 con mean pooling. Ottiene la migliore accuracy, il miglior weighted F1 sulle 9 classi e la migliore micro F1 sulle 7 azioni reali. La Macro F1 è leggermente inferiore rispetto a exp_21, ma il comportamento globale risulta più solido.

Il miglior compromesso attuale è quindi:

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

Rispetto agli esperimenti basati su ConvNeXt-Tiny, il passaggio a DINOv2-B/14 produce un miglioramento netto e stabile. Il passaggio successivo a DINOv2-L/14 migliora soprattutto le metriche globali quando il classificatore temporale viene reso più capiente in exp_19. Il passaggio a DINOv3-L/16 produce invece il miglioramento più importante sulle metriche complessive e sulle classi rare, portando la Macro F1 massima da 0.5865 di exp_16 a 0.6256 di exp_21.

Le classi più problematiche restano:

```text
tiroDaDue1
tiroDaTre1
tiroLibero1
```

`tiroDaTre1` ha solo 3 esempi in validation, quindi la sua metrica è molto instabile: exp_21 ed exp_22 riescono a classificarne correttamente almeno uno, mentre exp_19 ed exp_20 non ne riconoscono nessuno. Per questo motivo exp_21 resta importante come miglior modello in termini di Macro F1, mentre exp_22 è più adatto come modello operativo complessivo.

Il prossimo passo consigliato non è aumentare ulteriormente la dimensione del modello, ma introdurre **data augmentation mirata sulle classi di tiro** durante la fase di feature extraction. L'augmentation dovrebbe essere applicata solo al training set e principalmente alle classi rare, evitando di aumentare ulteriormente `passaggio`, `idle` e `non-gioco`, già molto rappresentate nel dataset.
