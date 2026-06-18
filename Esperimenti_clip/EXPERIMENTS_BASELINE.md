# Esperimenti

Questo file tiene traccia sintetica delle principali versioni sperimentali del modello, delle impostazioni usate e delle prestazioni ottenute su validation.

## Tabella riassuntiva

|ID|Modello|Feature extractor|Epoche|Batch size|LR|Hidden dim|Dropout|Weight decay|Val Loss|Val Accuracy|Val Macro F1|Val Weighted F1|Output dir|
|-|-|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-|
|exp\_01|GRU bidirezionale|ResNet18 ImageNet frozen|50|16|1e-4|-|-|-|1.3913|0.67|0.37|0.66|outputs/resnet18\_bigru|
|exp\_02|BiGRU + attention pooling|ConvNeXt-Tiny ImageNet frozen|50|16|1e-4|256|0.3|1e-4|0.9888|0.75|0.43|0.72|outputs/convnext\_bigru\_attention\_weighted|
|exp\_03|BiGRU + attention pooling|ConvNeXt-Tiny ImageNet frozen|35|16|1e-4|192|0.5|1e-3|0.8866|0.73|0.43|0.70|outputs/convnext\_bigru\_attention\_regularized|
|exp\_04|BiGRU + attention pooling|ConvNeXt-Tiny ImageNet frozen|35|16|1e-4|256|0.4|5e-4|1.0516|0.73|0.44|0.71|outputs/convnext\_bigru\_attention\_mid\_regularized|
|exp\_05|BiGRU + attention pooling + temporal pyramid pooling|ConvNeXt-Tiny ImageNet frozen|35|16|1e-4|256|0.4|5e-4|0.7579|0.73|0.43|0.72|outputs/convnext\_bigru\_attention\_temporal\_pyramid|
|exp\_06|BiGRU + attention pooling|ConvNeXt-Tiny ImageNet frozen, stretched 320x320 senza center crop|50|16|1e-4|256|0.3|1e-4|0.9344|0.74|0.45|0.72|outputs/convnext\_stretched\_320\_bigru|
|exp\_07|BiGRU + attention pooling + weighted loss + WeightedRandomSampler|ConvNeXt-Tiny ImageNet frozen, stretched 320x320 senza center crop|50|16|1e-4|256|0.3|1e-4|0.9723|0.70|0.42|0.71|outputs/exp\_07|
|exp\_08|BiGRU + attention pooling + WeightedRandomSampler|ConvNeXt-Tiny ImageNet frozen, stretched 320x320 senza center crop|50|16|1e-4|256|0.3|1e-4|0.8120|0.74|0.44|0.73|outputs/exp\_08|
|exp\_09|BiGRU + attention pooling + weighted loss|ConvNeXt-Tiny ImageNet frozen, stretched 320x320 senza center crop|50|16|1e-4|256|0.3|1e-4|0.7883|0.74|0.46|0.73|outputs/exp\_09|

## Risultati aggregati su validation - 9 classi

|ID|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|
|exp\_01|0.67|0.37|0.38|0.37|0.65|0.67|0.66|
|exp\_02|0.75|0.45|0.44|0.43|0.72|0.75|0.72|
|exp\_03|0.73|0.51|0.44|0.43|0.71|0.73|0.70|
|exp\_04|0.73|0.51|0.43|0.44|0.70|0.73|0.71|
|exp\_05|0.73|0.46|0.43|0.43|0.71|0.73|0.72|
|exp\_06|0.74|0.48|0.44|0.45|0.73|0.74|0.72|
|exp\_07|0.70|0.42|0.46|0.42|0.73|0.70|0.71|
|exp\_08|0.74|0.52|0.46|0.44|0.75|0.74|0.73|
|exp\_09|0.74|0.46|0.46|0.46|0.74|0.74|0.73|

## Risultati aggregati sulle sole 7 azioni reali

|ID|Micro Precision|Micro Recall|Micro F1|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|exp\_02|0.73|0.82|0.77|0.38|0.40|0.38|0.70|0.82|0.75|
|exp\_03|0.73|0.78|0.75|0.47|0.40|0.38|0.72|0.78|0.72|
|exp\_04|0.75|0.73|0.74|0.48|0.37|0.38|0.72|0.73|0.71|
|exp\_05|0.73|0.76|0.75|0.40|0.37|0.37|0.71|0.76|0.72|
|exp\_06|0.72|0.78|0.75|0.42|0.39|0.39|0.71|0.78|0.74|
|exp\_07|0.66|0.75|0.70|0.33|0.42|0.36|0.70|0.75|0.72|
|exp\_08|0.70|0.78|0.73|0.46|0.42|0.38|0.73|0.78|0.74|
|exp\_09|0.74|0.70|0.72|0.40|0.39|0.39|0.74|0.70|0.72|

## Risultati per classe su validation

### exp\_01 - ResNet18 + GRU bidirezionale

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.77|0.80|0.78|212|
|tiroDaDue0|0.44|0.38|0.41|21|
|tiroDaDue1|0.10|0.09|0.10|11|
|tiroDaTre0|0.43|0.50|0.46|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.17|0.14|0.15|7|
|tiroLibero1|0.22|0.36|0.28|11|
|idle|0.34|0.25|0.29|93|
|non-gioco|0.82|0.88|0.85|170|

### exp\_02 - ConvNeXt-Tiny + BiGRU + attention pooling

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.79|0.94|0.86|212|
|tiroDaDue0|0.56|0.67|0.61|21|
|tiroDaDue1|0.43|0.27|0.33|11|
|tiroDaTre0|0.60|0.75|0.67|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.00|0.00|0.00|7|
|tiroLibero1|0.29|0.18|0.22|11|
|idle|0.52|0.26|0.35|93|
|non-gioco|0.85|0.91|0.88|170|

### exp\_03 - ConvNeXt-Tiny + BiGRU + attention pooling regolarizzato

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.78|0.92|0.84|212|
|tiroDaDue0|0.80|0.19|0.31|21|
|tiroDaDue1|0.23|0.27|0.25|11|
|tiroDaTre0|0.53|0.75|0.62|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.44|0.57|0.50|7|
|tiroLibero1|0.50|0.09|0.15|11|
|idle|0.43|0.34|0.38|93|
|non-gioco|0.85|0.85|0.85|170|

### exp\_04 - ConvNeXt-Tiny + BiGRU + attention pooling con regolarizzazione intermedia

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.79|0.85|0.82|212|
|tiroDaDue0|0.80|0.38|0.52|21|
|tiroDaDue1|0.00|0.00|0.00|11|
|tiroDaTre0|0.45|0.83|0.59|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|1.00|0.43|0.60|7|
|tiroLibero1|0.33|0.09|0.14|11|
|idle|0.43|0.35|0.39|93|
|non-gioco|0.82|0.94|0.87|170|

### exp\_05 - ConvNeXt-Tiny + BiGRU + attention pooling + temporal pyramid pooling

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.79|0.89|0.84|212|
|tiroDaDue0|0.69|0.43|0.53|21|
|tiroDaDue1|0.33|0.09|0.14|11|
|tiroDaTre0|0.53|0.67|0.59|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.30|0.43|0.35|7|
|tiroLibero1|0.17|0.09|0.12|11|
|idle|0.46|0.41|0.43|93|
|non-gioco|0.86|0.86|0.86|170|

### exp\_06 - ConvNeXt-Tiny stretched 320x320 + BiGRU + attention pooling

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.80|0.90|0.85|212|
|tiroDaDue0|0.48|0.57|0.52|21|
|tiroDaDue1|0.50|0.27|0.35|11|
|tiroDaTre0|0.67|0.50|0.57|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.25|0.14|0.18|7|
|tiroLibero1|0.24|0.36|0.29|11|
|idle|0.59|0.32|0.42|93|
|non-gioco|0.82|0.91|0.86|170|

### exp\_07 - ConvNeXt-Tiny stretched 320x320 + BiGRU + attention pooling + weighted loss + WeightedRandomSampler

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.82|0.87|0.84|212|
|tiroDaDue0|0.30|0.29|0.29|21|
|tiroDaDue1|0.25|0.27|0.26|11|
|tiroDaTre0|0.53|0.67|0.59|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.29|0.29|0.29|7|
|tiroLibero1|0.15|0.55|0.24|11|
|idle|0.56|0.43|0.49|93|
|non-gioco|0.86|0.76|0.81|170|

### exp\_08 - ConvNeXt-Tiny stretched 320x320 + BiGRU + attention pooling + WeightedRandomSampler

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.82|0.89|0.85|212|
|tiroDaDue0|0.39|0.43|0.41|21|
|tiroDaDue1|0.33|0.18|0.24|11|
|tiroDaTre0|0.48|0.83|0.61|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|1.00|0.14|0.25|7|
|tiroLibero1|0.21|0.45|0.29|11|
|idle|0.58|0.39|0.46|93|
|non-gioco|0.86|0.86|0.86|170|

### exp\_09 - ConvNeXt-Tiny stretched 320x320 + BiGRU + attention pooling + weighted loss

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.85|0.80|0.83|212|
|tiroDaDue0|0.50|0.38|0.43|21|
|tiroDaDue1|0.23|0.27|0.25|11|
|tiroDaTre0|0.50|0.50|0.50|12|
|tiroDaTre1|0.00|0.00|0.00|3|
|tiroLibero0|0.40|0.29|0.33|7|
|tiroLibero1|0.29|0.45|0.36|11|
|idle|0.53|0.55|0.54|93|
|non-gioco|0.84|0.89|0.86|170|

## Comandi utilizzati

### exp\_01

```bash
python -m src.training.train \\
  --features-root data/features/resnet18 \\
  --epochs 50 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --output-dir outputs/resnet18\_bigru
```

### exp\_02

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 50 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --output-dir outputs/convnext\_bigru\_attention\_weighted
```

### exp\_03

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 35 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --hidden-dim 192 \\
  --dropout 0.5 \\
  --weight-decay 1e-3 \\
  --output-dir outputs/convnext\_bigru\_attention\_regularized
```

### exp\_04

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 35 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --hidden-dim 256 \\
  --dropout 0.4 \\
  --weight-decay 5e-4 \\
  --output-dir outputs/convnext\_bigru\_attention\_mid\_regularized
```

### exp\_05

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny \\
  --epochs 35 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --hidden-dim 256 \\
  --dropout 0.4 \\
  --weight-decay 5e-4 \\
  --output-dir outputs/convnext\_bigru\_attention\_temporal\_pyramid
```

### exp\_06

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny\_stretched\_320 \\
  --epochs 50 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --output-dir outputs/convnext\_stretched\_320\_bigru
```

### exp\_07

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny\_stretched\_320 \\
  --epochs 50 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --output-dir outputs/exp\_07
```

### exp\_08

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny\_stretched\_320 \\
  --epochs 50 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --output-dir outputs/exp\_08 \\
  --no-class-weights
```

### exp\_09

```bash
python -m src.training.train \\
  --features-root data/features/convnext\_tiny\_stretched\_320 \\
  --epochs 50 \\
  --batch-size 16 \\
  --lr 1e-4 \\
  --output-dir outputs/exp\_09 \\
  --no-weighted-sampler
```

## Nota finale

Il miglior modello in termini di Val Macro F1 diventa **exp\_09**, con:

```text
Val Macro F1 = 0.46
```

Il miglior modello in termini di Val Accuracy rimane **exp\_02**, con:

```text
Val Accuracy = 0.75
```

Il miglior Val Weighted F1, arrotondato a due decimali, è ottenuto da **exp\_08** ed **exp\_09**, con:

```text
Val Weighted F1 = 0.73
```

**exp\_09** usa le feature ConvNeXt-Tiny estratte con resize stretched 320x320 senza center crop, mantiene attiva la weighted loss e disattiva il `WeightedRandomSampler`. Rispetto a **exp\_06** migliora leggermente la Val Macro F1, passando da 0.45 a 0.46, e mantiene la Macro F1 sulle sole 7 azioni reali a 0.39.

Il confronto tra **exp\_07**, **exp\_08** ed **exp\_09** suggerisce che, per questa architettura, la `Weighted CrossEntropyLoss` è più stabile del `WeightedRandomSampler`. La combinazione tra weighted loss e sampler pesato non porta benefici, mentre il sampler da solo migliora rispetto a exp\_07 ma resta sotto exp\_09. Il modello continua comunque ad avere difficoltà sulle classi rare, in particolare `tiroDaTre1`.

