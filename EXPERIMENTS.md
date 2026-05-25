# Esperimenti

Questo file tiene traccia sintetica delle principali versioni sperimentali del modello, delle modifiche introdotte e delle prestazioni ottenute.

## Tabella riassuntiva

| ID | Modello | Feature | Epoche | Batch size | LR | Modifiche principali | Val Loss | Val Accuracy | Val Macro Precision | Val Macro Recall | Val Macro F1 | Val Weighted Precision | Val Weighted Recall | Val Weighted F1 | Note |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exp_01 | GRU bidirezionale | ResNet18 ImageNet frozen | 50 | 16 | 1e-4 | Baseline iniziale con feature estratte tramite ResNet18 e classificatore temporale GRU bidirezionale | 1.3913 | 0.67 | 0.37 | 0.38 | 0.37 | 0.65 | 0.67 | 0.66 | Buone prestazioni su `passaggio` e `non-gioco`; difficoltà marcate sulle classi rare, soprattutto `tiroDaTre1`, `tiroDaDue1` e `tiroLibero0`. Si nota overfitting: train macro F1 finale 0.95 contro val macro F1 0.37. |

## Dettaglio metriche per classe su validation

| Classe | Precision | Recall | F1-score | Support | Note |
|---|---:|---:|---:|---:|---|
| `passaggio` | 0.77 | 0.80 | 0.78 | 212 | Classe riconosciuta bene, anche grazie all’alto numero di esempi |
| `tiroDaDue0` | 0.44 | 0.38 | 0.41 | 21 | Prestazioni discrete ma ancora instabili |
| `tiroDaDue1` | 0.10 | 0.09 | 0.10 | 11 | Classe molto debole, probabilmente penalizzata da pochi esempi |
| `tiroDaTre0` | 0.43 | 0.50 | 0.46 | 12 | Recall discreta, ma support basso |
| `tiroDaTre1` | 0.00 | 0.00 | 0.00 | 3 | Classe non appresa nella validazione |
| `tiroLibero0` | 0.17 | 0.14 | 0.15 | 7 | Prestazioni basse, classe rara |
| `tiroLibero1` | 0.22 | 0.36 | 0.28 | 11 | Recall migliore della precision, ma ancora debole |
| `idle` | 0.34 | 0.25 | 0.29 | 93 | Spesso confuso con `passaggio` e `non-gioco` |
| `non-gioco` | 0.82 | 0.88 | 0.85 | 170 | Classe riconosciuta molto bene |

## Metriche aggregate su validation

| Metrica | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Accuracy | - | - | 0.67 | 540 |
| Macro avg | 0.37 | 0.38 | 0.37 | 540 |
| Weighted avg | 0.65 | 0.67 | 0.66 | 540 |



