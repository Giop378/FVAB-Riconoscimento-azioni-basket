# Risultati exp_46 sul test set

File risultati originale: `outputs/exp_46_test/results.txt`

## Comando utilizzato

```bash
/home/vrlab/miniconda3/envs/fvab-basket/bin/python /home/vrlab/Scrivania/BasketAR/Gruppo10/FVAB_Gruppo10/FVAB-Riconoscimento-azioni-basket/src/evaluation/evaluate_hierarchical.py \
  --features-root data/features/dinov3_vitl16_336 \
  --split test \
  --batch-size 64 \
  --num-workers 2 \
  --output-dir outputs/exp_46_test \
  --l1-checkpoint outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt \
  --l2-checkpoint outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt \
  --l3-checkpoint outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt \
  --l1-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2_v4/tracking_sequences.npz \
  --l1-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2_v4/tracking_sequence_index.json \
  --l2-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequences.npz \
  --l2-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequence_index.json \
  --l3-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1_v4/tracking_sequences.npz \
  --l3-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1_v4/tracking_sequence_index.json \
  --tracking-missing-policy error
```

## Configurazione dell'esperimento

| Componente | Valore |
|---|---|
| Feature video | `data/features/dinov3_vitl16_336` |
| Split valutato | `test` |
| Numero campioni | 937 |
| Batch size | 64 |
| Device | `cuda` |
| Output dir | `outputs/exp_46_test` |
| Missing policy tracking | `error` |

## Configurazione gerarchica

| Livello | Checkpoint | Label space | Tracking | Numero feature tracking |
|---|---|---|---|---:|
| L1 | `outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt` | `passaggio`, `tiro`, `no-action` | YOLO v2 `temp43` | 43 |
| L2 | `outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt` | `tiroDaDue`, `tiroDaTre`, `tiroLibero` | YOLO v2 `temp29` | 29 |
| L3 | `outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt` | `tiro0`, `tiro1` | YOLO v1 `temp43` | 43 |

Tutti i tracking sono stati normalizzati usando le statistiche salvate nei rispettivi checkpoint:

```text
L1: Normalizzate con statistiche del checkpoint L1: True
L2: Normalizzate con statistiche del checkpoint L2: True
L3: Normalizzate con statistiche del checkpoint L3: True
```

---

# Metriche end-to-end su 8 classi

Le 8 classi finali sono le 7 azioni annotate più la classe di background `no-action`, ottenuta collassando `idle` e `non-gioco`.

| Metrica | Valore |
|---|---:|
| Accuracy 8 classi | 0.8549 |
| Macro F1 8 classi | 0.6749 |
| Weighted F1 8 classi | 0.8564 |

## Classification report - 8 classi finali con no-action

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.91 | 0.93 | 0.92 | 506 |
| tiroDaDue0 | 0.79 | 0.56 | 0.66 | 48 |
| tiroDaDue1 | 0.76 | 0.70 | 0.73 | 50 |
| tiroDaTre0 | 0.76 | 0.55 | 0.64 | 29 |
| tiroDaTre1 | 0.58 | 0.78 | 0.67 | 9 |
| tiroLibero0 | 0.36 | 0.62 | 0.46 | 13 |
| tiroLibero1 | 0.35 | 0.73 | 0.47 | 11 |
| no-action | 0.87 | 0.84 | 0.85 | 271 |
| accuracy |  |  | 0.85 | 937 |
| macro avg | 0.67 | 0.71 | 0.67 | 937 |
| weighted avg | 0.86 | 0.85 | 0.86 | 937 |

## Confusion matrix - 8 classi finali con no-action

Ordine classi:

```text
passaggio, tiroDaDue0, tiroDaDue1, tiroDaTre0, tiroDaTre1, tiroLibero0, tiroLibero1, no-action
```

```text
[[473   5   0   0   0   0   0  28]
 [  3  27  11   4   0   2   0   1]
 [  0   1  35   1   1   2  10   0]
 [  0   1   0  16   4   7   1   0]
 [  0   0   0   0   7   0   1   1]
 [  0   0   0   0   0   8   3   2]
 [  0   0   0   0   0   1   8   2]
 [ 42   0   0   0   0   2   0 227]]
```

---

# Metriche sulle sole 7 azioni finali

Questa valutazione esclude `no-action` dal report finale e misura il comportamento sulle sole azioni di interesse.

| Metrica | Valore |
|---|---:|
| Micro F1 7 azioni | 0.86 |
| Macro F1 7 azioni | 0.65 |
| Weighted F1 7 azioni | 0.86 |

## Classification report - solo 7 azioni finali

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.91 | 0.93 | 0.92 | 506 |
| tiroDaDue0 | 0.79 | 0.56 | 0.66 | 48 |
| tiroDaDue1 | 0.76 | 0.70 | 0.73 | 50 |
| tiroDaTre0 | 0.76 | 0.55 | 0.64 | 29 |
| tiroDaTre1 | 0.58 | 0.78 | 0.67 | 9 |
| tiroLibero0 | 0.36 | 0.62 | 0.46 | 13 |
| tiroLibero1 | 0.35 | 0.73 | 0.47 | 11 |
| micro avg | 0.85 | 0.86 | 0.86 | 666 |
| macro avg | 0.65 | 0.70 | 0.65 | 666 |
| weighted avg | 0.86 | 0.86 | 0.86 | 666 |

## Confusion matrix - solo 7 azioni finali

Ordine classi:

```text
passaggio, tiroDaDue0, tiroDaDue1, tiroDaTre0, tiroDaTre1, tiroLibero0, tiroLibero1
```

```text
[[473   5   0   0   0   0   0]
 [  3  27  11   4   0   2   0]
 [  0   1  35   1   1   2  10]
 [  0   1   0  16   4   7   1]
 [  0   0   0   0   7   0   1]
 [  0   0   0   0   0   8   3]
 [  0   0   0   0   0   1   8]]
```

---

# Valutazione collassata senza esito del tiro

In questa valutazione le classi di tiro vengono collassate ignorando l'esito del tiro:

```text
tiroDaDue0 + tiroDaDue1 -> tiroDaDue
tiroDaTre0 + tiroDaTre1 -> tiroDaTre
tiroLibero0 + tiroLibero1 -> tiroLibero
```

| Metrica | Valore |
|---|---:|
| Accuracy tipo azione | 0.8762 |
| Macro F1 tipo azione | 0.7898 |
| Weighted F1 tipo azione | 0.8784 |

## Classification report - tipo azione senza esito

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.91 | 0.93 | 0.92 | 506 |
| tiroDaDue | 0.93 | 0.76 | 0.83 | 98 |
| tiroDaTre | 0.82 | 0.71 | 0.76 | 38 |
| tiroLibero | 0.44 | 0.83 | 0.58 | 24 |
| no-action | 0.87 | 0.84 | 0.85 | 271 |
| accuracy |  |  | 0.88 | 937 |
| macro avg | 0.79 | 0.81 | 0.79 | 937 |
| weighted avg | 0.89 | 0.88 | 0.88 | 937 |

## Confusion matrix - tipo azione senza esito

Ordine classi:

```text
passaggio, tiroDaDue, tiroDaTre, tiroLibero, no-action
```

```text
[[473   5   0   0  28]
 [  3  74   6  14   1]
 [  0   1  27   9   1]
 [  0   0   0  20   4]
 [ 42   0   0   2 227]]
```

---

# Analisi sintetica

Il modello `exp_46` sul test set ottiene una buona accuratezza globale, pari a 0.8549 sulle 8 classi finali. La classe più solida resta `passaggio`, con F1-score 0.92 su 506 campioni. Anche `no-action` rimane stabile, con F1-score 0.85 su 271 campioni.

Le classi di tiro risultano più difficili, soprattutto per i tiri liberi con esito separato: `tiroLibero0` ottiene F1-score 0.46 e `tiroLibero1` F1-score 0.47. Questo è coerente con il supporto ridotto di queste classi nel test set, rispettivamente 13 e 11 campioni.

La valutazione collassata senza esito del tiro è più positiva: la Macro F1 sale a 0.7898 e la Weighted F1 a 0.8784. Questo indica che una parte rilevante degli errori riguarda la distinzione dell'esito del tiro oppure confusioni interne tra tipologie di tiro, mentre il riconoscimento del tipo generale di azione rimane più robusto.

In particolare, nella confusion matrix collassata:

- `passaggio` viene riconosciuto correttamente in 473 casi su 506;
- `tiroDaDue` viene riconosciuto correttamente in 74 casi su 98;
- `tiroDaTre` viene riconosciuto correttamente in 27 casi su 38;
- `tiroLibero` viene riconosciuto correttamente in 20 casi su 24;
- `no-action` viene riconosciuto correttamente in 227 casi su 271.

Il risultato finale da riportare come valutazione sul test set è quindi:

```text
Accuracy 8 classi:        0.8549
Macro F1 8 classi:        0.6749
Weighted F1 8 classi:     0.8564
Micro F1 7 azioni:        0.86
Macro F1 7 azioni:        0.65
Weighted F1 7 azioni:     0.86
Macro F1 tipo azione:     0.7898
Weighted F1 tipo azione:  0.8784
```
