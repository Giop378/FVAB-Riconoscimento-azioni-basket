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
| Macro Precision 8 classi | 0.6743 |
| Macro Recall 8 classi | 0.7134 |
| Macro F1 8 classi | 0.6749 |
| Weighted Precision 8 classi | 0.8642 |
| Weighted Recall 8 classi | 0.8549 |
| Weighted F1 8 classi | 0.8564 |

## Classification report - 8 classi finali con no-action

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.9131 | 0.9348 | 0.9238 | 506 |
| tiroDaDue0 | 0.7941 | 0.5625 | 0.6585 | 48 |
| tiroDaDue1 | 0.7609 | 0.7000 | 0.7292 | 50 |
| tiroDaTre0 | 0.7619 | 0.5517 | 0.6400 | 29 |
| tiroDaTre1 | 0.5833 | 0.7778 | 0.6667 | 9 |
| tiroLibero0 | 0.3636 | 0.6154 | 0.4571 | 13 |
| tiroLibero1 | 0.3478 | 0.7273 | 0.4706 | 11 |
| no-action | 0.8697 | 0.8376 | 0.8534 | 271 |
| accuracy |  |  | 0.8549 | 937 |
| macro avg | 0.6743 | 0.7134 | 0.6749 | 937 |
| weighted avg | 0.8642 | 0.8549 | 0.8564 | 937 |

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

Questa valutazione esclude `no-action` dal label space del report finale e misura il comportamento sulle sole azioni di interesse.

L'`Accuracy 7 azioni` è calcolata sui 666 campioni con ground truth appartenente alle 7 azioni finali; una predizione `no-action` su una vera azione viene conteggiata come errore.

I falsi positivi da `no-action` verso una classe di azione contribuiscono alla precision delle 7 classi, anche se non sono visibili nella confusion matrix a 7 classi.

| Metrica | Valore |
|---|---:|
| Accuracy 7 azioni | 0.8619 |
| Macro Precision 7 azioni | 0.6464 |
| Macro Recall 7 azioni | 0.6956 |
| Macro F1 7 azioni | 0.6494 |
| Weighted Precision 7 azioni | 0.8620 |
| Weighted Recall 7 azioni | 0.8619 |
| Weighted F1 7 azioni | 0.8577 |

## Classification report - solo 7 azioni finali

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.9131 | 0.9348 | 0.9238 | 506 |
| tiroDaDue0 | 0.7941 | 0.5625 | 0.6585 | 48 |
| tiroDaDue1 | 0.7609 | 0.7000 | 0.7292 | 50 |
| tiroDaTre0 | 0.7619 | 0.5517 | 0.6400 | 29 |
| tiroDaTre1 | 0.5833 | 0.7778 | 0.6667 | 9 |
| tiroLibero0 | 0.3636 | 0.6154 | 0.4571 | 13 |
| tiroLibero1 | 0.3478 | 0.7273 | 0.4706 | 11 |
| macro avg | 0.6464 | 0.6956 | 0.6494 | 666 |
| weighted avg | 0.8620 | 0.8619 | 0.8577 | 666 |

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
| Macro Precision tipo azione | 0.7941 |
| Macro Recall tipo azione | 0.8143 |
| Macro F1 tipo azione | 0.7898 |
| Weighted Precision tipo azione | 0.8860 |
| Weighted Recall tipo azione | 0.8762 |
| Weighted F1 tipo azione | 0.8784 |

## Classification report - tipo azione senza esito

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.9131 | 0.9348 | 0.9238 | 506 |
| tiroDaDue | 0.9250 | 0.7551 | 0.8315 | 98 |
| tiroDaTre | 0.8182 | 0.7105 | 0.7606 | 38 |
| tiroLibero | 0.4444 | 0.8333 | 0.5797 | 24 |
| no-action | 0.8697 | 0.8376 | 0.8534 | 271 |
| accuracy |  |  | 0.8762 | 937 |
| macro avg | 0.7941 | 0.8143 | 0.7898 | 937 |
| weighted avg | 0.8860 | 0.8762 | 0.8784 | 937 |

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

# Metriche binarie azione / no-action

Questa valutazione misura la capacità del modello di distinguere tra una clip contenente una delle 7 azioni finali e una clip di background `no-action`.

| Metrica | Valore |
|---|---:|
| Accuracy azione/no-action | 0.9168 |
| Macro Precision azione/no-action | 0.9023 |
| Macro Recall azione/no-action | 0.8933 |
| Macro F1 azione/no-action | 0.8976 |
| Weighted Precision azione/no-action | 0.9161 |
| Weighted Recall azione/no-action | 0.9168 |
| Weighted F1 azione/no-action | 0.9163 |

## Classification report - azione / no-action

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| azione | 0.9349 | 0.9489 | 0.9419 | 666 |
| no-action | 0.8697 | 0.8376 | 0.8534 | 271 |
| accuracy |  |  | 0.9168 | 937 |
| macro avg | 0.9023 | 0.8933 | 0.8976 | 937 |
| weighted avg | 0.9161 | 0.9168 | 0.9163 | 937 |

## Confusion matrix - azione / no-action

Ordine classi:

```text
azione, no-action
```

```text
[[632  34]
 [ 44 227]]
```

Interpretazione della confusion matrix:

- 632 azioni riconosciute come azione;
- 34 azioni perse e classificate come `no-action`;
- 44 campioni `no-action` classificati erroneamente come azione;
- 227 campioni `no-action` riconosciuti correttamente.

---

# Metriche diagnostiche per livello gerarchico

Queste metriche servono a capire in quale livello della gerarchia avvengono gli errori principali. Non sostituiscono le metriche end-to-end, ma aiutano a interpretare il comportamento del modello.

## L1 - passaggio / tiro / no-action

| Metrica | Valore |
|---|---:|
| Accuracy L1 | 0.9082 |
| Macro Precision L1 | 0.9129 |
| Macro Recall L1 | 0.9054 |
| Macro F1 L1 | 0.9090 |
| Weighted Precision L1 | 0.9078 |
| Weighted Recall L1 | 0.9082 |
| Weighted F1 L1 | 0.9079 |

### Classification report - L1

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.9131 | 0.9348 | 0.9238 | 506 |
| tiro | 0.9557 | 0.9437 | 0.9497 | 160 |
| no-action | 0.8697 | 0.8376 | 0.8534 | 271 |
| accuracy |  |  | 0.9082 | 937 |
| macro avg | 0.9129 | 0.9054 | 0.9090 | 937 |
| weighted avg | 0.9078 | 0.9082 | 0.9079 | 937 |

### Confusion matrix - L1

Ordine classi:

```text
passaggio, tiro, no-action
```

```text
[[473   5  28]
 [  3 151   6]
 [ 42   2 227]]
```

## L2 - tipo di tiro

La valutazione L2 è calcolata sui veri tiri per cui L1 ha attivato il ramo `tiro`: 151 campioni su 160 veri tiri. I tiri persi prima di L2 per errore L1 sono 9.

| Metrica | Valore |
|---|---:|
| Accuracy L2 | 0.8013 |
| Macro Precision L2 | 0.7567 |
| Macro Recall L2 | 0.8390 |
| Macro F1 L2 | 0.7607 |
| Weighted Precision L2 | 0.8763 |
| Weighted Recall L2 | 0.8013 |
| Weighted F1 L2 | 0.8183 |

### Classification report - L2

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| tiroDaDue | 0.9867 | 0.7872 | 0.8757 | 94 |
| tiroDaTre | 0.8182 | 0.7297 | 0.7714 | 37 |
| tiroLibero | 0.4651 | 1.0000 | 0.6349 | 20 |
| accuracy |  |  | 0.8013 | 151 |
| macro avg | 0.7567 | 0.8390 | 0.7607 | 151 |
| weighted avg | 0.8763 | 0.8013 | 0.8183 | 151 |

### Confusion matrix - L2

Ordine classi:

```text
tiroDaDue, tiroDaTre, tiroLibero
```

```text
[[74  6 14]
 [ 1 27  9]
 [ 0  0 20]]
```

## L3 - esito del tiro

La valutazione L3 è calcolata sugli stessi 151 veri tiri arrivati al ramo `tiro` dopo L1.

| Metrica | Valore |
|---|---:|
| Accuracy L3 | 0.8411 |
| Macro Precision L3 | 0.8470 |
| Macro Recall L3 | 0.8496 |
| Macro F1 L3 | 0.8410 |
| Weighted Precision L3 | 0.8562 |
| Weighted Recall L3 | 0.8411 |
| Weighted F1 L3 | 0.8414 |

### Classification report - L3

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| tiro0 | 0.9286 | 0.7738 | 0.8442 | 84 |
| tiro1 | 0.7654 | 0.9254 | 0.8378 | 67 |
| accuracy |  |  | 0.8411 | 151 |
| macro avg | 0.8470 | 0.8496 | 0.8410 | 151 |
| weighted avg | 0.8562 | 0.8411 | 0.8414 | 151 |

### Confusion matrix - L3

Ordine classi:

```text
tiro0, tiro1
```

```text
[[65 19]
 [ 5 62]]
```

---

# Analisi sintetica

Il modello `exp_46` sul test set ottiene una buona accuratezza globale, pari a 0.8549 sulle 8 classi finali. La classe più solida resta `passaggio`, con F1-score 0.9238 su 506 campioni. Anche `no-action` rimane stabile, con F1-score 0.8534 su 271 campioni.

Le classi di tiro risultano più difficili, soprattutto per i tiri liberi con esito separato: `tiroLibero0` ottiene F1-score 0.4571 e `tiroLibero1` F1-score 0.4706. Questo è coerente con il supporto ridotto di queste classi nel test set, rispettivamente 13 e 11 campioni.

La valutazione collassata senza esito del tiro è più positiva: la Macro F1 sale a 0.7898 e la Weighted F1 a 0.8784. Questo indica che una parte rilevante degli errori riguarda la distinzione dell'esito del tiro oppure confusioni interne tra tipologie di tiro, mentre il riconoscimento del tipo generale di azione rimane più robusto.

La valutazione binaria `azione/no-action` ottiene Accuracy 0.9168 e Macro F1 0.8976. Questo indica che il modello è abbastanza solido nel distinguere la presenza di azione dal background, anche se rimangono 34 azioni perse e 44 falsi positivi da `no-action` verso azione.

Dalle metriche diagnostiche gerarchiche emerge che L1 è stabile, con Macro F1 0.9090. L3 è discreto nella distinzione dell'esito del tiro, con Macro F1 0.8410. Il livello più critico è L2, con Macro F1 0.7607, perché gestisce la distinzione tra `tiroDaDue`, `tiroDaTre` e `tiroLibero`, dove sono presenti diverse confusioni.

Il risultato finale da riportare come valutazione sul test set è quindi:

```text
Accuracy 8 classi:                  0.8549
Macro Precision 8 classi:           0.6743
Macro Recall 8 classi:              0.7134
Macro F1 8 classi:                  0.6749
Weighted Precision 8 classi:        0.8642
Weighted Recall 8 classi:           0.8549
Weighted F1 8 classi:               0.8564

Accuracy 7 azioni:                  0.8619
Macro Precision 7 azioni:           0.6464
Macro Recall 7 azioni:              0.6956
Macro F1 7 azioni:                  0.6494
Weighted Precision 7 azioni:        0.8620
Weighted Recall 7 azioni:           0.8619
Weighted F1 7 azioni:               0.8577

Accuracy tipo azione:               0.8762
Macro Precision tipo azione:        0.7941
Macro Recall tipo azione:           0.8143
Macro F1 tipo azione:               0.7898
Weighted Precision tipo azione:     0.8860
Weighted Recall tipo azione:        0.8762
Weighted F1 tipo azione:            0.8784

Accuracy azione/no-action:          0.9168
Macro Precision azione/no-action:   0.9023
Macro Recall azione/no-action:      0.8933
Macro F1 azione/no-action:          0.8976
Weighted Precision azione/no-action: 0.9161
Weighted Recall azione/no-action:   0.9168
Weighted F1 azione/no-action:       0.9163

Accuracy L1:                        0.9082
Macro Precision L1:                 0.9129
Macro Recall L1:                    0.9054
Macro F1 L1:                        0.9090
Weighted Precision L1:              0.9078
Weighted Recall L1:                 0.9082
Weighted F1 L1:                     0.9079

Accuracy L2:                        0.8013
Macro Precision L2:                 0.7567
Macro Recall L2:                    0.8390
Macro F1 L2:                        0.7607
Weighted Precision L2:              0.8763
Weighted Recall L2:                 0.8013
Weighted F1 L2:                     0.8183

Accuracy L3:                        0.8411
Macro Precision L3:                 0.8470
Macro Recall L3:                    0.8496
Macro F1 L3:                        0.8410
Weighted Precision L3:              0.8562
Weighted Recall L3:                 0.8411
Weighted F1 L3:                     0.8414
```
