# Risultati exp_46 sul validation set

File risultati originale: `outputs/exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43/results.txt`

## Comando utilizzato

```bash
/home/vrlab/miniconda3/envs/fvab-basket/bin/python /home/vrlab/Scrivania/BasketAR/Gruppo10/FVAB_Gruppo10/FVAB-Riconoscimento-azioni-basket/src/evaluation/evaluate_hierarchical.py \
  --features-root data/features/dinov3_vitl16_336 \
  --split val \
  --batch-size 64 \
  --num-workers 2 \
  --output-dir outputs/exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43 \
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
| Split valutato | `val` |
| Numero campioni | 540 |
| Batch size | 64 |
| Device | `cuda` |
| Output dir | `outputs/exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43` |
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
| Accuracy 8 classi | 0.9056 |
| Macro Precision 8 classi | 0.8004 |
| Macro Recall 8 classi | 0.8318 |
| Macro F1 8 classi | 0.8079 |
| Weighted Precision 8 classi | 0.9107 |
| Weighted Recall 8 classi | 0.9056 |
| Weighted F1 8 classi | 0.9062 |

## Classification report - 8 classi finali con no-action

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.8899 | 0.9528 | 0.9203 | 212 |
| tiroDaDue0 | 0.9000 | 0.8571 | 0.8780 | 21 |
| tiroDaDue1 | 0.6000 | 0.8182 | 0.6923 | 11 |
| tiroDaTre0 | 0.9091 | 0.8333 | 0.8696 | 12 |
| tiroDaTre1 | 0.5000 | 0.6667 | 0.5714 | 3 |
| tiroLibero0 | 0.7778 | 1.0000 | 0.8750 | 7 |
| tiroLibero1 | 0.8750 | 0.6364 | 0.7368 | 11 |
| no-action | 0.9512 | 0.8897 | 0.9194 | 263 |
| accuracy |  |  | 0.9056 | 540 |
| macro avg | 0.8004 | 0.8318 | 0.8079 | 540 |
| weighted avg | 0.9107 | 0.9056 | 0.9062 | 540 |

## Confusion matrix - 8 classi finali con no-action

Ordine classi:

```text
passaggio, tiroDaDue0, tiroDaDue1, tiroDaTre0, tiroDaTre1, tiroLibero0, tiroLibero1, no-action
```

```text
[[202   0   0   1   0   0   0   9]
 [  0  18   2   0   0   1   0   0]
 [  0   0   9   0   1   0   1   0]
 [  0   0   0  10   1   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   7   0   0]
 [  0   0   2   0   0   0   7   2]
 [ 25   2   1   0   0   1   0 234]]
```

---

# Metriche sulle sole 7 azioni finali

Questa valutazione esclude `no-action` dal label space del report finale e misura il comportamento sulle sole azioni di interesse.

L'`Accuracy 7 azioni` è calcolata sui 277 campioni con ground truth appartenente alle 7 azioni finali; una predizione `no-action` su una vera azione viene conteggiata come errore.

| Metrica | Valore |
|---|---:|
| Accuracy 7 azioni | 0.9206 |
| Macro Precision 7 azioni | 0.7788 |
| Macro Recall 7 azioni | 0.8235 |
| Macro F1 7 azioni | 0.7919 |
| Weighted Precision 7 azioni | 0.8723 |
| Weighted Recall 7 azioni | 0.9206 |
| Weighted F1 7 azioni | 0.8936 |

## Classification report - solo 7 azioni finali

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.8899 | 0.9528 | 0.9203 | 212 |
| tiroDaDue0 | 0.9000 | 0.8571 | 0.8780 | 21 |
| tiroDaDue1 | 0.6000 | 0.8182 | 0.6923 | 11 |
| tiroDaTre0 | 0.9091 | 0.8333 | 0.8696 | 12 |
| tiroDaTre1 | 0.5000 | 0.6667 | 0.5714 | 3 |
| tiroLibero0 | 0.7778 | 1.0000 | 0.8750 | 7 |
| tiroLibero1 | 0.8750 | 0.6364 | 0.7368 | 11 |
| macro avg | 0.7788 | 0.8235 | 0.7919 | 277 |
| weighted avg | 0.8723 | 0.9206 | 0.8936 | 277 |

## Confusion matrix - solo 7 azioni finali

Ordine classi:

```text
passaggio, tiroDaDue0, tiroDaDue1, tiroDaTre0, tiroDaTre1, tiroLibero0, tiroLibero1
```

```text
[[202   0   0   1   0   0   0]
 [  0  18   2   0   0   1   0]
 [  0   0   9   0   1   0   1]
 [  0   0   0  10   1   0   0]
 [  0   0   1   0   2   0   0]
 [  0   0   0   0   0   7   0]
 [  0   0   2   0   0   0   7]]
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
| Accuracy tipo azione | 0.9111 |
| Macro Precision tipo azione | 0.8720 |
| Macro Recall tipo azione | 0.8787 |
| Macro F1 tipo azione | 0.8744 |
| Weighted Precision tipo azione | 0.9133 |
| Weighted Recall tipo azione | 0.9111 |
| Weighted F1 tipo azione | 0.9111 |

## Classification report - tipo azione senza esito

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.8899 | 0.9528 | 0.9203 | 212 |
| tiroDaDue | 0.8286 | 0.9062 | 0.8657 | 32 |
| tiroDaTre | 0.8667 | 0.8667 | 0.8667 | 15 |
| tiroLibero | 0.8235 | 0.7778 | 0.8000 | 18 |
| no-action | 0.9512 | 0.8897 | 0.9194 | 263 |
| accuracy |  |  | 0.9111 | 540 |
| macro avg | 0.8720 | 0.8787 | 0.8744 | 540 |
| weighted avg | 0.9133 | 0.9111 | 0.9111 | 540 |

## Confusion matrix - tipo azione senza esito

Ordine classi:

```text
passaggio, tiroDaDue, tiroDaTre, tiroLibero, no-action
```

```text
[[202   0   1   0   9]
 [  0  29   1   2   0]
 [  0   1  13   0   1]
 [  0   2   0  14   2]
 [ 25   3   0   1 234]]
```

---

# Metriche binarie azione / no-action

Questa valutazione misura la capacità del modello di distinguere tra una clip contenente una delle 7 azioni finali e una clip di background `no-action`.

| Metrica | Valore |
|---|---:|
| Accuracy azione/no-action | 0.9241 |
| Macro Precision azione/no-action | 0.9263 |
| Macro Recall azione/no-action | 0.9232 |
| Macro F1 azione/no-action | 0.9238 |
| Weighted Precision azione/no-action | 0.9256 |
| Weighted Recall azione/no-action | 0.9241 |
| Weighted F1 azione/no-action | 0.9239 |

## Classification report - azione / no-action

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| azione | 0.9014 | 0.9567 | 0.9282 | 277 |
| no-action | 0.9512 | 0.8897 | 0.9194 | 263 |
| accuracy |  |  | 0.9241 | 540 |
| macro avg | 0.9263 | 0.9232 | 0.9238 | 540 |
| weighted avg | 0.9256 | 0.9241 | 0.9239 | 540 |

## Confusion matrix - azione / no-action

Ordine classi:

```text
azione, no-action
```

```text
[[265  12]
 [ 29 234]]
```

Interpretazione della confusion matrix:

- 265 azioni riconosciute come azione;
- 12 azioni perse e classificate come `no-action`;
- 29 campioni `no-action` classificati erroneamente come azione;
- 234 campioni `no-action` riconosciuti correttamente.

---

# Metriche diagnostiche per livello gerarchico

Queste metriche servono a capire in quale livello della gerarchia avvengono gli errori principali. Non sostituiscono le metriche end-to-end, ma aiutano a interpretare il comportamento del modello.

Per il validation set vengono riportate:

- **metriche condizionate nella gerarchia**: per L2 e L3 considerano solo i veri tiri che arrivano effettivamente al ramo `tiro` dopo L1;
- **metriche standalone sui veri tiri**: valutano il singolo livello direttamente su tutti i veri tiri del validation set, quando la valutazione standalone è disponibile per lo stesso checkpoint usato nella gerarchia.

Nel validation set sono presenti 65 veri tiri. Di questi, 62 arrivano ai rami L2/L3 perché L1 li predice come `tiro`, mentre 3 vengono persi prima dei livelli successivi perché L1 li classifica come `no-action`.

## L1 - passaggio / tiro / no-action

L1 viene applicato a tutte le 540 clip del validation set. Per questo livello la metrica nella gerarchia coincide con la valutazione del livello su tutte le clip.

| Metrica | Valore |
|---|---:|
| Accuracy L1 | 0.9222 |
| Macro Precision L1 | 0.9222 |
| Macro Recall L1 | 0.9321 |
| Macro F1 L1 | 0.9264 |
| Weighted Precision L1 | 0.9240 |
| Weighted Recall L1 | 0.9222 |
| Weighted F1 L1 | 0.9222 |

### Classification report - L1

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.8899 | 0.9528 | 0.9203 | 212 |
| tiro | 0.9254 | 0.9538 | 0.9394 | 65 |
| no-action | 0.9512 | 0.8897 | 0.9194 | 263 |
| accuracy |  |  | 0.9222 | 540 |
| macro avg | 0.9222 | 0.9321 | 0.9264 | 540 |
| weighted avg | 0.9240 | 0.9222 | 0.9222 | 540 |

### Confusion matrix - L1

Ordine classi:

```text
passaggio, tiro, no-action
```

```text
[[202   1   9]
 [  0  62   3]
 [ 25   4 234]]
```

Interpretazione principale:

- 62 veri tiri su 65 vengono inviati correttamente ai rami L2/L3;
- 3 veri tiri vengono persi prima dei rami L2/L3 perché classificati come `no-action`;
- 4 campioni `no-action` vengono inviati erroneamente al ramo `tiro`;
- 1 `passaggio` viene inviato erroneamente al ramo `tiro`.

## L2 - tipo di tiro

### L2 condizionato nella gerarchia

Questa valutazione misura il comportamento di L2 solo sui veri tiri che L1 ha riconosciuto come `tiro`.

```text
Veri tiri totali nel validation set: 65
Veri tiri arrivati a L2: 62
Tiri persi prima di L2 per errore L1: 3
```

| Metrica | Valore |
|---|---:|
| Accuracy L2 condizionata | 0.9032 |
| Macro Precision L2 condizionata | 0.9033 |
| Macro Recall L2 condizionata | 0.9033 |
| Macro F1 L2 condizionata | 0.9033 |
| Weighted Precision L2 condizionata | 0.9032 |
| Weighted Recall L2 condizionata | 0.9032 |
| Weighted F1 L2 condizionata | 0.9032 |

#### Classification report - L2 condizionato

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| tiroDaDue | 0.9062 | 0.9062 | 0.9062 | 32 |
| tiroDaTre | 0.9286 | 0.9286 | 0.9286 | 14 |
| tiroLibero | 0.8750 | 0.8750 | 0.8750 | 16 |
| accuracy |  |  | 0.9032 | 62 |
| macro avg | 0.9033 | 0.9033 | 0.9033 | 62 |
| weighted avg | 0.9032 | 0.9032 | 0.9032 | 62 |

#### Confusion matrix - L2 condizionato

Ordine classi:

```text
tiroDaDue, tiroDaTre, tiroLibero
```

```text
[[29  1  2]
 [ 1 13  0]
 [ 2  0 14]]
```

### L2 standalone su tutti i veri tiri

Questa valutazione misura il checkpoint L2 da solo, dando direttamente in input tutti i 65 veri tiri del validation set. In questo caso L2 viene valutato indipendentemente dagli errori di L1.

| Metrica | Valore |
|---|---:|
| Accuracy L2 standalone | 0.9077 |
| Macro Precision L2 standalone | 0.9095 |
| Macro Recall L2 standalone | 0.9095 |
| Macro F1 L2 standalone | 0.9095 |
| Weighted Precision L2 standalone | 0.9077 |
| Weighted Recall L2 standalone | 0.9077 |
| Weighted F1 L2 standalone | 0.9077 |

#### Classification report - L2 standalone

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| tiroDaDue | 0.9062 | 0.9062 | 0.9062 | 32 |
| tiroDaTre | 0.9333 | 0.9333 | 0.9333 | 15 |
| tiroLibero | 0.8889 | 0.8889 | 0.8889 | 18 |
| accuracy |  |  | 0.9077 | 65 |
| macro avg | 0.9095 | 0.9095 | 0.9095 | 65 |
| weighted avg | 0.9077 | 0.9077 | 0.9077 | 65 |

#### Confusion matrix - L2 standalone

Ordine classi:

```text
tiroDaDue, tiroDaTre, tiroLibero
```

```text
[[29  1  2]
 [ 1 14  0]
 [ 2  0 16]]
```

## L3 - esito del tiro

### L3 condizionato nella gerarchia

Questa valutazione misura il comportamento di L3 solo sui veri tiri che L1 ha riconosciuto come `tiro`.

```text
Veri tiri totali nel validation set: 65
Veri tiri arrivati a L3: 62
Tiri persi prima di L3 per errore L1: 3
```

| Metrica | Valore |
|---|---:|
| Accuracy L3 condizionata | 0.9516 |
| Macro Precision L3 condizionata | 0.9423 |
| Macro Recall L3 condizionata | 0.9615 |
| Macro F1 L3 condizionata | 0.9494 |
| Weighted Precision L3 condizionata | 0.9572 |
| Weighted Recall L3 condizionata | 0.9516 |
| Weighted F1 L3 condizionata | 0.9521 |

#### Classification report - L3 condizionato

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| tiro0 | 1.0000 | 0.9231 | 0.9600 | 39 |
| tiro1 | 0.8846 | 1.0000 | 0.9388 | 23 |
| accuracy |  |  | 0.9516 | 62 |
| macro avg | 0.9423 | 0.9615 | 0.9494 | 62 |
| weighted avg | 0.9572 | 0.9516 | 0.9521 | 62 |

#### Confusion matrix - L3 condizionato

Ordine classi:

```text
tiro0, tiro1
```

```text
[[36  3]
 [ 0 23]]
```

### L3 standalone su tutti i veri tiri

Questa valutazione misura il checkpoint L3 da solo, dando direttamente in input tutti i 65 veri tiri del validation set. In questo caso L3 viene valutato indipendentemente dagli errori di L1.

Checkpoint valutato:

```text
outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt
```

| Metrica | Valore |
|---|---:|
| Accuracy L3 standalone | 0.9385 |
| Macro Precision L3 standalone | 0.9310 |
| Macro Recall L3 standalone | 0.9500 |
| Macro F1 L3 standalone | 0.9366 |
| Weighted Precision L3 standalone | 0.9469 |
| Weighted Recall L3 standalone | 0.9385 |
| Weighted F1 L3 standalone | 0.9391 |

#### Classification report - L3 standalone

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| tiro0 | 1.0000 | 0.9000 | 0.9474 | 40 |
| tiro1 | 0.8621 | 1.0000 | 0.9259 | 25 |
| accuracy |  |  | 0.9385 | 65 |
| macro avg | 0.9310 | 0.9500 | 0.9366 | 65 |
| weighted avg | 0.9469 | 0.9385 | 0.9391 | 65 |

#### Confusion matrix - L3 standalone

Ordine classi:

```text
tiro0, tiro1
```

```text
[[36  4]
 [ 0 25]]
```

Interpretazione principale:

- 61 tiri su 65 hanno l'esito riconosciuto correttamente dal solo livello L3;
- 4 tiri con esito `tiro0` vengono classificati come `tiro1`;
- nessun `tiro1` viene classificato come `tiro0`.

# Analisi sintetica

Il modello `exp_46` sul validation set ottiene una buona accuratezza globale, pari a 0.9056 sulle 8 classi finali. La classe `passaggio` raggiunge F1-score 0.9203 su 212 campioni, mentre `no-action` ottiene F1-score 0.9194 su 263 campioni.

Le classi di tiro hanno risultati complessivamente buoni sul validation set, ma il supporto di alcune classi è molto basso. In particolare, `tiroDaTre1` ha solo 3 campioni e F1-score 0.5714, quindi questa metrica va interpretata con cautela.

La valutazione collassata senza esito del tiro è più stabile: la Macro F1 sale a 0.8744 e la Weighted F1 a 0.9111. Questo indica che il riconoscimento del tipo generale di azione è più robusto rispetto alla classificazione completa con esito del tiro separato.

La valutazione binaria `azione/no-action` ottiene Accuracy 0.9241 e Macro F1 0.9238. Rimangono 12 azioni perse e 29 falsi positivi da `no-action` verso azione.

Dalle metriche gerarchiche emerge che L1 è stabile su tutte le clip, con Macro F1 0.9264. L2 ottiene Macro F1 0.9033 quando viene valutato in modo condizionato nella gerarchia sui veri tiri arrivati al ramo `tiro`; valutato standalone su tutti i 65 veri tiri del validation set ottiene Macro F1 0.9095. L3 ottiene Macro F1 condizionata 0.9494 sui tiri arrivati al ramo `tiro`; valutato standalone su tutti i 65 veri tiri con lo stesso checkpoint usato in `exp_46` ottiene Macro F1 0.9366.

Il valore L3 condizionato resta leggermente più alto della metrica standalone perché considera solo i veri tiri che L1 invia effettivamente al ramo `tiro`. La metrica standalone su tutti i veri tiri fornisce invece una lettura indipendente del solo classificatore L3.

Il risultato finale da riportare come valutazione sul validation set è quindi:

```text
Accuracy 8 classi:                       0.9056
Macro Precision 8 classi:                0.8004
Macro Recall 8 classi:                   0.8318
Macro F1 8 classi:                       0.8079
Weighted Precision 8 classi:             0.9107
Weighted Recall 8 classi:                0.9056
Weighted F1 8 classi:                    0.9062

Accuracy 7 azioni:                       0.9206
Macro Precision 7 azioni:                0.7788
Macro Recall 7 azioni:                   0.8235
Macro F1 7 azioni:                       0.7919
Weighted Precision 7 azioni:             0.8723
Weighted Recall 7 azioni:                0.9206
Weighted F1 7 azioni:                    0.8936

Accuracy tipo azione:                    0.9111
Macro Precision tipo azione:             0.8720
Macro Recall tipo azione:                0.8787
Macro F1 tipo azione:                    0.8744
Weighted Precision tipo azione:          0.9133
Weighted Recall tipo azione:             0.9111
Weighted F1 tipo azione:                 0.9111

Accuracy azione/no-action:               0.9241
Macro Precision azione/no-action:        0.9263
Macro Recall azione/no-action:           0.9232
Macro F1 azione/no-action:               0.9238
Weighted Precision azione/no-action:     0.9256
Weighted Recall azione/no-action:        0.9241
Weighted F1 azione/no-action:            0.9239

Accuracy L1 su tutte le clip:            0.9222
Macro Precision L1 su tutte le clip:     0.9222
Macro Recall L1 su tutte le clip:        0.9321
Macro F1 L1 su tutte le clip:            0.9264
Weighted Precision L1 su tutte le clip:  0.9240
Weighted Recall L1 su tutte le clip:     0.9222
Weighted F1 L1 su tutte le clip:         0.9222

Accuracy L2 condizionata:                0.9032
Macro Precision L2 condizionata:         0.9033
Macro Recall L2 condizionata:            0.9033
Macro F1 L2 condizionata:                0.9033
Weighted Precision L2 condizionata:      0.9032
Weighted Recall L2 condizionata:         0.9032
Weighted F1 L2 condizionata:             0.9032

Accuracy L2 standalone:                  0.9077
Macro Precision L2 standalone:           0.9095
Macro Recall L2 standalone:              0.9095
Macro F1 L2 standalone:                  0.9095
Weighted Precision L2 standalone:        0.9077
Weighted Recall L2 standalone:           0.9077
Weighted F1 L2 standalone:               0.9077

Accuracy L3 condizionata:                0.9516
Macro Precision L3 condizionata:         0.9423
Macro Recall L3 condizionata:            0.9615
Macro F1 L3 condizionata:                0.9494
Weighted Precision L3 condizionata:      0.9572
Weighted Recall L3 condizionata:         0.9516
Weighted F1 L3 condizionata:             0.9521

Accuracy L3 standalone:                  0.9385
Macro Precision L3 standalone:           0.9310
Macro Recall L3 standalone:              0.9500
Macro F1 L3 standalone:                  0.9366
Weighted Precision L3 standalone:        0.9469
Weighted Recall L3 standalone:           0.9385
Weighted F1 L3 standalone:               0.9391
```
