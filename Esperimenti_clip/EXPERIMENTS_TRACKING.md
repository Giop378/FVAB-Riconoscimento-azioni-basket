# Esperimenti Tracking Palla/Canestro

Questo file tiene traccia degli esperimenti in cui le feature di tracking di **palla** e **canestro/rim** vengono combinate con le feature video **DINOv3-L/16 frozen** per migliorare il riconoscimento delle azioni nel dataset BasketAR.

La gerarchia usata negli esperimenti è:

```text
L1: passaggio / tiro / no-action
L2: tiroDaDue / tiroDaTre / tiroLibero        solo se L1 = tiro
L3: tiro0 / tiro1                             solo se L1 = tiro
```

Il tracking viene usato in tre forme:

```text
aggregate39: 39 feature aggregate per clip, replicate su tutti i timestep DINOv3
temp29:      29 feature temporali per frame, interpolate alla lunghezza della sequenza DINOv3
temp43:      43 feature temporali per frame, versione estesa delle feature palla/canestro
```

Con `aggregate39` l'input effettivo passa da `[T, 1024]` a `[T, 1063]`; con `temp29` passa da `[T, 1024]` a `[T, 1053]`; con `temp43` passa da `[T, 1024]` a `[T, 1067]`.

## Nota sulle classi

Il dataset originale contiene 9 classi:

```text
passaggio, tiroDaDue0, tiroDaDue1, tiroDaTre0, tiroDaTre1, tiroLibero0, tiroLibero1, idle, non-gioco
```

Negli esperimenti di questo file, però, non viene mantenuta una valutazione finale a 9 classi:

- i singoli livelli usano 3 classi per L1, 3 classi per L2 e 2 classi per L3;
- gli esperimenti end-to-end usano 8 classi finali, perché `idle` e `non-gioco` vengono collassate in `no-action`;
- per il report finale delle azioni si considerano le 7 azioni reali, lasciando `no-action` come classe di background/scarto.

Quindi gli esperimenti qui riportati non sono da leggere come risultati a 9 classi. Gli eventuali risultati precedenti a 9 classi vanno tenuti separati, perché non sono direttamente confrontabili con gli esperimenti a 8 classi con `idle/non-gioco -> no-action`.

---

# Parte 1 - Esperimenti storici prima del confronto tra versioni YOLO

Questa parte contiene gli esperimenti iniziali in cui è stato introdotto il tracking palla/canestro usando il detector YOLO storico. Qui l'obiettivo principale era verificare se il tracking potesse migliorare soprattutto lo **Stadio 3**, cioè la distinzione tra tiro segnato e tiro sbagliato.

## Feature tracking storiche

| Feature root | Detector | Scope | Tipo feature | Uso |
|---|---|---|---|---|
| `data/features/ball_rim_tracking_features_v1` | YOLO v1 storico | principalmente tiri | `aggregate39` | primi esperimenti L3 e gerarchia |
| `data/features/ball_rim_tracking_temporal_v1` | YOLO v1 storico | principalmente tiri | `temp29` | primo esperimento temporale L3 |
| `data/features/ball_rim_tracking_all_train_val` | YOLO v1 storico | tutte le clip train/val | `aggregate39` | baseline non gerarchica con no-action |

## Singolo livello storico - Stadio 3

|ID|Tracking|Label mode|Classi|Val Loss|Val Accuracy|Val Macro F1|Val Weighted F1|Output dir|
|-|-|-|-:|-:|-:|-:|-:|-|
|`exp_l3_yolo_v1_aggregate39_shots_d256_mean`|YOLO v1 `aggregate39`|`shot_outcome_only`|2|0.4656|0.8154|0.8132|0.8179|`outputs/exp_l3_yolo_v1_aggregate39_shots_d256_mean`|
|`exp_l3_yolo_v1_aggregate39_shots_d384_mean_s04`|YOLO v1 `aggregate39`|`shot_outcome_only`|2|0.4559|0.7846|0.7842|0.7865|`outputs/exp_l3_yolo_v1_aggregate39_shots_d384_mean_s04`|
|`exp_l3_tracking_temporal_v1`|YOLO v1 storico `temp29`|`shot_outcome_only`|2|0.4576|**0.8769**|**0.8733**|**0.8782**|`outputs/exp_l3_tracking_temporal_v1`|

Nota: `exp_l3_tracking_temporal_v1` è il nome/output storico del primo L3 temporale con YOLO v1, usato poi nella valutazione end-to-end `exp_40`. Nelle sezioni di confronto successive il riferimento YOLO v1 temporale viene indicato anche come `exp_l3_yolo_v1_temp29_shots_d256_mean`, ma le metriche riportate per lo Stadio 3 sono le stesse.

### `exp_l3_yolo_v1_aggregate39_shots_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.94|0.75|0.83|40|
|tiro1|0.70|0.92|0.79|25|

Confusion matrix:

```text
[[30 10]
 [ 2 23]]
```

### `exp_l3_yolo_v1_aggregate39_shots_d384_mean_s04`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.96|0.68|0.79|40|
|tiro1|0.65|0.96|0.77|25|

Confusion matrix:

```text
[[27 13]
 [ 1 24]]
```

### `exp_l3_tracking_temporal_v1`

Best epoch: 48. Questo è il modello L3 temporale storico usato in `exp_40`.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.94|0.85|0.89|40|
|tiro1|0.79|0.92|0.85|25|

Confusion matrix:

```text
[[34  6]
 [ 2 23]]
```

Le feature temporali `temp29` migliorano nettamente lo Stadio 3 rispetto alle feature aggregate: la Macro F1 passa da 0.8132 a 0.8733.

## Esperimenti end-to-end storici e baseline globale

|ID|Tipo|Configurazione|Accuracy 8 classi|Macro F1 8 classi|Weighted F1 8 classi|Micro F1 7 azioni|Macro F1 7 azioni|Weighted F1 7 azioni|Macro F1 collassato senza esito|Output dir|
|-|-|-|-:|-:|-:|-:|-:|-:|-:|-|
|`exp_38_dinov3_vitl16_hierarchical_tracking_l3`|Gerarchico|L1 `exp_28`, L2 `exp_29`, L3 YOLO v1 `aggregate39`|0.8759|0.6996|0.8766|0.85|0.67|0.85|0.8153|`outputs/exp_38_dinov3_vitl16_hierarchical_tracking_l3`|
|`exp_39_nonhierarchical_dinov3_tracking_noaction`|Non gerarchico|DINOv3 + YOLO v1 `aggregate39`, `idle/non-gioco -> no-action`|0.8426|0.5620|0.8447|0.80|0.52|0.80|0.7497*|`outputs/exp_39_nonhierarchical_dinov3_tracking_noaction`|
|`exp_40_dinov3_hierarchical_tracking_temporal_l3`|Gerarchico|L1 `exp_28`, L2 `exp_29`, L3 YOLO v1 `temp29`|0.8815|0.7223|0.8808|0.86|0.70|0.86|0.8153|`outputs/exp_40_dinov3_hierarchical_tracking_temporal_l3`|

`*` La metrica collassata di `exp_39` non era stampata direttamente nel file `results.txt`; è stata ricavata dalla confusion matrix a 8 classi collassando le classi di tiro per tipo.

## Risultati per classe - esperimenti storici end-to-end

### `exp_38_dinov3_vitl16_hierarchical_tracking_l3` - 8 classi finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.92|0.52|0.67|21|
|tiroDaDue1|0.43|0.82|0.56|11|
|tiroDaTre0|0.75|0.75|0.75|12|
|tiroDaTre1|0.33|0.67|0.44|3|
|tiroLibero0|0.83|0.71|0.77|7|
|tiroLibero1|0.83|0.45|0.59|11|
|no-action|0.95|0.86|0.90|263|

Confusion matrix - 8 classi finali con no-action:

```text
[[205   0   1   0   0   0   0   6]
 [  0  11   6   2   1   0   0   1]
 [  0   0   9   1   1   0   0   0]
 [  0   1   0   9   1   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   5   1   1]
 [  0   0   1   0   1   0   5   4]
 [ 32   0   3   0   0   1   0 227]]
```

Confusion matrix - tipo azione senza esito:

```text
[[205   1   0   0   6]
 [  0  26   5   0   1]
 [  0   2  12   0   1]
 [  0   1   1  11   5]
 [ 32   3   0   1 227]]
```

### `exp_39_nonhierarchical_dinov3_tracking_noaction` - 8 classi finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.84|0.96|0.90|212|
|tiroDaDue0|0.52|0.52|0.52|21|
|tiroDaDue1|0.38|0.45|0.42|11|
|tiroDaTre0|0.60|0.75|0.67|12|
|tiroDaTre1|0.25|0.33|0.29|3|
|tiroLibero0|1.00|0.29|0.44|7|
|tiroLibero1|0.31|0.45|0.37|11|
|no-action|0.96|0.83|0.89|263|

Confusion matrix - 8 classi con no-action:

```text
[[204   3   0   0   0   0   0   5]
 [  0  11   6   1   0   0   3   0]
 [  0   3   5   1   1   0   1   0]
 [  0   1   1   9   1   0   0   0]
 [  0   2   0   0   1   0   0   0]
 [  0   0   0   1   0   2   3   1]
 [  0   0   0   2   1   0   5   3]
 [ 38   1   1   1   0   0   4 218]]
```

Valutazione solo sulle 7 azioni reali:

```text
micro avg:    0.76 precision, 0.86 recall, 0.80 F1
macro avg:    0.56 precision, 0.54 recall, 0.52 F1
weighted avg: 0.77 precision, 0.86 recall, 0.80 F1
```

Valutazione collassata senza esito, ricavata dalla confusion matrix 8 classi:

```text
Accuracy tipo azione:    0.8667
Macro F1 tipo azione:    0.7497
Weighted F1 tipo azione: 0.8676
```

Confusion matrix - tipo azione senza esito, ricavata dalla confusion matrix 8 classi:

```text
[[204   3   0   0   5]
 [  0  25   3   4   0]
 [  0   4  11   0   0]
 [  0   0   4  10   4]
 [ 38   2   1   4 218]]
```

### `exp_40_dinov3_hierarchical_tracking_temporal_l3` - 8 classi finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.88|0.67|0.76|21|
|tiroDaDue1|0.53|0.82|0.64|11|
|tiroDaTre0|0.71|0.83|0.77|12|
|tiroDaTre1|0.50|0.67|0.57|3|
|tiroLibero0|0.80|0.57|0.67|7|
|tiroLibero1|0.71|0.45|0.56|11|
|no-action|0.95|0.86|0.90|263|

Confusion matrix - 8 classi finali con no-action:

```text
[[205   0   1   0   0   0   0   6]
 [  0  14   3   3   0   0   0   1]
 [  0   0   9   1   1   0   0   0]
 [  0   1   0  10   0   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   4   2   1]
 [  0   0   1   0   1   0   5   4]
 [ 32   1   2   0   0   1   0 227]]
```

Confusion matrix - tipo azione senza esito:

```text
[[205   1   0   0   6]
 [  0  26   5   0   1]
 [  0   2  12   0   1]
 [  0   1   1  11   5]
 [ 32   3   0   1 227]]
```

Il passaggio da `aggregate39` a `temp29` su L3 migliora l'end-to-end: Macro F1 8 classi da 0.6996 a 0.7223.

---

# Parte 2 - Confronto tra differenti versioni YOLO e integrazione progressiva nella gerarchia

Questa parte parte dal momento in cui sono state confrontate più versioni del detector YOLO per il tracking palla/canestro. L'obiettivo è capire quale versione usare su L1, L2 e L3.

## Feature tracking complete usate nel confronto YOLO

| Feature root corrente | Detector YOLO | Scope | Tipo feature | Uso previsto |
|---|---|---|---|---|
|`data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1`|YOLO v1|train/val/test, tutte le clip|`temp29`|riferimento storico e Stadio 3|
|`data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2`|YOLO v2|train/val/test, tutte le clip|`temp29`|migliore su L2|
|`data/features/ball_rim_tracking_temporal_clip_complete_yolo_v3`|YOLO v3|train/val/test, tutte le clip|`temp29`|migliore di YOLO v2 su L3, ma sotto YOLO v1 storico|
|`data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2_v4`|YOLO v2|train/val/test, tutte le clip|`temp43`|versione estesa, migliore su L1; testata anche su L2|
|`data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1_v4`|YOLO v1|train/val/test, tutte le clip|`temp43`|versione estesa, migliore su L3|

## Tabella riassuntiva - singoli livelli nel confronto YOLO

|ID|Livello|Tracking|Label mode|Classi|Epoche|Val Loss|Val Accuracy|Val Macro F1|Val Weighted F1|Output dir|
|-|-|-|-|-:|-:|-:|-:|-:|-:|-|
|`exp_l1_yolo_v1_temp29_allclips_d256_mean`|L1|YOLO v1 `temp29`|`action_noaction`|3|100|0.2665|0.9000|0.9113|0.8999|`outputs/exp_l1_yolo_v1_temp29_allclips_d256_mean`|
|`exp_l1_yolo_v2_temp29_allclips_d256_mean`|L1|YOLO v2 `temp29`|`action_noaction`|3|100|0.2974|0.9130|0.9211|0.9128|`outputs/exp_l1_yolo_v2_temp29_allclips_d256_mean`|
|`exp_l1_yolo_v3_temp29_allclips_d256_mean`|L1|YOLO v3 `temp29`|`action_noaction`|3|100|0.2833|0.9000|0.9078|0.9000|`outputs/exp_l1_yolo_v3_temp29_allclips_d256_mean`|
|`exp_l1_yolo_v2_temp43_allclips_d256_mean`|L1|YOLO v2 `temp43`|`action_noaction`|3|100|0.3198|**0.9222**|**0.9264**|**0.9222**|`outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean`|
|`exp_l2_yolo_v1_temp29_allclips_d256_mean`|L2|YOLO v1 `temp29`|`shot_type_only`|3|100|0.6552|0.8462|0.8401|0.8450|`outputs/exp_l2_yolo_v1_temp29_allclips_d256_mean`|
|`exp_l2_yolo_v2_temp29_allclips_d256_mean`|L2|YOLO v2 `temp29`|`shot_type_only`|3|100|0.4127|**0.9077**|**0.9095**|**0.9077**|`outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean`|
|`exp_l2_yolo_v3_temp29_allclips_d256_mean`|L2|YOLO v3 `temp29`|`shot_type_only`|3|100|0.6945|0.8923|0.8915|0.8921|`outputs/exp_l2_yolo_v3_temp29_allclips_d256_mean`|
|`exp_l2_yolo_v2_temp43_allclips_d256_mean`|L2|YOLO v2 `temp43`|`shot_type_only`|3|100|0.4614|0.8923|0.8921|0.8930|`outputs/exp_l2_yolo_v2_temp43_allclips_d256_mean`|
|`exp_l3_yolo_v1_temp29_shots_d256_mean`|L3|YOLO v1 `temp29`|`shot_outcome_only`|2|100|0.4576|0.8769|0.8733|0.8782|`outputs/exp_l3_yolo_v1_temp29_shots_d256_mean`|
|`exp_l3_yolo_v2_temp29_allclips_d256_mean`|L3|YOLO v2 `temp29`|`shot_outcome_only`|2|100|0.4430|0.8462|0.8397|0.8471|`outputs/exp_l3_yolo_v2_temp29_allclips_d256_mean`|
|`exp_l3_yolo_v3_temp29_allclips_d256_mean`|L3|YOLO v3 `temp29`|`shot_outcome_only`|2|100|0.4764|0.8615|0.8594|0.8634|`outputs/exp_l3_yolo_v3_temp29_allclips_d256_mean`|
|`exp_l3_yolo_v1_temp43_shots_d256_mean`|L3|YOLO v1 `temp43`|`shot_outcome_only`|2|100|0.2162|**0.9385**|**0.9366**|**0.9391**|`outputs/exp_l3_yolo_v1_temp43_shots_d256_mean`|

Scelte migliori per livello:

```text
L1: YOLO v2 temp43
L2: YOLO v2 temp29
L3: YOLO v1 temp43
```

## Risultati aggregati per livello

### Stadio 1

|ID|Tracking|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-|-:|-:|-:|-:|-:|-:|-:|
|`exp_28_dinov3_vitl16_transformer_mean_action_noaction`|no|0.9074|0.91|0.91|0.9073|0.91|0.91|0.9072|
|`exp_l1_yolo_v1_temp29_allclips_d256_mean`|YOLO v1 `temp29`|0.9000|0.91|0.92|0.9113|0.91|0.90|0.8999|
|`exp_l1_yolo_v2_temp29_allclips_d256_mean`|YOLO v2 `temp29`|0.9130|0.92|0.93|0.9211|0.92|0.91|0.9128|
|`exp_l1_yolo_v3_temp29_allclips_d256_mean`|YOLO v3 `temp29`|0.9000|0.90|0.92|0.9078|0.91|0.90|0.9000|
|`exp_l1_yolo_v2_temp43_allclips_d256_mean`|YOLO v2 `temp43`|**0.9222**|**0.92**|**0.93**|**0.9264**|**0.92**|**0.92**|**0.9222**|

### Stadio 2

|ID|Tracking|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-|-:|-:|-:|-:|-:|-:|-:|
|`exp_29_dinov3_vitl16_transformer_mean_shot_type_only`|no|0.8462|0.85|0.85|0.8410|0.87|0.85|0.8520|
|`exp_l2_yolo_v1_temp29_allclips_d256_mean`|YOLO v1 `temp29`|0.8462|0.88|0.82|0.8401|0.86|0.85|0.8450|
|`exp_l2_yolo_v2_temp29_allclips_d256_mean`|YOLO v2 `temp29`|**0.9077**|**0.91**|**0.91**|**0.9095**|**0.91**|**0.91**|**0.9077**|
|`exp_l2_yolo_v3_temp29_allclips_d256_mean`|YOLO v3 `temp29`|0.8923|0.91|0.88|0.8915|0.90|0.89|0.8921|
|`exp_l2_yolo_v2_temp43_allclips_d256_mean`|YOLO v2 `temp43`|0.8923|0.90|0.89|0.8921|0.90|0.89|0.8930|

### Stadio 3

|ID|Tracking|Accuracy|Macro Precision|Macro Recall|Macro F1|Weighted Precision|Weighted Recall|Weighted F1|
|-|-|-:|-:|-:|-:|-:|-:|-:|
|`exp_l3_yolo_v1_temp29_shots_d256_mean`|YOLO v1 `temp29`|0.8769|0.87|0.89|0.8733|0.89|0.88|0.8782|
|`exp_l3_yolo_v2_temp29_allclips_d256_mean`|YOLO v2 `temp29`|0.8462|0.84|0.84|0.8397|0.85|0.85|0.8471|
|`exp_l3_yolo_v3_temp29_allclips_d256_mean`|YOLO v3 `temp29`|0.8615|0.86|0.88|0.8594|0.89|0.86|0.8634|
|`exp_l3_yolo_v1_temp43_shots_d256_mean`|YOLO v1 `temp43`|**0.9385**|**0.93**|**0.95**|**0.9366**|**0.95**|**0.94**|**0.9391**|

## Risultati per classe - singoli livelli del confronto YOLO

### L1 - `exp_l1_yolo_v1_temp29_allclips_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.84|0.96|0.89|212|
|tiro|0.94|0.95|0.95|65|
|no-action|0.96|0.84|0.89|263|

Confusion matrix:

```text
[[204   1   7]
 [  0  62   3]
 [ 40   3 220]]
```

### L1 - `exp_l1_yolo_v2_temp29_allclips_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.85|0.98|0.91|212|
|tiro|0.94|0.95|0.95|65|
|no-action|0.97|0.85|0.91|263|

Confusion matrix:

```text
[[208   1   3]
 [  0  62   3]
 [ 37   3 223]]
```

### L1 - `exp_l1_yolo_v3_temp29_allclips_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.85|0.94|0.89|212|
|tiro|0.90|0.97|0.93|65|
|no-action|0.95|0.85|0.90|263|

Confusion matrix:

```text
[[199   4   9]
 [  0  63   2]
 [ 36   3 224]]
```


### L1 - `exp_l1_yolo_v2_temp43_allclips_d256_mean`

Questo esperimento usa il detector YOLO v2, ma con la versione estesa delle feature temporali palla/canestro a 43 dimensioni per frame. Il miglior modello è stato salvato alla epoch 49.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.89|0.95|0.92|212|
|tiro|0.93|0.95|0.94|65|
|no-action|0.95|0.89|0.92|263|

Confusion matrix:

```text
[[202   1   9]
 [  0  62   3]
 [ 25   4 234]]
```

Rispetto a `exp_l1_yolo_v2_temp29_allclips_d256_mean`, la versione `temp43` migliora L1 da Accuracy 0.9130 / Macro F1 0.9211 a Accuracy 0.9222 / Macro F1 0.9264.

### L2 - `exp_l2_yolo_v1_temp29_allclips_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiroDaDue|0.79|0.94|0.86|32|
|tiroDaTre|0.92|0.73|0.81|15|
|tiroLibero|0.93|0.78|0.85|18|

Confusion matrix:

```text
[[30  1  1]
 [ 4 11  0]
 [ 4  0 14]]
```

### L2 - `exp_l2_yolo_v2_temp29_allclips_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiroDaDue|0.91|0.91|0.91|32|
|tiroDaTre|0.93|0.93|0.93|15|
|tiroLibero|0.89|0.89|0.89|18|

Confusion matrix:

```text
[[29  1  2]
 [ 1 14  0]
 [ 2  0 16]]
```

### L2 - `exp_l2_yolo_v3_temp29_allclips_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiroDaDue|0.86|0.94|0.90|32|
|tiroDaTre|0.93|0.87|0.90|15|
|tiroLibero|0.94|0.83|0.88|18|

Confusion matrix:

```text
[[30  1  1]
 [ 2 13  0]
 [ 3  0 15]]
```


### L2 - `exp_l2_yolo_v2_temp43_allclips_d256_mean`

Questo esperimento testa YOLO v2 con feature temporali estese `temp43` anche sullo Stadio 2. Il miglior modello è stato salvato alla epoch 34.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiroDaDue|0.88|0.91|0.89|32|
|tiroDaTre|0.82|0.93|0.88|15|
|tiroLibero|1.00|0.83|0.91|18|

Confusion matrix:

```text
[[29  3  0]
 [ 1 14  0]
 [ 3  0 15]]
```

Su L2, `temp43` non migliora la configurazione migliore precedente: `exp_l2_yolo_v2_temp29_allclips_d256_mean` resta superiore con Accuracy 0.9077 e Macro F1 0.9095, contro Accuracy 0.8923 e Macro F1 0.8921 della versione `temp43`.

### L3 - `exp_l3_yolo_v1_temp29_shots_d256_mean` / `exp_l3_tracking_temporal_v1`

Il risultato storico `exp_l3_tracking_temporal_v1` è il riferimento YOLO v1 temporale usato come miglior L3. Nel confronto tra detector viene mantenuto come baseline YOLO v1 per lo Stadio 3.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.94|0.85|0.89|40|
|tiro1|0.79|0.92|0.85|25|

Confusion matrix:

```text
[[34  6]
 [ 2 23]]
```

### L3 - `exp_l3_yolo_v2_temp29_allclips_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.89|0.85|0.87|40|
|tiro1|0.78|0.84|0.81|25|

Confusion matrix:

```text
[[34  6]
 [ 4 21]]
```

### L3 - `exp_l3_yolo_v3_temp29_allclips_d256_mean`

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|0.97|0.80|0.88|40|
|tiro1|0.75|0.96|0.84|25|

Confusion matrix:

```text
[[32  8]
 [ 1 24]]
```


### L3 - `exp_l3_yolo_v1_temp43_shots_d256_mean`

Questo esperimento usa YOLO v1 con feature temporali estese `temp43` per lo Stadio 3, cioè la distinzione tra tiro sbagliato e tiro segnato. Il miglior modello è stato salvato alla epoch 29.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|tiro0|1.00|0.90|0.95|40|
|tiro1|0.86|1.00|0.93|25|

Confusion matrix:

```text
[[36  4]
 [ 0 25]]
```

Rispetto al precedente miglior L3 YOLO v1 `temp29`, la versione `temp43` migliora in modo netto: Accuracy da 0.8769 a 0.9385 e Macro F1 da 0.8733 a 0.9366.

## Tabella riassuntiva - end-to-end dopo il confronto YOLO

|ID|Configurazione|Accuracy 8 classi|Macro F1 8 classi|Weighted F1 8 classi|Micro F1 7 azioni|Macro F1 7 azioni|Weighted F1 7 azioni|Macro F1 collassato senza esito|Output dir|
|-|-|-:|-:|-:|-:|-:|-:|-:|-|
|`exp_41_hier_dinov3_l3_yolo_v3_temp29_allclips`|L1 `exp_28`, L2 `exp_29`, L3 YOLO v3 `temp29`|0.8778|0.6979|0.8774|0.86|0.67|0.85|0.8153|`outputs/exp_41_hier_dinov3_l3_yolo_v3_temp29_allclips`|
|`exp_42_hier_dinov3_l2_yolo_v2_l3_yolo_v3_temp29_allclips`|L1 `exp_28`, L2 YOLO v2 `temp29`, L3 YOLO v3 `temp29`|0.8852|0.7438|0.8849|0.87|0.72|0.87|0.8456|`outputs/exp_42_hier_dinov3_l2_yolo_v2_l3_yolo_v3_temp29_allclips`|
|`exp_43_hier_dinov3_l1_yolo_v2_l2_yolo_v2_l3_yolo_v3_temp29_allclips`|L1 YOLO v2 `temp29`, L2 YOLO v2 `temp29`, L3 YOLO v3 `temp29`|0.8889|0.7607|0.8894|0.87|0.74|0.87|0.8724|`outputs/exp_43_hier_dinov3_l1_yolo_v2_l2_yolo_v2_l3_yolo_v3_temp29_allclips`|
|`exp_44_hier_dinov3_l1_yolo_v2_l2_yolo_v2_l3_yolo_v1_temp29_allclips`|L1 YOLO v2 `temp29`, L2 YOLO v2 `temp29`, L3 YOLO v1 `temp29`|0.8907|0.7762|0.8908|0.88|0.76|0.88|0.8724|`outputs/exp_44_hier_dinov3_l1_yolo_v2_l2_yolo_v2_l3_yolo_v1_temp29_allclips`|
|`exp_45_hier_l1_yolo_v2_passaggio_threshold_sweep`|`exp_44` + soglia L1 su `passaggio`|0.8926|0.7767|0.8927|0.88|0.76|0.88|0.8733|`outputs/exp_45_hier_l1_yolo_v2_passaggio_threshold_sweep`|
|`exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43`|L1 YOLO v2 `temp43`, L2 YOLO v2 `temp29`, L3 YOLO v1 `temp43`|**0.9056**|**0.8079**|**0.9062**|**0.89**|**0.79**|**0.89**|**0.8744**|`outputs/exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43`|

`exp_46` diventa il miglior risultato operativo consigliato: usa la migliore configurazione osservata per ciascun livello, cioè L1 `temp43`, L2 `temp29` e L3 `temp43`. `exp_45` resta una semplice ablation di post-processing perché migliora pochissimo `exp_44` e usa una soglia scelta su validation.

## Risultati per classe - end-to-end dopo il confronto YOLO

### `exp_41_hier_dinov3_l3_yolo_v3_temp29_allclips` - 8 classi finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.76|0.62|0.68|21|
|tiroDaDue1|0.56|0.82|0.67|11|
|tiroDaTre0|0.75|0.75|0.75|12|
|tiroDaTre1|0.33|0.67|0.44|3|
|tiroLibero0|0.80|0.57|0.67|7|
|tiroLibero1|0.71|0.45|0.56|11|
|no-action|0.95|0.86|0.90|263|

Confusion matrix - 8 classi finali con no-action:

```text
[[205   1   0   0   0   0   0   6]
 [  0  13   4   3   0   0   0   1]
 [  0   0   9   0   2   0   0   0]
 [  0   1   0   9   1   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   4   2   1]
 [  0   0   1   0   1   0   5   4]
 [ 32   2   1   0   0   1   0 227]]
```

Confusion matrix - tipo azione senza esito:

```text
[[205   1   0   0   6]
 [  0  26   5   0   1]
 [  0   2  12   0   1]
 [  0   1   1  11   5]
 [ 32   3   0   1 227]]
```

### `exp_42_hier_dinov3_l2_yolo_v2_l3_yolo_v3_temp29_allclips` - 8 classi finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.84|0.76|0.80|21|
|tiroDaDue1|0.56|0.82|0.67|11|
|tiroDaTre0|0.91|0.83|0.87|12|
|tiroDaTre1|0.50|0.67|0.57|3|
|tiroLibero0|1.00|0.57|0.73|7|
|tiroLibero1|0.56|0.45|0.50|11|
|no-action|0.95|0.86|0.90|263|

Confusion matrix - 8 classi finali con no-action:

```text
[[205   0   0   1   0   0   0   6]
 [  0  16   3   0   0   0   1   1]
 [  0   0   9   0   1   0   1   0]
 [  0   0   0  10   1   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   4   2   1]
 [  0   0   2   0   0   0   5   4]
 [ 32   3   1   0   0   0   0 227]]
```

Confusion matrix - tipo azione senza esito:

```text
[[205   0   1   0   6]
 [  0  28   1   2   1]
 [  0   1  13   0   1]
 [  0   2   0  11   5]
 [ 32   4   0   0 227]]
```

### `exp_43_hier_dinov3_l1_yolo_v2_l2_yolo_v2_l3_yolo_v3_temp29_allclips` - 8 classi finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.85|0.98|0.91|212|
|tiroDaDue0|0.89|0.81|0.85|21|
|tiroDaDue1|0.60|0.82|0.69|11|
|tiroDaTre0|0.91|0.83|0.87|12|
|tiroDaTre1|0.50|0.67|0.57|3|
|tiroLibero0|0.71|0.71|0.71|7|
|tiroLibero1|0.60|0.55|0.57|11|
|no-action|0.97|0.85|0.91|263|

Confusion matrix - 8 classi finali con no-action:

```text
[[208   0   0   1   0   0   0   3]
 [  0  17   3   0   0   0   1   0]
 [  0   0   9   0   1   0   1   0]
 [  0   0   0  10   1   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   5   2   0]
 [  0   0   2   0   0   1   6   2]
 [ 37   2   0   0   0   1   0 223]]
```

Confusion matrix - tipo azione senza esito:

```text
[[208   0   1   0   3]
 [  0  29   1   2   0]
 [  0   1  13   0   1]
 [  0   2   0  14   2]
 [ 37   2   0   1 223]]
```

### `exp_44_hier_dinov3_l1_yolo_v2_l2_yolo_v2_l3_yolo_v1_temp29_allclips` - 8 classi finali

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.85|0.98|0.91|212|
|tiroDaDue0|0.86|0.86|0.86|21|
|tiroDaDue1|0.62|0.73|0.67|11|
|tiroDaTre0|0.92|0.92|0.92|12|
|tiroDaTre1|0.67|0.67|0.67|3|
|tiroLibero0|0.71|0.71|0.71|7|
|tiroLibero1|0.60|0.55|0.57|11|
|no-action|0.97|0.85|0.91|263|

Confusion matrix - 8 classi finali con no-action:

```text
[[208   0   0   1   0   0   0   3]
 [  0  18   2   0   0   0   1   0]
 [  0   1   8   0   1   0   1   0]
 [  0   0   0  11   0   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   5   2   0]
 [  0   0   2   0   0   1   6   2]
 [ 37   2   0   0   0   1   0 223]]
```

Confusion matrix - tipo azione senza esito:

```text
[[208   0   1   0   3]
 [  0  29   1   2   0]
 [  0   1  13   0   1]
 [  0   2   0  14   2]
 [ 37   2   0   1 223]]
```

Rispetto a `exp_43`, `exp_44` migliora soprattutto le classi di tiro con esito (`tiroDaDue0`, `tiroDaTre0`, `tiroDaTre1`) perché usa il miglior L3 storico YOLO v1.

### `exp_45_hier_l1_yolo_v2_passaggio_threshold_sweep` - 8 classi finali

Questo esperimento parte dalla stessa configurazione di `exp_44`, ma aggiunge un post-processing su L1. Lo sweep converte una predizione `passaggio` in `no-action` solo se `P(passaggio)` è sotto soglia e `P(no-action) > P(tiro)`.

La migliore soglia selezionata su validation secondo `macro_f1_8` è `0.65`, con 7 conversioni `passaggio -> no-action`.

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.86|0.97|0.91|212|
|tiroDaDue0|0.86|0.86|0.86|21|
|tiroDaDue1|0.62|0.73|0.67|11|
|tiroDaTre0|0.92|0.92|0.92|12|
|tiroDaTre1|0.67|0.67|0.67|3|
|tiroLibero0|0.71|0.71|0.71|7|
|tiroLibero1|0.60|0.55|0.57|11|
|no-action|0.96|0.86|0.91|263|

Confusion matrix - 8 classi finali con no-action:

```text
[[205   0   0   1   0   0   0   6]
 [  0  18   2   0   0   0   1   0]
 [  0   1   8   0   1   0   1   0]
 [  0   0   0  11   0   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   5   2   0]
 [  0   0   2   0   0   1   6   2]
 [ 33   2   0   0   0   1   0 227]]
```

Tabella dello sweep soglia L1 per `passaggio`:

|Threshold|Conversioni passaggio -> no-action|Accuracy 8|Macro F1 8|Weighted F1 8|Macro F1 7|Macro F1 tipo azione|Recall passaggio|Recall no-action|
|-|-:|-:|-:|-:|-:|-:|-:|-:|
|baseline|0|0.890741|0.776206|0.890759|0.757592|0.872449|0.981132|0.847909|
|0.50|0|0.890741|0.776206|0.890759|0.757592|0.872449|0.981132|0.847909|
|0.55|4|0.887037|0.775145|0.887118|0.756848|0.870752|0.966981|0.851711|
|0.60|5|0.888889|0.775671|0.888979|0.757134|0.871593|0.966981|0.855513|
|0.65|7|**0.892593**|**0.776723**|**0.892698**|**0.757710**|**0.873277**|0.966981|0.863118|
|0.70|8|0.890741|0.776193|0.890860|0.757363|0.872428|0.962264|0.863118|
|0.75|10|0.890741|0.776188|0.890878|0.757305|0.872420|0.957547|0.866920|
|0.80|15|0.885185|0.774583|0.885364|0.756187|0.869852|0.938679|0.870722|
|0.85|23|0.888889|0.775603|0.889076|0.756583|0.871484|0.924528|0.889734|
|0.90|30|0.890741|0.776089|0.890902|0.756691|0.872263|0.910377|0.904943|

Il miglioramento rispetto a `exp_44` è minimo: Macro F1 8 classi da 0.7762 a 0.7767. Inoltre la soglia riduce il recall di `passaggio` da 0.9811 a 0.9670 e viene scelta direttamente su validation. Per questo `exp_45` viene mantenuto come ablation, ma non scelto come modello principale.


### `exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43` - 8 classi finali

Configurazione:

```text
L1: outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt
    tracking: data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2_v4, temp43

L2: outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt
    tracking: data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2, temp29

L3: outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt
    tracking: data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1_v4, temp43
```

Metriche end-to-end:

```text
Accuracy 8 classi:    0.9056
Macro F1 8 classi:    0.8079
Weighted F1 8 classi: 0.9062
```

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.89|0.95|0.92|212|
|tiroDaDue0|0.90|0.86|0.88|21|
|tiroDaDue1|0.60|0.82|0.69|11|
|tiroDaTre0|0.91|0.83|0.87|12|
|tiroDaTre1|0.50|0.67|0.57|3|
|tiroLibero0|0.78|1.00|0.88|7|
|tiroLibero1|0.88|0.64|0.74|11|
|no-action|0.95|0.89|0.92|263|

Confusion matrix - 8 classi finali con no-action:

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

Valutazione solo sulle 7 azioni reali:

```text
micro avg F1:    0.89
macro avg F1:    0.79
weighted avg F1: 0.89
```

Confusion matrix - solo 7 azioni finali:

```text
[[202   0   0   1   0   0   0]
 [  0  18   2   0   0   1   0]
 [  0   0   9   0   1   0   1]
 [  0   0   0  10   1   0   0]
 [  0   0   1   0   2   0   0]
 [  0   0   0   0   0   7   0]
 [  0   0   2   0   0   0   7]]
```

Valutazione collassata senza esito del tiro:

```text
Accuracy tipo azione:    0.9111
Macro F1 tipo azione:    0.8744
Weighted F1 tipo azione: 0.9111
```

|Classe|Precision|Recall|F1-score|Support|
|-|-:|-:|-:|-:|
|passaggio|0.89|0.95|0.92|212|
|tiroDaDue|0.83|0.91|0.87|32|
|tiroDaTre|0.87|0.87|0.87|15|
|tiroLibero|0.82|0.78|0.80|18|
|no-action|0.95|0.89|0.92|263|

Confusion matrix - tipo azione senza esito:

```text
[[202   0   1   0   9]
 [  0  29   1   2   0]
 [  0   1  13   0   1]
 [  0   2   0  14   2]
 [ 25   3   0   1 234]]
```

Rispetto a `exp_44`, `exp_46` migliora sensibilmente le metriche end-to-end: Accuracy 8 classi da 0.8907 a 0.9056, Macro F1 8 classi da 0.7762 a 0.8079 e Weighted F1 8 classi da 0.8908 a 0.9062. Il miglioramento è dovuto soprattutto alla sostituzione di L1 con `temp43` e di L3 con `temp43`, mentre L2 resta sulla versione `temp29` perché più forte della variante `temp43`.

## Comandi principali

### Estrazione feature temporali complete con YOLO v1/v2/v3

Le tre estrazioni correnti usano lo stesso comando, cambiando solo pesi YOLO e cartella di output. Esempio per YOLO v3:

```bash
python -m src.features.extract_ball_rim_tracking_features \
  --dataset-root data/datasets/dataset_basket_v1 \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --yolo-weights runs/detect/outputs/ball_rim_detector/yolo11m_1280_v3/weights/best.pt \
  --output-dir data/features/ball_rim_tracking_temporal_clip_complete_yolo_v3 \
  --splits train val test \
  --num-frames 48 \
  --sample-mode uniform \
  --imgsz 1280 \
  --conf 0.10 \
  --iou 0.50 \
  --device 0 \
  --batch-size 16 \
  --save-temporal-sequences \
  --overwrite
```

Per YOLO v1 e YOLO v2 si usa lo stesso comando sostituendo:

```text
yolo11m_1280_v3 -> yolo11m_1280_v1 / yolo11m_1280_v2
ball_rim_tracking_temporal_clip_complete_yolo_v3 -> ball_rim_tracking_temporal_clip_complete_yolo_v1 / ball_rim_tracking_temporal_clip_complete_yolo_v2
```

### Training L3 storico YOLO v1 temporale (`exp_l3_tracking_temporal_v1`)

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --output-dir outputs/exp_l3_tracking_temporal_v1 \
  --label-mode shot_outcome_only \
  --tracking-sequences-npz data/features/ball_rim_tracking_temporal_v1/tracking_sequences.npz \
  --tracking-sequence-index data/features/ball_rim_tracking_temporal_v1/tracking_sequence_index.json \
  --epochs 100 \
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
  --sampler-power 0.5 \
  --num-workers 0 \
  --seed 42
```

### Training L2 YOLO v2

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --output-dir outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean \
  --label-mode shot_type_only \
  --tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequences.npz \
  --tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequence_index.json \
  --epochs 100 \
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
  --sampler-power 0.5 \
  --tracking-missing-policy error \
  --num-workers 0 \
  --seed 42
```

### Training L2 YOLO v3

```bash
python -m src.training.train \
  --features-root data/features/dinov3_vitl16_336 \
  --output-dir outputs/exp_l2_yolo_v3_temp29_allclips_d256_mean \
  --label-mode shot_type_only \
  --tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v3/tracking_sequences.npz \
  --tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v3/tracking_sequence_index.json \
  --epochs 100 \
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
  --sampler-power 0.5 \
  --tracking-missing-policy error \
  --num-workers 0 \
  --seed 42
```

### Valutazione exp_44

```bash
python -m src.evaluation.evaluate_hierarchical \
  --features-root data/features/dinov3_vitl16_336 \
  --split val \
  --batch-size 64 \
  --num-workers 2 \
  --output-dir outputs/exp_44_hier_dinov3_l1_yolo_v2_l2_yolo_v2_l3_yolo_v1_temp29_allclips \
  --l1-checkpoint outputs/exp_l1_yolo_v2_temp29_allclips_d256_mean/best_model.pt \
  --l2-checkpoint outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt \
  --l3-checkpoint outputs/exp_l3_yolo_v1_temp29_shots_d256_mean/best_model.pt \
  --l1-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequences.npz \
  --l1-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequence_index.json \
  --l2-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequences.npz \
  --l2-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequence_index.json \
  --l3-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1/tracking_sequences.npz \
  --l3-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1/tracking_sequence_index.json \
  --tracking-missing-policy error
```

### Valutazione exp_45

```bash
python -m src.evaluation.evaluate_hierarchical \
  --features-root data/features/dinov3_vitl16_336 \
  --split val \
  --batch-size 64 \
  --num-workers 2 \
  --output-dir outputs/exp_45_hier_l1_yolo_v2_passaggio_threshold_sweep \
  --l1-checkpoint outputs/exp_l1_yolo_v2_temp29_allclips_d256_mean/best_model.pt \
  --l2-checkpoint outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt \
  --l3-checkpoint outputs/exp_l3_yolo_v1_temp29_shots_d256_mean/best_model.pt \
  --l1-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequences.npz \
  --l1-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequence_index.json \
  --l2-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequences.npz \
  --l2-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v2/tracking_sequence_index.json \
  --l3-tracking-sequences-npz data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1/tracking_sequences.npz \
  --l3-tracking-sequence-index data/features/ball_rim_tracking_temporal_clip_complete_yolo_v1/tracking_sequence_index.json \
  --tracking-missing-policy error \
  --l1-passaggio-thresholds 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 \
  --l1-passaggio-threshold-policy noaction_gt_tiro \
  --threshold-select-metric macro_f1_8
```


### Valutazione exp_46

```bash
python -m src.evaluation.evaluate_hierarchical \
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

## Nota finale

Il tracking palla/canestro produce un miglioramento progressivo soprattutto quando viene integrato nei singoli stadi più adatti:

```text
exp_38: L1 vecchio, L2 vecchio, L3 YOLO v1 aggregate39             -> Macro F1 8 classi = 0.6996
exp_40: L1 vecchio, L2 vecchio, L3 YOLO v1 temp29                  -> Macro F1 8 classi = 0.7223
exp_42: L1 vecchio, L2 YOLO v2 temp29, L3 YOLO v3 temp29           -> Macro F1 8 classi = 0.7438
exp_43: L1 YOLO v2 temp29, L2 YOLO v2 temp29, L3 YOLO v3 temp29    -> Macro F1 8 classi = 0.7607
exp_44: L1 YOLO v2 temp29, L2 YOLO v2 temp29, L3 YOLO v1 temp29    -> Macro F1 8 classi = 0.7762
exp_45: exp_44 + soglia L1 passaggio                              -> Macro F1 8 classi = 0.7767
exp_46: L1 YOLO v2 temp43, L2 YOLO v2 temp29, L3 YOLO v1 temp43    -> Macro F1 8 classi = 0.8079
```

La configurazione consigliata diventa:

```text
L1: outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt
L2: outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt
L3: outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt
```

Il miglior output operativo da usare come riferimento è:

```text
outputs/exp_46_hier_best_per_level_l1temp43_l2temp29_l3temp43
```

`exp_46` è preferibile a `exp_45` perché ottiene un miglioramento molto più consistente senza introdurre soglie selezionate su validation. `exp_45` resta documentato come ablation di post-processing, ma non viene scelto come modello finale.

Il miglior esperimento effettuato fino ad ora è quindi strutturato così:

```text
L1 migliore: YOLO v2 temp43
L2 migliore: YOLO v2 temp29
L3 migliore: YOLO v1 temp43
End-to-end migliore: exp_46
```


