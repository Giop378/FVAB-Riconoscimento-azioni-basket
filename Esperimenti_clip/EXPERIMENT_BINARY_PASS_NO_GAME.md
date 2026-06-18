# Esperimento 47 - Correttore binario L1 passaggio/no-action

## Obiettivo dell'esperimento

L'esperimento introduce un quarto modello nella pipeline gerarchica con lo scopo di ridurre la confusione tra `passaggio` e `no-action`, emersa come uno dei principali limiti della configurazione `exp_46`.

La gerarchia di riferimento era:

```text
L1: passaggio / tiro / no-action
L2: tiroDaDue / tiroDaTre / tiroLibero
L3: tiro0 / tiro1
```

Il correttore binario viene applicato solo ai campioni che il livello L1 predice come `passaggio` o `no-action`. I campioni predetti come `tiro` non vengono modificati e continuano normalmente verso L2 e L3.

La logica diventa quindi:

```text
se L1 predice tiro:
    usa L2 e L3 normalmente

se L1 predice passaggio oppure no-action:
    usa il correttore binario passaggio/no-action
    sostituisci la predizione finale di L1
```

---

## Training del correttore binario

### Configurazione

Il correttore binario è stato addestrato con il seguente output directory:

```text
outputs/exp_l1_binary_passaggio_noaction_yolo_v2_temp43_d256_mean
```

La modalità di etichettatura utilizzata è:

```text
--label-mode passaggio_noaction_only
```

Il mapping delle classi è:

```text
passaggio -> passaggio
idle -> no-action
non-gioco -> no-action
classi di tiro -> escluse
```

Sono state usate le feature DINOv3-L/16 combinate con le sequenze temporali di tracking palla/canestro `YOLO v2 temp43`:

```text
Feature video: 1024
Feature tracking temporali per frame: 43
Input totale modello: 1067
```

### Dataset usato

| Split | Campioni originali | Campioni usati |
|---|---:|---:|
| Train | 4236 | 3603 |
| Validation | 540 | 475 |

Distribuzione delle classi nel training:

| Classe | Campioni train |
|---|---:|
| passaggio | 1787 |
| no-action | 1816 |

Le classi risultano quindi quasi perfettamente bilanciate.

### Miglior checkpoint

Il miglior modello è stato selezionato alla **epoca 68** sulla base della Macro F1 di validation.

| Metrica | Valore |
|---|---:|
| Best val loss | 0.5495 |
| Best val accuracy | 0.9179 |
| Best val macro-F1 | 0.9176 |
| Best val weighted-F1 | 0.9181 |

### Classification report del correttore binario

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.87 | 0.96 | 0.91 | 212 |
| no-action | 0.96 | 0.89 | 0.92 | 263 |
| accuracy |  |  | 0.92 | 475 |
| macro avg | 0.92 | 0.92 | 0.92 | 475 |
| weighted avg | 0.92 | 0.92 | 0.92 | 475 |

### Matrice di confusione del correttore binario

Ordine classi:

```text
[passaggio, no-action]
```

```text
[[203   9]
 [ 30 233]]
```

Interpretazione:

- `203` passaggi vengono classificati correttamente come `passaggio`.
- `9` passaggi vengono scambiati per `no-action`.
- `233` no-action vengono classificati correttamente come `no-action`.
- `30` no-action vengono scambiati per `passaggio`.

Il problema principale rimane quindi la direzione `no-action -> passaggio`, che è proprio l'errore che si voleva ridurre.

---

## Valutazione end-to-end con correttore binario - exp_47

### Configurazione end-to-end

Output directory:

```text
outputs/exp_47_hier_exp46_with_l1_binary_corrector
```

Modelli utilizzati:

| Livello | Checkpoint | Tracking |
|---|---|---|
| L1 | `outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/best_model.pt` | YOLO v2 temp43 |
| L2 | `outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/best_model.pt` | YOLO v2 temp29 |
| L3 | `outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/best_model.pt` | YOLO v1 temp43 |
| Correttore L1 | `outputs/exp_l1_binary_passaggio_noaction_yolo_v2_temp43_d256_mean/best_model.pt` | YOLO v2 temp43 |

### Uso effettivo del correttore

| Quantità | Valore |
|---|---:|
| Campioni passati al correttore | 473 |
| Predizioni modificate dal correttore | 11 |
| Conversioni `passaggio -> no-action` | 4 |
| Conversioni `no-action -> passaggio` | 7 |

Il correttore viene applicato a quasi tutti i campioni non predetti come tiro da L1. Tuttavia modifica solo 11 predizioni. Il dato più importante è che produce più conversioni `no-action -> passaggio` che conversioni `passaggio -> no-action`, peggiorando la direzione di errore più problematica.

---

## Metriche finali exp_47

### Valutazione a 8 classi finali

| Metrica | Valore |
|---|---:|
| Accuracy 8 classi | 0.9000 |
| Macro F1 8 classi | 0.8063 |
| Weighted F1 8 classi | 0.9006 |

Classification report:

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.88 | 0.95 | 0.91 | 212 |
| tiroDaDue0 | 0.90 | 0.86 | 0.88 | 21 |
| tiroDaDue1 | 0.60 | 0.82 | 0.69 | 11 |
| tiroDaTre0 | 0.91 | 0.83 | 0.87 | 12 |
| tiroDaTre1 | 0.50 | 0.67 | 0.57 | 3 |
| tiroLibero0 | 0.78 | 1.00 | 0.88 | 7 |
| tiroLibero1 | 0.88 | 0.64 | 0.74 | 11 |
| no-action | 0.95 | 0.88 | 0.91 | 263 |
| accuracy |  |  | 0.90 | 540 |
| macro avg | 0.80 | 0.83 | 0.81 | 540 |
| weighted avg | 0.91 | 0.90 | 0.90 | 540 |

### Matrice di confusione a 8 classi

Ordine classi:

```text
[passaggio, tiroDaDue0, tiroDaDue1, tiroDaTre0, tiroDaTre1, tiroLibero0, tiroLibero1, no-action]
```

```text
[[202   0   0   1   0   0   0   9]
 [  0  18   2   0   0   1   0   0]
 [  0   0   9   0   1   0   1   0]
 [  0   0   0  10   1   0   0   1]
 [  0   0   1   0   2   0   0   0]
 [  0   0   0   0   0   7   0   0]
 [  0   0   2   0   0   0   7   2]
 [ 28   2   1   0   0   1   0 231]]
```

L'errore principale rimane `no-action -> passaggio`, che compare 28 volte.

---

## Valutazione solo sulle 7 azioni finali

| Metrica | Valore |
|---|---:|
| Micro F1 7 azioni | 0.89 |
| Macro F1 7 azioni | 0.79 |
| Weighted F1 7 azioni | 0.89 |

Classification report:

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.88 | 0.95 | 0.91 | 212 |
| tiroDaDue0 | 0.90 | 0.86 | 0.88 | 21 |
| tiroDaDue1 | 0.60 | 0.82 | 0.69 | 11 |
| tiroDaTre0 | 0.91 | 0.83 | 0.87 | 12 |
| tiroDaTre1 | 0.50 | 0.67 | 0.57 | 3 |
| tiroLibero0 | 0.78 | 1.00 | 0.88 | 7 |
| tiroLibero1 | 0.88 | 0.64 | 0.74 | 11 |

### Matrice di confusione sulle 7 azioni

Ordine classi:

```text
[passaggio, tiroDaDue0, tiroDaDue1, tiroDaTre0, tiroDaTre1, tiroLibero0, tiroLibero1]
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

## Valutazione collassata senza esito del tiro

In questa valutazione le classi di tiro vengono collassate nel solo tipo di tiro, ignorando l'esito realizzato/sbagliato.

| Metrica | Valore |
|---|---:|
| Accuracy tipo azione | 0.9056 |
| Macro F1 tipo azione | 0.8719 |
| Weighted F1 tipo azione | 0.9056 |

Classification report:

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| passaggio | 0.88 | 0.95 | 0.91 | 212 |
| tiroDaDue | 0.83 | 0.91 | 0.87 | 32 |
| tiroDaTre | 0.87 | 0.87 | 0.87 | 15 |
| tiroLibero | 0.82 | 0.78 | 0.80 | 18 |
| no-action | 0.95 | 0.88 | 0.91 | 263 |
| accuracy |  |  | 0.91 | 540 |
| macro avg | 0.87 | 0.88 | 0.87 | 540 |
| weighted avg | 0.91 | 0.91 | 0.91 | 540 |

### Matrice di confusione collassata senza esito

Ordine classi:

```text
[passaggio, tiroDaDue, tiroDaTre, tiroLibero, no-action]
```

```text
[[202   0   1   0   9]
 [  0  29   1   2   0]
 [  0   1  13   0   1]
 [  0   2   0  14   2]
 [ 28   3   0   1 231]]
```

---

## Confronto con exp_46

La configurazione precedente `exp_46` era:

```text
L1: YOLO v2 temp43
L2: YOLO v2 temp29
L3: YOLO v1 temp43
```

L'esperimento `exp_47` aggiunge a questa configurazione il correttore binario passaggio/no-action.

| Metrica | exp_46 | exp_47 | Differenza |
|---|---:|---:|---:|
| Accuracy 8 classi | 0.9056 | 0.9000 | -0.0056 |
| Macro F1 8 classi | 0.8079 | 0.8063 | -0.0016 |
| Weighted F1 8 classi | 0.9062 | 0.9006 | -0.0056 |
| Macro F1 tipo azione | 0.8744 | 0.8719 | -0.0025 |
| Accuracy tipo azione | 0.9111 | 0.9056 | -0.0055 |

Confronto specifico sugli errori `passaggio`/`no-action`:

| Errore | exp_46 | exp_47 |
|---|---:|---:|
| `passaggio -> no-action` | 9 | 9 |
| `no-action -> passaggio` | 25 | 28 |
| `no-action` corretti | 234 | 231 |
| `passaggio` corretti | 202 | 202 |

Il correttore non riduce gli errori `passaggio -> no-action` e peggiora gli errori `no-action -> passaggio`, che aumentano da 25 a 28.

---

## Conclusione

L'esperimento `exp_47` **non rappresenta un miglioramento rispetto a `exp_46`**.

Il correttore binario standalone ottiene metriche apparentemente buone, con Macro F1 pari a 0.9176 sulla validation binaria, ma quando viene inserito nella pipeline end-to-end peggiora leggermente le metriche globali:

- Accuracy 8 classi: `0.9056 -> 0.9000`
- Macro F1 8 classi: `0.8079 -> 0.8063`
- Weighted F1 8 classi: `0.9062 -> 0.9006`
- Macro F1 tipo azione: `0.8744 -> 0.8719`

Inoltre, il problema principale `no-action -> passaggio` peggiora da 25 a 28 casi.

Per questi motivi, `exp_47` va considerato una **ablation negativa**. La configurazione consigliata rimane `exp_46`, senza correttore binario.

Il risultato suggerisce che aggiungere un secondo classificatore sulle stesse feature non è sufficiente per risolvere la confusione tra `passaggio` e `no-action`. Per migliorare realmente questa parte del sistema, la strada più promettente è introdurre nuove informazioni, ad esempio feature relative ai giocatori, al movimento dei giocatori o alla relazione palla-giocatore.
