# Estrazione dei frame per annotazioni ball/rim

## Obiettivo

Questo documento descrive la procedura utilizzata per estrarre i frame da annotare in CVAT per l'addestramento e la validazione del detector **palla/canestro**. L'obiettivo è ampliare il dataset del detector già costruito sui tiri, aggiungendo anche frame provenienti da clip di `passaggio`, `idle` e `non-gioco`, in modo da rendere il tracking della palla più robusto anche in contesti diversi dalla conclusione a canestro.

L'estrazione è stata eseguita tramite lo script:

```bash
python src/annotations/extract_ball_rim_frames.py
```

I parametri sono stati fissati direttamente nel codice, quindi non sono stati passati da riga di comando.

## Parametri utilizzati

| Parametro | Valore |
|---|---|
| Dataset root | `data/datasets/dataset_basket_v1` |
| Manifest | `data/datasets/dataset_basket_v1/manifest.csv` |
| Output directory | `data/annotations/ball_rim_frames_sample` |
| Seed | `42` |
| Percentuali per i tiri | `0.70, 0.85, 0.95, 1.00` |
| Percentuali per passaggio/idle/non-gioco | `0.05, 0.35, 0.65, 0.95` |
| Max clip train per classe di tiro | `50` |
| Max clip train per passaggio | `50` |
| Max clip train per idle | `35` |
| Max clip train per non-gioco | `35` |
| Max clip validation per classe | `10` |

Il seed `42` rende riproducibile il campionamento delle clip dal manifest.

## Criterio generale di estrazione dei frame

Per ogni clip selezionata sono stati estratti fino a quattro frame. L'indice del frame è calcolato come posizione relativa all'interno della clip:

```text
frame_idx = round((n_frames - 1) * percentuale)
```

Nel caso di clip molto brevi, due percentuali diverse possono produrre lo stesso indice di frame. In questo caso lo script evita di salvare duplicati per la stessa clip. Per questo motivo il numero finale di immagini può essere leggermente inferiore a `numero_clip × 4`.

## Estrazione dei frame di training

### Tiri

Per le sei classi di tiro sono stati mantenuti gli stessi criteri usati nella precedente estrazione dei frame. Le classi considerate sono:

- `tiroDaDue0`
- `tiroDaDue1`
- `tiroDaTre0`
- `tiroDaTre1`
- `tiroLibero0`
- `tiroLibero1`

Per ogni classe è stato richiesto un massimo di 50 clip dal training set. Per `tiroDaTre1` erano disponibili solo 46 clip nel training set, quindi sono state utilizzate tutte quelle disponibili.

| Classe | Clip train selezionate |
|---|---:|
| `tiroDaDue0` | 50 |
| `tiroDaDue1` | 50 |
| `tiroDaTre0` | 50 |
| `tiroDaTre1` | 46 |
| `tiroLibero0` | 50 |
| `tiroLibero1` | 50 |
| **Totale** | **296** |

I frame dei tiri sono stati estratti alle percentuali:

```text
0.70, 0.85, 0.95, 1.00
```

Questa scelta è stata mantenuta perché, nelle clip di tiro, la parte finale è la più informativa per il detector palla/canestro: la palla tende ad avvicinarsi al ferro e diventa più rilevante per distinguere la traiettoria e l'interazione con il canestro.

Le clip di tiro del training sono state divise in due parti bilanciate per classe, così da ottenere due zip separati per CVAT.

| Zip | Numero immagini | Descrizione |
|---|---:|---|
| `train_part_01_images_for_cvat.zip` | 592 | Prima parte dei frame di tiro train |
| `train_part_02_images_for_cvat.zip` | 591 | Seconda parte dei frame di tiro train |
| **Totale tiri train** | **1183** | Frame di tiro per il training |

Il totale atteso sarebbe stato 296 clip × 4 = 1184 immagini, ma una clip ha prodotto un frame duplicato a causa della durata ridotta; lo script ha quindi salvato 1183 immagini effettive.

### Passaggio, idle e non-gioco

Per ampliare il detector a contesti diversi dal tiro sono state estratte clip anche dalle classi:

- `passaggio`
- `idle`
- `non-gioco`

La distribuzione scelta per il training è la seguente:

| Classe | Clip train selezionate | Frame estratti |
|---|---:|---:|
| `passaggio` | 50 | 200 |
| `idle` | 35 | 140 |
| `non-gioco` | 35 | 140 |
| **Totale** | **120** | **480** |

Per queste classi non è stato usato il criterio temporale dei tiri. I frame sono stati estratti in modo più uniforme lungo la clip, usando le percentuali:

```text
0.05, 0.35, 0.65, 0.95
```

Questa scelta permette di osservare la palla in momenti diversi dell'azione o della fase di gioco, evitando di concentrarsi solo sulla parte finale della clip. Inoltre, usare `0.05` e `0.95` invece di `0.00` e `1.00` riduce il rischio di prendere frame troppo vicini ai tagli temporali della clip.

I frame di `passaggio`, `idle` e `non-gioco` sono stati salvati in un unico zip:

| Zip | Numero immagini | Descrizione |
|---|---:|---|
| `train_context_images_for_cvat.zip` | 480 | Frame train di passaggio, idle e non-gioco |

### Totale training

| Gruppo | Immagini |
|---|---:|
| Tiri train | 1183 |
| Passaggio/idle/non-gioco train | 480 |
| **Totale training** | **1663** |

## Estrazione dei frame di validation

Per la validation sono state considerate tutte le 9 classi del dataset:

- `passaggio`
- `idle`
- `non-gioco`
- `tiroDaDue0`
- `tiroDaDue1`
- `tiroDaTre0`
- `tiroDaTre1`
- `tiroLibero0`
- `tiroLibero1`

È stato richiesto un massimo di 10 clip per classe. Alcune classi rare non avevano 10 clip disponibili nello split di validation; in questi casi sono state usate tutte le clip disponibili.

| Classe | Clip validation selezionate | Frame estratti |
|---|---:|---:|
| `idle` | 10 | 40 |
| `non-gioco` | 10 | 40 |
| `passaggio` | 10 | 40 |
| `tiroDaDue0` | 10 | 40 |
| `tiroDaDue1` | 10 | 40 |
| `tiroDaTre0` | 10 | 40 |
| `tiroDaTre1` | 3 | 12 |
| `tiroLibero0` | 7 | 28 |
| `tiroLibero1` | 10 | 40 |
| **Totale** | **80** | **320** |

Per i tiri in validation è stato mantenuto lo stesso criterio usato per i tiri in training:

```text
0.70, 0.85, 0.95, 1.00
```

Per `passaggio`, `idle` e `non-gioco` è stato usato il campionamento più uniforme:

```text
0.05, 0.35, 0.65, 0.95
```

Tutti i frame di validation sono stati salvati in un unico zip:

| Zip | Numero immagini | Descrizione |
|---|---:|---|
| `val_images_for_cvat.zip` | 320 | Frame di validation da tutte le classi |

## Output generati

Lo script ha generato i seguenti output principali:

```text
data/annotations/ball_rim_frames_sample/train/train_part_01_images_for_cvat.zip
data/annotations/ball_rim_frames_sample/train/train_part_02_images_for_cvat.zip
data/annotations/ball_rim_frames_sample/train/train_context_images_for_cvat.zip
data/annotations/ball_rim_frames_sample/val/val_images_for_cvat.zip
```

Sono stati inoltre generati i file di mapping CSV, utili per risalire da ogni immagine alla clip originale, alla label, allo split, al frame estratto e alla percentuale temporale usata:

```text
data/annotations/ball_rim_frames_sample/train/train_frame_mapping.csv
data/annotations/ball_rim_frames_sample/train/train_part_01_frame_mapping.csv
data/annotations/ball_rim_frames_sample/train/train_part_02_frame_mapping.csv
data/annotations/ball_rim_frames_sample/train/train_context_frame_mapping.csv
data/annotations/ball_rim_frames_sample/val/val_frame_mapping.csv
data/annotations/ball_rim_frames_sample/all_frame_mapping.csv
```

## Protocollo di annotazione in CVAT

L'annotazione viene effettuata in CVAT sui frame estratti. Le annotazioni finali devono rappresentare la ground truth per l'addestramento e la validazione del detector YOLO palla/canestro.

### Oggetti da annotare

Le classi da annotare sono:

- **palla**: il pallone da basket visibile nel frame;
- **canestro/rim**: il ferro del canestro visibile nel frame.

Quando in un frame sono presenti più palloni o più canestri chiaramente visibili, vanno annotati tutti gli oggetti visibili appartenenti alle classi previste. Questo evita che oggetti reali non annotati vengano trattati come background durante il training del detector.

### Regole per la palla

La palla va annotata quando è visibile e localizzabile in modo affidabile. La bounding box deve racchiudere il pallone in modo stretto, includendo solo l'oggetto visibile e non la scia del movimento o parti dei giocatori.

Casi principali:

- palla completamente visibile: annotare con box stretta;
- palla parzialmente occlusa ma chiaramente riconoscibile: annotare la parte visibile con una box coerente;
- palla molto sfocata ma ancora distinguibile: annotare solo se la posizione è chiara;
- palla non visibile o non localizzabile con sicurezza: non annotare;
- oggetto ambiguo che potrebbe non essere la palla: non annotare.

### Regole per il canestro/rim

Il canestro va annotato quando il ferro è visibile e localizzabile. La bounding box deve racchiudere il rim/ferro, non l'intero tabellone o l'intera struttura del canestro.

Casi principali:

- ferro completamente visibile: annotare con box stretta;
- ferro parzialmente visibile ma riconoscibile: annotare la parte visibile;
- solo tabellone visibile senza ferro chiaramente localizzabile: non annotare il rim;
- canestro fuori inquadratura: non annotare.

### Frame senza oggetti

Nei frame di `idle`, `non-gioco` o anche `passaggio` può capitare che la palla e/o il canestro non siano visibili. In questi casi il frame deve rimanere senza bounding box per gli oggetti assenti. Questi frame sono comunque utili perché funzionano come esempi negativi e aiutano a ridurre i falsi positivi del detector.

### Uso di pre-annotazioni

Se vengono usate predizioni del detector precedente come pre-annotazioni, queste devono essere sempre controllate manualmente. In particolare:

- correggere box spostate o non aderenti all'oggetto;
- aggiungere box mancanti quando palla o rim sono visibili;
- eliminare falsi positivi;
- correggere eventuali label sbagliate;
- lasciare senza annotazioni i frame in cui gli oggetti non sono presenti.

La pre-annotazione serve solo a velocizzare il lavoro, ma il dataset esportato da CVAT deve essere considerato valido solo dopo revisione manuale.

## Utilizzo previsto

Dopo l'annotazione e l'esportazione da CVAT in formato YOLO, i frame di training saranno usati per addestrare il nuovo detector YOLO11m palla/canestro. I frame di validation saranno usati come validation set durante l'addestramento, così da monitorare il comportamento del detector sia sui tiri sia sui contesti di passaggio, idle e non-gioco.

Il nuovo detector potrà poi essere confrontato con il detector precedente, in particolare per verificare che l'aggiunta dei nuovi contesti migliori la robustezza generale senza peggiorare la qualità del tracking sui tiri, che rimane fondamentale per il livello 3 della pipeline gerarchica.
