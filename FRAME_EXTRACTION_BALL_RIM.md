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


## Preparazione del dataset YOLO dopo l'annotazione CVAT

Dopo la revisione manuale delle annotazioni in CVAT, i quattro export YOLO sono stati unificati in un unico dataset per l'addestramento del detector palla/canestro tramite lo script:

```bash
python src/annotations/prepare_ball_rim_yolo_dataset.py
```

Gli export CVAT utilizzati sono stati:

| Export CVAT | Split finale | Immagini | Note |
|---|---|---:|---|
| `train_part_01_cvat_yolo.zip` | train | 592 | Tiri train, parte 1 |
| `train_part_02_cvat_yolo.zip` | train | 591 | Tiri train, parte 2 |
| `train_context_cvat_yolo.zip` | train | 480 | Passaggio, idle e non-gioco train |
| `val_cvat_yolo.zip` | val | 320 | Validation con tutte le classi |

Durante la preparazione del dataset è stato controllato che l'ordine delle classi fosse coerente in tutti gli export:

```text
0 = ball
1 = rim
```

Il dataset finale è stato salvato in:

```text
data/datasets/ball_rim_yolo
```

con la seguente struttura:

```text
data/datasets/ball_rim_yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
├── dataset_summary.txt
└── dataset_summary.csv
```

### Conteggi finali del dataset YOLO

| Split | Immagini totali | Immagini con box | Immagini senza box |
|---|---:|---:|---:|
| train | 1663 | 1655 | 8 |
| val | 320 | 305 | 15 |

Le immagini senza box sono state mantenute nel dataset, perché rappresentano esempi negativi utili per ridurre falsi positivi, soprattutto nei frame di `idle` e `non-gioco` in cui palla e/o canestro non sono visibili.


## Detector YOLO11m v1 precedente

Prima dell'ampliamento del dataset palla/canestro è stato addestrato un primo detector YOLO11m, indicato come versione `v1`, usando esclusivamente frame estratti da clip di tiro del training set. Questo modello è stato utilizzato come primo generatore di feature tracking per gli esperimenti iniziali sullo Stadio 3 della gerarchia.

Il dataset del modello precedente era composto da **1183 frame totali**, ottenuti dalle sei classi di tiro:

```text
tiroDaDue0, tiroDaDue1, tiroDaTre0, tiroDaTre1, tiroLibero0, tiroLibero1
```

I frame erano stati estratti solo dalla parte finale delle clip di tiro, usando le percentuali:

```text
0.70, 0.85, 0.95, 1.00
```

Questa scelta era coerente con l'obiettivo iniziale: migliorare la distinzione tra tiro segnato e tiro sbagliato, concentrandosi sui momenti in cui la palla si avvicina al canestro e l'esito diventa osservabile. Tuttavia, il dataset non conteneva esempi provenienti da `passaggio`, `idle` e `non-gioco`, quindi il detector non era stato addestrato in modo esplicito a gestire contesti senza tiro.

### Split usato nel modello precedente

Nel modello precedente il **10% dei 1183 frame** è stato usato come validation interna YOLO, mentre il restante 90% è stato usato per il training. È importante sottolineare che questi frame provenivano comunque tutti da clip dello split `train` del dataset di action recognition.

Di conseguenza, la validation del detector `v1` non era una validation completamente indipendente rispetto al dataset video: serviva soprattutto a monitorare l'addestramento YOLO, ma poteva produrre metriche più ottimistiche perché i frame di training e validation provenivano dallo stesso insieme di clip di tiro. Per questo motivo il confronto diretto con il detector `v2` deve essere interpretato con cautela.

### Risultati del modello precedente

Il file `results.csv` del training YOLO del modello precedente riporta metriche aggregate sulle due classi `ball` e `rim`. Il training è arrivato a **150 epoche**. La metrica più severa, `mAP50-95`, ha raggiunto il valore massimo all'epoca 133.

| Riferimento | Epoca | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Miglior mAP50-95 | 133 | 0.9506 | 0.9388 | 0.9714 | 0.5268 |
| Miglior mAP50 | 79 | 0.9653 | 0.9562 | 0.9881 | 0.5049 |
| Ultima epoca | 150 | 0.9548 | 0.9414 | 0.9754 | 0.5240 |

Le metriche indicano che il detector precedente era già buono sui frame di tiro: la precision e la recall aggregate restano alte e il valore di mAP50 è elevato. Il limite principale non è quindi la qualità apparente sulle immagini di validation interna, ma la composizione del dataset usato per addestrarlo e validarlo: essendo basato solo sui tiri e con validation derivata dalle clip di train, il modello `v1` era meno adatto a essere applicato ai livelli della gerarchia che lavorano anche su `passaggio`, `idle` e `non-gioco`.

Per questo motivo è stato successivamente costruito il dataset esteso con 1663 immagini di training e 320 immagini di validation provenienti anche dallo split `val`, includendo frame di contesto. Il detector `v2` è quindi più indicato per gli esperimenti futuri su Stadio 1 e Stadio 2, mentre il modello `v1` resta una baseline storica utile per il confronto.

## Addestramento del detector YOLO11m v2

Il nuovo detector è stato addestrato sul dataset YOLO unificato, partendo dal modello pre-addestrato standard `yolo11m.pt` e non dal detector precedente. Questa scelta rende l'esperimento più pulito: il modello apprende dal nuovo dataset completo, che include sia i frame di tiro già utilizzati in precedenza sia i nuovi contesti di passaggio, idle e non-gioco.

Il training è stato avviato con:

```bash
python src/training/train_ball_rim_yolo.py
```

### Configurazione di training

| Parametro | Valore |
|---|---|
| Modello iniziale | `yolo11m.pt` |
| Dataset | `data/datasets/ball_rim_yolo/data.yaml` |
| Risoluzione | `1280` |
| Epoche massime | `150` |
| Batch size | `8` |
| Device | `0` |
| Workers | `8` |
| Seed | `42` |
| Patience early stopping | `30` |
| Cache | `False` |
| Project | `runs/detect/outputs/ball_rim_detector` |
| Nome esperimento | `yolo11m_1280_v2` |

Le principali augmentation mantenute sono state:

| Parametro | Valore |
|---|---:|
| `hsv_h` | 0.015 |
| `hsv_s` | 0.5 |
| `hsv_v` | 0.3 |
| `translate` | 0.05 |
| `scale` | 0.30 |
| `fliplr` | 0.5 |
| `mosaic` | 0.5 |
| `close_mosaic` | 15 |

Il training è stato eseguito su GPU NVIDIA RTX 5000 Ada Generation. YOLO ha caricato il modello `yolo11m.pt`, adattando il numero di classi da 80 a 2 e trasferendo 643/649 pesi dal modello pre-addestrato.

### Risultati del training

L'addestramento si è fermato automaticamente dopo 67 epoche per early stopping. Il miglior risultato è stato osservato all'epoca 37, e il modello migliore è stato salvato come `best.pt`.

```text
EarlyStopping: Training stopped early as no improvement observed in last 30 epochs.
Best results observed at epoch 37, best model saved as best.pt.
```

La validazione finale del `best.pt` ha prodotto i seguenti risultati:

| Classe | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 320 | 567 | 0.974 | 0.942 | 0.965 | 0.564 |
| ball | 271 | 271 | 0.961 | 0.908 | 0.943 | 0.561 |
| rim | 296 | 296 | 0.987 | 0.976 | 0.987 | 0.568 |

Il tempo medio riportato in validazione è stato:

| Fase | Tempo per immagine |
|---|---:|
| Preprocess | 0.2 ms |
| Inference | 5.9 ms |
| Postprocess | 0.5 ms |

### Considerazioni sull'early stopping

L'early stopping non sembra aver penalizzato in modo rilevante le prestazioni del detector. Il modello migliore è stato individuato all'epoca 37, con mAP50-95 pari a circa 0.567 durante il training. Le epoche successive hanno mostrato ancora valori molto buoni, in alcuni casi con mAP50 leggermente superiore, ma senza migliorare stabilmente la metrica più severa mAP50-95. La validazione finale del `best.pt` ha infatti confermato un valore mAP50-95 pari a 0.564.

Questo indica che proseguire fino a 150 epoche avrebbe probabilmente aumentato il tempo di training senza un miglioramento chiaro della qualità delle box. La scelta di `patience=30` appare quindi ragionevole per questo esperimento.

### Confronto con il detector precedente

Il confronto con il detector precedente deve essere interpretato con cautela, perché il modello precedente era stato addestrato su 1183 frame di tiro e validato usando il 10% di quegli stessi frame, tutti provenienti da clip dello split `train` del dataset di action recognition. Di conseguenza le sue metriche di validation erano più ottimistiche e non direttamente confrontabili con quelle del nuovo modello.

Il nuovo detector, invece, è stato validato su 320 frame provenienti dallo split `val`, con clip diverse da quelle di training e con tutte le 9 classi rappresentate dove disponibili. Le metriche ottenute sono quindi più affidabili come stima della generalizzazione, anche se il validation set resta relativamente piccolo e alcune classi rare hanno pochi esempi.

Dal punto di vista pratico, il risultato è positivo: il detector mantiene metriche alte sia sulla palla sia sul canestro, con una recall della palla pari a 0.908 e una recall del rim pari a 0.976. La palla rimane l'oggetto più difficile, come atteso, perché è piccola, spesso in movimento, parzialmente occlusa o lontana dal canestro.

### Nota sul path di salvataggio

Nel log di training il modello viene salvato da Ultralytics nel path effettivo:

```text
runs/detect/runs/detect/outputs/ball_rim_detector/yolo11m_1280_v2/weights/best.pt
```

mentre lo script stampa e controlla il path:

```text
runs/detect/outputs/ball_rim_detector/yolo11m_1280_v2/weights/best.pt
```

Per questo motivo, al termine del training compare il warning:

```text
[WARN] best.pt non trovato. Controllare la cartella dell'esperimento.
```

Il training è comunque completato correttamente e il `best.pt` è stato salvato; il warning dipende solo dalla discrepanza tra il path atteso dallo script e il path effettivo generato da Ultralytics.

## Utilizzo previsto

Il detector YOLO11m v2 addestrato su questo dataset verrà utilizzato per estrarre feature di tracking palla/canestro sulle clip del dataset. Le feature potranno essere impiegate nei modelli di action recognition, sia per i livelli che distinguono le azioni generali sia per il livello relativo all'esito del tiro.

Il modello precedente resta utile come baseline storica, ma il nuovo detector è più adatto agli esperimenti successivi perché è stato addestrato anche su passaggio, idle e non-gioco ed è stato validato su frame provenienti dallo split `val`, quindi su clip diverse da quelle di training.
