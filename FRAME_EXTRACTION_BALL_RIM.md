# Estrazione frame per annotazione palla/canestro

Per introdurre una branch di tracking palla-canestro nel terzo livello del modello gerarchico, cioè nella classificazione `tiro0` / `tiro1`, sono stati estratti alcuni frame dalle sole clip di tiro del training set.

La selezione dei frame è stata effettuata in modo automatico e riproducibile usando il file `manifest.csv` del dataset. Poiché le clip di tiro sono annotate dall'inizio del movimento fino al momento in cui l'esito è determinato, sono stati scelti frame concentrati nella parte finale della clip, dove è più probabile osservare l'interazione tra palla e canestro.

Per ogni clip di tiro selezionata sono stati estratti 4 frame nelle seguenti posizioni relative:

```text
70%, 85%, 95%, 100%
```

Sono state considerate solo le classi di tiro:

```text
tiroDaDue0
tiroDaDue1
tiroDaTre0
tiroDaTre1
tiroLibero0
tiroLibero1
```

Per ridurre il costo di annotazione, è stato impostato un massimo di 50 clip per ciascuna classe di tiro. Nel caso di classi con meno di 50 esempi disponibili, vengono usate tutte le clip presenti. Il lavoro è stato inoltre diviso in due parti bilanciate, generando due file `.zip` separati per l'annotazione in CVAT.

Le label da annotare in CVAT sono:

```text
ball
rim
```

Le annotazioni sono bounding box. La palla o il ferro vanno annotati solo se chiaramente visibili; in caso di occlusione, oggetto fuori inquadratura o frame ambiguo, la bounding box non viene inserita.

## Comando utilizzato per l'estrazione dei frame

```bash
python src/annotations/extract_ball_rim_frames.py \
  --dataset-root data/datasets/dataset_basket_v1 \
  --manifest data/datasets/dataset_basket_v1/manifest.csv \
  --output-dir data/annotations/ball_rim_frames_sample \
  --splits train \
  --percents 0.70 0.85 0.95 1.0 \
  --max-clips-per-class 50 \
  --num-parts 2 \
  --seed 42
```

## Output prodotto dall'estrazione

Lo script produce una cartella con le immagini estratte, un file di mapping per mantenere la tracciabilità tra immagini e clip originali, e due archivi `.zip` da caricare separatamente in CVAT:

```text
data/annotations/ball_rim_frames_sample/train/
├── images/
├── train_frame_mapping.csv
├── train_part_01_frame_mapping.csv
├── train_part_02_frame_mapping.csv
├── train_part_01_images_for_cvat.zip
└── train_part_02_images_for_cvat.zip
```

I due file `.zip` permettono di dividere il lavoro tra due annotatori, mantenendo tutti i frame della stessa clip nella stessa parte.

## Annotazione in CVAT

I due archivi sono stati caricati su CVAT come due task separati. Per entrambi i task sono state usate le stesse label, nello stesso ordine:

```text
0 = ball
1 = rim
```

Sono state completate circa 1200 annotazioni complessive sui frame del training set, divise tra i due annotatori. Le annotazioni sono state esportate da CVAT in formato YOLO/Ultralytics YOLO, includendo anche le immagini.

Gli zip esportati da CVAT vanno salvati nella seguente cartella:

```text
data/annotations/ball_rim_cvat_exports/train/
├── train_part_01_yolo.zip
└── train_part_02_yolo.zip
```

Esempio di comandi per creare la cartella e copiare gli zip:

```bash
mkdir -p data/annotations/ball_rim_cvat_exports/train

cp /path/dove/sono/gli/zip/train_part_01_yolo.zip \
  data/annotations/ball_rim_cvat_exports/train/

cp /path/dove/sono/gli/zip/train_part_02_yolo.zip \
  data/annotations/ball_rim_cvat_exports/train/
```

## Creazione del dataset YOLO

I due export CVAT vengono uniti e divisi automaticamente in training e validation interna usando lo script:

```text
src/annotations/prepare_ball_rim_yolo_dataset.py
```

Il dataset YOLO finale viene salvato in:

```text
data/datasets/ball_rim_yolo_v1/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Comando utilizzato:

```bash
python src/annotations/prepare_ball_rim_yolo_dataset.py \
  --zips \
    data/annotations/ball_rim_cvat_exports/train/train_part_01_yolo.zip \
    data/annotations/ball_rim_cvat_exports/train/train_part_02_yolo.zip \
  --out data/datasets/ball_rim_yolo_v1 \
  --val-ratio 0.10 \
  --seed 42 \
  --names ball rim
```

Se la cartella di output esiste già e si vuole rigenerare il dataset:

```bash
python src/annotations/prepare_ball_rim_yolo_dataset.py \
  --zips \
    data/annotations/ball_rim_cvat_exports/train/train_part_01_yolo.zip \
    data/annotations/ball_rim_cvat_exports/train/train_part_02_yolo.zip \
  --out data/datasets/ball_rim_yolo_v1 \
  --val-ratio 0.10 \
  --seed 42 \
  --names ball rim \
  --overwrite
```

La validation ottenuta con `--val-ratio 0.10` è solo una validation interna ricavata dai frame di training annotati. Serve per monitorare il primo addestramento del detector, non come valutazione finale definitiva.

## Training del detector YOLO

Il detector palla-canestro viene addestrato con lo script:

```text
src/training/train_ball_rim_yolo.py
```

Prima del training è necessario installare o aggiornare Ultralytics:

```bash
pip install -U ultralytics
```

Comando di training consigliato per il primo esperimento:

```bash
python src/training/train_ball_rim_yolo.py \
  --data data/datasets/ball_rim_yolo_v1/data.yaml \
  --model yolo11m.pt \
  --imgsz 1280 \
  --epochs 150 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --project outputs/ball_rim_detector \
  --name yolo11m_1280_v1
```

Il modello scelto è `yolo11m.pt`, con risoluzione `1280`, perché la palla è un oggetto piccolo e può essere difficile da rilevare a risoluzioni basse.

Se il training genera errore di memoria GPU, si può ridurre il batch size:

```bash
python src/training/train_ball_rim_yolo.py \
  --data data/datasets/ball_rim_yolo_v1/data.yaml \
  --model yolo11m.pt \
  --imgsz 1280 \
  --epochs 150 \
  --batch 4 \
  --device 0 \
  --workers 8 \
  --project outputs/ball_rim_detector \
  --name yolo11m_1280_v1_batch4
```

Se anche con `batch=4` il training è troppo pesante, si può ridurre la risoluzione:

```bash
python src/training/train_ball_rim_yolo.py \
  --data data/datasets/ball_rim_yolo_v1/data.yaml \
  --model yolo11m.pt \
  --imgsz 960 \
  --epochs 150 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --project outputs/ball_rim_detector \
  --name yolo11m_960_v1
```

Il miglior modello addestrato viene salvato in:

```text
outputs/ball_rim_detector/yolo11m_1280_v1/weights/best.pt
```

## Uso successivo del modello

Il primo modello addestrato sui frame di training verrà usato per pre-annotare automaticamente i frame delle clip di validation. Le predizioni saranno poi corrette manualmente in CVAT, in modo da velocizzare la creazione di un set di validation annotato e ottenere una prima valutazione qualitativa degli errori del detector.

Dopo la correzione manuale delle annotazioni di validation, sarà possibile creare una nuova versione del dataset:

```text
data/datasets/ball_rim_yolo_v2/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

In questa seconda versione, il training set conterrà i frame annotati manualmente del training, mentre la validation conterrà i frame di validation pre-annotati dal modello e corretti manualmente.
