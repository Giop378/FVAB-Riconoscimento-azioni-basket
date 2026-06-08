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

## Comando utilizzato

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

## Output prodotto

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
