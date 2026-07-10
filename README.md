# BasketAR

**Riconoscimento gerarchico di azioni cestistiche su clip annotate e video continui**

Progetto per il corso di **Fondamenti di Visione Artificiale e Biometria**  
Università degli Studi di Salerno — A.A. 2025/2026

**Autori:** Giovanni Paolo Chierchia, Sebastiano Caliendo

---

## Panoramica

BasketAR riconosce azioni di pallacanestro da video RGB ripresi con una singola telecamera.

Il progetto comprende due pipeline:

- **clip-level**, per classificare clip già ritagliate e annotate;
- **long-video**, per localizzare e classificare eventi in segmenti video continui.

La configurazione finale combina:

- **DINOv3 ViT-L/16** come estrattore di feature visuali;
- **Temporal Transformer Encoder** come classificatore temporale;
- classificazione gerarchica su tre livelli;
- tracking di palla e canestro tramite **YOLO**;
- sliding window multi-scala e post-processing temporale per i video lunghi.

---

## Architettura finale

La classificazione è suddivisa in tre livelli:

| Livello | Compito | Output | Tracking |
|---|---|---|---|
| L1 | Azione principale | `passaggio`, `tiro`, `no-action` | YOLO v2 + `temp43` |
| L2 | Tipo di tiro | `tiroDaDue`, `tiroDaTre`, `tiroLibero` | YOLO v2 + `temp29` |
| L3 | Esito del tiro | `tiro0`, `tiro1` | YOLO v1 + `temp43` |

Le classi `idle` e `non-gioco` vengono accorpate in `no-action`. Le sette classi di azione finali sono:

```text
passaggio
tiroDaDue0
tiroDaDue1
tiroDaTre0
tiroDaTre1
tiroLibero0
tiroLibero1
```

---

## Risultati principali

### Clip-level — `exp_46`

| Split | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Validation | 0.9056 | 0.8079 | 0.9062 |
| Test | 0.8549 | 0.6749 | 0.8564 |

Valutando soltanto le sette azioni reali:

| Split | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Validation | 0.9206 | 0.7919 | 0.8936 |
| Test | 0.8619 | 0.6494 | 0.8577 |

Senza distinguere l'esito del tiro:

| Split | Macro F1 |
|---|---:|
| Validation | 0.8744 |
| Test | 0.7898 |

### Long-video — `exp_long_13`

Valutazione event-level con temporal IoU pari a `0.20`:

| Split | Precision | Recall | F1 globale |
|---|---:|---:|---:|
| Validation | 0.5250 | 0.5915 | 0.5563 |
| Test | 0.4207 | 0.5610 | 0.4808 |

La F1 long-video riportata è calcolata globalmente sui conteggi complessivi di true positive, false positive e false negative.

---

## Installazione

La repository usa **Git LFS** per le feature store e i checkpoint necessari all'inferenza long-video.

```powershell
git clone https://github.com/Giop378/FVAB-Riconoscimento-azioni-basket.git
cd FVAB_BasketAR

git lfs install
git lfs pull

conda env create -f environment.yml
conda activate fvab-basket
```

---

## DINOv3

La pipeline usa l'implementazione ufficiale Meta di DINOv3 caricata localmente tramite PyTorch Hub.

| Proprietà | Valore |
|---|---|
| Repository | `facebookresearch/dinov3` |
| Directory attesa | `third_party/dinov3/` |
| Modello | `dinov3_vitl16` |
| Architettura | ViT-L/16 |
| Pesi | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| Input | `336 × 336` |
| Dimensione feature | `1024` |

Il checkpoint è atteso in:

```text
checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

DINOv3 e i relativi pesi servono soltanto per rigenerare le feature. Non sono necessari per eseguire la pipeline long-video a partire dalle feature store incluse.

---

## Artefatti inclusi

La repository include tramite Git LFS:

```text
data/features_long/primaparte_0215_1215_exp46/
data/features_long/psa_converted_0010_1010_exp46/

outputs/exp_l1_yolo_v2_temp43_allclips_d256_mean/
outputs/exp_l2_yolo_v2_temp29_allclips_d256_mean/
outputs/exp_l3_yolo_v1_temp43_shots_d256_mean/
```

Questi artefatti permettono di eseguire la pipeline long-video a partire dalla costruzione delle finestre, senza ripetere DINOv3 e YOLO.

Non sono inclusi:

- video e clip originali;
- dataset e annotazioni YOLO;
- pesi DINOv3 e YOLO;
- feature clip-level;
- output completi degli esperimenti;
- preview video generate.

I video originali sono necessari soltanto per rigenerare le feature store o produrre le preview annotate.

---

## Riproduzione degli esperimenti

I comandi completi sono raccolti nei seguenti file:

| File | Contenuto |
|---|---|
| [`final_commands_exp_46.md`](final_commands_exp_46.md) | Estrazione delle feature, tracking, training L1/L2/L3 e valutazione clip-level |
| [`final_commands_exp_long_13.md`](final_commands_exp_long_13.md) | Finestre, inferenza, post-processing, valutazione e preview long-video |

Per riprodurre la pipeline long-video usando gli artefatti già inclusi, seguire `final_commands_exp_long_13.md` partendo dalla sezione dedicata alla **costruzione delle finestre**.

I file nelle cartelle `Esperimenti_clip/` ed `Esperimenti_long_clip/` documentano configurazioni e prove storiche; non costituiscono il riferimento operativo finale.

---

## Struttura della repository

```text
FVAB_BasketAR/
├── data/
│   ├── datasets/          # Manifest e struttura dei dataset
│   ├── features/          # Feature di tracking clip-level generate localmente
│   └── features_long/     # Feature store long-video incluse tramite LFS
├── Esperimenti_clip/      # Documentazione degli esperimenti clip-level
├── Esperimenti_long_clip/ # Documentazione degli esperimenti long-video
├── notebooks/             # Analisi esplorative
├── outputs/               # Checkpoint finali e output locali
├── src/
│   ├── annotations/       # Estrazione dei frame per annotazione
│   ├── data/              # Dataset e utilità video
│   ├── evaluation/        # Valutazione gerarchica clip-level
│   ├── features/          # DINOv3 e tracking palla-canestro
│   ├── long_video/        # Pipeline long-video
│   ├── models/            # Temporal Transformer
│   └── training/          # Training dei classificatori e di YOLO
├── environment.yml
├── final_commands_exp_46.md
└── final_commands_exp_long_13.md
```

---

## Dataset

Il dataset principale è indicizzato da:

```text
data/datasets/dataset_basket_v1/manifest.csv
```

e contiene:

| Split | Numero di clip |
|---|---:|
| Train | 4236 |
| Validation | 540 |
| Test | 937 |

Gli split sono definiti per video, in modo da evitare la presenza di clip dello stesso video in train e valutazione.

Le clip e i video originali non sono versionati nella repository.

---

## Pipeline

### Clip-level

1. estrazione delle feature DINOv3;
2. estrazione del tracking temporale per L1, L2 e L3;
3. training dei tre classificatori;
4. valutazione gerarchica su validation e test.

### Long-video

1. estrazione della feature store;
2. costruzione delle finestre multi-scala;
3. inferenza con `exp_46`;
4. post-processing `exp_long_13`;
5. valutazione event-level;
6. esportazione del report e generazione della preview.

La configurazione finale long-video usa finestre da:

```text
1.0, 1.5, 2.0, 2.5, 3.0 secondi
```

con stride pari a:

```text
0.25 secondi
```

---

## Output principali

| File | Descrizione |
|---|---|
| `best_model.pt` | Checkpoint di uno dei livelli gerarchici |
| `predictions.csv` | Predizioni clip-level |
| `windows_manifest.csv` | Finestre temporali del video lungo |
| `window_predictions_raw.csv` | Predizioni raw per finestra |
| `events_postprocessed.csv` | Eventi finali dopo il post-processing |
| `event_metrics.md` | Metriche event-level |
| `BasketAR_*_report_events_exp13.csv` | Report compatto degli eventi |
| `preview_annotated.mp4` | Video con eventi sovrapposti |

---

## Note

- Il training usa seed `42`.
- Il best checkpoint viene selezionato sulla Macro F1 di validation.
- Gli artefatti binari inclusi devono essere scaricati con `git lfs pull`.
- Se un file `.npy`, `.npz` o `.pt` pesa soltanto pochi byte, è presente soltanto il puntatore Git LFS.
- Per rigenerare completamente le feature sono necessari DINOv3, i pesi YOLO e i video originali.
