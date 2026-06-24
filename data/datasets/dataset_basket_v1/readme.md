# Dataset Action Recognition Basket

Dataset di clip video annotate per il training di un modello di action recognition su partite di basket. Costruito a partire dalle annotazioni prodotte dai gruppi del corso.

## Struttura del dataset

```
dataset/
├── manifest.csv         # File indice con tutte le clip e i loro metadati
├── train/               # Clip di training organizzate per classe
│   ├── passaggio/
│   │   ├── clip_000000.mp4
│   │   └── ...
│   ├── tiroDaDue0/
│   ├── tiroDaDue1/
│   ├── tiroDaTre0/
│   ├── tiroDaTre1/
│   ├── tiroLibero0/
│   ├── tiroLibero1/
│   ├── idle/
│   └── non-gioco/
├── val/                 # Stessa struttura di train, per validazione
│   └── ...
├── test/                # Stessa struttura di train, per test finale
│   └── ...
└── README.md            # Questo file
```

La struttura è compatibile con loader standard (es. `torchvision.datasets.ImageFolder` o equivalenti per video). In alternativa, il `manifest.csv` permette di caricare le clip via path esplicito.

## Formato del manifest

Il file `manifest.csv` contiene una riga per ogni clip. Colonne:

| Colonna | Tipo | Descrizione |
|---|---|---|
| `clip_id` | stringa | Identificativo univoco della clip |
| `path` | stringa | Path relativo alla clip dalla root del dataset (es. `train/passaggio/clip_000000.mp4`) |
| `video_id` | stringa | Video sorgente da cui è stata estratta la clip |
| `start_time` | float | Timestamp (secondi) di inizio nel video sorgente |
| `end_time` | float | Timestamp (secondi) di fine nel video sorgente |
| `label` | stringa | Classe dell'azione (vedi tabella sotto) |
| `split` | stringa | Assegnazione: `train`, `val` oppure `test` |

Le clip hanno **durata variabile**, tipicamente tra 0.15s e 5s, in base alla durata reale dell'azione annotata. Le clip `non-gioco` originariamente più lunghe sono state spezzate automaticamente in clip più corte.

## Classi

Il dataset contiene 9 classi:

| Label | Descrizione |
|---|---|
| `passaggio` | Trasferimento intenzionale della palla tra compagni |
| `tiroDaDue0` | Tiro da 2 punti fallito |
| `tiroDaDue1` | Tiro da 2 punti segnato |
| `tiroDaTre0` | Tiro da 3 punti fallito |
| `tiroDaTre1` | Tiro da 3 punti segnato |
| `tiroLibero0` | Tiro libero fallito |
| `tiroLibero1` | Tiro libero segnato |
| `idle` | Gioco attivo senza azione rilevante (campionato automaticamente) |
| `non-gioco` | Riscaldamento, intervallo, interruzioni (gioco fermo) |

**Importante**: nel report finale del modello vanno riportate solo le 7 classi di azione. `idle` e `non-gioco` servono al modello per riconoscere quando NON c'è un'azione e non vanno mai inserite nel report finale.

## Split

Lo split è **per video** (non per clip): tutte le clip di uno stesso video stanno in un solo split. Questo previene data leakage (il modello non vede in test frame simili a quelli del training).

- **train**: 4 video
- **val**: 1 video
- **test**: 1 video

Con solo 1 video in val/test alcune classi rare hanno pochi campioni di validazione (es. tiroDaTre1 in val). Tenetelo presente quando interpretate le metriche.

## Distribuzione delle classi

Il dataset è **fortemente sbilanciato**:

- `passaggio` ~53%
- `idle` ~15%
- `non-gioco` ~14%
- Tiri vari ~18% complessivo (con tiroDaTre1 ~1.2%, classe più rara)

Strategie consigliate per gestire lo sbilanciamento:
- **Class weights** nella loss (inverso della frequenza)
- **Weighted random sampler** per garantire presenza delle classi rare in ogni batch
- **Augmentation aggressiva** sulle classi rare (flip orizzontale, color jitter, crop temporali)

### Pipeline di inferenza per il report finale

L'obiettivo finale è applicare il modello su un video intero di una partita e produrre un report con le azioni rilevate (tipo + timestamp). Approccio consigliato:

1. Sliding window sul video con finestra fissa (es. 2s) e overlap (es. 50%)
2. Classificazione di ogni finestra
3. Filtraggio: scartare predizioni `idle` e `non-gioco`
4. Post-processing temporale: unire finestre adiacenti con la stessa classe e applicare smoothing per ridurre falsi positivi
5. Output: lista di azioni con classe, tempo di inizio, tempo di fine

## Note finali

- Il dataset è frutto di annotazioni manuali e contiene errori inevitabili: alcune azioni mancanti, alcuni timestamp imprecisi
- Le clip molto brevi (< 0.5s) sono spesso passaggi: tenetelo in conto se decidete di usare strategie di padding/resample
- Per il deploy del modello tenete presente che il video di input avrà fasi di non-gioco (riscaldamento, intervallo): la classe `non-gioco` serve proprio per gestire questi casi