# Evoluzione degli esperimenti long-video: exp_long_01 → exp_long_18

Questo documento riassume l'evoluzione degli esperimenti sulla pipeline long-video per il riconoscimento di azioni di basket. L'obiettivo era applicare il modello gerarchico addestrato su clip a un video lungo, trasformando le predizioni su finestre temporali in eventi finali valutabili rispetto alle annotazioni del manifest.

La valutazione è stata eseguita sul segmento di validation:

```text
Video: PrimaParte.mp4
Video ID manifest: prima_parte
Segmento: 135s → 735s
Numero eventi GT sulle 7 azioni: 71
IoU threshold: 0.20
Pred time mode: absolute
```

Le classi valutate sono:

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

## Obiettivo della pipeline long-video

La pipeline parte da predizioni su finestre temporali e deve produrre eventi finali con:

- classe dell'azione;
- tempo di inizio e fine;
- confidence;
- numero di eventi non eccessivo;
- buona copertura delle azioni reali;
- particolare attenzione alla demo video, dove troppi falsi positivi rendono il risultato poco credibile.

Durante gli esperimenti è emerso che il problema principale non era solo riconoscere le azioni, ma trasformare correttamente molte finestre sovrapposte in pochi eventi puliti.

---

## Tabella riassuntiva degli esperimenti

| Esperimento | Idea principale | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU | Center MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp_long_01 | Prima configurazione ufficiale con molte scale e stride fitto | 225 | 54 | 171 | 17 | 0.2400 | 0.7606 | 0.3649 | 0.3968 | 0.2825s |
| exp_long_02 | Finestre più corte per passaggi, rimosso 0.75s, stride 0.20 | 208 | 55 | 153 | 16 | 0.2644 | 0.7746 | 0.3943 | 0.4065 | 0.2708s |
| exp_long_03 | Finestre ancora più corte per passaggi | 289 | 55 | 234 | 16 | 0.1903 | 0.7746 | 0.3056 | 0.4359 | 0.2631s |
| exp_long_04 | Finestre lunghe per passaggi e stride 0.25 | 97 | 44 | 53 | 27 | 0.4536 | 0.6197 | 0.5238 | 0.3326 | 0.3706s |
| exp_long_05 | Come exp_long_04, ma max-duration-passaggio ridotta a 1.50s | 113 | 48 | 65 | 23 | 0.4248 | 0.6761 | 0.5217 | 0.3657 | 0.3315s |
| exp_long_06 | Peak-based passaggi, versione molto selettiva | 81 | 36 | 45 | 35 | 0.4444 | 0.5070 | 0.4737 | 0.5021 | 0.3289s |
| exp_long_07 | Peak-based meno aggressivo | 91 | 42 | 49 | 29 | 0.4615 | 0.5915 | 0.5185 | 0.4646 | 0.3205s |
| exp_long_08 | Peak-based più permissivo, max 4 picchi/gruppo | 112 | 45 | 67 | 26 | 0.4018 | 0.6338 | 0.4918 | 0.4807 | 0.3036s |
| exp_long_09 | No-action margin forte, configurazione pulita | 89 | 44 | 45 | 27 | 0.4944 | 0.6197 | 0.5500 | 0.3789 | 0.3012s |
| exp_long_10 | No-action margin molto alto, 50.0 | 68 | 37 | 31 | 34 | 0.5441 | 0.5211 | 0.5324 | 0.4009 | 0.2447s |
| exp_long_11 | Finestre più permissive + no-action margin 50.0 | 98 | 45 | 53 | 26 | 0.4592 | 0.6338 | 0.5325 | 0.4024 | 0.2666s |
| exp_long_12 | Finestre permissive + no-action margin 75.0 | 89 | 42 | 47 | 29 | 0.4719 | 0.5915 | 0.5250 | 0.4037 | 0.2732s |
| exp_long_13 | Miglior compromesso globale, no-action margin 20.0 | 80 | 42 | 38 | 29 | 0.5250 | 0.5915 | 0.5563 | 0.3886 | 0.2765s |
| exp_long_14 | Come exp_long_13, ma min-conf-tiro 0.30 | 80 | 42 | 38 | 29 | 0.5250 | 0.5915 | 0.5563 | 0.3886 | 0.2765s |
| exp_long_15 | Diagnostico con allow-overlaps | 142 | 52 | 90 | 19 | 0.3662 | 0.7324 | 0.4883 | 0.4235 | 0.3185s |
| exp_long_16 | Priorità ai tiri rispetto ai passaggi negli overlap | 73 | 39 | 34 | 32 | 0.5342 | 0.5493 | 0.5417 | 0.4110 | 0.3154s |
| exp_long_17 | Come exp_long_16, min-conf-tiro 0.50 | 73 | 39 | 34 | 32 | 0.5342 | 0.5493 | 0.5417 | 0.4110 | 0.3154s |
| exp_long_18 | Margini separati passaggi/tiri + priorità controllata ai tiri | 77 | 40 | 37 | 31 | 0.5195 | 0.5634 | 0.5405 | 0.3931 | 0.3330s |

---

# Evoluzione e motivazioni

## exp_long_01 — Prima configurazione ufficiale

La prima configurazione ufficiale usava molte scale temporali:

```text
window-sizes = 0.4 0.5 0.75 1.0 1.5 2.0 2.5 3.0
stride-sec = 0.15
```

L'obiettivo era massimizzare la possibilità di intercettare eventi di durata diversa. Il risultato aveva recall alta, ma troppi falsi positivi:

```text
Pred = 225
TP / FP / FN = 54 / 171 / 17
F1 = 0.3649
```

Il problema principale era sui passaggi: il modello prediceva spesso `passaggio` anche in zone di non-gioco o transizione, generando molti eventi consecutivi falsi.

**Conclusione:** configurazione troppo permissiva e non adatta alla demo.

---

## exp_long_02 — Finestre più corte per i passaggi

Si è provato a ridurre il rumore sui passaggi togliendo la finestra da `0.75s` e usando stride meno fitto:

```text
window-sizes = 0.4 0.5 1.0 1.5 2.0 2.5 3.0
stride-sec = 0.20
max-window-sec-passaggio = 0.50
```

Risultato:

```text
Pred = 208
TP / FP / FN = 55 / 153 / 16
F1 = 0.3943
```

Il risultato migliora rispetto a exp_long_01, ma i falsi positivi restano troppi.

**Conclusione:** togliere scale intermedie aiuta, ma non risolve il problema.

---

## exp_long_03 — Finestre ancora più corte

Si è provato a rendere i passaggi ancora più localizzati con finestre da `0.3s` e `0.4s`.

Risultato:

```text
Pred = 289
TP / FP / FN = 55 / 234 / 16
F1 = 0.3056
```

La recall resta alta, ma i falsi positivi esplodono.

**Conclusione:** finestre troppo corte rendono il modello più rumoroso. Questa direzione è stata scartata.

---

## exp_long_04 — Finestre lunghe per i passaggi

È stata provata l'idea opposta: usare finestre più lunghe per i passaggi, in modo da dare al modello più contesto e ridurre le predizioni spurie.

Configurazione indicativa:

```text
window-sizes = 1.0 1.5 2.0 2.5 3.0
stride-sec = 0.25
max-window-sec-passaggio = 2.00
max-duration-passaggio = 2.00
```

Risultato:

```text
Pred = 97
TP / FP / FN = 44 / 53 / 27
F1 = 0.5238
```

Questo è stato il primo salto importante: i falsi positivi si riducono molto e il numero di predizioni diventa più realistico.

**Conclusione:** per i passaggi, più contesto temporale è meglio di finestre molto corte.

---

## exp_long_05 — Riduzione della durata massima dei passaggi

Partendo da exp_long_04, è stata ridotta la durata massima degli eventi `passaggio`:

```text
max-duration-passaggio = 1.50
```

Risultato:

```text
Pred = 113
TP / FP / FN = 48 / 65 / 23
F1 = 0.5217
Mean IoU = 0.3657
```

Rispetto a exp_long_04 vengono recuperati più veri positivi e migliora la localizzazione, ma aumentano anche i falsi positivi.

**Conclusione:** exp_long_05 è un buon compromesso, soprattutto come configurazione di partenza per gli esperimenti successivi.

---

## exp_long_06, exp_long_07, exp_long_08 — Strategia peak-based

È stata introdotta una strategia alternativa per i passaggi: invece di trasformare un gruppo lungo di finestre in uno o più eventi lunghi, si cerca il picco di confidence e si genera un evento breve centrato sul picco.

L'obiettivo era separare:

```text
classificazione: fatta su finestre lunghe
localizzazione: fatta attorno al picco di confidenza
```

### exp_long_06

Configurazione molto selettiva:

```text
1 picco massimo per gruppo
peak duration ≈ 0.80s
min distance ≈ 1.00s
```

Risultato:

```text
Pred = 81
TP / FP / FN = 36 / 45 / 35
F1 = 0.4737
Mean IoU = 0.5021
```

La localizzazione migliora molto, ma si perdono troppi eventi veri.

### exp_long_07

La strategia viene resa meno aggressiva:

```text
peak duration ≈ 0.90s
min distance ≈ 0.70s
max peaks/group = 2
```

Risultato:

```text
Pred = 91
TP / FP / FN = 42 / 49 / 29
F1 = 0.5185
Mean IoU = 0.4646
```

Migliora rispetto a exp_long_06, ma non supera chiaramente exp_long_05.

### exp_long_08

Si prova a essere ancora più permissivi:

```text
peak duration ≈ 0.70s
min distance ≈ 0.50s
max peaks/group = 4
```

Risultato:

```text
Pred = 112
TP / FP / FN = 45 / 67 / 26
F1 = 0.4918
```

Aumentano i veri positivi, ma anche troppi falsi positivi.

**Conclusione:** la strategia peak-based migliora la localizzazione, ma non migliora abbastanza la detection complessiva. Non è stata scelta come configurazione finale.

---

## exp_long_09 — No-action margin forte

Si è provato a rendere più severo il filtro tra azione e no-action:

```text
--require-action-gt-noaction
--noaction-margin 10.0
```

Risultato:

```text
Pred = 89
TP / FP / FN = 44 / 45 / 27
Precision = 0.4944
Recall = 0.6197
F1 = 0.5500
```

Questo esperimento migliora il compromesso globale: meno falsi positivi e F1 più alto rispetto a exp_long_05.

**Conclusione:** il filtro action/no-action è molto utile per ottenere una pipeline più pulita.

---

## exp_long_10 — No-action margin molto alto

Si è provato un margine molto più aggressivo:

```text
--noaction-margin 50.0
```

Risultato:

```text
Pred = 68
TP / FP / FN = 37 / 31 / 34
F1 = 0.5324
```

La precision aumenta e il numero di falsi positivi scende, ma la recall cala troppo.

**Conclusione:** margine 50 è troppo severo sulla configurazione con finestre lunghe.

---

## exp_long_11 e exp_long_12 — Finestre più permissive + no-action margin alto

È stata provata una combinazione diversa: usare una configurazione di finestre più permissiva, ma filtrare con no-action margin alto.

### exp_long_11

```text
base: predizioni exp_long_02
noaction-margin = 50.0
```

Risultato:

```text
Pred = 98
TP / FP / FN = 45 / 53 / 26
F1 = 0.5325
```

### exp_long_12

```text
base: predizioni exp_long_02
noaction-margin = 75.0
```

Risultato:

```text
Pred = 89
TP / FP / FN = 42 / 47 / 29
F1 = 0.5250
```

**Conclusione:** questa strada non supera exp_long_09. Il margine alto aiuta, ma la configurazione più permissiva porta comunque troppi falsi positivi.

---

## exp_long_13 — Miglior compromesso globale

Si è scelto un valore intermedio del no-action margin:

```text
base: predizioni exp_long_04
noaction-margin = 20.0
```

Risultato:

```text
Pred = 80
TP / FP / FN = 42 / 38 / 29
Precision = 0.5250
Recall = 0.5915
F1 = 0.5563
```

Questo è il miglior risultato globale tra gli esperimenti provati. Il numero di eventi predetti è vicino al numero reale e i falsi positivi sono abbastanza contenuti.

Metriche principali per classe:

```text
passaggio: GT 53, Pred 60, TP 35, FP 25, FN 18, F1 0.6195
```

Sui tiri, però, la recall resta limitata: vengono riconosciuti circa 7 tiri veri su 18.

**Conclusione:** exp_long_13 è la configurazione migliore dal punto di vista quantitativo globale.

---

## exp_long_14 — Abbassamento soglia tiri

Si è provato ad abbassare la soglia dei tiri:

```text
min-conf-tiro = 0.30
```

Risultato:

```text
Pred = 80
TP / FP / FN = 42 / 38 / 29
F1 = 0.5563
```

Il risultato è identico a exp_long_13.

**Conclusione:** la soglia dei tiri non era il collo di bottiglia principale.

---

## exp_long_15 — Diagnostica con allow-overlaps

È stata fatta una prova diagnostica permettendo eventi sovrapposti:

```text
--allow-overlaps
```

Risultato:

```text
Pred = 142
TP / FP / FN = 52 / 90 / 19
Precision = 0.3662
Recall = 0.7324
F1 = 0.4883
```

La cosa importante è che i tiri migliorano molto:

```text
tiri GT = 18
tiri TP = 17
tiri FN = 1
```

Però vengono generati troppi falsi tiri.

**Conclusione:** i tiri sono presenti nelle predizioni raw, ma spesso vengono eliminati dalla soppressione degli overlap o competono con i passaggi. Non conviene usare allow-overlaps nella demo, ma è utile per capire il problema.

---

## exp_long_16 — Priorità ai tiri negli overlap

Sulla base della diagnostica di exp_long_15, è stata modificata la soppressione degli overlap:

```text
se un tiro e un passaggio si sovrappongono,
preferisci il tiro
```

Risultato:

```text
Pred = 73
TP / FP / FN = 39 / 34 / 32
Precision = 0.5342
Recall = 0.5493
F1 = 0.5417
```

La demo sui tiri migliora:

```text
tiri GT = 18
tiri TP = 12
tiri FN = 6
```

Il prezzo pagato è una perdita sui passaggi:

```text
passaggio: GT 53, Pred 38, TP 27, FP 11, FN 26
```

**Conclusione:** exp_long_16 è molto interessante per la demo, perché mostra più tiri e mantiene il numero totale di eventi molto realistico.

---

## exp_long_17 — Soglia tiri più alta con priorità ai tiri

È stata provata una soglia tiri più alta:

```text
min-conf-tiro = 0.50
```

Risultato:

```text
Pred = 73
TP / FP / FN = 39 / 34 / 32
F1 = 0.5417
```

Il risultato globale è praticamente identico a exp_long_16.

**Conclusione:** alzare la soglia tiri non cambia in modo significativo il comportamento.

---

## exp_long_18 — Margini separati per passaggi e tiri

Ultima modifica: separare il margine action/no-action per passaggi e tiri, e rendere controllata la priorità ai tiri.

Configurazione:

```text
noaction-margin = 20.0
noaction-margin-passaggio = 20.0
noaction-margin-tiro = 5.0
prefer-shots-over-passaggi = True
prefer-shots-min-confidence = 0.55
min-conf-tiro = 0.50
```

Risultato:

```text
Pred = 77
TP / FP / FN = 40 / 37 / 31
Precision = 0.5195
Recall = 0.5634
F1 = 0.5405
```

Sui tiri:

```text
tiri GT = 18
tiri Pred = 37
tiri TP = 12
tiri FP = 25
tiri FN = 6
```

Quindi exp_long_18 non migliora il F1 globale rispetto a exp_long_13, ma mantiene una buona capacità di mostrare i tiri nella demo.

**Conclusione:** exp_long_18 è una variante orientata alla demo, soprattutto se è importante mostrare il riconoscimento dei tiri.

---

# Esperimenti principali

Ecco i risultati rappresentativi.

### 1. exp_long_02 — baseline permissiva

Serve a mostrare il problema iniziale:

```text
alta recall
ma troppi falsi positivi
```

### 2. exp_long_04 / exp_long_05 — finestre più lunghe

Serve a mostrare la prima svolta:

```text
più contesto temporale
meno falsi positivi
numero di eventi più realistico
```

### 3. exp_long_13 — miglior risultato quantitativo globale

È il candidato principale per le metriche:

```text
Pred = 80
TP / FP / FN = 42 / 38 / 29
F1 = 0.5563
```

### 4. exp_long_18 — variante orientata alla demo sui tiri

È il candidato principale se nella demo si vuole mostrare meglio il riconoscimento dei tiri:

```text
Pred = 77
TP / FP / FN = 40 / 37 / 31
Tiri TP = 12 / 18
```

---

# Test finale

Dopo la selezione dei due candidati principali sulla validation, sono state eseguite entrambe le configurazioni sul video di test:

```text
Video: PSA_converted.mp4
Video ID manifest: psa_converted
Segmento: 10s → 610s
Numero eventi GT sulle 7 azioni: 123
IoU threshold: 0.20
Pred time mode: absolute
```

L'obiettivo del test era verificare quale delle due configurazioni generalizzasse meglio su un video non usato per la taratura del post-processing.

## Risultati test

| Configurazione test | Idea principale | Pred | TP | FP | FN | Precision | Recall | F1 | Macro F1 | Mean IoU | Center MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_exp_long_13 | Miglior compromesso globale scelto su validation | 164 | 69 | 95 | 54 | 0.4207 | 0.5610 | 0.4808 | 0.4357 | 0.4064 | 0.3076s |
| test_exp_long_18 | Variante con margini separati e priorità controllata ai tiri | 143 | 56 | 87 | 67 | 0.3916 | 0.4553 | 0.4211 | 0.3102 | 0.4257 | 0.2897s |

## Test con configurazione exp_long_13

La configurazione `exp_long_13` sul test produce più eventi rispetto al numero reale:

```text
GT = 123
Pred = 164
TP / FP / FN = 69 / 95 / 54
Precision = 0.4207
Recall = 0.5610
F1 = 0.4808
Macro F1 = 0.4357
Mean IoU = 0.4064
Center MAE = 0.3076s
```

Rispetto alla validation, il risultato cala soprattutto in precisione: il modello continua a recuperare una quantità ragionevole di eventi, ma produce più falsi positivi. Il comportamento è comunque coerente con la natura del problema long-video: sul test ci sono più eventi, più passaggi e maggiore variabilità temporale.

Metriche principali per classe:

```text
passaggio: GT 96, Pred 145, TP 61, FP 84, FN 35, F1 0.5062
tiroDaDue0: GT 5, Pred 8, TP 2, FP 6, FN 3, F1 0.3077
tiroDaDue1: GT 7, Pred 3, TP 2, FP 1, FN 5, F1 0.4000
tiroDaTre0: GT 8, Pred 2, TP 1, FP 1, FN 7, F1 0.2000
tiroDaTre1: GT 3, Pred 0, TP 0, FP 0, FN 3
tiroLibero0: GT 2, Pred 3, TP 1
tiroLibero1: GT 2, Pred 3, TP 2
```

Il limite principale resta sui tiri da tre e sulla classificazione fine degli esiti, mentre sui passaggi la configurazione mantiene una buona recall ma con molti falsi positivi.

**Conclusione test exp_long_13:** è rumoroso, ma è il miglior compromesso globale sul test perché mantiene la recall più alta e il miglior F1 micro/macro tra le due configurazioni finali.

## Test con configurazione exp_long_18

La configurazione `exp_long_18` era stata introdotta per favorire maggiormente i tiri, separando i margini passaggio/tiro e dando priorità controllata ai tiri negli overlap. Sul test però non generalizza meglio di exp_long_13:

```text
GT = 123
Pred = 143
TP / FP / FN = 56 / 87 / 67
Precision = 0.3916
Recall = 0.4553
F1 = 0.4211
Macro F1 = 0.3102
Mean IoU = 0.4257
Center MAE = 0.2897s
```

Rispetto a `test_exp_long_13`, riduce leggermente il numero di predizioni e falsi positivi, ma perde troppi eventi veri:

```text
TP: 69 → 56
FN: 54 → 67
F1: 0.4808 → 0.4211
Macro F1: 0.4357 → 0.3102
```

Metriche principali per classe:

```text
passaggio: GT 96, Pred 90, TP 45, FP 45, FN 51, F1 0.4839
tiroDaDue0: GT 5, Pred 16, TP 3, FP 13, FN 2, F1 0.2857
tiroDaDue1: GT 7, Pred 7, TP 3, FP 4, FN 4, F1 0.4286
tiroDaTre0: GT 8, Pred 8, TP 1, FP 7, FN 7, F1 0.1250
tiroDaTre1: GT 3, Pred 6, TP 1, FP 5, FN 2, F1 0.2222
tiroLibero0: GT 2, Pred 9, TP 1, FP 8, FN 1, F1 0.1818
tiroLibero1: GT 2, Pred 7, TP 2, FP 5, FN 0, F1 0.4444
```

La localizzazione degli eventi matchati è leggermente migliore rispetto a exp_long_13, come mostrano `Mean IoU = 0.4257` e `Center MAE = 0.2897s`, ma questo vantaggio non compensa il calo di recall e l'aumento dei falsi tiri.

**Conclusione test exp_long_18:** è più selettivo sui passaggi e localizza leggermente meglio gli eventi che riconosce, ma perde troppi eventi veri e introduce molti falsi positivi sui tiri. Non viene scelto come configurazione finale.

## Scelta finale sul test

La configurazione finale scelta è:

```text
exp_long_13
```

Motivazione:

```text
- migliore F1 micro sul test: 0.4808 contro 0.4211
- migliore recall: 0.5610 contro 0.4553
- migliore macro F1: 0.4357 contro 0.3102
- migliore compromesso complessivo tra azioni recuperate e falsi positivi
```

`exp_long_18` resta utile come esperimento di ablazione, perché mostra l'effetto della priorità ai tiri e dei margini separati, ma non viene adottato come configurazione finale.

---

# Interpretazione finale

L'evoluzione degli esperimenti mostra che:

1. Le finestre troppo corte aumentano il rumore.
2. Le finestre più lunghe riducono i falsi positivi, soprattutto sui passaggi.
3. Il filtro action/no-action è fondamentale per rendere la preview credibile.
4. La gestione degli overlap incide molto sui tiri.
5. Preferire i tiri ai passaggi aiuta la demo, ma può penalizzare i passaggi.
6. Separare la logica tra passaggi e tiri è utile, ma il margine di miglioramento via post-processing è ormai limitato.

Il limite principale rimasto non sembra essere solo il post-processing, ma la classificazione fine delle sottoclassi di tiro, in particolare casi come `tiroDaDue1`, che rimangono difficili da riconoscere correttamente.

---

# Conclusione operativa

Dopo il confronto anche sul video di test, la configurazione finale scelta è:

```text
exp_long_13
```

`exp_long_13` viene scelto come configurazione finale perché mantiene il miglior compromesso quantitativo sul test: F1 micro più alto, recall più alta e macro F1 migliore rispetto alla variante `exp_long_18`.

`exp_long_18` può essere citato come esperimento di ablazione orientato ai tiri, ma non viene scelto come risultato finale perché sul test perde troppi eventi veri e introduce molti falsi positivi sulle classi di tiro.

