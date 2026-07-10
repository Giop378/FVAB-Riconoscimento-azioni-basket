"""
Valutazione gerarchica end-to-end della configurazione finale exp_46.

Lo script carica tre checkpoint addestrati separatamente:
- L1: passaggio / tiro / no-action
- L2: tiroDaDue / tiroDaTre / tiroLibero
- L3: tiro0 / tiro1

Se un checkpoint richiede tracking, vengono usate solo sequenze temporali
palla/canestro in formato NPZ + JSON. Il supporto alle vecchie feature
aggregate CSV è stato rimosso perché non serve per exp_46.
"""
# Collegamenti con la pipeline:
# - FeatureDataset carica le feature .pt prodotte da extract_features.py;
# - i tre checkpoint sono generati da training/train.py con label mode differenti;
# - TrackingSequenceFeatureStore recupera le sequenze NPZ/JSON prodotte da
#   extract_ball_rim_tracking_features.py e applica le statistiche del checkpoint;
# - TemporalTransformerActionClassifier ricostruisce esattamente ciascun livello;
# - predictions.csv conserva predizioni, confidenze e disponibilità del tracking.


from pathlib import Path
import argparse
import csv
import shlex
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.feature_dataset import FeatureDataset, collate_features, IDX_TO_LABEL
from src.models.temporal_transformer_classifier import TemporalTransformerActionClassifier
from src.features.tracking_sequence_store import TrackingSequenceFeatureStore


# Spazi di label usati per la valutazione completa e per la variante collassata
# che ignora l’esito del tiro.
FINAL_LABELS = [
    "passaggio",
    "tiroDaDue0",
    "tiroDaDue1",
    "tiroDaTre0",
    "tiroDaTre1",
    "tiroLibero0",
    "tiroLibero1",
    "no-action",
]

FINAL_ACTION_LABELS = FINAL_LABELS[:-1]

FINAL_TYPE_LABELS = [
    "passaggio",
    "tiroDaDue",
    "tiroDaTre",
    "tiroLibero",
    "no-action",
]


class Tee:
    """Duplica stdout/stderr sia su terminale sia su file di log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_reconstructed_command() -> str:
    parts = [sys.executable] + sys.argv
    return " ".join(shlex.quote(str(part)) for part in parts)


def normalize_idx_to_label(idx_to_label):
    if isinstance(idx_to_label, dict):
        return {int(k): str(v) for k, v in idx_to_label.items()}

    if isinstance(idx_to_label, (list, tuple)):
        return {idx: str(label) for idx, label in enumerate(idx_to_label)}

    raise TypeError(f"Formato idx_to_label non supportato: {type(idx_to_label)}")


def fallback_idx_to_label(label_mode: str):
    mappings = {
        "action_noaction": {
            0: "passaggio",
            1: "tiro",
            2: "no-action",
        },
        "shot_type_only": {
            0: "tiroDaDue",
            1: "tiroDaTre",
            2: "tiroLibero",
        },
        "shot_outcome_only": {
            0: "tiro0",
            1: "tiro1",
        },
    }

    if label_mode not in mappings:
        raise ValueError(f"Label mode non gestita per fallback: {label_mode}")

    return mappings[label_mode]


def original_idx_to_label():
    return {idx: IDX_TO_LABEL[idx] for idx in range(len(IDX_TO_LABEL))}


def original_to_final_label(original_label: str) -> str:
    if original_label in {"idle", "non-gioco"}:
        return "no-action"

    if original_label in FINAL_ACTION_LABELS:
        return original_label

    raise ValueError(f"Label originale non riconosciuta: {original_label}")


def final_to_type_label(final_label: str) -> str:
    if final_label in {"passaggio", "no-action"}:
        return final_label

    if final_label.startswith("tiroDaDue"):
        return "tiroDaDue"

    if final_label.startswith("tiroDaTre"):
        return "tiroDaTre"

    if final_label.startswith("tiroLibero"):
        return "tiroLibero"

    raise ValueError(f"Label finale non riconosciuta: {final_label}")


def get_sample_path(dataset, idx: int) -> str:
    if hasattr(dataset, "items"):
        item = dataset.items[idx]

        if isinstance(item, dict):
            if "path" in item:
                return str(item["path"])
            return str(item)

        return str(item)

    return ""


# Ricostruisce l’architettura dai metadati salvati da train.py, carica i pesi
# e recupera il vocabolario delle classi dello specifico livello gerarchico.
def load_checkpoint_model(checkpoint_path: str, device: torch.device, label_mode: str):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_config" not in checkpoint:
        raise KeyError(
            f"Il checkpoint {checkpoint_path} non contiene 'model_config'. "
            "Serve un checkpoint salvato da train.py."
        )

    config = checkpoint["model_config"]
    num_classes = int(config["num_classes"])
    idx_to_label = checkpoint.get("idx_to_label")

    if idx_to_label is None:
        idx_to_label = fallback_idx_to_label(label_mode)
    else:
        idx_to_label = normalize_idx_to_label(idx_to_label)

    if len(idx_to_label) != num_classes:
        idx_to_label = fallback_idx_to_label(label_mode)

    model = TemporalTransformerActionClassifier(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        dim_feedforward=int(config.get("dim_feedforward", config.get("ff_dim"))),
        num_classes=num_classes,
        dropout=float(config["dropout"]),
        pooling=str(config["pooling"]),
        max_len=int(config.get("max_len", 1024)),
        last_mean_ratio=float(config.get("last_mean_ratio", 0.30)),
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model, idx_to_label, checkpoint, config


def get_checkpoint_tracking_requirements(checkpoint, config):
    """
    Ricava se uno stadio richiede tracking temporale.

    Per exp_46 sono supportate solo feature tracking temporali. Se un checkpoint
    richiede ancora tracking aggregate CSV, viene sollevato un errore esplicito.
    """
    tracking_config = checkpoint.get("tracking_config") or config.get("tracking_config")
    tracking_input_dim = int(config.get("tracking_input_dim", 0))

    if tracking_input_dim <= 0:
        return tracking_config, "none", tracking_input_dim

    tracking_type = "temporal_sequence"
    if tracking_config:
        tracking_type = tracking_config.get("type", "temporal_sequence")

    if tracking_type != "temporal_sequence":
        raise ValueError(
            "Il checkpoint richiede feature tracking non temporali "
            f"('{tracking_type}'), ma questa versione supporta solo exp_46 "
            "con tracking temporale NPZ + JSON."
        )

    return tracking_config, tracking_type, tracking_input_dim


# Ogni livello può avere una configurazione tracking indipendente: il relativo
# store viene creato solo quando il checkpoint dichiara tracking_input_dim > 0.
def load_tracking_store_for_level(
    level_name: str,
    checkpoint,
    config,
    tracking_sequences_npz: str | None = None,
    tracking_sequence_index: str | None = None,
    missing_policy: str = "zeros",
):
    """
    Carica lo store tracking temporale richiesto da uno specifico livello.

    Se il checkpoint non richiede tracking, restituisce None.
    Se il checkpoint richiede tracking, usa prima i path passati da riga di
    comando; se mancanti, prova i path salvati nel checkpoint.
    """
    tracking_config, tracking_type, tracking_input_dim = get_checkpoint_tracking_requirements(
        checkpoint=checkpoint,
        config=config,
    )

    if tracking_type == "none" or tracking_input_dim <= 0:
        print(f"\n# Feature tracking per {level_name}")
        print("Non richieste dal checkpoint.")
        return None

    npz_path = tracking_sequences_npz
    index_path = tracking_sequence_index

    if tracking_config:
        npz_path = npz_path or tracking_config.get("npz_path")
        index_path = index_path or tracking_config.get("index_path")

    if npz_path is None:
        raise ValueError(
            f"Il checkpoint {level_name} richiede sequenze tracking temporali, "
            f"ma non è stato passato il file NPZ."
        )

    feature_names = tracking_config.get("feature_names") if tracking_config else None
    mean = tracking_config.get("mean") if tracking_config else None
    std = tracking_config.get("std") if tracking_config else None
    normalized = bool(tracking_config.get("normalized", False)) if tracking_config else False

    store = TrackingSequenceFeatureStore(
        npz_path,
        index_path=index_path,
        feature_names=feature_names,
        mean=mean,
        std=std,
        normalized=normalized,
    )

    print(f"\n# Feature tracking temporali per {level_name}")
    print(f"NPZ tracking: {npz_path}")
    print(f"Indice tracking: {store.index_path}")
    print(f"Numero feature tracking per frame: {store.num_features}")
    print(f"Normalizzate con statistiche del checkpoint {level_name}: {normalized}")
    print(f"Missing policy: {missing_policy}")
    return store


# Allinea le sequenze di tracking alla lunghezza reale delle feature DINOv3 e
# le porta al Tmax del batch con padding a zero.
def build_tracking_sequence_batch(
    tracking_store: TrackingSequenceFeatureStore | None,
    dataset,
    sample_offset: int,
    batch_size: int,
    features: torch.Tensor,
    lengths: torch.Tensor,
    missing_policy: str,
    device: torch.device,
):
    """
    Costruisce il tensore tracking temporale [B, Tmax, K] per un batch.

    Restituisce:
    - tracking_sequences oppure None;
    - tracking_available: lista booleana, una per campione.
    """
    tracking_available = [False] * batch_size

    if tracking_store is None:
        return None, tracking_available

    max_seq_len = features.shape[1]
    tracking_sequence_vectors = []

    for i in range(batch_size):
        global_idx = sample_offset + i
        sample_path = get_sample_path(dataset, global_idx)
        tracking_available[i] = tracking_store.has(sample_path)

        real_len = int(lengths[i].item())
        sequence = tracking_store.get(
            sample_path,
            target_len=real_len,
            missing_policy=missing_policy,
        )

        padded = np.zeros((max_seq_len, tracking_store.num_features), dtype=np.float32)
        padded[:real_len] = sequence[:real_len]
        tracking_sequence_vectors.append(padded)

    tracking_sequences = torch.tensor(
        np.stack(tracking_sequence_vectors, axis=0),
        dtype=features.dtype,
        device=device,
    )
    return tracking_sequences, tracking_available


def append_tracking_sequence_to_features(
    features: torch.Tensor,
    tracking_sequences: torch.Tensor | None,
) -> torch.Tensor:
    """Concatena una sequenza tracking [B, T, K] alle feature video [B, T, D]."""
    if tracking_sequences is None:
        return features

    if tracking_sequences.ndim != 3:
        raise ValueError(
            f"tracking_sequences deve avere forma [B, T, K], "
            f"ricevuta {tuple(tracking_sequences.shape)}."
        )

    if tracking_sequences.shape[:2] != features.shape[:2]:
        raise ValueError(
            f"Shape tracking temporale non coerente: features {tuple(features.shape)}, "
            f"tracking {tuple(tracking_sequences.shape)}."
        )

    return torch.cat([features, tracking_sequences.to(features.device, dtype=features.dtype)], dim=2)


# Esegue il routing gerarchico: L1 decide se la clip è un tiro; soltanto le
# clip instradate come tiro vengono sottoposte in parallelo a L2 e L3.
@torch.no_grad()
def predict_hierarchical_batch(
    features,
    lengths,
    model_l1,
    model_l2,
    model_l3,
    idx_to_label_l1,
    idx_to_label_l2,
    idx_to_label_l3,
    l1_tracking_sequences=None,
    l2_tracking_sequences=None,
    l3_tracking_sequences=None,
):
    features_l1 = append_tracking_sequence_to_features(features, l1_tracking_sequences)

    logits_l1 = model_l1(features_l1, lengths)
    probs_l1 = torch.softmax(logits_l1, dim=1)
    preds_l1 = probs_l1.argmax(dim=1)

    batch_size = features.size(0)

    pred_l1_labels = [idx_to_label_l1[int(idx)] for idx in preds_l1.cpu().tolist()]
    pred_l2_labels = [""] * batch_size
    pred_l3_labels = [""] * batch_size
    final_preds = [""] * batch_size

    p_l1 = probs_l1.max(dim=1).values.detach().cpu().tolist()
    p_l2 = [None] * batch_size
    p_l3 = [None] * batch_size

    # Gli indici preservano la posizione nel batch originale per ricomporre poi
    # la label finale tipo+esito nella stessa posizione del campione sorgente.
    shot_indices = [idx for idx, label in enumerate(pred_l1_labels) if label == "tiro"]

    for idx, label_l1 in enumerate(pred_l1_labels):
        if label_l1 == "passaggio":
            final_preds[idx] = "passaggio"
        elif label_l1 == "no-action":
            final_preds[idx] = "no-action"
        elif label_l1 == "tiro":
            final_preds[idx] = "__pending_shot__"
        else:
            raise ValueError(f"Predizione L1 non riconosciuta: {label_l1}")

    if shot_indices:
        shot_indices_tensor = torch.tensor(shot_indices, dtype=torch.long, device=features.device)
        shot_features = features.index_select(0, shot_indices_tensor)
        shot_lengths = lengths.index_select(0, shot_indices_tensor)

        shot_l2_tracking_sequences = None
        if l2_tracking_sequences is not None:
            shot_l2_tracking_sequences = l2_tracking_sequences.index_select(0, shot_indices_tensor)

        shot_features_l2 = append_tracking_sequence_to_features(
            shot_features,
            shot_l2_tracking_sequences,
        )

        logits_l2 = model_l2(shot_features_l2, shot_lengths)
        probs_l2 = torch.softmax(logits_l2, dim=1)
        preds_l2 = probs_l2.argmax(dim=1)

        shot_l3_tracking_sequences = None
        if l3_tracking_sequences is not None:
            shot_l3_tracking_sequences = l3_tracking_sequences.index_select(0, shot_indices_tensor)

        shot_features_l3 = append_tracking_sequence_to_features(
            shot_features,
            shot_l3_tracking_sequences,
        )

        logits_l3 = model_l3(shot_features_l3, shot_lengths)
        probs_l3 = torch.softmax(logits_l3, dim=1)
        preds_l3 = probs_l3.argmax(dim=1)

        pred_l2_shots = [idx_to_label_l2[int(idx)] for idx in preds_l2.cpu().tolist()]
        pred_l3_shots = [idx_to_label_l3[int(idx)] for idx in preds_l3.cpu().tolist()]

        p_l2_shots = probs_l2.max(dim=1).values.detach().cpu().tolist()
        p_l3_shots = probs_l3.max(dim=1).values.detach().cpu().tolist()

        for local_idx, global_idx in enumerate(shot_indices):
            shot_type = pred_l2_shots[local_idx]
            shot_outcome = pred_l3_shots[local_idx]

            if shot_outcome == "tiro0":
                suffix = "0"
            elif shot_outcome == "tiro1":
                suffix = "1"
            else:
                raise ValueError(f"Predizione L3 non riconosciuta: {shot_outcome}")

            final_preds[global_idx] = f"{shot_type}{suffix}"
            pred_l2_labels[global_idx] = shot_type
            pred_l3_labels[global_idx] = shot_outcome
            p_l2[global_idx] = p_l2_shots[local_idx]
            p_l3[global_idx] = p_l3_shots[local_idx]

    return final_preds, pred_l1_labels, pred_l2_labels, pred_l3_labels, p_l1, p_l2, p_l3


def print_report(title: str, y_true, y_pred, labels):
    print(f"\n{title}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            zero_division=0,
        )
    )

    print(f"Confusion matrix - {title}:")
    print(confusion_matrix(y_true, y_pred, labels=labels))


# Pipeline end-to-end: crea il DataLoader, carica i tre livelli e i relativi
# store tracking, accumula le predizioni e calcola le metriche finali.
def run_evaluation(args) -> None:
    print("# Comando utilizzato")
    print(get_reconstructed_command())
    print("\n" + "=" * 80 + "\n")

    print("# Configurazione esperimento")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("\n" + "=" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    dataset = FeatureDataset(args.features_root, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Split: {args.split}")
    print(f"Numero campioni: {len(dataset)}")

    model_l1, idx_to_label_l1, ckpt_l1, config_l1 = load_checkpoint_model(
        args.l1_checkpoint,
        device,
        label_mode="action_noaction",
    )
    model_l2, idx_to_label_l2, ckpt_l2, config_l2 = load_checkpoint_model(
        args.l2_checkpoint,
        device,
        label_mode="shot_type_only",
    )
    model_l3, idx_to_label_l3, ckpt_l3, config_l3 = load_checkpoint_model(
        args.l3_checkpoint,
        device,
        label_mode="shot_outcome_only",
    )

    print("\n# Modelli caricati")
    print(f"L1 checkpoint: {args.l1_checkpoint}")
    print(f"L1 idx_to_label: {idx_to_label_l1}")
    print(f"L2 checkpoint: {args.l2_checkpoint}")
    print(f"L2 idx_to_label: {idx_to_label_l2}")
    print(f"L3 checkpoint: {args.l3_checkpoint}")
    print(f"L3 idx_to_label: {idx_to_label_l3}")

    l1_tracking_store = load_tracking_store_for_level(
        level_name="L1",
        checkpoint=ckpt_l1,
        config=config_l1,
        tracking_sequences_npz=args.l1_tracking_sequences_npz,
        tracking_sequence_index=args.l1_tracking_sequence_index,
        missing_policy=args.tracking_missing_policy,
    )
    l2_tracking_store = load_tracking_store_for_level(
        level_name="L2",
        checkpoint=ckpt_l2,
        config=config_l2,
        tracking_sequences_npz=args.l2_tracking_sequences_npz,
        tracking_sequence_index=args.l2_tracking_sequence_index,
        missing_policy=args.tracking_missing_policy,
    )
    l3_tracking_store = load_tracking_store_for_level(
        level_name="L3",
        checkpoint=ckpt_l3,
        config=config_l3,
        tracking_sequences_npz=args.l3_tracking_sequences_npz,
        tracking_sequence_index=args.l3_tracking_sequence_index,
        missing_policy=args.tracking_missing_policy,
    )

    original_mapping = original_idx_to_label()

    y_true_final = []
    y_pred_final = []
    rows = []
    sample_offset = 0

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].cpu().tolist()
        batch_size = len(labels)

        l1_tracking_sequences, l1_tracking_available = build_tracking_sequence_batch(
            tracking_store=l1_tracking_store,
            dataset=dataset,
            sample_offset=sample_offset,
            batch_size=batch_size,
            features=features,
            lengths=lengths,
            missing_policy=args.tracking_missing_policy,
            device=device,
        )
        l2_tracking_sequences, l2_tracking_available = build_tracking_sequence_batch(
            tracking_store=l2_tracking_store,
            dataset=dataset,
            sample_offset=sample_offset,
            batch_size=batch_size,
            features=features,
            lengths=lengths,
            missing_policy=args.tracking_missing_policy,
            device=device,
        )
        l3_tracking_sequences, l3_tracking_available = build_tracking_sequence_batch(
            tracking_store=l3_tracking_store,
            dataset=dataset,
            sample_offset=sample_offset,
            batch_size=batch_size,
            features=features,
            lengths=lengths,
            missing_policy=args.tracking_missing_policy,
            device=device,
        )

        (
            final_preds,
            pred_l1_labels,
            pred_l2_labels,
            pred_l3_labels,
            p_l1,
            p_l2,
            p_l3,
        ) = predict_hierarchical_batch(
            features=features,
            lengths=lengths,
            model_l1=model_l1,
            model_l2=model_l2,
            model_l3=model_l3,
            idx_to_label_l1=idx_to_label_l1,
            idx_to_label_l2=idx_to_label_l2,
            idx_to_label_l3=idx_to_label_l3,
            l1_tracking_sequences=l1_tracking_sequences,
            l2_tracking_sequences=l2_tracking_sequences,
            l3_tracking_sequences=l3_tracking_sequences,
        )

        for i, original_idx in enumerate(labels):
            global_idx = sample_offset + i
            original_label = original_mapping[int(original_idx)]
            true_final = original_to_final_label(original_label)
            pred_final = final_preds[i]

            y_true_final.append(true_final)
            y_pred_final.append(pred_final)

            rows.append(
                {
                    "sample_idx": global_idx,
                    "path": get_sample_path(dataset, global_idx),
                    "original_label": original_label,
                    "true_final": true_final,
                    "pred_l1": pred_l1_labels[i],
                    "pred_l2": pred_l2_labels[i],
                    "pred_l3": pred_l3_labels[i],
                    "pred_final": pred_final,
                    "p_l1": f"{p_l1[i]:.6f}",
                    "p_l2": "" if p_l2[i] is None else f"{p_l2[i]:.6f}",
                    "p_l3": "" if p_l3[i] is None else f"{p_l3[i]:.6f}",
                    "tracking_used_l1": int(l1_tracking_store is not None),
                    "tracking_available_l1": int(l1_tracking_available[i]),
                    "tracking_used_l2": int(l2_tracking_store is not None),
                    "tracking_available_l2": int(l2_tracking_available[i]),
                    "tracking_used_l3": int(l3_tracking_store is not None),
                    "tracking_available_l3": int(l3_tracking_available[i]),
                    "correct": int(true_final == pred_final),
                }
            )

        sample_offset += batch_size

    if not rows:
        raise RuntimeError("Nessun campione valutato: controlla features-root e split.")

    acc = accuracy_score(y_true_final, y_pred_final)
    macro_f1 = f1_score(y_true_final, y_pred_final, labels=FINAL_LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_final, y_pred_final, labels=FINAL_LABELS, average="weighted", zero_division=0)

    print("\n" + "=" * 80)
    print("\n# Valutazione gerarchica end-to-end")
    print(f"Accuracy 8 classi: {acc:.4f}")
    print(f"Macro F1 8 classi: {macro_f1:.4f}")
    print(f"Weighted F1 8 classi: {weighted_f1:.4f}")

    print_report(
        title="Classification report - 8 classi finali con no-action",
        y_true=y_true_final,
        y_pred=y_pred_final,
        labels=FINAL_LABELS,
    )

    print_report(
        title="Classification report - solo 7 azioni finali",
        y_true=y_true_final,
        y_pred=y_pred_final,
        labels=FINAL_ACTION_LABELS,
    )

    # Seconda lettura dei risultati: collassa 0/1 per misurare il riconoscimento
    # del tipo di azione senza penalizzare gli errori sul solo esito del tiro.
    y_true_type = [final_to_type_label(label) for label in y_true_final]
    y_pred_type = [final_to_type_label(label) for label in y_pred_final]

    type_acc = accuracy_score(y_true_type, y_pred_type)
    type_macro_f1 = f1_score(y_true_type, y_pred_type, labels=FINAL_TYPE_LABELS, average="macro", zero_division=0)
    type_weighted_f1 = f1_score(y_true_type, y_pred_type, labels=FINAL_TYPE_LABELS, average="weighted", zero_division=0)

    print("\n" + "=" * 80)
    print("\n# Valutazione collassata senza esito del tiro")
    print(f"Accuracy tipo azione: {type_acc:.4f}")
    print(f"Macro F1 tipo azione: {type_macro_f1:.4f}")
    print(f"Weighted F1 tipo azione: {type_weighted_f1:.4f}")

    print_report(
        title="Classification report - tipo azione senza esito",
        y_true=y_true_type,
        y_pred=y_pred_type,
        labels=FINAL_TYPE_LABELS,
    )

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "predictions.csv"

    # Salva una riga per clip per rendere verificabili errori, routing e tracking.
    with open(predictions_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPredizioni salvate in: {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valutazione gerarchica finale exp_46 con tracking temporale."
    )

    parser.add_argument("--features-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs/exp_46_hierarchical_end_to_end")

    parser.add_argument("--l1-checkpoint", type=str, required=True)
    parser.add_argument("--l2-checkpoint", type=str, required=True)
    parser.add_argument("--l3-checkpoint", type=str, required=True)

    for level in ["l1", "l2", "l3"]:
        parser.add_argument(
            f"--{level}-tracking-sequences-npz",
            type=str,
            default=None,
            help=(
                f"NPZ con sequenze tracking temporali da usare per {level.upper()} "
                "se il checkpoint di quel livello è stato addestrato con tracking temporale."
            ),
        )
        parser.add_argument(
            f"--{level}-tracking-sequence-index",
            type=str,
            default=None,
            help=(
                f"JSON indice associato al file NPZ tracking temporale per {level.upper()}. "
                "Default: tracking_sequence_index.json nella stessa cartella del file NPZ."
            ),
        )

    parser.add_argument(
        "--tracking-missing-policy",
        type=str,
        default="zeros",
        choices=["zeros", "error"],
        help="Comportamento se una clip non ha feature tracking associate.",
    )

    parser.add_argument("--cpu", action="store_true", help="Forza l'esecuzione su CPU.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.txt"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(results_path, "w", encoding="utf-8") as results_file:
        sys.stdout = Tee(original_stdout, results_file)
        sys.stderr = Tee(original_stderr, results_file)

        try:
            print(f"File results.txt: {results_path}")
            print()
            run_evaluation(args)

        except Exception:
            print("\nERRORE DURANTE L'ESECUZIONE:", file=sys.stderr)
            traceback.print_exc()
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Risultati salvati in: {results_path}")


if __name__ == "__main__":
    main()
