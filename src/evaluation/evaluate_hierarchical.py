from pathlib import Path
import argparse
import csv
import shlex
import sys
import traceback

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.data.feature_dataset import FeatureDataset, collate_features, IDX_TO_LABEL
from src.models.temporal_transformer_classifier import TemporalTransformerActionClassifier


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

    # Alcuni checkpoint vecchi potrebbero avere idx_to_label non coerente.
    # In quel caso usiamo il mapping atteso per quello specifico livello.
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
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model, idx_to_label, checkpoint, config


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
):
    logits_l1 = model_l1(features, lengths)
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

        logits_l2 = model_l2(shot_features, shot_lengths)
        probs_l2 = torch.softmax(logits_l2, dim=1)
        preds_l2 = probs_l2.argmax(dim=1)

        logits_l3 = model_l3(shot_features, shot_lengths)
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


def run_evaluation(args):
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

    original_mapping = original_idx_to_label()

    y_true_final = []
    y_pred_final = []
    rows = []

    sample_offset = 0

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].cpu().tolist()

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
                    "correct": int(true_final == pred_final),
                }
            )

        sample_offset += len(labels)

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

    with open(predictions_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPredizioni salvate in: {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs/exp_31_hierarchical_end_to_end")

    parser.add_argument("--l1-checkpoint", type=str, required=True)
    parser.add_argument("--l2-checkpoint", type=str, required=True)
    parser.add_argument("--l3-checkpoint", type=str, required=True)

    parser.add_argument("--cpu", action="store_true", help="Forza l'esecuzione su CPU.")

    return parser.parse_args()


def main():
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
