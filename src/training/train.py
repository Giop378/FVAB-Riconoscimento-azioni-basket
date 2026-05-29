from pathlib import Path
import argparse
import random
import sys
import shlex
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.data.feature_dataset import FeatureDataset, collate_features, IDX_TO_LABEL, LABEL_TO_IDX
from src.models.temporal_transformer_classifier import TemporalTransformerActionClassifier


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_labels_and_counts(dataset, num_classes: int = 9):
    labels = []
    counts = torch.zeros(num_classes, dtype=torch.float)

    if hasattr(dataset, "items"):
        for item in dataset.items:
            item_path = Path(item)
            label_name = item_path.parent.name

            if label_name not in LABEL_TO_IDX:
                raise ValueError(f"Label non riconosciuta: {label_name}")

            label_idx = LABEL_TO_IDX[label_name]
            labels.append(label_idx)
            counts[label_idx] += 1
    else:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            label_idx = int(sample["label"])
            labels.append(label_idx)
            counts[label_idx] += 1

    labels = torch.tensor(labels, dtype=torch.long)
    return labels, counts


def compute_class_weights_from_counts(counts: torch.Tensor, power: float = 0.5):
    weights = 1.0 / torch.pow(counts.clamp(min=1.0), power)
    weights = weights / weights.mean()
    return weights


def build_weighted_sampler(labels: torch.Tensor, counts: torch.Tensor, power: float, seed: int):
    class_sample_weights = 1.0 / torch.pow(counts.clamp(min=1.0), power)
    sample_weights = class_sample_weights[labels]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def print_class_stats(counts, class_weights=None):
    print("Class counts:")
    for idx in range(len(counts)):
        print(f"  {IDX_TO_LABEL[idx]}: {int(counts[idx].item())}")

    if class_weights is not None:
        print("\nClass weights:")
        for idx in range(len(class_weights)):
            print(f"  {IDX_TO_LABEL[idx]}: {class_weights[idx].item():.4f}")


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float = 1.0):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(features, lengths)
        loss = criterion(logits, labels)

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, macro_f1, weighted_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(features, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, macro_f1, weighted_f1, all_labels, all_preds


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/temporal_transformer")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--max-len", type=int, default=1024)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disattiva i pesi di classe nella CrossEntropyLoss.",
    )

    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument(
        "--no-weighted-sampler",
        action="store_true",
        help="Disattiva il WeightedRandomSampler e usa shuffle=True.",
    )

    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)

    return parser.parse_args()


def build_model(args, device):
    model = TemporalTransformerActionClassifier(
        input_dim=args.input_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.ff_dim,
        num_classes=9,
        dropout=args.dropout,
        pooling=args.pooling,
        max_len=args.max_len,
    ).to(device)

    model_config = {
        "model_type": "temporal_transformer",
        "input_dim": args.input_dim,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dim_feedforward": args.ff_dim,
        "num_classes": 9,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "max_len": args.max_len,
    }

    return model, model_config


def run_training(args):
    print("# Comando utilizzato")
    print(get_reconstructed_command())
    print("\n" + "=" * 80 + "\n")

    print("# Configurazione esperimento")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("\n" + "=" * 80 + "\n")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    train_dataset = FeatureDataset(args.features_root, split="train")
    val_dataset = FeatureDataset(args.features_root, split="val")

    train_labels, train_counts = get_dataset_labels_and_counts(
        train_dataset,
        num_classes=9,
    )

    print("\n# Distribuzione classi")

    if args.no_class_weights:
        class_weights = None
        criterion = nn.CrossEntropyLoss()
        print_class_stats(train_counts)
        print("\nWeighted CrossEntropyLoss disattivata.")
    else:
        class_weights_cpu = compute_class_weights_from_counts(
            train_counts,
            power=args.class_weight_power,
        )
        class_weights = class_weights_cpu.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print_class_stats(train_counts, class_weights_cpu)
        print("\nWeighted CrossEntropyLoss attiva.")

    if args.no_weighted_sampler:
        train_sampler = None
        train_shuffle = True
        sampler_enabled = False
        print("\nWeightedRandomSampler disattivato. Uso shuffle=True.")
    else:
        train_sampler = build_weighted_sampler(
            labels=train_labels,
            counts=train_counts,
            power=args.sampler_power,
            seed=args.seed,
        )
        train_shuffle = False
        sampler_enabled = True
        print("\nWeightedRandomSampler attivato.")

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=data_loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=data_loader_generator,
    )

    model, model_config = build_model(args, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    print("\n# Modello")
    print(model)

    print("\n# Training")
    print(
        f"Scheduler: ReduceLROnPlateau(mode='max', "
        f"factor={args.scheduler_factor}, "
        f"patience={args.scheduler_patience}, "
        f"min_lr={args.scheduler_min_lr})"
    )

    best_macro_f1 = -1.0
    best_weighted_f1 = 0.0
    best_val_loss = None
    best_val_acc = None
    best_epoch = None
    best_val_labels = None
    best_val_preds = None

    output_dir = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        current_lr = get_current_lr(optimizer)

        train_loss, train_acc, train_f1, train_weighted_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_loss, val_acc, val_f1, val_weighted_f1, val_labels, val_preds = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"lr {current_lr:.8f} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"macroF1 {train_f1:.4f} weightedF1 {train_weighted_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} "
            f"macroF1 {val_f1:.4f} weightedF1 {val_weighted_f1:.4f}"
        )

        scheduler.step(val_f1)

        new_lr = get_current_lr(optimizer)
        if new_lr != current_lr:
            print(f"Learning rate aggiornato: {current_lr:.8f} -> {new_lr:.8f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_weighted_f1 = val_weighted_f1
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_labels = val_labels
            best_val_preds = val_preds

            checkpoint_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "best_weighted_f1": best_weighted_f1,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "epoch": best_epoch,
                    "idx_to_label": IDX_TO_LABEL,
                    "model_config": model_config,
                    "training_config": vars(args),
                    "class_weights": class_weights.detach().cpu()
                    if class_weights is not None
                    else None,
                    "weighted_sampler": sampler_enabled,
                    "scheduler": {
                        "name": "ReduceLROnPlateau",
                        "mode": "max",
                        "factor": args.scheduler_factor,
                        "patience": args.scheduler_patience,
                        "min_lr": args.scheduler_min_lr,
                    },
                    "seed": args.seed,
                    "command_reconstructed": get_reconstructed_command(),
                    "argv": sys.argv,
                },
                checkpoint_path,
            )

            print(f"Salvato nuovo best model: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("\nValutazione finale su validation usando il miglior modello salvato:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best val macro-F1: {best_macro_f1:.4f}")
    print(f"Best val weighted-F1: {best_weighted_f1:.4f}")

    print("\nClassification report - 9 classi:")
    print(
        classification_report(
            best_val_labels,
            best_val_preds,
            labels=list(range(9)),
            target_names=[IDX_TO_LABEL[i] for i in range(9)],
            zero_division=0,
        )
    )

    print("Confusion matrix - 9 classi:")
    print(
        confusion_matrix(
            best_val_labels,
            best_val_preds,
            labels=list(range(9)),
        )
    )

    print("\nClassification report - solo 7 azioni reali:")
    print(
        classification_report(
            best_val_labels,
            best_val_preds,
            labels=list(range(7)),
            target_names=[IDX_TO_LABEL[i] for i in range(7)],
            zero_division=0,
        )
    )


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
            run_training(args)

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