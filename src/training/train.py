from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.data.feature_dataset import FeatureDataset, collate_features, IDX_TO_LABEL, LABEL_TO_IDX
from src.models.gru_classifier import GRUActionClassifier


def compute_class_weights(dataset, num_classes: int = 9, power: float = 0.5) -> torch.Tensor:
    """
    Calcola pesi moderati per gestire lo sbilanciamento delle classi.

    power=0.5 usa 1 / sqrt(freq), meno aggressivo di 1 / freq.
    """
    counts = torch.zeros(num_classes, dtype=torch.float)

    if hasattr(dataset, "items"):
        for item in dataset.items:
            item_path = Path(item)
            label_name = item_path.parent.name

            if label_name not in LABEL_TO_IDX:
                raise ValueError(f"Label non riconosciuta: {label_name}")

            label_idx = LABEL_TO_IDX[label_name]
            counts[label_idx] += 1
    else:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            label_idx = int(sample["label"])
            counts[label_idx] += 1

    weights = 1.0 / torch.pow(counts.clamp(min=1.0), power)
    weights = weights / weights.mean()

    print("Class counts:")
    for idx in range(num_classes):
        print(f"  {IDX_TO_LABEL[idx]}: {int(counts[idx].item())}")

    print("Class weights:")
    for idx in range(num_classes):
        print(f"  {IDX_TO_LABEL[idx]}: {weights[idx].item():.4f}")

    return weights


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

    return avg_loss, acc, macro_f1


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

    return avg_loss, acc, macro_f1, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs/convnext_bigru_attention")
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disattiva i pesi di classe nella CrossEntropyLoss.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = FeatureDataset(args.features_root, split="train")
    val_dataset = FeatureDataset(args.features_root, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_features,
        pin_memory=True,
    )

    model = GRUActionClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=9,
        bidirectional=True,
        dropout=args.dropout,
    ).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        class_weights = None
        print("Class weights disattivati.")
    else:
        class_weights = compute_class_weights(
            train_dataset,
            num_classes=9,
            power=args.class_weight_power,
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_macro_f1 = 0.0
    best_val_labels = None
    best_val_preds = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_loss, val_acc, val_f1, val_labels, val_preds = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} macroF1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} macroF1 {val_f1:.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_val_labels = val_labels
            best_val_preds = val_preds

            checkpoint_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "epoch": epoch,
                    "idx_to_label": IDX_TO_LABEL,
                    "model_config": {
                        "input_dim": args.input_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "num_classes": 9,
                        "bidirectional": True,
                        "dropout": args.dropout,
                    },
                    "class_weights": class_weights.detach().cpu() if class_weights is not None else None,
                },
                checkpoint_path,
            )

            print(f"Salvato nuovo best model: {checkpoint_path}")

    if best_val_labels is None or best_val_preds is None:
        best_val_labels = val_labels
        best_val_preds = val_preds

    print("\nValutazione finale su validation usando il miglior modello salvato:")
    print(f"Best val macro-F1: {best_macro_f1:.4f}")

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


if __name__ == "__main__":
    main()