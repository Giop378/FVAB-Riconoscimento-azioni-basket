from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.data.feature_dataset import FeatureDataset, collate_features, IDX_TO_LABEL
from src.models.gru_classifier import GRUActionClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
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
    parser.add_argument("--output-dir", type=str, default="outputs/gru_baseline")
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
        num_workers=2,
        collate_fn=collate_features,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_features,
        pin_memory=True,
    )

    model = GRUActionClassifier(
        input_dim=512,
        hidden_dim=256,
        num_layers=1,
        num_classes=9,
        bidirectional=True,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_macro_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1, val_labels, val_preds = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} macroF1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} macroF1 {val_f1:.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            checkpoint_path = output_dir / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "epoch": epoch,
                    "idx_to_label": IDX_TO_LABEL,
                },
                checkpoint_path,
            )

            print(f"Salvato nuovo best model: {checkpoint_path}")

    print("\nValutazione finale su validation:")
    print(
        classification_report(
            val_labels,
            val_preds,
            target_names=[IDX_TO_LABEL[i] for i in range(9)],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(val_labels, val_preds))


if __name__ == "__main__":
    main()