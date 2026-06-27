"""
Training dei classificatori temporali per la gerarchia finale di BasketAR.

Lo script addestra un Transformer temporale sulle feature DINOv3 già estratte
per le clip. Supporta solo i tre livelli usati nella configurazione finale:

- L1: action_noaction    -> passaggio / tiro / no-action
- L2: shot_type_only     -> tiroDaDue / tiroDaTre / tiroLibero
- L3: shot_outcome_only  -> tiro0 / tiro1

Per exp_46 usa feature tracking temporali palla/canestro salvate in formato
NPZ + JSON e le concatena frame-per-frame alle feature DINOv3.
"""

from pathlib import Path
import argparse
import random
import shlex
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.feature_dataset import FeatureDataset, collate_features, IDX_TO_LABEL
from src.models.temporal_transformer_classifier import TemporalTransformerActionClassifier
from src.features.tracking_sequence_store import TrackingSequenceFeatureStore


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_original_idx_to_label() -> dict[int, str]:
    return {idx: IDX_TO_LABEL[idx] for idx in range(len(IDX_TO_LABEL))}


def build_label_mapping(label_mode: str):
    """
    Costruisce il mapping tra label originali e label usate da uno specifico
    livello della gerarchia finale.
    """
    if label_mode == "action_noaction":
        idx_to_label = {
            0: "passaggio",
            1: "tiro",
            2: "no-action",
        }
        label_mapping = {
            "passaggio": "passaggio",
            "tiroDaDue0": "tiro",
            "tiroDaDue1": "tiro",
            "tiroDaTre0": "tiro",
            "tiroDaTre1": "tiro",
            "tiroLibero0": "tiro",
            "tiroLibero1": "tiro",
            "idle": "no-action",
            "non-gioco": "no-action",
        }

    elif label_mode == "shot_type_only":
        idx_to_label = {
            0: "tiroDaDue",
            1: "tiroDaTre",
            2: "tiroLibero",
        }
        label_mapping = {
            "tiroDaDue0": "tiroDaDue",
            "tiroDaDue1": "tiroDaDue",
            "tiroDaTre0": "tiroDaTre",
            "tiroDaTre1": "tiroDaTre",
            "tiroLibero0": "tiroLibero",
            "tiroLibero1": "tiroLibero",
        }

    elif label_mode == "shot_outcome_only":
        idx_to_label = {
            0: "tiro0",
            1: "tiro1",
        }
        label_mapping = {
            "tiroDaDue0": "tiro0",
            "tiroDaTre0": "tiro0",
            "tiroLibero0": "tiro0",
            "tiroDaDue1": "tiro1",
            "tiroDaTre1": "tiro1",
            "tiroLibero1": "tiro1",
        }

    else:
        raise ValueError(
            f"Label mode non supportata: {label_mode}. "
            "Sono ammessi solo: action_noaction, shot_type_only, shot_outcome_only."
        )

    return label_mapping, idx_to_label


def get_label_to_idx(idx_to_label: dict[int, str]) -> dict[str, int]:
    return {label: idx for idx, label in idx_to_label.items()}


def normalize_sample_label(label_value) -> int:
    if isinstance(label_value, torch.Tensor):
        return int(label_value.item())
    return int(label_value)


def infer_original_label_from_item(item) -> str:
    """
    Ricava la label originale senza caricare il tensore delle feature.
    Nel FeatureDataset corrente gli item sono path e la label coincide con
    il nome della cartella padre.
    """
    original_idx_to_label = get_original_idx_to_label()

    if isinstance(item, dict):
        if "label" in item:
            label = item["label"]
            if isinstance(label, str):
                return label
            return original_idx_to_label[normalize_sample_label(label)]

        if "path" in item:
            return Path(item["path"]).parent.name

    return Path(item).parent.name


class LabelMappedDataset(Dataset):
    """
    Wrapper di FeatureDataset che rimappa le label originali nel label space
    del livello gerarchico scelto. Per L2 e L3 filtra automaticamente le clip
    non di tiro.
    """

    def __init__(self, base_dataset: Dataset, label_mode: str):
        self.base_dataset = base_dataset
        self.label_mode = label_mode
        self.label_mapping, self.idx_to_label = build_label_mapping(label_mode)
        self.label_to_idx = get_label_to_idx(self.idx_to_label)
        self.original_idx_to_label = get_original_idx_to_label()

        self.indices = []
        self.mapped_labels = []
        self.original_labels = []

        if hasattr(base_dataset, "items"):
            for idx, item in enumerate(base_dataset.items):
                original_label = infer_original_label_from_item(item)
                self._try_add_item(idx, original_label)
        else:
            for idx in range(len(base_dataset)):
                sample = base_dataset[idx]
                if "label" not in sample:
                    raise KeyError("Il sample del dataset non contiene la chiave 'label'.")

                original_label_idx = normalize_sample_label(sample["label"])
                original_label = self.original_idx_to_label[original_label_idx]
                self._try_add_item(idx, original_label)

        if len(self.indices) == 0:
            raise ValueError(
                f"Nessun campione disponibile per label_mode='{label_mode}'. "
                "Controlla mapping, nomi delle cartelle e split del dataset."
            )

    def _try_add_item(self, idx: int, original_label: str) -> None:
        if original_label not in self.label_mapping:
            return

        mapped_label = self.label_mapping[original_label]
        mapped_idx = self.label_to_idx[mapped_label]

        self.indices.append(idx)
        self.mapped_labels.append(mapped_idx)
        self.original_labels.append(original_label)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        sample = dict(self.base_dataset[base_idx])
        sample["label"] = torch.tensor(self.mapped_labels[idx], dtype=torch.long)
        return sample


def get_base_item_path_from_label_dataset(label_dataset, idx: int) -> str:
    """Recupera il path originale associato a un elemento di LabelMappedDataset."""
    if hasattr(label_dataset, "indices") and hasattr(label_dataset, "base_dataset"):
        base_idx = label_dataset.indices[idx]
        base_dataset = label_dataset.base_dataset

        if hasattr(base_dataset, "items"):
            item = base_dataset.items[base_idx]
            if isinstance(item, dict):
                if "path" in item:
                    return str(item["path"])
                return str(item)
            return str(item)

    if hasattr(label_dataset, "base_dataset") and hasattr(label_dataset.base_dataset, "items"):
        item = label_dataset.base_dataset.items[idx]
        if isinstance(item, dict):
            return str(item.get("path", item))
        return str(item)

    return ""


class TemporalTrackingAugmentedDataset(Dataset):
    """
    Concatena a ogni timestep DINOv3 la sequenza tracking palla/canestro
    interpolata alla stessa lunghezza temporale.
    """

    def __init__(
        self,
        label_dataset: Dataset,
        tracking_store: TrackingSequenceFeatureStore,
        missing_policy: str = "zeros",
    ):
        self.label_dataset = label_dataset
        self.tracking_store = tracking_store
        self.missing_policy = missing_policy

        self.missing_count = 0
        for idx in range(len(label_dataset)):
            path = get_base_item_path_from_label_dataset(label_dataset, idx)
            if not tracking_store.has(path):
                self.missing_count += 1

        print("\n# TemporalTrackingAugmentedDataset")
        print(f"Campioni: {len(label_dataset)}")
        print(f"Feature tracking temporali per frame: {tracking_store.num_features}")
        print(f"Campioni senza tracking sequences: {self.missing_count}")
        print(f"Missing policy: {missing_policy}")

    def __len__(self):
        return len(self.label_dataset)

    def __getitem__(self, idx: int):
        sample = dict(self.label_dataset[idx])

        features = sample["features"]
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.float()

        if features.ndim != 2:
            path = get_base_item_path_from_label_dataset(self.label_dataset, idx)
            raise ValueError(
                f"Le feature video devono avere forma [T, D], "
                f"ma per {path} hanno forma {tuple(features.shape)}."
            )

        path = get_base_item_path_from_label_dataset(self.label_dataset, idx)
        tracking_sequence = self.tracking_store.get(
            path,
            target_len=features.shape[0],
            missing_policy=self.missing_policy,
        )
        tracking_tensor = torch.tensor(tracking_sequence, dtype=features.dtype)

        if tracking_tensor.shape[0] != features.shape[0]:
            raise ValueError(
                f"Lunghezza tracking non coerente per {path}: "
                f"features T={features.shape[0]}, tracking T={tracking_tensor.shape[0]}."
            )

        sample["features"] = torch.cat([features, tracking_tensor], dim=1)
        return sample


def get_dataset_labels_and_counts(dataset: Dataset, num_classes: int):
    labels = []
    counts = torch.zeros(num_classes, dtype=torch.float)

    if hasattr(dataset, "mapped_labels"):
        iterable_labels = dataset.mapped_labels
    else:
        iterable_labels = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            iterable_labels.append(normalize_sample_label(sample["label"]))

    for label_idx in iterable_labels:
        label_idx = int(label_idx)
        if label_idx < 0 or label_idx >= num_classes:
            raise ValueError(
                f"Label index fuori range: {label_idx}. "
                f"Numero classi corrente: {num_classes}."
            )
        labels.append(label_idx)
        counts[label_idx] += 1

    return torch.tensor(labels, dtype=torch.long), counts


def compute_class_weights_from_counts(counts: torch.Tensor, power: float = 0.5):
    weights = 1.0 / torch.pow(counts.clamp(min=1.0), power)
    return weights / weights.mean()


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


def print_label_mode_info(label_mode: str, label_mapping, idx_to_label) -> None:
    print("\n# Label mode")
    print(f"Label mode: {label_mode}")

    print("Classi usate nel training:")
    for idx in range(len(idx_to_label)):
        print(f"  {idx}: {idx_to_label[idx]}")

    print("\nMapping label originali -> label usate nel training:")
    for idx in range(len(IDX_TO_LABEL)):
        original_label = IDX_TO_LABEL[idx]
        if original_label in label_mapping:
            print(f"  {original_label} -> {label_mapping[original_label]}")
        else:
            print(f"  {original_label} -> esclusa")


def print_class_stats(counts, idx_to_label, class_weights=None) -> None:
    print("Class counts:")
    for idx in range(len(counts)):
        print(f"  {idx_to_label[idx]}: {int(counts[idx].item())}")

    if class_weights is not None:
        print("\nClass weights:")
        for idx in range(len(class_weights)):
            print(f"  {idx_to_label[idx]}: {class_weights[idx].item():.4f}")


def get_current_lr(optimizer) -> float:
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
    parser = argparse.ArgumentParser(
        description="Training dei livelli gerarchici finali con feature DINOv3 e tracking temporale."
    )

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
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "last_mean"])
    parser.add_argument(
        "--last-mean-ratio",
        type=float,
        default=0.30,
        help="Percentuale finale della clip da usare con --pooling last_mean.",
    )
    parser.add_argument("--max-len", type=int, default=1024)

    parser.add_argument(
        "--label-mode",
        type=str,
        required=True,
        choices=["action_noaction", "shot_type_only", "shot_outcome_only"],
        help="Livello gerarchico da addestrare.",
    )

    parser.add_argument(
        "--tracking-sequences-npz",
        type=str,
        default=None,
        help=(
            "File NPZ con sequenze temporali di tracking palla/canestro. "
            "Se indicato, ogni sequenza [S, K] viene interpolata a [T, K] "
            "e concatenata alle feature DINOv3 frame-per-frame."
        ),
    )
    parser.add_argument(
        "--tracking-sequence-index",
        type=str,
        default=None,
        help=(
            "JSON indice associato a --tracking-sequences-npz. "
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
    parser.add_argument(
        "--no-normalize-tracking-features",
        action="store_true",
        help="Disattiva la normalizzazione z-score del tracking calcolata sul train set.",
    )

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


def build_model(args, device, num_classes: int, tracking_dim: int = 0, tracking_config=None):
    input_dim = int(args.input_dim)
    actual_input_dim = input_dim + int(tracking_dim)

    model = TemporalTransformerActionClassifier(
        input_dim=actual_input_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.ff_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        pooling=args.pooling,
        max_len=args.max_len,
        last_mean_ratio=args.last_mean_ratio,
    ).to(device)

    model_config = {
        "model_type": "temporal_transformer",
        "input_dim": actual_input_dim,
        "base_input_dim": input_dim,
        "tracking_input_dim": int(tracking_dim),
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dim_feedforward": args.ff_dim,
        "num_classes": num_classes,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "last_mean_ratio": args.last_mean_ratio,
        "max_len": args.max_len,
        "tracking_config": tracking_config,
    }

    return model, model_config


def run_training(args) -> None:
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

    label_mapping, idx_to_label = build_label_mapping(args.label_mode)
    label_to_idx = get_label_to_idx(idx_to_label)
    num_classes = len(idx_to_label)

    print_label_mode_info(args.label_mode, label_mapping, idx_to_label)

    base_train_dataset = FeatureDataset(args.features_root, split="train")
    base_val_dataset = FeatureDataset(args.features_root, split="val")

    train_dataset = LabelMappedDataset(base_train_dataset, args.label_mode)
    val_dataset = LabelMappedDataset(base_val_dataset, args.label_mode)

    print("\n# Dataset")
    print(f"Train samples originali: {len(base_train_dataset)}")
    print(f"Train samples usati: {len(train_dataset)}")
    print(f"Val samples originali: {len(base_val_dataset)}")
    print(f"Val samples usati: {len(val_dataset)}")

    train_labels, train_counts = get_dataset_labels_and_counts(train_dataset, num_classes=num_classes)

    print("\n# Distribuzione classi")
    if args.no_class_weights:
        class_weights = None
        criterion = nn.CrossEntropyLoss()
        print_class_stats(train_counts, idx_to_label)
        print("\nWeighted CrossEntropyLoss disattivata.")
    else:
        class_weights_cpu = compute_class_weights_from_counts(
            train_counts,
            power=args.class_weight_power,
        )
        class_weights = class_weights_cpu.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print_class_stats(train_counts, idx_to_label, class_weights_cpu)
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

    tracking_dim = 0
    tracking_config = None

    if args.tracking_sequences_npz is not None:
        print("\n# Feature tracking palla/canestro temporali")
        tracking_store = TrackingSequenceFeatureStore(
            args.tracking_sequences_npz,
            index_path=args.tracking_sequence_index,
        )

        if args.no_normalize_tracking_features:
            print("Normalizzazione sequenze tracking disattivata.")
        else:
            train_tracking_paths = [
                get_base_item_path_from_label_dataset(train_dataset, idx)
                for idx in range(len(train_dataset))
            ]
            tracking_store.fit_normalizer_from_paths(train_tracking_paths)

        tracking_dim = tracking_store.num_features
        tracking_config = tracking_store.get_config()

        train_dataset = TemporalTrackingAugmentedDataset(
            train_dataset,
            tracking_store,
            missing_policy=args.tracking_missing_policy,
        )
        val_dataset = TemporalTrackingAugmentedDataset(
            val_dataset,
            tracking_store,
            missing_policy=args.tracking_missing_policy,
        )

        print(f"Input dim feature video: {args.input_dim}")
        print(f"Input dim feature tracking temporali: {tracking_dim}")
        print(f"Input dim totale modello: {args.input_dim + tracking_dim}")
    else:
        print("\n# Feature tracking palla/canestro")
        print("Feature tracking non usate.")

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

    model, model_config = build_model(
        args,
        device,
        num_classes=num_classes,
        tracking_dim=tracking_dim,
        tracking_config=tracking_config,
    )

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
                    "idx_to_label": idx_to_label,
                    "label_to_idx": label_to_idx,
                    "label_mode": args.label_mode,
                    "label_mapping": label_mapping,
                    "original_idx_to_label": get_original_idx_to_label(),
                    "model_config": model_config,
                    "tracking_config": tracking_config,
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

    if best_val_labels is None or best_val_preds is None:
        raise RuntimeError("Training completato senza salvare alcun best model.")

    print("\n" + "=" * 80)
    print("\nValutazione finale su validation usando il miglior modello salvato:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best val macro-F1: {best_macro_f1:.4f}")
    print(f"Best val weighted-F1: {best_weighted_f1:.4f}")

    labels = list(range(num_classes))
    target_names = [idx_to_label[i] for i in labels]

    print(f"\nClassification report - {num_classes} classi ({args.label_mode}):")
    print(
        classification_report(
            best_val_labels,
            best_val_preds,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    print(f"Confusion matrix - {num_classes} classi ({args.label_mode}):")
    print(
        confusion_matrix(
            best_val_labels,
            best_val_preds,
            labels=labels,
        )
    )


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
