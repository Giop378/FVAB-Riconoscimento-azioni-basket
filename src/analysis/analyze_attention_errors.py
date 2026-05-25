from pathlib import Path
import argparse
import csv

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.feature_dataset import FeatureDataset, IDX_TO_LABEL, LABEL_TO_IDX
from src.data.video_io import read_video_frames
from src.models.gru_classifier import GRUActionClassifier


def load_model(
    checkpoint_path: Path,
    device,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GRUActionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=9,
        bidirectional=True,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def analyze_errors(
    model,
    dataset,
    dataset_root: Path,
    output_dir: Path,
    device,
    top_k: int = 5,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "attention_errors.csv"

    rows = []

    for feature_path in tqdm(dataset.items):
        feature_path = Path(feature_path)

        item = torch.load(feature_path, map_location="cpu")

        features = item["features"]          # [T, 768]
        clip_id = item["clip_id"]
        label_name = item["label"]
        rel_video_path = item["path"]

        true_label = LABEL_TO_IDX[label_name]

        features_batch = features.unsqueeze(0).to(device)   # [1, T, 768]
        lengths = torch.tensor([features.shape[0]], device=device)

        logits, attention = model(
            features_batch,
            lengths,
            return_attention=True,
        )

        probs = F.softmax(logits, dim=1)
        pred_label = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_label].item())

        # Analizziamo solo le clip sbagliate
        if pred_label == true_label:
            continue

        attention = attention[0, : lengths.item()].detach().cpu()

        k = min(top_k, attention.numel())
        top_values, top_indices = torch.topk(attention, k=k)

        video_path = dataset_root / rel_video_path

        try:
            frames = read_video_frames(video_path)
        except Exception as e:
            print(f"Errore lettura video {video_path}: {e}")
            continue

        clip_output_dir = frames_dir / (
            f"{clip_id}_true-{label_name}_pred-{IDX_TO_LABEL[pred_label]}"
        )
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        for rank, frame_idx in enumerate(top_indices.tolist(), start=1):
            if frame_idx >= len(frames):
                continue

            attention_value = float(attention[frame_idx].item())

            save_path = clip_output_dir / (
                f"rank{rank}_frame{frame_idx}_att{attention_value:.4f}.jpg"
            )

            frames[frame_idx].save(save_path)

        rows.append(
            {
                "clip_id": clip_id,
                "true_label": label_name,
                "pred_label": IDX_TO_LABEL[pred_label],
                "confidence": confidence,
                "num_frames": int(features.shape[0]),
                "top_attention_frames": ";".join(map(str, top_indices.tolist())),
                "top_attention_values": ";".join(
                    [f"{v:.4f}" for v in top_values.tolist()]
                ),
                "video_path": str(video_path),
                "saved_frames_dir": str(clip_output_dir),
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_id",
            "true_label",
            "pred_label",
            "confidence",
            "num_frames",
            "top_attention_frames",
            "top_attention_values",
            "video_path",
            "saved_frames_dir",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nAnalisi completata.")
    print(f"Clip sbagliate trovate: {len(rows)}")
    print(f"CSV salvato in: {csv_path}")
    print(f"Frame salvati in: {frames_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-root", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-dir", type=str, default="outputs/attention_error_analysis")
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = FeatureDataset(args.features_root, split=args.split)

    model = load_model(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    analyze_errors(
        model=model,
        dataset=dataset,
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        device=device,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()