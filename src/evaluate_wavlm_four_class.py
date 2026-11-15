"""Evaluate a four-class WavLM model on clean/noisy validation or test actors."""

from pathlib import Path
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.train_wavlm import FeatureDataset, collate_batch
from src.wavlm_model import WavLMEmotionHead

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "wavlm"
DEFAULT_EXPERIMENT = ROOT / "outputs" / "wavlm_four_class"


def evaluate_condition(condition, split, experiment_dir, device):
    if condition == "clean":
        features_path = BASE_DIR / "features.npy"
        metadata_path = experiment_dir / "metadata.npz"
    else:
        features_path = BASE_DIR / "noise_augmented_features.npy"
        metadata_path = BASE_DIR / "noise_augmented_metadata.npz"
    for path in (features_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required {condition} data missing: {path}")

    features = np.load(features_path, mmap_mode="r")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        labels = metadata["labels"].astype(np.int64)
        actors = metadata["actor_ids"].astype(np.int64)
        splits = metadata["splits"].astype(str)
        filenames = metadata["filenames"].astype(str)
        lengths = metadata["frame_lengths"].astype(np.int64)
        class_names = metadata["class_names"].astype(str).tolist()
        snr_db = metadata["snr_db"].astype(np.float32) if "snr_db" in metadata else None
    indices = np.flatnonzero(splits == split)
    if not len(indices):
        raise ValueError(f"No {split} examples in {condition} data")

    with np.load(experiment_dir / "feature_normalization.npz", allow_pickle=False) as stats:
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)
    dataset = FeatureDataset(features, labels, lengths, indices, mean, std)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_batch)
    model = WavLMEmotionHead(features.shape[2], len(class_names)).to(device)
    model.load_state_dict(torch.load(experiment_dir / "best_head.pt", map_location=device, weights_only=True))
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for batch_features, batch_labels, batch_lengths in loader:
            logits = model(batch_features.to(device), batch_lengths)
            actual.extend(batch_labels.numpy().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    report_text = classification_report(actual, predicted, labels=list(range(len(class_names))), target_names=class_names, zero_division=0)
    report = classification_report(actual, predicted, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))
    metrics = {
        "split": split,
        "condition": condition,
        "examples": int(len(indices)),
        "actors": sorted(np.unique(actors[indices]).astype(int).tolist()),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    }
    if condition == "noisy" and snr_db is not None:
        metrics["synthetic_snr_db"] = float(np.mean(snr_db[indices]))
    prefix = f"{split}_{condition}"
    (experiment_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (experiment_dir / f"{prefix}_classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(report).T.to_csv(experiment_dir / f"{prefix}_classification_report.csv")
    pd.DataFrame({
        "file": filenames[indices],
        "actor_id": actors[indices],
        "actual": [class_names[index] for index in actual],
        "predicted": [class_names[index] for index in predicted],
        **({"synthetic_snr_db": snr_db[indices]} if snr_db is not None else {}),
    }).to_csv(experiment_dir / f"{prefix}_predictions.csv", index=False)
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Four-class WavLM {split} ({condition})")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    plt.savefig(experiment_dir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close()
    print(f"\n{condition.title()} {split} | examples: {metrics['examples']} | actors: {metrics['actors']}")
    if "synthetic_snr_db" in metrics:
        print(f"Synthetic noise SNR: {metrics['synthetic_snr_db']:.1f} dB")
    print(f"Accuracy: {metrics['accuracy']:.3f} | Macro F1: {metrics['macro_f1']:.3f} | Weighted F1: {metrics['weighted_f1']:.3f}")
    print(report_text)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--condition", choices=("clean", "noisy", "both"), default="clean")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    for path in (args.experiment_dir / "metadata.npz", args.experiment_dir / "best_head.pt", args.experiment_dir / "feature_normalization.npz"):
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact missing: {path}")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = ("clean", "noisy") if args.condition == "both" else (args.condition,)
    for condition in conditions:
        evaluate_condition(condition, args.split, args.experiment_dir, device)
    print(f"Reports saved under: {args.experiment_dir}")


if __name__ == "__main__":
    main()
