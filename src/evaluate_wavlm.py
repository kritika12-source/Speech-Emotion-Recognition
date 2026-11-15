"""Evaluate the WavLM head on validation or held-out test actors."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "wavlm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()

    features_path = EXPERIMENT_DIR / "features.npy"
    metadata_path = EXPERIMENT_DIR / "metadata.npz"
    checkpoint_path = EXPERIMENT_DIR / "best_head.pt"
    stats_path = EXPERIMENT_DIR / "feature_normalization.npz"
    for path in (features_path, metadata_path, checkpoint_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"Required WavLM artifact not found: {path}")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = np.load(features_path, mmap_mode="r")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        labels = metadata["labels"].astype(np.int64)
        actor_ids = metadata["actor_ids"].astype(np.int64)
        splits = metadata["splits"].astype(str)
        filenames = metadata["filenames"].astype(str)
        lengths = metadata["frame_lengths"].astype(np.int64)
        class_names = metadata["class_names"].astype(str).tolist()
    with np.load(stats_path, allow_pickle=False) as stats:
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)

    indices = np.flatnonzero(splits == args.split)
    if not len(indices):
        raise ValueError(f"No examples found for split {args.split!r}")
    dataset = FeatureDataset(features, labels, lengths, indices, mean, std)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = WavLMEmotionHead(features.shape[2], len(class_names)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for batch_features, batch_labels, batch_lengths in loader:
            logits = model(batch_features.to(device), batch_lengths)
            actual.extend(batch_labels.numpy().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    report_text = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))
    prefix = args.split
    metrics = {
        "split": args.split,
        "examples": int(len(indices)),
        "actors": sorted(np.unique(actor_ids[indices]).astype(int).tolist()),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    }
    (EXPERIMENT_DIR / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (EXPERIMENT_DIR / f"{prefix}_classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(report).T.to_csv(EXPERIMENT_DIR / f"{prefix}_classification_report.csv")
    pd.DataFrame(
        {
            "file": filenames[indices],
            "actor_id": actor_ids[indices],
            "actual": [class_names[index] for index in actual],
            "predicted": [class_names[index] for index in predicted],
        }
    ).to_csv(EXPERIMENT_DIR / f"{prefix}_predictions.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"WavLM {args.split} set: actor-disjoint confusion matrix")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close()

    print(f"Split: {args.split}")
    print(f"Examples: {metrics['examples']}")
    print(f"Actors: {metrics['actors']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print(report_text)
    print(f"Reports saved under: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
