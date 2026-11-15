"""Report validation performance by emotion without touching the test outputs."""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.model import EmotionCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    features_path = OUTPUT_DIR / "ravdess_mfcc.npz"
    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    stats_path = OUTPUT_DIR / "normalization.npz"
    for path in (features_path, checkpoint_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(features_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        lengths = data["frame_lengths"].astype(np.int64)
        splits = data["splits"].astype(str)
        class_names = data["class_names"].astype(str).tolist()
    with np.load(stats_path, allow_pickle=False) as stats:
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)

    indices = np.flatnonzero(splits == "validation")
    if not len(indices):
        raise ValueError("No validation examples found")
    features = ((X[indices] - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    for local_index, original_index in enumerate(indices):
        features[local_index, :, int(lengths[original_index]) :] = 0.0
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features[:, None, :, :]), torch.from_numpy(y[indices])),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = EmotionCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for batch_features, targets in loader:
            logits = model(batch_features.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            actual.extend(targets.numpy().tolist())

    text_report = classification_report(
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
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "validation_classification_report.csv")
    (OUTPUT_DIR / "validation_classification_report.txt").write_text(text_report, encoding="utf-8")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("RAVDESS validation set: actor-disjoint confusion matrix")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_confusion_matrix.png", dpi=180)
    plt.close()

    print(f"Validation examples: {len(indices)}")
    print(f"Accuracy: {accuracy_score(actual, predicted):.3f}")
    print(f"Macro F1: {f1_score(actual, predicted, average='macro', zero_division=0):.3f}")
    print(text_report)
    print(f"Validation-only reports saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
