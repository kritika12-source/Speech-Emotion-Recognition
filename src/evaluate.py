"""Evaluate the saved best checkpoint once on the held-out test actors."""

from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.model import EmotionCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "ravdess_mfcc.npz"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"
BATCH_SIZE = 32


def main() -> None:
    for required_path in (FEATURES_PATH, CHECKPOINT_PATH, OUTPUT_DIR / "normalization.npz"):
        if not required_path.exists():
            raise FileNotFoundError(f"Required artifact not found: {required_path}")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with np.load(FEATURES_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        lengths = data["frame_lengths"].astype(np.int64)
        splits = data["splits"].astype(str)
        filenames = data["filenames"].astype(str)
        actor_ids = data["actor_ids"].astype(np.int64)
        class_names = data["class_names"].astype(str).tolist()
    with np.load(OUTPUT_DIR / "normalization.npz", allow_pickle=False) as stats:
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)

    test_indices = np.flatnonzero(splits == "test")
    if not len(test_indices):
        raise ValueError("No test examples found in the saved feature file")

    test_features = (
        (X[test_indices] - mean[None, :, None]) / std[None, :, None]
    ).astype(np.float32)
    for local_index, original_index in enumerate(test_indices):
        test_features[local_index, :, int(lengths[original_index]) :] = 0.0

    test_dataset = TensorDataset(
        torch.from_numpy(test_features[:, None, :, :]),
        torch.from_numpy(y[test_indices]),
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = EmotionCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()

    actual = []
    predicted = []
    with torch.no_grad():
        for features, targets in test_loader:
            logits = model(features.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            actual.extend(targets.numpy().tolist())

    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    weighted_f1 = f1_score(actual, predicted, average="weighted", zero_division=0)
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))

    metrics = {
        "test_examples": int(len(test_indices)),
        "test_actors": sorted(np.unique(actor_ids[test_indices]).astype(int).tolist()),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    (OUTPUT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "classification_report.csv")
    (OUTPUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(
        {
            "file": filenames[test_indices],
            "actor_id": actor_ids[test_indices],
            "actual": [class_names[index] for index in actual],
            "predicted": [class_names[index] for index in predicted],
            "correct": np.asarray(actual) == np.asarray(predicted),
        }
    ).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("RAVDESS test set: actor-disjoint confusion matrix")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    history_path = OUTPUT_DIR / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(history["epoch"], history["train_loss"], label="Train")
        axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["validation_macro_f1"], color="darkorange")
        axes[1].set_title("Validation macro F1")
        axes[1].set_xlabel("Epoch")
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=180)
        plt.close(figure)

    print(f"Device: {device}")
    print(f"Test examples: {metrics['test_examples']}")
    print(f"Test actors: {metrics['test_actors']}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")
    print("\nPer-class report:")
    print(report_text)
    print(f"Reports and plots saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
