"""Train the MFCC CNN using actor-disjoint training and validation sets."""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.model import EmotionCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "ravdess_mfcc.npz"
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 1e-3


def load_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run `python -m src.prepare_data` first."
        )
    with np.load(FEATURES_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        lengths = data["frame_lengths"].astype(np.int64)
        splits = data["splits"].astype(str)
        class_names = data["class_names"].astype(str)
    return X, y, lengths, splits, class_names


def normalize_from_training(X, lengths, train_indices):
    """Fit per-MFCC normalization on valid training frames only."""
    train_values = X[train_indices]
    time_indices = np.arange(X.shape[2])[None, :]
    valid_frames = time_indices < lengths[train_indices, None]
    mask = valid_frames[:, None, :]
    frame_count = int(valid_frames.sum())

    sums = (train_values * mask).sum(axis=(0, 2), dtype=np.float64)
    squared_sums = (np.square(train_values) * mask).sum(axis=(0, 2), dtype=np.float64)
    mean = sums / frame_count
    variance = np.maximum(squared_sums / frame_count - np.square(mean), 1e-8)
    std = np.sqrt(variance)

    normalized = ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    for sample_index, frame_length in enumerate(lengths):
        normalized[sample_index, :, int(frame_length) :] = 0.0
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_items += targets.size(0)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return {
        "loss": total_loss / total_items,
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(all_targets, all_predictions, average="macro", zero_division=0),
    }


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X, y, lengths, splits, class_names = load_features()
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Training or validation split is empty")

    normalized, mean, std = normalize_from_training(X, lengths, train_indices)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "normalization.npz", mean=mean, std=std)
    with (OUTPUT_DIR / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(class_names.tolist(), file, indent=2)

    tensor_dataset = TensorDataset(
        torch.from_numpy(normalized[:, None, :, :]),
        torch.from_numpy(y),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        Subset(tensor_dataset, train_indices.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(tensor_dataset, validation_indices.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    class_counts = np.bincount(y[train_indices], minlength=len(class_names))
    class_weights = len(train_indices) / (len(class_names) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model = EmotionCNN(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_items = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * targets.size(0)
            total_train_items += targets.size(0)

        validation = evaluate(model, validation_loader, criterion, device)
        scheduler.step(validation["loss"])
        train_loss = total_train_loss / total_train_items
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "learning_rate": learning_rate,
            }
        )
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {validation['loss']:.4f} | "
            f"val accuracy {validation['accuracy']:.3f} | "
            f"val macro F1 {validation['macro_f1']:.3f}"
        )

        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping: validation macro F1 did not improve.")
                break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"Best validation macro F1: {best_macro_f1:.3f}")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Training history saved to: {OUTPUT_DIR / 'training_history.csv'}")


if __name__ == "__main__":
    main()
