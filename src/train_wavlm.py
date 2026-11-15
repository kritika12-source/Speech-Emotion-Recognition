"""Train the BiLSTM-attention head on cached frozen WavLM features."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.wavlm_model import WavLMEmotionHead

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "wavlm"
FEATURES_PATH = EXPERIMENT_DIR / "features.npy"
METADATA_PATH = EXPERIMENT_DIR / "metadata.npz"
SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 9
LEARNING_RATE = 5e-4


class FeatureDataset(Dataset):
    def __init__(self, features, labels, lengths, indices, mean, std):
        self.features = features
        self.labels = labels
        self.lengths = lengths
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        length = int(self.lengths[index])
        values = np.array(self.features[index, :length], dtype=np.float32, copy=True)
        values = (values - self.mean[None, :]) / self.std[None, :]
        return torch.from_numpy(values), int(self.labels[index]), length


def collate_batch(rows):
    lengths = torch.tensor([row[2] for row in rows], dtype=torch.long)
    max_length = int(lengths.max())
    feature_size = rows[0][0].shape[1]
    features = torch.zeros(len(rows), max_length, feature_size, dtype=torch.float32)
    labels = torch.empty(len(rows), dtype=torch.long)
    for index, (values, label, length) in enumerate(rows):
        features[index, :length] = values
        labels[index] = label
    return features, labels, lengths


def fit_normalization(features, lengths, train_indices):
    sums = None
    squared_sums = None
    frame_count = 0
    for index in train_indices:
        length = int(lengths[index])
        values = np.asarray(features[index, :length], dtype=np.float32)
        local_sum = values.sum(axis=0, dtype=np.float64)
        local_squared_sum = np.square(values).sum(axis=0, dtype=np.float64)
        if sums is None:
            sums = local_sum
            squared_sums = local_squared_sum
        else:
            sums += local_sum
            squared_sums += local_squared_sum
        frame_count += length
    mean = sums / frame_count
    variance = np.maximum(squared_sums / frame_count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    item_count = 0
    targets_all = []
    predictions_all = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels, lengths in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features, lengths)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            item_count += labels.size(0)
            targets_all.extend(labels.detach().cpu().numpy().tolist())
            predictions_all.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return {
        "loss": total_loss / item_count,
        "accuracy": accuracy_score(targets_all, predictions_all),
        "macro_f1": f1_score(targets_all, predictions_all, average="macro", zero_division=0),
    }


def main() -> None:
    if not FEATURES_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.extract_wavlm_features` first")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training BiLSTM-attention head on: {device}")

    features = np.load(FEATURES_PATH, mmap_mode="r")
    with np.load(METADATA_PATH, allow_pickle=False) as metadata:
        labels = metadata["labels"].astype(np.int64)
        lengths = metadata["frame_lengths"].astype(np.int64)
        splits = metadata["splits"].astype(str)
        class_names = metadata["class_names"].astype(str)
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Training or validation split is empty")

    mean, std = fit_normalization(features, lengths, train_indices)
    np.savez(EXPERIMENT_DIR / "feature_normalization.npz", mean=mean, std=std)
    train_data = FeatureDataset(features, labels, lengths, train_indices, mean, std)
    validation_data = FeatureDataset(features, labels, lengths, validation_indices, mean, std)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_batch,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    class_counts = np.bincount(labels[train_indices], minlength=len(class_names))
    class_weights = len(train_indices) / (len(class_names) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model = WavLMEmotionHead(input_size=features.shape[2], num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_macro_f1 = -1.0
    stale_epochs = 0
    history = []
    checkpoint_path = EXPERIMENT_DIR / "best_head.pt"
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        validation_metrics = run_epoch(
            model, validation_loader, criterion, optimizer, device, training=False
        )
        scheduler.step(validation_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss {row['train_loss']:.4f} | "
            f"val loss {row['validation_loss']:.4f} | "
            f"val accuracy {row['validation_accuracy']:.3f} | "
            f"val macro F1 {row['validation_macro_f1']:.3f}"
        )
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print("Early stopping on validation macro F1.")
                break

    pd.DataFrame(history).to_csv(EXPERIMENT_DIR / "training_history.csv", index=False)
    print(f"Best validation macro F1: {best_macro_f1:.3f}")
    print(f"Best head saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
