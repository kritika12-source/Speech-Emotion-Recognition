"""Train a four-class WavLM head using the already cached RAVDESS features."""

from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from src.train_wavlm import FeatureDataset, collate_batch, run_epoch
from src.wavlm_model import WavLMEmotionHead

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "outputs" / "wavlm"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "wavlm_four_class"
FEATURES_PATH = BASE_DIR / "features.npy"
METADATA_PATH = BASE_DIR / "metadata.npz"
CLASS_NAMES = ["angry", "happy", "sad", "neutral"]
LABEL_MAP = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "calm": "neutral",
}
SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 40
PATIENCE = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise-augmented",
        action="store_true",
        help="mix waveform-noise-augmented features into training",
    )
    args = parser.parse_args()
    experiment_dir = EXPERIMENT_DIR / "noise_augmented" if args.noise_augmented else EXPERIMENT_DIR
    if not FEATURES_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.extract_wavlm_features` first")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "clean + noisy waveform augmentation" if args.noise_augmented else "clean audio only"
    print(f"Training four-class WavLM head on: {device} ({mode})")

    features = np.load(FEATURES_PATH, mmap_mode="r")
    with np.load(METADATA_PATH, allow_pickle=False) as source:
        original_labels = source["labels"].astype(np.int64)
        actor_ids = source["actor_ids"].astype(np.int64)
        original_splits = source["splits"].astype(str)
        filenames = source["filenames"].astype(str)
        lengths = source["frame_lengths"].astype(np.int64)
        original_classes = source["class_names"].astype(str).tolist()

    old_id_to_name = {index: name for index, name in enumerate(original_classes)}
    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = np.full_like(original_labels, fill_value=-1)
    selected = np.zeros(len(original_labels), dtype=bool)
    for row_index, old_label in enumerate(original_labels):
        mapped_name = LABEL_MAP.get(old_id_to_name[int(old_label)])
        if mapped_name is not None:
            labels[row_index] = class_to_id[mapped_name]
            selected[row_index] = True
    split_names = np.where(selected, original_splits, "excluded")
    train_indices = np.flatnonzero(split_names == "train")
    validation_indices = np.flatnonzero(split_names == "validation")
    test_indices = np.flatnonzero(split_names == "test")

    experiment_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        experiment_dir / "metadata.npz",
        labels=labels,
        actor_ids=actor_ids,
        splits=split_names,
        filenames=filenames,
        frame_lengths=lengths,
        class_names=np.asarray(CLASS_NAMES),
        source_model_id=np.asarray("microsoft/wavlm-base-plus"),
    )
    training_parts = [(features, lengths, train_indices, labels)]
    if args.noise_augmented:
        noisy_features_path = BASE_DIR / "noise_augmented_features.npy"
        noisy_metadata_path = BASE_DIR / "noise_augmented_metadata.npz"
        if not noisy_features_path.exists() or not noisy_metadata_path.exists():
            raise FileNotFoundError(
                "Build the noise feature cache first: python -m src.extract_noise_augmented_features"
            )
        noisy_features = np.load(noisy_features_path, mmap_mode="r")
        with np.load(noisy_metadata_path, allow_pickle=False) as noisy_metadata:
            noisy_labels = noisy_metadata["labels"].astype(np.int64)
            noisy_splits = noisy_metadata["splits"].astype(str)
            noisy_lengths = noisy_metadata["frame_lengths"].astype(np.int64)
        noisy_train_indices = np.flatnonzero(noisy_splits == "train")
        if not np.array_equal(noisy_labels[noisy_train_indices], labels[train_indices]):
            raise ValueError("Noisy feature labels/order do not match clean training data.")
        training_parts.append((noisy_features, noisy_lengths, noisy_train_indices, noisy_labels))

    sums = np.zeros(features.shape[2], dtype=np.float64)
    squared_sums = np.zeros(features.shape[2], dtype=np.float64)
    frame_count = 0
    for part_features, part_lengths, part_indices, _ in training_parts:
        for index in part_indices:
            length = int(part_lengths[index])
            values = np.asarray(part_features[index, :length], dtype=np.float32)
            sums += values.sum(axis=0, dtype=np.float64)
            squared_sums += np.square(values).sum(axis=0, dtype=np.float64)
            frame_count += length
    mean = sums / frame_count
    variance = np.maximum(squared_sums / frame_count - np.square(mean), 1e-8)
    mean, std = mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
    np.savez(experiment_dir / "feature_normalization.npz", mean=mean, std=std)

    train_data = ConcatDataset([
        FeatureDataset(part_features, part_labels, part_lengths, part_indices, mean, std)
        for part_features, part_lengths, part_indices, part_labels in training_parts
    ])
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

    class_counts = np.bincount(labels[train_indices], minlength=len(CLASS_NAMES))
    class_weights = len(train_indices) / (len(CLASS_NAMES) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model = WavLMEmotionHead(features.shape[2], len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_macro_f1 = -1.0
    stale_epochs = 0
    history = []
    checkpoint_path = experiment_dir / "best_head.pt"
    history_path = experiment_dir / "training_history.csv"
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        validation_metrics = run_epoch(model, validation_loader, criterion, optimizer, device, training=False)
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
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss {row['train_loss']:.4f} | "
            f"val accuracy {row['validation_accuracy']:.3f} | "
            f"val macro F1 {row['validation_macro_f1']:.3f}"
        )
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("Early stopping on validation macro F1.")
                break

    print(f"Selected {int(selected.sum())} clips across four classes.")
    print(f"Split sizes: train={len(train_indices)}, validation={len(validation_indices)}, test={len(test_indices)}")
    print(f"Best validation macro F1: {best_macro_f1:.3f}")
    print(f"Best checkpoint: {checkpoint_path}")
    if args.noise_augmented:
        print("Model was trained with synthetic waveform noise; compare clean and noisy held-out metrics.")
    print("The held-out test split has not been evaluated by this script.")


if __name__ == "__main__":
    main()
