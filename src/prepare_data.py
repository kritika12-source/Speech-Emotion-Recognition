"""Create MFCC features and reproducible actor-disjoint RAVDESS splits."""

from io import BytesIO
from pathlib import Path
import re

import librosa
import numpy as np
import pandas as pd
from datasets import Audio, load_dataset

DATASET_ID = "AbstractTTS/RAVDESS"
EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
LABEL_TO_ID = {emotion: index for index, emotion in enumerate(EMOTIONS)}
ACTORS_BY_SPLIT = {
    "train": set(range(1, 17)),
    "validation": set(range(17, 21)),
    "test": set(range(21, 25)),
}
SAMPLE_RATE = 16_000
N_MFCC = 40
HOP_LENGTH = 160  # 10 ms at 16 kHz
MAX_FRAMES = 544  # 5.44 seconds; covers the clips in this dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def get_actor_id(filename: str) -> int:
    match = re.search(r"Actor(\d+)", filename)
    if not match:
        raise ValueError(f"Could not parse actor ID from filename: {filename}")
    return int(match.group(1))


def get_split(actor_id: int) -> str:
    for split_name, actor_ids in ACTORS_BY_SPLIT.items():
        if actor_id in actor_ids:
            return split_name
    raise ValueError(f"Actor ID {actor_id} is not assigned to a split")


def main() -> None:
    print(f"Loading {DATASET_ID} from Hugging Face...")
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    required_columns = {"file", "audio", "emotion"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    count = len(dataset)
    features = np.zeros((count, N_MFCC, MAX_FRAMES), dtype=np.float32)
    labels = np.zeros(count, dtype=np.int64)
    actor_ids = np.zeros(count, dtype=np.int64)
    frame_lengths = np.zeros(count, dtype=np.int64)
    filenames: list[str] = []
    splits: list[str] = []
    manifest_rows: list[dict[str, object]] = []

    for index, row in enumerate(dataset):
        filename = row["file"]
        emotion = row["emotion"]
        actor_id = get_actor_id(filename)
        split_name = get_split(actor_id)
        audio_bytes = row["audio"].get("bytes")
        if audio_bytes is None:
            raise ValueError(f"No audio bytes found for {filename}")
        if emotion not in LABEL_TO_ID:
            raise ValueError(f"Unexpected emotion label {emotion!r} in {filename}")

        waveform, _ = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=512,
            hop_length=HOP_LENGTH,
            n_mels=64,
        ).astype(np.float32)
        frame_count = min(mfcc.shape[1], MAX_FRAMES)
        features[index, :, :frame_count] = mfcc[:, :frame_count]
        labels[index] = LABEL_TO_ID[emotion]
        actor_ids[index] = actor_id
        frame_lengths[index] = frame_count
        filenames.append(filename)
        splits.append(split_name)
        manifest_rows.append(
            {"file": filename, "actor_id": actor_id, "emotion": emotion, "split": split_name}
        )

        if (index + 1) % 200 == 0 or index + 1 == count:
            print(f"Processed {index + 1}/{count} clips")

    manifest = pd.DataFrame(manifest_rows)
    actor_sets = {
        split_name: set(manifest.loc[manifest["split"] == split_name, "actor_id"])
        for split_name in ACTORS_BY_SPLIT
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if actor_sets[first] & actor_sets[second]:
            raise RuntimeError(f"Actor leakage between {first} and {second} splits")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)
    pd.crosstab(manifest["split"], manifest["emotion"]).reindex(
        index=["train", "validation", "test"], columns=EMOTIONS, fill_value=0
    ).to_csv(OUTPUT_DIR / "split_emotion_counts.csv")
    np.savez_compressed(
        OUTPUT_DIR / "ravdess_mfcc.npz",
        X=features,
        y=labels,
        actor_ids=actor_ids,
        frame_lengths=frame_lengths,
        filenames=np.asarray(filenames),
        splits=np.asarray(splits),
        class_names=np.asarray(EMOTIONS),
        sample_rate=np.asarray(SAMPLE_RATE),
        hop_length=np.asarray(HOP_LENGTH),
    )

    print("\nExamples by split:")
    print(manifest.groupby("split").size().reindex(["train", "validation", "test"]).to_string())
    print("\nEmotion counts by split:")
    print(pd.crosstab(manifest["split"], manifest["emotion"]).reindex(
        index=["train", "validation", "test"], columns=EMOTIONS, fill_value=0
    ).to_string())
    print(f"\nSaved features and split metadata under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
