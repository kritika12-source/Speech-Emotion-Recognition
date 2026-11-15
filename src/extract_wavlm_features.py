"""Extract frozen WavLM Base+ features for actor-disjoint RAVDESS clips."""

from io import BytesIO
from pathlib import Path
import re

import librosa
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

DATASET_ID = "AbstractTTS/RAVDESS"
MODEL_ID = "microsoft/wavlm-base-plus"
EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
LABEL_TO_ID = {emotion: index for index, emotion in enumerate(EMOTIONS)}
ACTORS_BY_SPLIT = {
    "train": set(range(1, 17)),
    "validation": set(range(17, 21)),
    "test": set(range(21, 25)),
}
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 5.44
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "wavlm"


def actor_and_split(filename: str) -> tuple[int, str]:
    match = re.search(r"Actor(\d+)", filename)
    if not match:
        raise ValueError(f"Could not parse actor ID from filename: {filename}")
    actor_id = int(match.group(1))
    for split_name, actor_ids in ACTORS_BY_SPLIT.items():
        if actor_id in actor_ids:
            return actor_id, split_name
    raise ValueError(f"Actor ID {actor_id} is not assigned to a split")


def feature_length(input_length: int, config) -> int:
    length = input_length
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        length = (length - kernel) // stride + 1
    return length


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading pretrained encoder: {MODEL_ID}")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    print(f"Loading {DATASET_ID} from Hugging Face...")
    dataset = load_dataset(DATASET_ID, split="train").cast_column("audio", Audio(decode=False))
    count = len(dataset)
    hidden_size = encoder.config.hidden_size
    max_frames = feature_length(MAX_AUDIO_SAMPLES, encoder.config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = OUTPUT_DIR / "features.npy"
    feature_array = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(count, max_frames, hidden_size),
    )
    frame_lengths = np.zeros(count, dtype=np.int32)
    labels = np.zeros(count, dtype=np.int64)
    actor_ids = np.zeros(count, dtype=np.int64)
    splits: list[str] = []
    filenames: list[str] = []

    for index, row in enumerate(dataset):
        filename = row["file"]
        emotion = row["emotion"]
        actor_id, split_name = actor_and_split(filename)
        audio_bytes = row["audio"].get("bytes")
        if audio_bytes is None:
            raise ValueError(f"No audio bytes found for {filename}")
        if emotion not in LABEL_TO_ID:
            raise ValueError(f"Unexpected emotion label {emotion!r} in {filename}")

        waveform, _ = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        waveform = waveform[:MAX_AUDIO_SAMPLES]
        inputs = extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = encoder(**inputs).last_hidden_state[0]

        valid_frames = min(hidden.shape[0], max_frames)
        feature_array[index, :valid_frames] = hidden[:valid_frames].cpu().numpy().astype(np.float16)
        frame_lengths[index] = valid_frames
        labels[index] = LABEL_TO_ID[emotion]
        actor_ids[index] = actor_id
        splits.append(split_name)
        filenames.append(filename)

        if (index + 1) % 50 == 0 or index + 1 == count:
            feature_array.flush()
            print(f"Extracted {index + 1}/{count} clips")

    feature_array.flush()
    np.savez_compressed(
        OUTPUT_DIR / "metadata.npz",
        labels=labels,
        actor_ids=actor_ids,
        splits=np.asarray(splits),
        filenames=np.asarray(filenames),
        frame_lengths=frame_lengths,
        class_names=np.asarray(EMOTIONS),
        model_id=np.asarray(MODEL_ID),
        sample_rate=np.asarray(SAMPLE_RATE),
    )
    print(f"Saved frozen WavLM features to: {feature_path}")
    print(f"Saved metadata to: {OUTPUT_DIR / 'metadata.npz'}")
    print("Feature cache may use several hundred MB of disk space.")


if __name__ == "__main__":
    main()
