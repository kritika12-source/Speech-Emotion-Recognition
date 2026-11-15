"""Create waveform-noise-augmented WavLM features for robustness training."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

from src.extract_wavlm_features import DATASET_ID, MODEL_ID, SAMPLE_RATE, feature_length


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "wavlm"
FOUR_CLASS_DIR = ROOT / "outputs" / "wavlm_four_class"
OUTPUT_FEATURES = BASE_DIR / "noise_augmented_features.npy"
OUTPUT_METADATA = BASE_DIR / "noise_augmented_metadata.npz"
SNR_RANGE_DB = (5.0, 20.0)
VALIDATION_TEST_SNR_DB = 10.0


def make_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.choice(("white", "pink", "hum"))
    if kind == "white":
        noise = rng.normal(size=length)
    elif kind == "pink":
        white = rng.normal(size=length)
        spectrum = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(length, d=1 / SAMPLE_RATE)
        spectrum /= np.sqrt(np.maximum(frequencies, 1.0))
        noise = np.fft.irfft(spectrum, n=length)
    else:
        time = np.arange(length, dtype=np.float32) / SAMPLE_RATE
        base = float(rng.choice((50.0, 60.0)))
        noise = sum(
            (1.0 / harmonic) * np.sin(2 * np.pi * base * harmonic * time + rng.uniform(0, 2 * np.pi))
            for harmonic in (1, 2, 3)
        )
        noise += 0.2 * rng.normal(size=length)
    noise = np.asarray(noise, dtype=np.float32)
    noise -= noise.mean()
    noise /= max(float(np.sqrt(np.mean(noise**2))), 1e-8)
    return noise


def add_noise(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    speech = np.asarray(waveform, dtype=np.float32)
    noise = make_noise(len(speech), rng)
    speech_rms = float(np.sqrt(np.mean(speech**2)))
    if speech_rms < 1e-7:
        return speech
    noise *= speech_rms / (10 ** (snr_db / 20.0))
    augmented = speech + noise
    peak = float(np.max(np.abs(augmented)))
    if peak > 1.0:
        augmented = augmented / peak * 0.98
    return augmented.astype(np.float32)


def main() -> None:
    if not FOUR_CLASS_DIR.joinpath("metadata.npz").exists():
        raise FileNotFoundError("Train the four-class experiment once before building noise features.")
    rng = np.random.default_rng(2026)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    with np.load(FOUR_CLASS_DIR / "metadata.npz", allow_pickle=False) as metadata:
        names = metadata["filenames"].astype(str)
        labels = metadata["labels"].astype(np.int64)
        actors = metadata["actor_ids"].astype(np.int64)
        splits = metadata["splits"].astype(str)
        class_names = metadata["class_names"].astype(str)
    name_to_row = {name: index for index, name in enumerate(names)}
    selected_rows = np.flatnonzero(np.isin(splits, ("train", "validation", "test")))
    max_frames = feature_length(int(SAMPLE_RATE * 5.44), encoder.config)
    feature_path = np.lib.format.open_memmap(
        OUTPUT_FEATURES,
        mode="w+",
        dtype=np.float16,
        shape=(len(selected_rows), max_frames, encoder.config.hidden_size),
    )
    output_labels = np.empty(len(selected_rows), dtype=np.int64)
    output_actors = np.empty(len(selected_rows), dtype=np.int64)
    output_splits = np.empty(len(selected_rows), dtype="U10")
    output_names = np.empty(len(selected_rows), dtype="U80")
    output_lengths = np.zeros(len(selected_rows), dtype=np.int32)
    snrs = np.empty(len(selected_rows), dtype=np.float32)
    dataset = load_dataset(DATASET_ID, split="train").cast_column("audio", Audio(decode=False))
    processed = 0

    for row in dataset:
        filename = row["file"]
        metadata_index = name_to_row.get(filename)
        if metadata_index is None or splits[metadata_index] == "excluded":
            continue
        audio_bytes = row["audio"].get("bytes")
        if audio_bytes is None:
            continue
        waveform, _ = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        waveform = waveform[: int(SAMPLE_RATE * 5.44)]
        split = splits[metadata_index]
        snr_db = float(rng.uniform(*SNR_RANGE_DB)) if split == "train" else VALIDATION_TEST_SNR_DB
        noisy = add_noise(waveform, snr_db, rng)
        inputs = extractor(noisy, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = encoder(**inputs).last_hidden_state[0]
        length = min(hidden.shape[0], max_frames)
        feature_path[processed, :length] = hidden[:length].cpu().numpy().astype(np.float16)
        output_labels[processed] = labels[metadata_index]
        output_actors[processed] = actors[metadata_index]
        output_splits[processed] = split
        output_names[processed] = filename
        output_lengths[processed] = length
        snrs[processed] = snr_db
        processed += 1
        if processed % 50 == 0 or processed == len(selected_rows):
            feature_path.flush()
            print(f"Encoded noisy clips: {processed}/{len(selected_rows)}")

    if processed != len(selected_rows):
        raise RuntimeError(f"Expected {len(selected_rows)} selected clips, encoded {processed}.")
    np.savez_compressed(
        OUTPUT_METADATA,
        labels=output_labels,
        actor_ids=output_actors,
        splits=output_splits,
        filenames=output_names,
        frame_lengths=output_lengths,
        class_names=class_names,
        snr_db=snrs,
        noise_types=np.asarray("synthetic white/pink/hum mixture"),
    )
    print(f"Saved {processed} noise-augmented feature sequences to {OUTPUT_FEATURES}")
    print("Training clips have randomized 5–20 dB SNR; validation/test clips use synthetic 10 dB SNR.")


if __name__ == "__main__":
    main()
