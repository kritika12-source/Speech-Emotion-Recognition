"""Predict emotions for a WAV audio file using the trained MFCC CNN."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch

from src.model import EmotionCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_RATE = 16_000
N_MFCC = 40
HOP_LENGTH = 160
MAX_FRAMES = 544


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="Path to a WAV audio file")
    args = parser.parse_args()

    checkpoint = OUTPUT_DIR / "best_model.pt"
    stats_path = OUTPUT_DIR / "normalization.npz"
    class_names_path = OUTPUT_DIR / "class_names.json"
    for required_path in (args.audio, checkpoint, stats_path, class_names_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    import json

    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    with np.load(stats_path, allow_pickle=False) as stats:
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)

    waveform, _ = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=HOP_LENGTH,
        n_mels=64,
    ).astype(np.float32)
    frame_count = min(mfcc.shape[1], MAX_FRAMES)
    features = np.zeros((1, N_MFCC, MAX_FRAMES), dtype=np.float32)
    features[0, :, :frame_count] = (
        mfcc[:, :frame_count] - mean[:, None]
    ) / std[:, None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features[:, None, :, :]).to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = np.argsort(probabilities)[::-1][:3]
    print(f"Audio: {args.audio}")
    print("Top predictions (scores are not calibrated probabilities):")
    for index in top_indices:
        print(f"  {class_names[index]:10s} {probabilities[index]:.3f}")


if __name__ == "__main__":
    main()
