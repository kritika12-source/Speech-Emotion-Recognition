"""Predict emotion for one WAV file using a saved WavLM experiment."""

from pathlib import Path
import argparse

from src.wavlm_inference import WavLMPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="Path to a WAV audio file")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/wavlm"),
        help="Directory containing best_head.pt, metadata.npz, and feature_normalization.npz",
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    predictor = WavLMPredictor(args.experiment_dir)
    result = predictor.predict_file(args.audio)
    top_indices = result["top_indices"]
    scores = result["probabilities"]
    print(f"Audio: {args.audio}")
    print(f"Windows analyzed: {result['chunks']}")
    print("Top predictions (scores are not calibrated probabilities):")
    for index in top_indices:
        index = int(index)
        print(f"  {predictor.class_names[index]:10s} {scores[index]:.3f}")


if __name__ == "__main__":
    main()
