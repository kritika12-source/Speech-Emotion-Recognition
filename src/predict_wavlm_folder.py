"""Batch-predict emotion for WAV files in a folder and save one CSV."""

from pathlib import Path
import argparse

import librosa
import pandas as pd

from src.wavlm_inference import WavLMPredictor, SAMPLE_RATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing WAV files")
    parser.add_argument("--output", type=Path, help="Destination CSV path")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/wavlm"),
        help="Directory containing a trained WavLM experiment",
    )
    parser.add_argument("--recursive", action="store_true", help="Search subfolders too")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input folder not found: {args.input_dir}")

    pattern = "**/*.wav" if args.recursive else "*.wav"
    audio_files = sorted(path for path in args.input_dir.glob(pattern) if path.is_file())
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {args.input_dir}")

    predictor = WavLMPredictor(args.experiment_dir)
    rows = []
    for index, audio_path in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] {audio_path.name}")
        try:
            waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            result = predictor.predict_waveform(waveform)
            top_indices = result["top_indices"]
            scores = result["probabilities"]
            rows.append(
                {
                    "file": str(audio_path),
                    "prediction": result["prediction"],
                    "score": result["score"],
                    "top_2": predictor.class_names[int(top_indices[1])],
                    "top_2_score": float(scores[int(top_indices[1])]),
                    "top_3": predictor.class_names[int(top_indices[2])],
                    "top_3_score": float(scores[int(top_indices[2])]),
                    "windows_analyzed": result["chunks"],
                    "error": "",
                }
            )
        except Exception as error:
            rows.append({"file": str(audio_path), "error": f"{type(error).__name__}: {error}"})

    output = args.output or (args.experiment_dir / "batch_predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} results to: {output}")
    print("Scores are not calibrated probabilities; review errors and predictions before relying on them.")


if __name__ == "__main__":
    main()
