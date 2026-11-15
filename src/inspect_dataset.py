"""Download and inspect the Hugging Face RAVDESS dataset metadata."""

from pathlib import Path

import pandas as pd
from datasets import Audio, load_dataset

DATASET_ID = "AbstractTTS/RAVDESS"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    print(f"Loading {DATASET_ID} from Hugging Face...")
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    required_columns = {"file", "audio", "emotion"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing)}. "
            f"Available columns: {dataset.column_names}"
        )

    table = pd.DataFrame(
        {
            "file": dataset["file"],
            "emotion": dataset["emotion"],
        }
    )
    table["actor"] = table["file"].str.extract(r"(Actor\d+)", expand=False)
    if table["actor"].isna().any():
        examples = table.loc[table["actor"].isna(), "file"].head().tolist()
        raise ValueError(f"Could not parse actor IDs from filenames: {examples}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emotion_counts = table["emotion"].value_counts().sort_index()
    actor_counts = table["actor"].value_counts().sort_index()
    emotion_counts.rename_axis("emotion").to_csv(OUTPUT_DIR / "emotion_counts.csv")
    actor_counts.rename_axis("actor").to_csv(OUTPUT_DIR / "actor_counts.csv")

    first_audio = dataset[0]["audio"]
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"Actors: {table['actor'].nunique()} ({', '.join(sorted(table['actor'].unique()))})")
    print("Emotion counts:")
    print(emotion_counts.to_string())
    print(f"Audio metadata keys: {sorted(first_audio.keys())}")
    print(f"Audio bytes available: {first_audio.get('bytes') is not None}")
    print(f"Summary CSVs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
