"""Shared WavLM inference helpers for single-file and folder prediction."""

from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from src.wavlm_model import WavLMEmotionHead

MODEL_ID = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 5.44
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
WINDOW_STRIDE = WINDOW_SAMPLES // 2


class WavLMPredictor:
    def __init__(self, experiment_dir: Path | None = None) -> None:
        self.experiment_dir = experiment_dir or Path(__file__).resolve().parents[1] / "outputs" / "wavlm"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metadata_path = self.experiment_dir / "metadata.npz"
        stats_path = self.experiment_dir / "feature_normalization.npz"
        checkpoint_path = self.experiment_dir / "best_head.pt"
        for path in (metadata_path, stats_path, checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(f"Required WavLM artifact not found: {path}")

        with np.load(metadata_path, allow_pickle=False) as metadata:
            self.class_names = metadata["class_names"].astype(str).tolist()
        with np.load(stats_path, allow_pickle=False) as stats:
            self.mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(self.device)
            self.std = torch.from_numpy(stats["std"].astype(np.float32)).to(self.device)

        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self.encoder = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.classifier = WavLMEmotionHead(
            input_size=self.encoder.config.hidden_size,
            num_classes=len(self.class_names),
        ).to(self.device)
        self.classifier.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.classifier.eval()

    @staticmethod
    def _window_starts(sample_count: int) -> list[int]:
        if sample_count <= WINDOW_SAMPLES:
            return [0]
        final_start = sample_count - WINDOW_SAMPLES
        starts = list(range(0, final_start + 1, WINDOW_STRIDE))
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def predict_waveform(self, waveform: np.ndarray) -> dict[str, object]:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("Audio contains no samples")

        probabilities = []
        for start in self._window_starts(waveform.size):
            chunk = waveform[start : start + WINDOW_SAMPLES]
            inputs = self.extractor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                hidden = self.encoder(**inputs).last_hidden_state[0]
                normalized = (hidden - self.mean) / self.std
                logits = self.classifier(
                    normalized.unsqueeze(0),
                    torch.tensor([len(normalized)]),
                )
                probabilities.append(torch.softmax(logits, dim=1)[0].cpu().numpy())

        mean_probabilities = np.mean(probabilities, axis=0)
        top_indices = np.argsort(mean_probabilities)[::-1][:3]
        return {
            "prediction": self.class_names[int(top_indices[0])],
            "score": float(mean_probabilities[top_indices[0]]),
            "top_indices": top_indices,
            "probabilities": mean_probabilities,
            "chunks": len(probabilities),
        }

    def predict_file(self, audio_path: Path) -> dict[str, object]:
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return self.predict_waveform(waveform)
