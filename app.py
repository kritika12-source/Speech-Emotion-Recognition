"""Browser UI for recording/uploading audio and extracting audio from MP4."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import imageio_ffmpeg

from src.wavlm_inference import WavLMPredictor
from src.audio_quality import analyze_audio_quality, format_quality_report


ROOT = Path(__file__).resolve().parent
BASE_EXPERIMENT_DIR = ROOT / "outputs" / "wavlm_four_class"
ROBUST_EXPERIMENT_DIR = BASE_EXPERIMENT_DIR / "noise_augmented"
EXPERIMENT_DIR = (
    ROBUST_EXPERIMENT_DIR
    if (ROBUST_EXPERIMENT_DIR / "best_head.pt").exists()
    else BASE_EXPERIMENT_DIR
)
SUPPORTED_AUDIO = {".wav"}
_predictor: WavLMPredictor | None = None


def get_predictor() -> WavLMPredictor:
    global _predictor
    if _predictor is None:
        _predictor = WavLMPredictor(EXPERIMENT_DIR)
    return _predictor


def audio_from_mp4(video_path: str, output_path: Path) -> None:
    """Extract mono 16 kHz PCM audio from an MP4 using the bundled FFmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        details = result.stderr.strip().splitlines()
        reason = details[-1] if details else "No audio stream was found."
        raise ValueError(f"Could not extract audio from this MP4. {reason}")


def predict(audio_path: str | None, video_path: str | None) -> tuple[str, dict[str, float] | None, str]:
    if bool(audio_path) == bool(video_path):
        raise gr.Error("Provide either a WAV recording or an MP4 video, not both.")

    source = Path(audio_path or video_path or "")
    if not source.is_file():
        raise gr.Error("The selected file could not be found. Please upload it again.")

    if audio_path:
        if source.suffix.lower() not in SUPPORTED_AUDIO:
            raise gr.Error("Audio upload currently supports WAV files. You can also upload an MP4 video.")
        inference_path = source
        source_description = "WAV audio"
        return _predict(inference_path, source_description)

    if source.suffix.lower() != ".mp4":
        raise gr.Error("Video upload currently supports MP4 files.")

    with tempfile.TemporaryDirectory(prefix="speech_emotion_") as temp_dir:
        wav_path = Path(temp_dir) / "extracted_audio.wav"
        try:
            audio_from_mp4(str(source), wav_path)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return _predict(wav_path, "audio extracted from MP4")


def _predict(audio_path: Path, source_description: str) -> tuple[str, dict[str, float], str]:
    try:
        quality_report = analyze_audio_quality(audio_path)
        predictor = get_predictor()
        result = predictor.predict_file(audio_path)
    except FileNotFoundError as exc:
        raise gr.Error(
            "The four-class model files are missing. Run `python -m src.train_wavlm_four_class` first."
        ) from exc
    except Exception as exc:
        raise gr.Error(f"Prediction failed: {exc}") from exc

    scores = result["probabilities"]
    ranking = sorted(
        ((name, float(scores[index])) for index, name in enumerate(predictor.class_names)),
        key=lambda item: item[1],
        reverse=True,
    )
    top = ranking[0]
    close_score_note = ""
    if len(ranking) > 1 and top[1] - ranking[1][1] < 0.10:
        close_score_note = (
            "\n\n**Uncertain result:** the top two model scores are close. "
            "Try a shorter, clearer speech segment and compare the prediction."
        )
    details = (
        f"## Prediction: **{top[0].title()}**\n\n"
        f"Analyzed {result['chunks']} audio window(s) from {source_description}.\n\n"
        "The displayed scores are model ranking scores, not calibrated probabilities. "
        "The model recognizes four broad classes: angry, happy, sad, and neutral."
        f"{close_score_note}"
    )
    return details, dict(ranking), format_quality_report(quality_report)


with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.Markdown(
        """
        # Speech Emotion Recognition
        Record or upload a **WAV** clip, or upload an **MP4** and analyze its audio track.
        This model predicts one of four broad emotions: angry, happy, sad, or neutral.
        """
    )
    with gr.Row():
        audio_input = gr.Audio(
            label="Record or upload WAV audio",
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
        )
        video_input = gr.Video(label="Or upload an MP4 video", sources=["upload"])
    predict_button = gr.Button("Predict emotion", variant="primary")
    prediction_output = gr.Markdown()
    score_output = gr.Label(label="Emotion ranking scores", num_top_classes=4)
    quality_output = gr.Markdown()
    predict_button.click(
        fn=predict,
        inputs=[audio_input, video_input],
        outputs=[prediction_output, score_output, quality_output],
    )
    gr.Markdown(
        "**Note:** Predictions can be wrong, especially for noisy or very different recordings. "
        "MP4 files must contain an audio track. This project analyzes audio, not facial expressions."
    )


if __name__ == "__main__":
    demo.queue().launch()
