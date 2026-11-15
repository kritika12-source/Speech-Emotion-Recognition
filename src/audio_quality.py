"""Lightweight recording-quality diagnostics (heuristics, not a calibrated SNR meter)."""

from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000


def _dbfs(values: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) if values.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-8))


def analyze_audio_quality(path: Path) -> dict[str, object]:
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError("The selected file has no usable audio samples.")

    clipped_mask = np.abs(waveform) >= 0.999
    clipped_percent = float(np.mean(clipped_mask) * 100.0)
    edges = np.diff(np.concatenate(([0], clipped_mask.astype(np.int8), [0])))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)
    longest_clipped_run = int(np.max(run_ends - run_starts)) if len(run_starts) else 0
    frame = int(0.03 * SAMPLE_RATE)
    hop = int(0.015 * SAMPLE_RATE)
    active_ranges = librosa.effects.split(
        waveform, top_db=30, frame_length=frame, hop_length=hop
    )
    active_mask = np.zeros(waveform.size, dtype=bool)
    for start, end in active_ranges:
        active_mask[int(start) : int(end)] = True

    active_samples = waveform[active_mask]
    quiet_samples = waveform[~active_mask]
    active_dbfs = _dbfs(active_samples)
    overall_dbfs = _dbfs(waveform)

    notes: list[str] = []
    if active_dbfs < -38:
        level = "very quiet"
        notes.append("Speech level is very low; move closer to the microphone or raise input gain.")
    elif active_dbfs < -28:
        level = "quiet"
        notes.append("Speech is on the quiet side; try recording closer to the microphone.")
    else:
        level = "usable"

    snr_estimate = None
    noise_label = "not enough quiet gaps to estimate separately"
    if quiet_samples.size >= int(0.4 * SAMPLE_RATE):
        quiet_dbfs = _dbfs(quiet_samples)
        snr_estimate = float(active_dbfs - quiet_dbfs)
        if snr_estimate < 10:
            noise_label = "likely high"
            notes.append("Background noise may be competing with speech; try a quieter room.")
        elif snr_estimate < 20:
            noise_label = "noticeable"
            notes.append("Some background noise may be present; reduce fan/music and record closer.")
        else:
            noise_label = "low in detected quiet gaps"

    if clipped_percent >= 0.1 or longest_clipped_run >= 4:
        notes.append("Audio may be clipping; lower the microphone input level and re-record.")
    if len(active_samples) < int(0.25 * SAMPLE_RATE):
        notes.append("Very little audio above the silence threshold was detected; check the recording.")
    if not notes:
        notes.append("No strong volume or clipping issue was detected. Use a quiet room and speak clearly.")

    return {
        "audio_level": level,
        "noise_estimate": noise_label,
        "speech_to_quiet_gap_db": snr_estimate,
        "clipped_percent": clipped_percent,
        "active_seconds": len(active_samples) / SAMPLE_RATE,
        "duration_seconds": len(waveform) / SAMPLE_RATE,
        "recommendations": notes,
        "overall_dbfs": overall_dbfs,
    }


def format_quality_report(report: dict[str, object]) -> str:
    ratio = report["speech_to_quiet_gap_db"]
    ratio_text = f"~{ratio:.1f} dB (rough estimate)" if ratio is not None else str(report["noise_estimate"])
    tips = "\n".join(f"- {note}" for note in report["recommendations"])
    return (
        "### Recording quality (rough checks)\n"
        f"- Audio level (loudness check, not a clarity score): **{report['audio_level']}**\n"
        f"- Background/noise estimate: **{report['noise_estimate']}**\n"
        f"- Speech-to-quiet-gap level difference: **{ratio_text}**\n"
        f"- Clipped samples: **{report['clipped_percent']:.3f}%**\n\n"
        f"**Recording tips**\n{tips}\n\n"
        "*These are signal heuristics, not a certified noise meter or intelligibility score. "
        "Continuous background noise may not be separable from speech.*"
    )
