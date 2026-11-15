# Speech Emotion Recognition

A PyTorch speech emotion recognition project using Hugging Face datasets and pretrained WavLM features. It compares a small MFCC CNN baseline with a BiLSTM and multi-head attention classifier over frozen WavLM representations. The evaluation keeps actors disjoint across training, validation, and test.

## Results

The held-out test set contains 240 clips from RAVDESS actors 21–24, none of whom appear in training or validation.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MFCC CNN baseline | 36.2% | 0.320 | 47.5% | 0.388 | 0.414 |
| WavLM + BiLSTM + attention | 61.3% | 0.599 | 70.0% | 0.699 | 0.699 |

WavLM per-emotion test F1:

| Emotion | F1 |
| --- | ---: |
| Angry | 0.84 |
| Calm | 0.67 |
| Disgust | 0.83 |
| Fearful | 0.73 |
| Happy | 0.65 |
| Neutral | 0.69 |
| Sad | 0.54 |
| Surprised | 0.64 |

These are the measured results for this project. The 90%+ goal has **not** been reached. The actor-disjoint test is meant to measure performance on unseen speakers, which is harder than a random clip split where the same actors can occur in train and test. Do not present the scores from other papers as this model's results.

## Dataset and protocol

- Dataset: [AbstractTTS/RAVDESS on Hugging Face](https://huggingface.co/datasets/AbstractTTS/RAVDESS), 1,440 speech clips from 24 actors with eight emotion labels.
- Actor split: 01–16 train, 17–20 validation, 21–24 test. Each split has equal male/female actor counts.
- The split is fixed and reproducible. Feature normalization is fitted on training actors only. The best checkpoint is selected by validation macro F1.
- RAVDESS is licensed for non-commercial use under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Credit Livingstone and Russo and check the dataset terms before redistributing audio.
- Paper: Livingstone and Russo (2018), [The Ryerson Audio-Visual Database of Emotional Speech and Song](https://doi.org/10.1371/journal.pone.0196391).

## Models

### MFCC CNN baseline

Audio is resampled to 16 kHz; 40 MFCC coefficients are extracted with a 10 ms hop. A compact 2D CNN is trained from scratch with class-weighted cross entropy.

### WavLM transfer learning

The [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus) encoder is frozen to extract frame-level speech features. A trainable BiLSTM, multi-head attention, and attention-pooling head is trained in PyTorch. The 378 MB pretrained encoder is downloaded from Hugging Face on first use. Feature arrays and model checkpoints are generated locally and are not committed to GitHub.

## Setup

The project was developed with Python 3.13 and CPU PyTorch. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements_wavlm.txt
```

## Reproduce the pipeline

```powershell
# Inspect the Hugging Face dataset
python -m src.inspect_dataset

# Prepare the MFCC CNN baseline
python -m src.prepare_data
python -m src.train
python -m src.validation_diagnostics
python -m src.evaluate

# Extract WavLM features, train the classifier, then evaluate
python -m src.extract_wavlm_features
python -m src.train_wavlm
python -m src.evaluate_wavlm --split validation
python -m src.evaluate_wavlm --split test
```

The WavLM test command evaluates the fixed held-out actor set. Select changes using validation data; keep the test result fixed when comparing model iterations.

## Try one audio file

```powershell
python -m src.predict_wavlm --audio "C:\path\to\speech.wav"
```

## Batch-predict a folder of WAV files

Place WAV files in `audio_samples`, then run:

```powershell
python -m src.predict_wavlm_folder --input-dir audio_samples --output outputs/wavlm/batch_predictions.csv
```

Add `--recursive` to search subfolders. The script loads the model once, writes one row per file to CSV, and divides longer clips into overlapping 5.44-second windows. Scores are raw model outputs, not calibrated confidence estimates. The model was trained on acted English speech, so unrelated languages, music, noise, or spontaneous conversation can produce unreliable labels.

## Project structure

```text
src/       Data inspection, preprocessing, model, training, evaluation, and prediction
 data/     Local data artifacts (ignored by Git)
outputs/   Features, checkpoints, metrics, predictions, and plots (ignored by Git)
audio_samples/  Optional local WAV files for batch prediction (ignored by Git)
```

## Next experiment

A broader model could train on the six emotions shared by RAVDESS and [CREMA-D](https://huggingface.co/datasets/cfahlgren1/crema-d), which contains 7,442 clips from 91 actors. This would require a new label map and a new actor-disjoint evaluation. Those results have not been measured in this repository yet.

## Four-class experiment

A simpler experiment reuses the cached WavLM features and predicts angry, happy, sad, and neutral. RAVDESS calm clips are grouped with neutral; disgust, fearful, and surprised clips are excluded. This changes the task, so its accuracy must be reported separately from the eight-class results above. The current four-class held-out test result is 72.9% accuracy and 0.706 macro F1 across actors 21–24 (144 clips). Validation was used to select the checkpoint; the test set was evaluated afterward.

```powershell
python -m src.train_wavlm_four_class
python -m src.evaluate_wavlm_four_class --split validation
python -m src.evaluate_wavlm_four_class --split test

# Use the four-class model for one file or a folder
python -m src.predict_wavlm --experiment-dir outputs/wavlm_four_class --audio "C:\path\to\speech.wav"
python -m src.predict_wavlm_folder --experiment-dir outputs/wavlm_four_class --input-dir audio_samples
```

## Record or upload audio/video in a browser

The Gradio app lets you record from a microphone or upload WAV audio. It can also accept an MP4 video and extract its audio track before predicting. It does not analyze video frames or facial expressions. Train the four-class model first, then install the UI dependencies and start the app:

```powershell
python -m pip install -r requirements_ui.txt
python app.py
```

Open the local URL printed by Gradio. Predictions are limited to angry, happy, sad, and neutral, and may be incorrect on recordings unlike RAVDESS.

The app also displays heuristic recording checks (level, clipping, and an estimate from detected quiet gaps). These checks cannot reliably separate continuous background noise from speech, and they are not an intelligibility score.

## Noise robustness experiment

To train a separate head with synthetic waveform noise mixed into the **training actors only**, create WavLM features, train the augmented head, then compare clean and synthetic-noisy conditions. Noise types are generated white/pink noise and electrical hum. Validation/test clips are held out by actor and receive a fixed synthetic 10 dB SNR mixture for the noisy condition; this is a controlled stress test, not a substitute for evaluation on real-world recordings.

```powershell
python -m src.extract_noise_augmented_features
python -m src.train_wavlm_four_class --noise-augmented
python -m src.evaluate_wavlm_four_class --split validation --condition both --experiment-dir outputs/wavlm_four_class/noise_augmented
python -m src.evaluate_wavlm_four_class --split test --condition both --experiment-dir outputs/wavlm_four_class/noise_augmented
```

The app automatically uses the noise-augmented model after its checkpoint exists. Restart `python app.py` to load it. On the held-out test actors (21–24, 144 clips), the measured comparison was:

| Model | Clean test accuracy / macro F1 | Synthetic 10 dB noisy test accuracy / macro F1 |
|---|---:|---:|
| Original four-class head | 72.9% / 0.706 | 23.6% / 0.112 |
| Noise-augmented head | 78.5% / 0.783 | 78.5% / 0.766 |

The noisy test uses the same **kind of synthetic noise augmentation** as training, at a fixed 10 dB SNR; it shows robustness to this controlled corruption and does not guarantee the same result for real rooms, phones, music, traffic, or other noise. The earlier clean four-class score remains documented separately as the original baseline.
