# Speech Emotion Recognition + Speaker Identification

Two-part audio ML project. Part one classifies human emotion from speech. Part two identifies the speaker. Both run in real time from a microphone.

---

## Results

| Task | Model | Accuracy | Dataset |
|------|-------|----------|---------|
| Emotion Recognition (8 classes) | Wav2Vec 2.0 fine-tuned | **88.98%** | RAVDESS |
| Speaker Identification | SVM + MFCC | **94%** | Custom dataset |
| Emotion Recognition (baseline) | CNN | ~82% | RAVDESS |

The Wav2Vec model beat the CNN baseline by over 6 percentage points. The speaker ID module was built from scratch on a custom-recorded dataset.

---

## The 8 emotion classes

Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised

---

## How it works

**Emotion recognition**
- Pre-trained Wav2Vec 2.0 (facebook/wav2vec2-base) loaded from HuggingFace
- Fine-tuned on RAVDESS with a classification head added on top
- Data augmentation during training: noise injection, pitch shifting, time stretching
- Real-time inference via microphone using PyAudio

**Speaker identification**
- MFCC features extracted per audio frame using Librosa
- SVM classifier trained on custom-labelled speaker recordings
- Delta and delta-delta MFCC features included for temporal context

---

## Tech stack

```
Audio processing   →  Librosa · PyAudio · SoundFile
Feature extraction →  MFCC · Delta-MFCC
Deep learning      →  PyTorch · HuggingFace Transformers · Wav2Vec 2.0
Classical ML       →  Scikit-learn · SVM
Data augmentation  →  audiomentations
```

---

## Run it

```bash
git clone https://github.com/SArora2712/[repo-name].git
cd [repo-name]
pip install -r requirements.txt

# Run emotion recognition on a WAV file
python predict_emotion.py --audio sample.wav

# Run real-time microphone inference
python realtime_inference.py
```

---

## Dataset

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) — 24 actors, 8 emotions, 1440 audio files
- **Custom speaker dataset** — recorded and labelled manually for speaker ID task

---

## Training details

- Base model: `facebook/wav2vec2-base`
- Fine-tuning epochs: 20
- Batch size: 16
- Optimizer: AdamW with warmup scheduler
- Train/val/test split: 70/15/15

---

## Project context

This started as an experiment to see how well a pre-trained speech model could be adapted for emotion classification with relatively limited data. The 88.98% result on RAVDESS exceeded what I expected going in — the CNN baseline I had was sitting around 82%, and the Wav2Vec fine-tuning approach closed that gap significantly.

The speaker ID module was a natural extension — once you have per-frame audio features, identifying speakers is a different but related classification problem.

---

*Python · PyTorch · Wav2Vec 2.0 · Scikit-learn · Real-time inference*
