#  Hybrid Speech Intelligence System
### Speech Emotion Recognition + Speaker Identification

> A production-grade AI system that analyses a single audio clip and simultaneously detects **who is speaking**, **what emotion they're expressing**, and **what type of speech it is** — delivered as a natural language sentence.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![Flask](https://img.shields.io/badge/Flask-API-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

##  Results at a Glance

| Task | Model | Accuracy | F1-Score |
|---|---|---|---|
| **Speech Emotion Recognition** | Wav2Vec 2.0 (fine-tuned) | **88.94%** | **88.68%** |
| **Speaker Identification** | SVM (RBF Kernel) | **94.1%** | **94.1%** |
| CNN+GRU (emotion baseline) | CNN + Bidirectional GRU | 74.73% | 74.76% |
| SVM (emotion baseline) | SVM on MFCC | 70.5% | 70.2% |

**10 out of 17 speakers identified with 100% accuracy.**  
Wav2Vec 2.0 outperforms all classical baselines by 18–41 percentage points.

---

##  Demo

<!-- Paste your demo video/GIF here -->
> 📹 **[Watch the full demo →]** *https://github.com/SArora2712/Speech-Emotion-Speaker_Identification_5351/blob/main/demo.mp4*

**Sample system outputs:**

| Audio Input | Detected Speaker | Emotion (Confidence) | Final Output |
|---|---|---|---|
| sample_01.wav | Manas | Neutral (97.9%) | *"Manas is speaking in a neutral tone while narrating a story."* |
| sample_02.wav | Daivik | Disgust (98.2%) | *"Daivik is speaking in a disgusted tone while in a conversation."* |
| sample_03.wav | Tarinee | Calm (99.2%) | *"Tarinee is speaking in a calm tone while storytelling."* |

---

##  How It Works

A single `.wav` file enters a **6-stage synchronous pipeline** with **3 parallel inference streams**:

```
Audio Input
    │
    ▼
Preprocessing  (normalisation · silence trimming · dual-rate resampling)
    │
    ├────────────────────────────────────────────┐
    │                                            │
    ▼                                            ▼
MFCC Features (40-dim)         Wav2Vec 2.0 raw waveform (16kHz)
    │                          + Mel Spectrogram (128×128)
    ▼                                            ▼
Speaker ID (SVM-RBF)           Emotion Detection (Wav2Vec 2.0)
                                                 │
                                                 ▼
                                  Speech Type (Lexicon Scorer)
    │                                            │
    └────────────────────────────────────────────┘
                          │
                          ▼
          Natural Language Generation (NLG)
  "[Speaker] is speaking in a [emotion] tone while [speech type]."
```

### The Three Inference Engines

**1. Emotion Recognition — Wav2Vec 2.0 (Primary Engine)**
- Fine-tuned `facebook/wav2vec2-large` (317M parameters) on RAVDESS dataset
- Bottom 10 transformer layers frozen; upper 14 fine-tuned for emotional prosody
- Custom **Attention Pooling** module weights emotionally-salient time steps (pitch spikes, trembles)
- 4 augmentation strategies: pitch shifting, time stretching, Gaussian noise, time masking
- Converged in 25 epochs; training loss 2.1 → 0.15

**2. Speaker Identification — SVM (Biometric Engine)**
- 40-dimensional MFCC mean vectors as "spectral fingerprints" of vocal tract shape
- SVM with RBF kernel (C=10, γ='scale') trained on 680 recordings from 17 local speakers
- Dual-rate DSP: biometric stream at 22,050 Hz · transformer stream at 16,000 Hz

**3. Speech Type Classifier — Lexicon Engine**
- Heuristic lexicon scoring on Google STT transcript
- Four categories: **Instructing** · **Sharing Knowledge** · **Storytelling** · **Conversation**
- Scores token matches per category; highest-weight category wins; defaults to "Conversation"

---

##  Detailed Results

### Emotion Recognition — Per-Class Accuracy (Wav2Vec 2.0)

| Emotion | Accuracy | Notes |
|---|---|---|
| Calm | **98.2%** | Highest — distinct low-arousal prosody |
| Fearful | **94.5%** | High-frequency jitter is distinctive |
| Disgust | **94.4%** | Unique spectral changes in low frequencies |
| Surprised | 92.6% | Sudden pitch peaks aid detection |
| Angry | 90.7% | High energy, sharp onset |
| Sad | 85.5% | Some confusion with fearful |
| Happy | 85.2% | Minor confusion with surprised |
| Neutral | 60.0% | Hardest class — lacks distinct spectral peaks |

> **Insight:** Neutral is the primary challenge, often confused with Calm or Sad due to shared low-arousal acoustics. This is a known open problem in affective computing.

### Emotion Recognition — All Models Compared

| Model | Accuracy | Notes |
|---|---|---|
| **Wav2Vec 2.0** | **88.94%** | Processes raw waveform via self-supervised learning |
| CNN + GRU | 74.73% | Spatial-temporal on Mel-Spectrograms |
| SVM (MFCC) | 70.5% | Best classical baseline |
| XGBoost | 68.3% | Tabular MFCC features |
| Random Forest | 67.4% | 200 trees |
| Logistic Regression | 65.2% | Multinomial, lbfgs solver |
| MLP | 62.8% | Shallow network |

### Speaker Identification — All Models Compared

| Model | Accuracy | F1-Score |
|---|---|---|
| **SVM (RBF Kernel)** | **94.1%** | **94.1%** |
| Logistic Regression | 92.6% | 92.7% |
| MLP Classifier | 91.2% | 91.7% |
| Random Forest | 89.7% | 89.7% |
| XGBoost | 87.5% | 87.5% |

### Speaker Identification — Per-Speaker Performance (SVM)

| Accuracy | Speakers |
|---|---|
| **100%** | Daivik, Harshita, Millee, Parul, Sanad, Shehbaaz, Shreya, Shubham, Sukhman, Tarinee |
| **87.5%** | Gauri, Manas, Mehak, Rumani, Vishesh |
| 62.5% | Sukarman *(statistical outlier)* |

16 of 17 speakers achieved ≥87.5% recognition accuracy.

---

##  Project Structure

```
Speech-Emotion-Speaker_Identification/
│
├── data/
│   ├── Ravdess/                    # RAVDESS dataset (1,440 speech files used)
│   ├── Custom_Dataset/             # 680 recordings from 17 local speakers
│   ├── custom_features.csv         # Pre-extracted MFCC features (custom dataset)
│   ├── mfcc_features.csv           # Pre-extracted MFCC features (RAVDESS)
│   └── wav2vec_ready_dataset.csv   # Processed dataset for Wav2Vec training
│
├── src/                            # Core model implementations
│   ├── fine_tunew2v2.py            # Wav2Vec 2.0 fine-tuning script (main)
│   ├── wav2vec_emotion.py          # Wav2Vec inference module
│   ├── cnn_gru.py                  # CNN + Bidirectional GRU hybrid
│   ├── cnn.py                      # CNN-only baseline
│   ├── train_emotion.py            # Emotion model training pipeline
│   ├── train_speaker.py            # Speaker ID training pipeline
│   ├── predict.py                  # Unified inference engine
│   ├── hybrid.py                   # Parallel inference orchestrator
│   ├── speech_type_classifier.py   # Lexicon-based speech type engine
│   └── txt_emotion_classifier.py   # DistilRoBERTa text sentiment module
│
├── scripts/                        # Data preparation and analysis
│   ├── preprocessing.py            # DSP pipeline (normalise, trim, resample)
│   ├── feature_extraction.py       # MFCC + Mel-Spectrogram extraction
│   ├── prepare_v2w2.py             # Dataset preparation for Wav2Vec
│   ├── record.py                   # Custom dataset recording script
│   ├── Custom_data_features.py     # Feature extraction for custom dataset
│   ├── dataset_analysis.py         # Dataset visualisation and statistics
│   └── audio-analysis.ipynb        # Exploratory analysis notebook
│
├── outputs/                        # Model outputs and visualisations
│   ├── confusion_matrix_wav2vec.png
│   ├── confusion_matrix_cnn_gru.png
│   ├── confusion_matrix_svm.png
│   ├── wav2vec__fnal_training_curves.png
│   ├── per_speaker_accuracy.png
│   ├── speaker_model_comparison.png
│   ├── emotion_model_comparison.png
│   ├── mfcc_visualization.png
│   ├── custom_mfcc_heatmaps.png
│   └── spectogram.png
│
├── frontend_SER_SI/                # React / Next.js dashboard
│   ├── app/                        # Next.js app router
│   ├── components/                 # AudioDashboard, Visualizer, ResultCards
│   ├── hooks/                      # React state hooks
│   └── styles/                     # Tailwind CSS
│
├── tdd/                            # Unit tests
├── main.py                         # Entry point — boots Flask + loads models
└── Requirement.txt                 # Python dependencies
```

---

##  Installation & Setup

### Prerequisites
- Python 3.9+
- NVIDIA GPU with 8GB+ VRAM recommended (CPU fallback available, ~10x slower)
- Node.js 18+ (for frontend only)

### Backend

```bash
# 1. Clone the repo
git clone https://github.com/SArora2712/Speech-Emotion-Speaker_Identification_5351.git
cd Speech-Emotion-Speaker_Identification_5351

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r Requirement.txt

# 4. Start the Flask API server
python main.py
# API available at http://localhost:5000
```

### Frontend

```bash
cd frontend_SER_SI
npm install
npm run dev
# Dashboard at http://localhost:3000
```

### Direct Inference (no UI)

```bash
python src/predict.py --audio path/to/your/audio.wav
```

---

##  Technical Details

### Preprocessing Pipeline

```
Raw Audio
    ↓  Peak normalisation → [-1, +1]       (removes microphone distance bias)
    ↓  VAD silence trimming (20dB threshold) (focuses model on speech)
    ↓  Dual-stream resampling:
         → 22,050 Hz  (biometric MFCC stream)
         → 16,000 Hz  (Wav2Vec transformer stream)
```

### Wav2Vec 2.0 Fine-Tuning Architecture

```
facebook/wav2vec2-large  (317M parameters)
├── Feature Encoder         [FROZEN]
├── Transformer Layers 1-10 [FROZEN — preserves phoneme representations]
├── Transformer Layers 11-24[FINE-TUNED — learns emotional prosody]
└── Attention Pooling Layer [TRAINED — weights emotionally salient frames]
    └── Linear Classifier   [TRAINED — 8-class output]
```

### MFCC Extraction

```python
mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40,
                               n_fft=2048, hop_length=512)
mfcc_mean = np.mean(mfccs, axis=1)   # 40-dimensional fingerprint vector
```

---

##  Datasets

### RAVDESS — Emotion Recognition
- 1,440 speech files (from 2,452 total; speech only, no song)
- 24 professional actors, studio-controlled, 10+ independent raters per label
- [Download →](https://zenodo.org/record/1188976)

### Custom Speaker Dataset — Speaker Identification
- 680 recordings · 17 local university students · ~40 per speaker
- Each speaker recorded all 8 RAVDESS emotions for emotion-invariant biometric training
- 22,050 Hz mono WAV, laptop microphone, quiet environment

| | RAVDESS | Custom Dataset |
|---|---|---|
| Files | 1,440 | 680 |
| Speakers | 24 (professional) | 17 (local students) |
| Emotion classes | 8 | 8 |
| Primary use | Emotion model training | Speaker ID training |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.0, Hugging Face Transformers |
| Audio Processing | Librosa, Wav2Vec 2.0 |
| Classical ML | Scikit-learn (SVM, LogReg, RF, XGBoost) |
| NLP / STT | DistilRoBERTa, Google Speech Recognition API |
| Backend | Flask, Flask-CORS |
| Frontend | React 18, Next.js 14, Tailwind CSS, Axios |
| Visualisation | Matplotlib, Framer Motion, Web Audio API |
| Version Control | Git, GitHub |
| API Testing | Postman |

---

##  Roadmap

- [ ] Real-time streaming via WebSocket + MediaRecorder API
- [ ] Multilingual support — Hindi (IITKGP-SEH) and Punjabi corpora
- [ ] Edge deployment — model quantisation → Wav2Vec 2.0 Tiny for mobile/Raspberry Pi
- [ ] Multi-modal fusion — combine audio with facial expression (CV)
- [ ] Anti-spoofing — liveness detection to block replay attacks on Speaker ID
- [ ] Hardware calibration — acoustic normalisation profiles for mixed-device environments

---



##  Author

**Sukhman Arora**  
B.Tech Computer Science (AI & ML) · Amity University Punjab  
Supervised by Dr. Lokesh Pawar, ASET

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/sukhman-arora-45b5272b6)
[![GitHub](https://img.shields.io/badge/GitHub-SArora2712-black?style=flat-square&logo=github)](https://github.com/SArora2712)
[![Email](https://img.shields.io/badge/Email-sukhmanarora01%40gmail.com-red?style=flat-square&logo=gmail)](mailto:sukhmanarora01@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

> ⭐ If this project was useful or interesting, star the repo!
