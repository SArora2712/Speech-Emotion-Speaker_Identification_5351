import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import librosa
import joblib
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from speech_type_classifier import classify_speech_type, generate_output

# ================= CONFIG =================
MODEL_PATH = "models/wav2vec2_emotion_final"
SPEAKER_MODEL_PATH = "models/speaker_model.pkl"
LE_SPEAKER_PATH = "models/label_encoder_speaker.pkl"
SCALER_SPEAKER_PATH = "models/scaler_speaker.pkl"
LE_EMOTION_PATH = "models/label_encoder_emotion.pkl"

class FinalEmotiVoice:
    def __init__(self):
        print("\n" + "=" * 80)
        print("EMOTIVOICE - FINAL SYSTEM")
        print("=" * 80)
        self.load_models()

    def load_models(self):
        try:
            print(f"Loading model from: {MODEL_PATH}")

            self.processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
            self.model.eval()

            self.speaker_model = joblib.load(SPEAKER_MODEL_PATH)
            self.le_speaker = joblib.load(LE_SPEAKER_PATH)
            self.scaler_speaker = joblib.load(SCALER_SPEAKER_PATH)

            self.le_emotion = joblib.load(LE_EMOTION_PATH)

            print("Models loaded successfully!")

        except Exception as e:
            print("Model loading failed:", repr(e))
            raise e

    # ================= EMOTION (16kHz REQUIRED) =================
    def predict_emotion(self, filepath):
        try:
            # 🔥 wav2vec2 requires 16kHz
            waveform, _ = librosa.load(filepath, sr=16000)

            waveform = waveform.astype(np.float32)

            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_id].item()

            emotion = self.le_emotion.inverse_transform([predicted_id])[0]

            return emotion, confidence

        except Exception as e:
            print("[Emotion ERROR]:", repr(e))
            return "neutral", 0.0

    # ================= SPEAKER (ORIGINAL SR) =================
    def predict_speaker(self, filepath):
        try:
            #  keep original sample rate (important)
            signal, sr = librosa.load(filepath, sr=22050)

            signal, _ = librosa.effects.trim(signal, top_db=20)

            if len(signal) < 1000:
                return "Unknown"

            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
            features = np.mean(mfcc, axis=1).reshape(1, -1)

            scaled = self.scaler_speaker.transform(features)
            speaker_id = self.speaker_model.predict(scaled)[0]

            return self.le_speaker.inverse_transform([speaker_id])[0]

        except Exception as e:
            print("[Speaker ERROR]:", repr(e))
            return "Unknown"

    # ================= MAIN =================
    def predict(self, filepath):
        print(f"\nAnalyzing: {os.path.basename(filepath)}")

        try:
            speaker = self.predict_speaker(filepath)
            emotion, confidence = self.predict_emotion(filepath)

            speech_type, transcript, scores, conf_level = classify_speech_type(filepath)
            sentence = generate_output(speaker, emotion, speech_type)

            result = {
                "speaker": speaker,
                "emotion": emotion,
                "confidence": confidence,
                "speech_type": speech_type,
                "speech_confidence": conf_level,
                "sentence": sentence
            }

            self.print_result(result)
            return result

        except Exception as e:
            print("[PREDICT ERROR]:", repr(e))

            return {
                "speaker": "Unknown",
                "emotion": "neutral",
                "confidence": 0.0,
                "speech_type": "unknown",
                "speech_confidence": "low",
                "sentence": "Could not analyze audio properly."
            }

    # ================= PRINT =================
    def print_result(self, result):
        print("\n" + "=" * 80)
        print("FINAL PREDICTION RESULT")
        print("=" * 80)
        print(f"Speaker   : {result['speaker']}")
        print(f"Emotion   : {result['emotion']} ({result['confidence']*100:.2f}%)")
        print(f"Speech Type : {result['speech_type']} ({result['speech_confidence']})")
        print(f"Output      : {result['sentence']}")
        print("=" * 80)


# ================= API HELPER =================
def analyze_audio(audio_path: str):
    if not hasattr(analyze_audio, "predictor"):
        analyze_audio.predictor = FinalEmotiVoice()
    return analyze_audio.predictor.predict(audio_path)