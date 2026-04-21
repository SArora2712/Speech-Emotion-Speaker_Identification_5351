
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import librosa
import joblib
import sounddevice as sd
import soundfile as sf
import time
from datetime import datetime
import torch
import warnings
warnings.filterwarnings("ignore")
import soundfile as sf


from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from speech_type_classifier import classify_speech_type ,generate_output
# ================= CONFIG =================
MODEL_PATH = "models/wav2vec2_emotion_final"
SPEAKER_MODEL_PATH = "models/speaker_model.pkl"
LE_SPEAKER_PATH = "models/label_encoder_speaker.pkl"
SCALER_SPEAKER_PATH = "models/scaler_speaker.pkl"
LE_EMOTION_PATH = "models/label_encoder_emotion.pkl"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5


class FinalEmotiVoice:
    def __init__(self):
        print("\n" + "="*80)
        print("EMOTIVOICE - FINAL SYSTEM")
        print("="*80)
        self.load_models()

    def load_models(self):
        try:
            print(f"Loading model from: {MODEL_PATH}")

            #  ALWAYS load from your trained folder
            self.processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)

            self.model.eval()

           

            # Speaker Model
            self.speaker_model = joblib.load(SPEAKER_MODEL_PATH)
            
            self.le_speaker = joblib.load(LE_SPEAKER_PATH)
            self.scaler_speaker = joblib.load(SCALER_SPEAKER_PATH)
            # Emotion Label Encoder (FIX)
            self.le_emotion = joblib.load(LE_EMOTION_PATH)

            print(" Models loaded successfully!")

        except Exception as e:
            print(f" Error loading models: {e}")
            exit()

    def predict_emotion(self, filepath):
        try:
            #  Load audio properly
            try:
                waveform, sr = sf.read(filepath)
            except:
                waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE)

            #  Process input
            inputs = self.processor(
                waveform,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )

            #  Predict correctly3
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_id].item()

            emotion = self.le_emotion.inverse_transform([predicted_id])[0]

            return emotion, confidence

        except Exception as e:
            print(f"Emotion prediction error: {e}")
            return "neutral", 0.0

    def predict_speaker(self, filepath):
        try:
            signal, sr = librosa.load(filepath, sr=22050)
            signal, _ = librosa.effects.trim(signal, top_db=20)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
            features = np.mean(mfcc, axis=1).reshape(1, -1)
            scaled = self.scaler_speaker.transform(features)
            speaker_id = self.speaker_model.predict(scaled)[0]
            speaker = self.le_speaker.inverse_transform([speaker_id])[0]
            return speaker
        except:
            return "Unknown"

    def predict(self, filepath):
        print(f"\nAnalyzing: {os.path.basename(filepath)}")

        speaker = self.predict_speaker(filepath)
        emotion, confidence = self.predict_emotion(filepath)

       # 🧠 Speech type detection
        speech_type, transcript, scores, conf_level = classify_speech_type(filepath)

        # 🎯 Final sentence with speech type
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

    def print_result(self, result):
        print("\n" + "="*80)
        print("FINAL PREDICTION RESULT")
        print("="*80)
        print(f"Speaker   : {result['speaker']}")
        print(f"Emotion   : {result['emotion']} ({result['confidence']*100:.2f}%)")
        print(f"Speech Type : {result['speech_type']} ({result['speech_confidence']})")
        print(f"Output      : {result['sentence']}")
        print("="*80)


def main():
    predictor = FinalEmotiVoice()

    while True:
        print("\n1. Analyze .wav file")
        print("2. Record from microphone (5 seconds)")
        print("3. Exit")
        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            path = input("\nEnter .wav file path: ").strip().strip('"')
            if os.path.exists(path):
                predictor.predict(path)
            else:
                print(" File not found!")

        elif choice == "2":
            print(f"\n Recording for {RECORD_SECONDS} seconds...")
            for i in range(3, 0, -1):
                print(f"Starting in {i}...", end="\r")
                time.sleep(1)

            audio = sd.rec(int(RECORD_SECONDS * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()

            temp_file = f"temp_rec_{datetime.now().strftime('%H%M%S')}.wav"
            sf.write(temp_file, audio.flatten(), 16000)

            predictor.predict(temp_file)

        elif choice == "3":
            print("Thank you! Goodbye.")
            break



# ================= NEW: Clean function for API =================
def analyze_audio(audio_path: str):
    """Clean function to be called from FastAPI"""
    if not hasattr(analyze_audio, "predictor"):
        analyze_audio.predictor = FinalEmotiVoice()
    
    return analyze_audio.predictor.predict(audio_path)


# Keep your original main() for local testing if you want
if __name__ == "__main__":
    main()

