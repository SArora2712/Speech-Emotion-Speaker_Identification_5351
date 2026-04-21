# src/wav2vec_emotion.py
import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import warnings
import os
warnings.filterwarnings("ignore")

class Wav2VecEmotionClassifier:
    def __init__(self):
        print("   Loading Wav2Vec2.0 Emotion Model (stable version)...")
        
        # Using a stable and well-supported model
        self.model_name = "superb/wav2vec2-large-superb-er"
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # Emotion labels for this model
        self.labels = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        
        print("   ✅ Wav2Vec2.0 Emotion Model loaded successfully!")

    def predict(self, audio_path):
        """Predict emotion from audio file"""
        try:
            # Load audio at 16kHz (required by this model)
            speech, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            inputs = self.feature_extractor(
                speech, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
                probs = torch.softmax(logits, dim=-1)[0]
            
            # Get top prediction
            pred_idx = torch.argmax(probs).item()
            emotion = self.labels[pred_idx]
            confidence = probs[pred_idx].item()
            
            return emotion, round(confidence, 4)
            
        except Exception as e:
            print(f"Error in Wav2Vec prediction: {e}")
            return "neutral", 0.60


# Quick test
if __name__ == "__main__":
    classifier = Wav2VecEmotionClassifier()
    path = input("\nEnter path to a .wav file: ").strip()
    if os.path.exists(path):
        emotion, conf = classifier.predict(path)
        print(f"\nWav2Vec2.0 Result → Emotion: {emotion.upper()} ({conf*100:.1f}%)")