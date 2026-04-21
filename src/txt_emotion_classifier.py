# src/text_emotion_classifier.py
import os
import speech_recognition as sr
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import pipeline


class TextEmotionClassifier:
    def __init__(self):
        print("   Loading BERT-based Text Emotion Classifier...")
        from transformers import pipeline

        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={"use_safetensors": True}
        )
        self.recognizer = sr.Recognizer()
        print("   ✅ Text Emotion Classifier loaded successfully!")

    def speech_to_text(self, audio_file):
        """Convert speech to text"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language="en-IN")
                return text.strip()
        except:
            return ""

    def classify_emotion(self, text):
        """Classify emotion from text"""
        if len(text.strip()) < 5:
            return "neutral", 0.75, text

        results = self.emotion_pipeline(text)[0]
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        top_label = sorted_results[0]['label']
        confidence = sorted_results[0]['score']

        emotion_map = {
            "joy": "happy",
            "sadness": "sad",
            "anger": "angry",
            "fear": "fearful",
            "disgust": "disgust",
            "surprise": "surprised",
            "neutral": "neutral"
        }

        emotion = emotion_map.get(top_label, "neutral")
        return emotion, round(confidence, 4), text

    def predict_from_audio(self, audio_file):
        """Full pipeline: audio → text → emotion"""
        transcript = self.speech_to_text(audio_file)
        emotion, confidence, final_text = self.classify_emotion(transcript)
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "transcript": final_text,
            "method": "Text-based BERT"
        }


# Quick test
if __name__ == "__main__":
    classifier = TextEmotionClassifier()
    file_path = input("Enter .wav file path: ").strip()
    if os.path.exists(file_path):
        result = classifier.predict_from_audio(file_path)
        print(f"\nEmotion: {result['emotion'].upper()} ({result['confidence']*100:.1f}%)")
        print(f"Text: {result['transcript']}")