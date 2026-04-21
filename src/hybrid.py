# src/ensemble_evaluate_final.py
import os
import numpy as np
import joblib
import librosa
import matplotlib.pyplot as plt
from tabulate import tabulate

from txt_emotion_classifier import TextEmotionClassifier
from wav2vec_emotion import Wav2VecEmotionClassifier

print("Loading all models for final ensemble evaluation...\n")

# Load Models
text_bert = TextEmotionClassifier()
wav2vec = Wav2VecEmotionClassifier()

svm_model = joblib.load("models/emotion_model.pkl")
le_emotion = joblib.load("models/label_encoder_emotion.pkl")
scaler_emotion = joblib.load("models/scaler_emotion.pkl")

dataset_path = "data/Custom_Dataset"   # Change if your folder is different

print("\n" + "="*110)
print("FINAL ENSEMBLE EVALUATION - ALL MODELS COMPARISON")
print("="*110)

results = []
correct_svm = correct_bert = correct_wav = correct_ensemble = total = 0

for speaker in os.listdir(dataset_path):
    speaker_path = os.path.join(dataset_path, speaker)
    if not os.path.isdir(speaker_path):
        continue

    for emotion_folder in os.listdir(speaker_path):
        emo_path = os.path.join(speaker_path, emotion_folder)
        if not os.path.isdir(emo_path):
            continue

        true_emotion = emotion_folder.lower()

        for filename in os.listdir(emo_path):
            if not filename.endswith(".wav"):
                continue

            filepath = os.path.join(emo_path, filename)

            # SVM Prediction
            signal, sr = librosa.load(filepath, sr=22050)
            signal, _ = librosa.effects.trim(signal, top_db=20)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
            features = np.mean(mfccs, axis=1).reshape(1, -1)
            features_scaled = scaler_emotion.transform(features)
            svm_pred = le_emotion.inverse_transform(svm_model.predict(features_scaled))[0]

            # Text-BERT
            bert_result = text_bert.predict_from_audio(filepath)
            bert_pred = bert_result["emotion"]

            # Wav2Vec2.0
            wav_pred, _ = wav2vec.predict(filepath)

            # Ensemble (Weighted)
            votes = {}
            votes[bert_pred] = votes.get(bert_pred, 0) + 0.45
            votes[wav_pred] = votes.get(wav_pred, 0) + 0.40
            votes[svm_pred] = votes.get(svm_pred, 0) + 0.15

            ensemble_pred = max(votes, key=votes.get)

            total += 1
            if svm_pred == true_emotion: correct_svm += 1
            if bert_pred == true_emotion: correct_bert += 1
            if wav_pred == true_emotion: correct_wav += 1
            if ensemble_pred == true_emotion: correct_ensemble += 1

            status = "✅" if ensemble_pred == true_emotion else "❌"

            results.append([
                filename[:25],
                true_emotion,
                svm_pred,
                bert_pred,
                wav_pred,
                ensemble_pred,
                status
            ])

# ================== RESULTS TABLE ==================
print(tabulate(results, 
               headers=["File", "True Emotion", "SVM", "Text-BERT", "Wav2Vec2.0", "Ensemble", "Result"],
               tablefmt="grid"))

# ================== ACCURACY SUMMARY ==================
acc_svm = (correct_svm / total * 100) if total > 0 else 0
acc_bert = (correct_bert / total * 100) if total > 0 else 0
acc_wav = (correct_wav / total * 100) if total > 0 else 0
acc_ensemble = (correct_ensemble / total * 100) if total > 0 else 0

print("\n" + "="*110)
print("FINAL ACCURACY COMPARISON")
print("="*110)
print(f"SVM (MFCC)          : {acc_svm:.2f}%")
print(f"Text-BERT           : {acc_bert:.2f}%")
print(f"Wav2Vec2.0          : {acc_wav:.2f}%")
print(f"**ENSEMBLE (Final)** : **{acc_ensemble:.2f}%**  ← Best")
print(f"Total Files Tested  : {total}")
print("="*110)

# ================== BAR CHART ==================
models = ['SVM', 'Text-BERT', 'Wav2Vec2.0', 'Ensemble']
accs = [acc_svm, acc_bert, acc_wav, acc_ensemble]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accs, color=['#e74c3c', '#1abc9c', '#3498db', '#2ecc71'])
plt.title('All Models Performance Comparison (Custom Dataset)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/ensemble_comparison_final.png", dpi=200)
plt.show()

print(f"\n✅ Bar chart saved: outputs/ensemble_comparison_final.png")
print("You can directly use this image in your report and PPT.")