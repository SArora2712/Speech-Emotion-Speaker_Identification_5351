import os
import sys
import numpy as np
import librosa
import joblib
import sounddevice as sd
import soundfile as sf
import tempfile
import time
from speech_type_classifier import Speech_type_keywords,speech_type_desc
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False


model_folder="models"
sample_rate = 22050
N_mfcc=40
record_seconds=5
emotion_map={
    "01":"neutral",
    "02":"calm",
    "03":"happy",
    "04":"sad",
    "05":"angry",
    "06":"fearful",
    "07":"disgust",
    "08":"surprised"

}

#---Load Models
def load_model():
    print("\n"+"-"*50)
    print(f"Loading the Saved models from the {model_folder} folder")
    print("-"*50)
    
    required={
        "emotion_model": "emotion_model.pkl",
        "label_encoder_emotion": "label_encoder_emotion.pkl",
        "scaler_emotion": "scaler_emotion.pkl",
        "speaker_model": "speaker_model.pkl",
        "label_encoder_speaker": "label_encoder_speaker.pkl",
        "scaler_speaker": "scaler_speaker.pkl"
    }

    models={}
    all_found=True

    for key,filename in required.items():
        fpath=os.path.join(model_folder,filename)
        if os.path.exists(fpath):
            models[key]=joblib.load(fpath)
            print(f"Loaded {filename} successfully.")

        else:
            print(f"Error: {filename} not found in {model_folder} folder.")
            all_found=False

    if not all_found:
        print("Please ensure all required model files are present in the models folder.")
        print("Run the training script\"train_emotion.py\" to generate the missing files.")
        return None
    
    le_speaker=models["label_encoder_speaker"]
    le_emotion=models["label_encoder_emotion"]
    print(f"Known Speakers: {list(le_speaker.classes_)}")
    print(f"Known Emotions: {list(le_emotion.classes_)}")
    print("All models loaded successfully.\n")
    return models

def preprocessing(filepath):
    signal,sr=librosa.load(filepath,sr=sample_rate)
    if len(signal)==0:
        raise ValueError("Audio file is empty or could not be loaded.")
    
    max_amp=np.max(np.abs(signal))
    if max_amp>0:
        signal=signal/max_amp
    
    signal,_=librosa.effects.trim(signal,top_db=20)

    if len(signal)==0:
        raise ValueError("Audio file is silent after trimming.")
    
    return signal,sr

def extract_features(signal,sr):
    mfccs=librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=N_mfcc)
    print("MFCC raw shape:", mfccs.shape)
    mfccs_mean=np.mean(mfccs,axis=1)
    print("MFCC mean shape:", mfccs_mean.shape)
    return mfccs_mean.reshape(1,-1)


def predict_emotion(features,models):
    scaler=models["scaler_emotion"]
    model=models["emotion_model"]
    le=models["label_encoder_emotion"]
    features_scaled=scaler.transform(features)
    pred_encoded=model.predict(features_scaled)[0]
    emotion=le.inverse_transform([pred_encoded])[0]

    confidence=None

    if hasattr(model,"predict_proba"):
        proba=model.predict_proba(features_scaled)[0]
        confidence=proba[pred_encoded]*100

    return emotion,confidence

def predict_speaker(features,models):
    scaler=models["scaler_speaker"]
    model=models["speaker_model"]
    le=models["label_encoder_speaker"]
    features_scaled=scaler.transform(features)
    pred_encoded=model.predict(features_scaled)[0]
    speaker=le.inverse_transform([pred_encoded])[0]

    confidence=None

    if hasattr(model,"predict_proba"):
        proba=model.predict_proba(features_scaled)[0]
        confidence=proba[pred_encoded]*100

    return speaker,confidence

def speech_to_text(filepath):
    if not SPEECH_RECOGNITION_AVAILABLE:
        return ""

    recognizer=sr.Recognizer()
    
    try:
        signal,sr_rate=librosa.load(filepath,sr=16000,mono=True)
        temp=tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp.name, signal, 16000)
        with sr.AudioFile(temp.name) as source:
            recognizer.adjust_for_ambient_noise(source,duration=3)
            audio=recognizer.record(source)

        text=recognizer.recognize_google(audio)
        return text.lower()
    

    except Exception as e:
        return ""
    
    finally:
        try:
            if os.path.exists(temp.name):
                os.remove(temp.name)
        except Exception as e:
            pass

def classify_speech_type(filepath):
    transcript=speech_to_text(filepath)
    scores     = {st: 0.0 for st in Speech_type_keywords}
    if not transcript:
        return "conversational","",scores
    
    for speech_type,data in Speech_type_keywords.items():
        score=0
        for keyword in data["keywords"]:
            if keyword in transcript:
                score+=1

        for phrase in data["phrases"]:
            if phrase in transcript:
                score+=2
        scores[speech_type]=score*data["weight"]

    if all(s==0 for s in scores.values()):
        return "conversational",transcript,scores
    
    speech_type=max(scores,key=scores.get)
    return speech_type,transcript,scores

def generate_sentence(speaker,emotion,speech_type):
    desc=speech_type_desc.get(speech_type,"during a conversation")
    return f"{speaker} is speaking with {emotion} emotion in a {speech_type} manner. {desc}"

def predict_from_file(file_path,models,show_detail=True):
    if not os.path.exists(file_path):
        if show_detail:
            print(f"Error:File not found")
        return None
    if show_detail:
        print(f"\n"+"-"*50)
        print(f"Analyzing Audio File")
        print("-"*50)
        print(f"File: {os.path.basename(file_path)}")

     
        
    try:
            #1->Preprocessing
        if show_detail:
            print("1. Preprocessing audio file...")
        signal,sr=preprocessing(file_path)
        duration=len(signal)/sr
        if show_detail:
            print(f" Duration after preprocessing: {duration:.2f} seconds")

            #2->Feature Extraction

        if show_detail:
            print("2. Extracting MFCC features...")
        features=extract_features(signal,sr)
        if show_detail:
            print(f" Extracted {features.shape[1]} features")

            #3->Predict Emotion
        if show_detail:
            print("3. Predicting Emotion...")
        emotion,emotion_conf=predict_emotion(features,models)
        if show_detail:
            conf_str=f" (Confidence: {emotion_conf:.1f}%)" if emotion_conf  else ""
            print(f" Predicted Emotion: {emotion}{conf_str}")

            #4->Predict Speaker
        if show_detail:
            print("4. Predicting Speaker...")
        speaker,speaker_conf=predict_speaker(features,models)
        if show_detail:
            conf_str=f" (Confidence: {speaker_conf:.1f}%)" if speaker_conf else ""
            print(f" Predicted Speaker: {speaker}{conf_str}")

            #5->Classify Speech Type
        if show_detail:
            print("5. Classifying Speech Type...")
        speech_type,transcript,speech_type_scores=classify_speech_type(file_path)

        if show_detail:
            if transcript:
                print(f" Transcribed Text: \"{transcript}\"")
            else:
                print("Transcribed Text: (No speech detected or transcription failed)")
                
            print(f" Predicted Speech Type: {speech_type}")

            #6->Generate Sentence
            sentence=generate_sentence(speaker,emotion,speech_type)
        if show_detail:
            print("\n"+"-"*50)
            print("Generated Sentence:")
            print("-"*50)
            print(sentence)

            print("\n Detailed Breakdown:")
            print(f" Speaker: {speaker} (Confidence: {speaker_conf:.1f}%)" if speaker_conf else f" Speaker: {speaker}")
            print(f" Emotion: {emotion} (Confidence: {emotion_conf:.1f}%)" if emotion_conf else f" Emotion: {emotion}")
            print(f" Speech Type: {speech_type}")
            if transcript:
                print(f" Transcribed Text: \"{transcript}\"")
            if any(s>0 for s in speech_type_scores.values()):
                print("\n Speech Type Scores:")
                for st,sc in sorted(speech_type_scores.items(),key=lambda x: x[1],reverse=True):
                    bar="█"*int(sc) if sc >0 else ""
                    marker="<--" if st==speech_type else ""
                    print(f" {st:<20    }: {sc:.1f} {bar} {marker}")
            return {
                "speaker": speaker,
                "speaker_confidence": speaker_conf,
                "emotion": emotion,
                "emotion_confidence": emotion_conf,
                "speech_type": speech_type,
                "transcript": transcript,
                "generated_sentence": sentence
                }
    except Exception as e:
        if show_detail:
            print(f"Error durng preiciton: {str(e)}")
        return None
        
def record_audio_and_predict(models):
    print("\n"+"-"*50)
    print("Recording Audio from Microphone")
    print("-"*50)

    if not SPEECH_RECOGNITION_AVAILABLE:
        print("SpeechRecognition library is not available. Please install it to use this feature.")

    print(f"Recording for {record_seconds} seconds...")
    for i in range(3,0,-1) :
        print(f" Starting in {i}...",end="\r")
        time.sleep(1)

    print(" Recording... ")
    audio=sd.rec(
        int(record_seconds*sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32")
    audio=audio.flatten()
    sd.wait()
    print("Recording complete. Analyzing...")
    temp = os.path.join(os.getcwd(), "temp_recording.wav")
    sf.write(temp, audio, sample_rate)
    result=predict_from_file(temp.name,models,show_detail=True)
    try:
        os.remove(temp.name)
    except Exception as e:
        pass
    return result

def predict_folder(folder_path,models):
    print("\n"+"-"*50)
    print(f"Predicting from Audio Files in Folder: {folder_path}")
    print("-"*50)

    if not os.path.exists(folder_path):
        print("Error: Folder not found.")
        return None
    
    wav_files=sorted([
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(".wav")
        ])

    if not wav_files:
        print("No audio files found in the folder.")
        return 
    
    print(f"Found {len(wav_files)} audio files. Processing each file...\n")

    results=[]
    for filename in wav_files:
        fpath=os.path.join(folder_path,filename)
        print(f"Analyzing File: {filename}")
        result=predict_from_file(fpath,models,show_detail=True)
        if result:
            results.append({"file": filename, **result})
            print(f"Result: {result['generated_sentence']}\n")
            spk_c=f"({result['speaker_confidence']:.0f}%)" if result['speaker_confidence'] else ""
            emo_c=f"({result['emotion_confidence']:.0f}%)" if result['emotion_confidence'] else ""  
            print(f"Summary: {result['speaker']} {spk_c} is speaking with {result['emotion']} emotion {emo_c} in a {result['speech_type']} manner.\n")
            print()
    
    if results:
        print("\n"+"-"*50)
        print("Summary of All Predictions:")
        print("-"*50)
        print(f"{'File':<20} {'Speaker':<15} {'Emotion':<15} {'Speech Type':<15}")
        print(" "+"-"*65)
        for res in results:
            fname_short=res['file'][:33]+"..." if len(res['file'])>35 else res['file']
            print(f"{fname_short:<20} {res['speaker']:<15} {res['emotion']:<15} {res['speech_type']:<15}")
            print(f"Total Files Processed: {len(results)}")
            print(f"-"*75)


def test_on_ravdess(models):
    ravdess_path=os.path.join("data","Ravdess","Audio_Speech_Actors_01-24")

    if not os.path.exists(ravdess_path):
        print("RAVDESS dataset not found. Please download and extract it to the data folder.")
        return
    
    print("\n"+"-"*50)
    print("Testing on RAVDESS Dataset") 
    print("-"*50)

    tested=0
    correct=0
    results=[]

    actor_folders = sorted(os.listdir(ravdess_path))[:5]
    print(f"  Testing on first 5 actors: {actor_folders}\n")

    for actor_folder in actor_folders:
        actor_path=os.path.join(ravdess_path,actor_folder)
        if not os.path.isdir(actor_path):
            continue

        files=sorted([f for f in os.listdir(actor_path)
                      if f.lower().endswith(".wav")])[:2]
        
        for fname in files:
            fpath=os.path.join(actor_path,fname)
            parts=fname.replace(".wav","").split("-")

            if len(parts)<3:
                continue
            actual_emotion=emotion_map.get(parts[2],"unknown")
            
            result=predict_from_file(fpath,models,show_detail=False)
            if result:
                predicted_emotion=result['emotion']
                match="OK" if predicted_emotion==actual_emotion else "WRONG"
                if predicted_emotion==actual_emotion:
                    correct+=1
                tested+=1
                results.append({
                    "file": fname,
                    "actual_emotion": actual_emotion,
                    "predicted_emotion": predicted_emotion,
                    "match": match
                })
                print(f"{match} {fname[:35]}")
                print(f"Actual: {actual_emotion:<12}"
                      f"predicted: {predicted_emotion}")
                
    print(f"\n test accuracy:{correct}/{tested}"
          f"={correct/tested*100:.1f}%" if tested>0 else "")
    
def main():
    print("\n"+"-"*55)
    print("  SPEECH EMOTION DETECTION AND")
    print("  SPEAKER IDENTIFICATION SYSTEM")
    print("  Minor Project — Sukhman Arora | A25305223154")
    print("  Amity University Punjab, Mohali")
    print("=" * 55)
 
    # Load all models once at startup
    models = load_model()
    if models is None:
        return
 
    while True:
        print("\n" + "=" * 55)
        print("  MAIN MENU")
        print("=" * 55)
        print("  1. Analyze a .wav file")
        print("  2. Record from microphone and analyze")
        print("  3. Batch analyze all files in a folder")
        print("  4. Test on RAVDESS sample files")
        print("  5. Exit")
 
        choice = input("\n  Enter choice (1/2/3/4/5): ").strip()
        if choice == "1":
            path = input("\n  Enter full path to .wav file: ").strip()
            path = path.strip('"')
            predict_from_file(path, models)
 
        elif choice == "2":
            record_audio_and_predict(models)
 
        elif choice == "3":
            folder = input("\n  Enter folder path: ").strip()
            folder = folder.strip('"')
            predict_folder(folder, models)
 
        elif choice == "4":
            test_on_ravdess(models)
 
        elif choice == "5":
            print("\n  Thank you. Goodbye!")
            break
 
        else:
            print("  Invalid choice. Enter 1, 2, 3, 4 or 5.")
 
        input("\n  Press ENTER to continue...")
 
 
if __name__ == "__main__":
    main()
