from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys
import librosa
import soundfile as sf
import numpy as np

# === Path setup ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.predict import FinalEmotiVoice

app = FastAPI(title="EmotiVoice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model once
predictor = FinalEmotiVoice()

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".webm")):
        raise HTTPException(400, detail="Only .wav or .webm files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        #  Load + normalize audio (FAST + SAFE)
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        print(f"[DEBUG] Audio length: {len(audio)}, SR: {sr}")

        #  Validation
        if audio is None or len(audio) < 1000:
            raise HTTPException(400, detail="Audio too short or invalid")

        if np.isnan(audio).any():
            raise HTTPException(400, detail="Invalid audio (NaN detected)")

        #  Pad audio (important for model stability)
        min_length = 16000 * 2  # 2 seconds
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))

        #  Convert to clean WAV
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        sf.write(wav_path, audio, sr, subtype='PCM_16')

        #  Safe prediction
        try:
            result = predictor.predict(wav_path)
        except Exception as model_error:
            print("[MODEL ERROR]:", model_error)
            raise HTTPException(500, detail="Model prediction failed")

        return result

    except HTTPException:
        raise
    except Exception as e:
        print("[SERVER ERROR]:", e)
        raise HTTPException(500, detail="Internal processing error")

    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)


@app.get("/")
def home():
    return {"message": "EmotiVoice API is running 🚀"}