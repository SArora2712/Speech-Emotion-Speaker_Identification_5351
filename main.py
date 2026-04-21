from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys
from pydub import AudioSegment
import librosa
import soundfile as sf

# Load audio (auto detects format)

# === IMPORTANT: Add paths so Python can find your files ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.predict import FinalEmotiVoice   # ← Updated import

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

#  Convert ANY audio to proper WAV
audio, sr = librosa.load(file_path, sr=None)
wav_path = file_path.rsplit(".", 1)[0] + ".wav"
sf.write(wav_path, audio, sr)

# Now send WAV to model
result = predictor.predict(wav_path)

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".webm")):
        raise HTTPException(400, detail="Only .wav or .webm files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predictor.predict(file_path)
        return result
    except Exception as e:
        raise HTTPException(500, detail=f"Analysis failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# Optional: Health check
@app.get("/")
def home():
    return {"message": "EmotiVoice API is running 🚀"}