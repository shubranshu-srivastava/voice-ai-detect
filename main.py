from fastapi import FastAPI, Header, HTTPException
import base64, uuid, os
import librosa
import numpy as np

API_KEY = os.getenv("API_KEY", "GUVI12345")


app = FastAPI()

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_var = np.var(pitch)
    
    energy = np.mean(librosa.feature.rms(y=y))

    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    return pitch_var, energy, flatness

@app.post("/detect")
def detect_voice(payload: dict, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        audio_b64 = payload["audio"]
        audio_bytes = base64.b64decode(audio_b64)

        filename = f"{uuid.uuid4()}.mp3"
        with open(filename, "wb") as f:
            f.write(audio_bytes)

        pitch_var, energy, flatness = extract_features(filename)
        os.remove(filename)

        if pitch_var < 5 and flatness > 0.2:
            label = "AI_GENERATED"
            confidence = 0.85
            reason = "Low pitch variation and high spectral flatness"
        else:
            label = "HUMAN"
            confidence = 0.80
            reason = "Natural pitch variation detected"

        return {
            "result": label,
            "confidence": confidence,
            "explanation": reason
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

from fastapi import UploadFile, File

@app.post("/detect-file")
async def detect_file(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    audio_bytes = await file.read()

    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_bytes)

    pitch_var, energy, flatness = extract_features("temp_audio.mp3")

    return {
        "pitch_variance": float(pitch_var),
        "energy": float(energy),
        "flatness": float(flatness),
        "result": "Human voice detected"
    }

