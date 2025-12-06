
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vocab.vocab import VocabMVP
from g2fmodules.mvpg2f import modelMVPG2F
from phonesPredictor.predictorPhones import PredictorPhones
from utils.normalizerMVP import NormalizerMVP
from reproductor.reproductor import Reproductor
from fastapi.responses import JSONResponse
import base64
from pathlib import Path
import logging

# from reproductor.reproductor import Reproductor
import os

app = FastAPI(title="Speech MVP API")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Enable CORS for the frontend development server
app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/data")
def read_root():
	return {
		"message": "Welcome to the Speech MVP API. Use /predict endpoint to get phonemes from text.",
		"data": "This is a sample API for text-to-phoneme prediction.",
	}

class PredictRequest(BaseModel):
	text: str

# Inicializar componentes reutilizando tu código
vocab = VocabMVP()
modelo = modelMVPG2F(vocab_size=len(vocab.word_to_index),
                     phone_size=len(vocab.phone_to_index),
                     embed_dim=128, hidden_dim=256)
predictor = PredictorPhones(model=modelo, vocab=vocab)
path_model = os.path.join(os.path.dirname(__file__), "..", "model100.pt")
predictor.load(path_model)
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model100.pt")
# try:
#     predictor.load(MODEL_PATH)
# except Exception as e:
#     # Si no hay modelo cargado, la ruta puede requerir ajuste
#     print("Aviso: no se pudo cargar modelo en", MODEL_PATH, e)

rep = Reproductor()
@app.post("/predict")
def predict(req: PredictRequest):
	logger.info(f"Received text: {req.text}")
	try:
		tokens={}
		tokens_text,fonos = predictor.obtener_fonos_oracion(req.text)
		tokens['tokens']=tokens_text
		tokens['fonos']=fonos
		output_dir = Path(__file__).resolve().parent / ".." / "tts_output"
		output_dir.mkdir(parents=True, exist_ok=True)
		audio_path = output_dir / "output_oracion.wav"
		rep.reproducir_fonemas_oracion(fonos, output=str(audio_path))
		logger.info(f"Generated audio at: {audio_path}")

		with open(audio_path, "rb") as f:
			audio_bytes = f.read()
			audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

		return {
			"token": tokens,
			"audio": {
				"base64": audio_b64,
				"mime": "audio/wav",
				"filename": audio_path.name,
			}
		}
	except Exception as e:
		logger.exception("Error in prediction")
		raise HTTPException(status_code=500, detail=str(e))

# @app.post("/speak")
# def speak(req: PredictRequest):
#     try:
#         fonos = predictor.obtener_fonos_oracion(req.text)
#         # convertir a minusculas como hace tu main.py
#         fonos_minuscular = [[ph.lower() for ph in w] if isinstance(w, list) else w for w in fonos]
#         out = reproductor.reproducir_fonemas_oracion(fonos_minuscular, output="output_oracion.wav")
#         return {"output_file": out}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# # ...existing code...