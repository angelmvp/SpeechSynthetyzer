
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vocab.vocab import VocabMVP
from g2fmodules.mvpg2f import modelMVPG2F
from phonesPredictor.predictorPhones import PredictorPhones
from utils.normalizerMVP import NormalizerMVP
from dataset.dataset import DatasetMVP
from dataloader.dataloader import DataLoaderMVP
from reproductor.reproductor import Reproductor
from fastapi.responses import JSONResponse
import base64
from pathlib import Path
import logging
import torch
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


N=10000
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256


vocabulario = VocabMVP()
print("VOCABULARIO_CREADO")

# dataset = DatasetMVP(vocabulario,N)
# print("DATASET_CREADO")

# dataloader = DataLoaderMVP(dataset,batch_size=BATCH_SIZE)
# print("DATALOADER_CREADO")	


vocabulario_size=len(vocabulario.word_to_index)
print("vocablakda SIISIISISZE",vocabulario_size)
fonos_size = len(vocabulario.phone_to_index)
modelo = modelMVPG2F(vocab_size=vocabulario_size,phone_size=fonos_size,embed_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM)
print("MODELO_CREADO")

predictor = PredictorPhones(model=modelo,vocab=vocabulario)
print("PREDICTOR CREADO")
path_model = "model50.pt"

predictor.load(path_model)
rep = Reproductor()
@app.post("/predict")
def predict(req: PredictRequest):
	logger.info(f"Received text: {req.text}")
	try:
		tokens={}
		tokens_text,fonos = predictor.obtener_fonos_oracion(req.text)
		tokens['tokens']=tokens_text
		tokens['fonos']=fonos
		output_dir = Path(__file__).resolve().parent / ".." / "output_synthesis"
		output_dir.mkdir(parents=True, exist_ok=True)
		audio_path = output_dir / "output_oracion.wav"
		rep.concatenar_fonemas(fonos, nombre_archivo_salida=str(audio_path.name))
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
