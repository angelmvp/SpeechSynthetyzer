
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vocab.vocab import VocabMVP
from g2fmodules.mvpg2f import modelMVPG2F
from phonesPredictor.predictorPhones import PredictorPhones
from reproductor.reproductor import Reproductor
import base64
from pathlib import Path
import logging
import torch
from prosodiamodules.prosodia import Prosodia_module
from utils.token import Token
from typing import List
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
logger.info("REPRODUCTOR CREADO")
prosodia = Prosodia_module()
print("PROSODIA CREADO")
OUTPUT_PATH = "./output_audios"
rep = Reproductor(output_path=OUTPUT_PATH)
print("REPRODUCTOR CREADO")
@app.post("/predict")
def predict(req: PredictRequest):
	logger.info(f"Received text: {req.text}")
	try:
		oracion = req.text
		print("Oracion Original",oracion )
		predictor.load(path_model)
		Tokens:List[Token] = []
		palabras = predictor.obtener_tokens_normalizados(oracion)
		print("\n\nPalabras Normalizadas:",palabras)
		palabras,fonos = predictor.obtener_fonos_oracion(palabras)
		print("\n\nFonos Obtenidos:",fonos)
		prosodia_stress = prosodia.obtener_indices_prosodia(palabras)
		print("\n\nPalabras con Prosodia:",palabras)
		print("\nprosodia",prosodia_stress)
		print("\n\n Todos los tokens con sus fonos obtenidos denustr omodelo y con la prosodia asignada:")
		for i,palabra in enumerate(palabras):
			tokenNuevo = Token(token=palabra,token_text=palabra,fonos=fonos[i],prosodia=prosodia_stress[i])
			tokenNuevo.to_string()
			Tokens.append(tokenNuevo)
		
		rep.generar_audios(Tokens)

		tokens={}
		for token in Tokens:
			tokens[token.get_token()] = {
				"token": token.token,
				"fonos": token.fonos,
				"stress_fono": token.stress_fono,
				"stress_prosodia": token.stress_prosodia,
				"fonos_prosodia": token.fonos_prosodia,
				"signo": token.signo,
			}
		output_dir = Path(__file__).resolve().parent / ".." / OUTPUT_PATH
		output_dir.mkdir(parents=True, exist_ok=True)
		audio_path = output_dir / "output_oracion.wav"
		audio_sin_prosodia = output_dir / "output.wav"
		audio_con_prosodia = output_dir / "output_prosodia.wav"
		logger.info(f"Generated audio at: {audio_sin_prosodia} and {audio_con_prosodia}")

		with open(audio_sin_prosodia, "rb") as f:
			audio_bytes = f.read()
			audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
		with open(audio_con_prosodia, "rb") as f:
			audio_bytes = f.read()
			audio_con_prosodia_b64 = base64.b64encode(audio_bytes).decode("ascii")

		return {
			"tokens": tokens,
			"audio": {
				"base64": audio_b64,
				"mime": "audio/wav",
				"filename": audio_path.name,
			},
			"audio_con_prosodia": {
				"base64": audio_con_prosodia_b64,
				"mime": "audio/wav",
				"filename": audio_con_prosodia.name,
			},
		}
	except Exception as e:
		logger.exception("Error in prediction")
		raise HTTPException(status_code=500, detail=str(e))
