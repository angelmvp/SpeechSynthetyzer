
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vocab.vocab import VocabMVP
from g2fmodules.mvpg2f import modelMVPG2F
from phonesPredictor.predictorPhones import PredictorPhones
# from reproductor.reproductor import Reproductor
import os

app = FastAPI(title="Speech MVP API")

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

# class PredictRequest(BaseModel):
#     text: str

# Inicializar componentes reutilizando tu código
vocab = VocabMVP()
modelo = modelMVPG2F(vocab_size=len(vocab.word_to_index),
                     phone_size=len(vocab.phone_to_index),
                     embed_dim=128, hidden_dim=256)
predictor = PredictorPhones(model=modelo, vocab=vocab)

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model100.pt")
# try:
#     predictor.load(MODEL_PATH)
# except Exception as e:
#     # Si no hay modelo cargado, la ruta puede requerir ajuste
#     print("Aviso: no se pudo cargar modelo en", MODEL_PATH, e)

# reproductor = Reproductor()

# @app.post("/predict")
# def predict(req: PredictRequest):
#     try:
#         fonos = predictor.obtener_fonos_oracion(req.text)
#         return {"text": req.text, "phonemes": fonos}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

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