from dataset.dataset import DatasetMVP
from vocab.vocab import VocabMVP
from dataloader.dataloader import DataLoaderMVP
from g2fmodules.mvpg2f import modelMVPG2F
from training.train import Train
from phonesPredictor.predictorPhones import PredictorPhones
import torch
from utils.normalizerMVP import NormalizerMVP
from reproductor.reproductor import Reproductor
from torch.utils.data import random_split
from prosodiamodules.prosodia import Prosodia_module
from utils.token import Token
from typing import List



N=10000
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256


vocabulario = VocabMVP()
print("VOCABULARIO_CREADO")

dataset = DatasetMVP(vocabulario,N)
print("DATASET_CREADO")

dataloader = DataLoaderMVP(dataset,batch_size=BATCH_SIZE)
print("DATALOADER_CREADO")	


vocabulario_size=len(vocabulario.word_to_index)
print("vocablakda SIISIISISZE",vocabulario_size)
fonos_size = len(vocabulario.phone_to_index)
modelo = modelMVPG2F(vocab_size=vocabulario_size,phone_size=fonos_size,embed_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM)
print("MODELO_CREADO")

predictor = PredictorPhones(model=modelo,vocab=vocabulario)
print("PREDICTOR CREADO")
prosodia = Prosodia_module()
print("PROSODIA CREADO")

def main_for_train():
	entrenamiento = Train()
	epocas = 10
	lr=1e-3


# 2. División del Dataset
	TRAIN_RATIO = 0.85  # 90% para entrenamiento
	train_size = int(TRAIN_RATIO * len(dataset))
	val_size = len(dataset) - train_size

	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	print(f"Tamaño Entrenamiento: {len(train_dataset)}, Tamaño Validación: {len(val_dataset)}")
	BATCH_SIZE = 64
	train_dataloader = DataLoaderMVP(train_dataset, batch_size=BATCH_SIZE)
	val_dataloader = DataLoaderMVP(val_dataset, batch_size=BATCH_SIZE) # Usamos el mismo BATCH_SIZE

	entrenamiento.train(modelo,vocabulario,train_dataloader,val_dataloader,BATCH_SIZE,epochs=epocas,learning_rate=lr)

def reproducirAudio(fonos):
	pass
def main(path_model,modelo,oracion):
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
	rep = Reproductor()
	rep.generar_audios(Tokens)


path = "model50.pt"
oracion = "Bijan Robinson, is a player who plays in the falcons . He was born on 21/09/2002 . His contract is worth 12 million dollars per year. He has the record of the youngest player to reach 1000 yards, and has recorded 4 touchdowns in just 2 games! , unbelievable."

main(path,modelo,oracion)

