from dataset.dataset import DatasetMVP
from vocab.vocab import VocabMVP
from dataloader.dataloader import DataLoaderMVP
from g2fmodules.mvpg2f import modelMVPG2F
from training.train import Train
from phonesPredictor.predictorPhones import PredictorPhones
import torch
from utils.normalizerMVP import NormalizerMVP
from reproductor.reproductor import Reproductor

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
fonos_size = len(vocabulario.phone_to_index)
modelo = modelMVPG2F(vocab_size=vocabulario_size,phone_size=fonos_size,embed_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM)
print("MODELO_CREADO")

predictor = PredictorPhones(model=modelo,vocab=vocabulario)
print("PREDICTOR CREADO")


def main_for_train():
	entrenamiento = Train()
	entrenamiento.train(modelo,vocabulario,dataloader)

def reproducirAudio(fonos):
	pass
def main(path_model,modelo,oracion):
	predictor.load(path_model)
	fonos = predictor.obtener_fonos_oracion(oracion)
	print(fonos)
	for i in fonos:
		print (' '.join(i))
	rep = Reproductor()
	rep.reproducir_fonemas_oracion(fonos)


path = "model100.pt"
oracion = "Hello this is a banana of my brother"

main(path,modelo,oracion)

