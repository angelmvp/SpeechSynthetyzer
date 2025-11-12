from dataset.dataset import DatasetMVP
from vocab.vocab import VocabMVP
from dataloader.dataloader import DataLoaderMVP
from g2fmodules.mvpg2f import modelMVPG2F
from training.train import Train
import torch
import pronouncing
def main():
	vocabulario = VocabMVP()
	print("VOCABULARIO_CREADO")
	N=10000
	dataset = DatasetMVP(vocabulario,N)
	print("DATASET_CREADO")
	print(len(dataset.data))
	BATCH_SIZE = 64
	dataloader = DataLoaderMVP(dataset,batch_size=BATCH_SIZE)
	
	print("DATALOADER_CREADO")
	vocabulario_size=len(vocabulario.word_to_index)
	fonos_size = len(vocabulario.phone_to_index)
	embedding_dim = 128
	hidden_dim = 256
	modelo = modelMVPG2F(vocab_size=vocabulario_size,phone_size=fonos_size,embed_dim=embedding_dim,hidden_dim=hidden_dim)
	print("MODELO_CREADO")
	
	entrenamiento = Train()
	epocas = 10
	lr=1e-3
	entrenamiento.train(modelo,vocabulario,dataloader,BATCH_SIZE,epochs=epocas,learning_rate=lr)
def test():
	vocabulario = VocabMVP()
	print("VOCABULARIO_CREADO")


	vocabulario_size=len(vocabulario.word_to_index)
	fonos_size = len(vocabulario.phone_to_index)
	embedding_dim = 128
	hidden_dim = 256
	saved_model_path = "model"
	state_dict =torch.load(saved_model_path,map_location='cpu')
	modelo = modelMVPG2F(vocab_size=vocabulario_size,phone_size=fonos_size,embed_dim=embedding_dim,hidden_dim=hidden_dim)
	modelo.load_state_dict(state_dict=state_dict)
	modelo.eval()
	print("MODELO_CREADO")

	palabras = ['addiction','banana','surprise','huevon','maricon','sheinbaun','hilinski']
	for palabra in palabras:
		x=torch.tensor([vocabulario.word_to_indexes(palabra)],dtype=torch.long)
		with torch.no_grad():
			logits = modelo(x)
		prediccion_indices= torch.argmax(logits,dim=-1).squeeze(0).tolist()
		pred_phones = vocabulario.indexes_to_phones(prediccion_indices)
		fonos_libreria = pronouncing.phones_for_word(palabra)
		print(f"palabra: {palabra}  fonos MVP {pred_phones}  fonos libreria {fonos_libreria}")
		

test()

