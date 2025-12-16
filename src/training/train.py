from g2fmodules.mvpg2f import modelMVPG2F
from vocab.vocab import VocabMVP
from dataloader.dataloader import DataLoaderMVP
import torch.nn as nn
import torch

class Train:
	def __init__(self):
		self.lr=1e-3
		self.epochs=100
		self.criterion =nn.CrossEntropyLoss(ignore_index=0) 

	def train(self,model:modelMVPG2F ,vocab:VocabMVP,dataloader:DataLoaderMVP,val_dataloader,batch_size,epochs,learning_rate):
		"""
		Entrenamiento del modelo
		"""
		model.train()
		model.to(self.device)
		criterion =nn.CrossEntropyLoss(ignore_index=0)
		optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)
		for epoca in range(epochs):
			total_loss= 0
			for x_batch,y_batch in dataloader:
				x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

				# print(x_batch,y_batch)
				optimizer.zero_grad()
				predicho = model.forward(x_batch)
				loss= criterion(predicho.view(-1, predicho.size(-1)), y_batch.view(-1))
				loss.backward()

				optimizer.step()
				total_loss+=loss.item()
			avg_train_loss = total_loss / len(dataloader)

			## Validacion 
			avg_val_loss = self.evaluate(model, val_dataloader, criterion)

			print(f"Epoca {epoca+1}/{epochs}: Perdida Entrenamiento={avg_train_loss:.4f}, Perdida Validacion={avg_val_loss:.4f}")



	def evaluate(self, model, dataloader, criterion):
		"""Calcular la perdida en el conjutn ode validacione """
		model.eval()  
		total_val_loss = 0
		with torch.no_grad():
				for x_batch, y_batch in dataloader:
					x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
					predicho = model.forward(x_batch)
					loss = criterion(predicho.view(-1, predicho.size(-1)), y_batch.view(-1))
					total_val_loss += loss.item()

		
		return total_val_loss / len(dataloader)	
                