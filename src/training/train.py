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

			# --- FASE DE VALIDACIÓN ---
			avg_val_loss = self.evaluate(model, val_dataloader, criterion)

			print(f"⭐ Época {epoca+1}/{epochs}: Pérdida Entrenamiento={avg_train_loss:.4f} | Pérdida Validación={avg_val_loss:.4f}")



	def evaluate(self, model, dataloader, criterion):
			"""Calcula la pérdida en el conjunto de validación."""
			model.eval()  # Pone el modelo en modo evaluación (desactiva Dropout, etc.)
			total_val_loss = 0
			with torch.no_grad(): # Desactiva el cálculo de gradientes para ahorrar memoria y tiempo
					for x_batch, y_batch in dataloader:
						x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
						# El criterio de pérdida ignora automáticamente el índice 0 (padding)
						predicho = model.forward(x_batch)
						loss = criterion(predicho.view(-1, predicho.size(-1)), y_batch.view(-1))
						total_val_loss += loss.item()

			model.train() # Vuelve el modelo a modo entrenamiento
			return total_val_loss / len(dataloader)	
                