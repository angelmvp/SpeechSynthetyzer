from g2fmodules.mvpg2f import modelMVPG2F
from vocab.vocab import VocabMVP
from dataloader.dataloader import DataLoaderMVP
import torch.nn as nn
import torch

class Train:
	def __init__(self):
		pass
	def train(self,model:modelMVPG2F ,vocab:VocabMVP,dataloader:DataLoaderMVP,batch_size,epochs,learning_rate):
		criterion =nn.CrossEntropyLoss(ignore_index=0) 
		optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
		for epoca in range(epochs):
			total_loss= 0
			for x_batch,y_batch in dataloader:
				# print(x_batch,y_batch)	
				optimizer.zero_grad()
				predicho = model.forward(x_batch)
				loss= criterion(predicho.view(-1, predicho.size(-1)), y_batch.view(-1))
				loss.backward()

				optimizer.step()
				total_loss+=loss.item()	
			print(f"Epoch {epoca+1}: loss={total_loss/len(dataloader):.4f}")
		save_path="./model"
		torch.save(model.state_dict(), save_path)
		print(f"✅ Modelo guardado en: {save_path}")			
                