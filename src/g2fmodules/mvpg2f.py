import torch.nn as nn
import logging
class modelMVPG2F(nn.Module):
    def __init__(self,vocab_size,phone_size,embed_dim,hidden_dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.encoder= nn.LSTM(embed_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_dim*2,phone_size)

    def forward(self,x):
        x_embedding= self.embedding(x) #Batch size,seq len,embed dim
        outputs,zz = self.encoder(x_embedding) # (batch,seq n,hidden dim*2)
        logits= self.fc(outputs) 
        return logits
