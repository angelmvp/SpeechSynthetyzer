import torch.nn as nn

# class modelMVPG2F(nn.Module):
#     def __init__(self,vocab_size,phone_size,embed_dim,hidden_dim):
#         super().__init__()
#         self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
#         self.encoder= nn.LSTM(embed_dim,hidden_dim,batch_first=True,bidirectional=True)
#         self.fc=nn.Linear(hidden_dim*2,phone_size)
#         pass
#     def forward(self,x):
#         x_embedding= self.embedding(x) #Bathc siez,seqlen
#         outputs,zz = self.encoder(x_embedding)
#         logits= self.fc(outputs) # ( batc,seql n,phone size)
#         return logits 
        
#     def get_phonemes_from_text(self,text:str):
#         return "MACHIMBURRI"