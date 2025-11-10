import torch
import itertools
from vocab.vocab import VocabMVP
class DatasetMVP(torch.utils.data.Dataset):
    def __init__(self,vocab:VocabMVP):
        super().__init__()
        self.X=[]
        self.Y=[]
        self.data=[]
        self.vocab=vocab
        self.readDataset()
    def readDataset(self):
        fileName = "../data/cmudict.dict"
        with open(fileName,'r') as file:
            self.X =[]
            self.Y=[]
            for linea in file:
                linea = linea.strip().split(" ")
                palabra = linea[0]
                fonos = linea[1:]
                fonosNormalizados = self.vocab.normalizeFono(fonos)
                
                #print(fonos)
                palabraNormalizada=self.vocab.normalizeWord(palabra)
                
                x= self.vocab.word_to_indexes(palabraNormalizada)
                y = self.vocab.phones_to_indexes(fonosNormalizados)
                # print(palabraNormalizada,fonosNormalizados)
                # print(x,y)
                self.data.append((x,y))
    
    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return torch.tensor(x),torch.tensor(y)
    def __len__(self):
        return len(self.data)
    
