import torch
import itertools
from vocab.vocab import VocabMVP
import random
class DatasetMVP(torch.utils.data.Dataset):
    def __init__(self,vocab:VocabMVP,N=10000):
        super().__init__()
        self.data=[]
        self.vocab=vocab
        self.N=N
        self.readDataset()
    def readDataset(self):
        """
        Lee el archivo cmudict.dict y crea el dataset de pares (palabra, fonos)
        """
        fileName = "../data/cmudict.dict"
        with open(fileName,'r') as file:
            for linea in file:
                linea = linea.strip().split(" ")
                palabra = linea[0]
                fonos = linea[1:]
                fonosNormalizados = self.vocab.normalizeFono(fonos)

                #print(fonos)
                palabraNormalizada=self.vocab.normalizeWord(palabra)
                x= self.vocab.word_to_indexes(palabraNormalizada)
                y = self.vocab.phones_to_indexes(fonosNormalizados)
                y = y + [2]# añadimos el indice EOS
                # print(palabraNormalizada,fonosNormalizados)
                # print(x,y)
                self.data.append((x,y))
        random.shuffle(self.data) # Mezclamos los datos para mayor aleatoriedad
        self.data =self.data[:self.N]

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return torch.tensor(x),torch.tensor(y)
    def __len__(self):
        return len(self.data)

