import torch
import itertools
from src.vocab.vocab import Vocab
class Dataset(torch.utils.data.Dataset):
    def __init__(self,vocab:Vocab):
        super().__init__()
        self.X=[]
        self.Y=[]
        self.data=[]
        self.vocab=vocab
        self.readDataset()
    def readDataset(self):
        fileName = "./data/cmudict.dict"
        with open(fileName,'r') as file:
            self.X =[]
            self.Y=[]
            for linea in file:
                linea = linea.strip().split(" ")
                palabra = linea[0]
                fonos = linea[1:]
                y = self.vocab.phones_to_indexes(fonos)
                x= self.vocab.word_to_indexes(palabra)
                print(x)
                print(y)
                break
                self.X.append(palabra)
                self.data.append((palabra,y))

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1:]
        return torch.tensor(x),torch.tensor(y)
    def __len__(self):
        return len(self.data)
    

vocabulario = Vocab()
dataset = Dataset(vocab=vocabulario)
print(dataset.X[500:505])
print(dataset.Y[500:505])
print(dataset.__getitem__(500))