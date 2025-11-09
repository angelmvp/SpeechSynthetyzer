class Vocab:
    def __init__(self):
        self.phone_to_index = {'PAD': 0, 'UNK': 1}
        self.index_to_phone = {}
        self.word_to_index= {'PAD': 0, 'UNK': 1}
        self.index_to_word = {}
        self.phones  = self.read_phones()
        self.read_symbols()
    def read_phones(self):
        phones = {}
        filename="./data/cmudict.phones"
        file  = open(filename,'r',encoding='utf-8')
        for line in file:
            linea = line.strip().split("\t")
            tipo_fono = linea[1]
            fono= linea[0]
            print(tipo_fono,fono)
            if tipo_fono not in phones:
                phones[tipo_fono] = []
            phones[tipo_fono].append(fono)
        return phones
    def read_symbols(self):
        filename="./data/cmudict.symbols"
        file = open(filename,'r',encoding='utf-8')
        for i,line in enumerate(file):
            symbol = line.strip()
            if symbol not in self.phone_to_index:
                self.phone_to_index[symbol] = i + 2
        self.index_to_phone = {v:k for k,v in self.phone_to_index.items()}
    def creat_vocab_words(self):
        vocabulario = "ABCDEFGHIJKLMNOPQRSTUVWXYZ'()"
        for i,char in enumerate(vocabulario):
            if char not in self.word_to_index:
                self.word_to_index[char] = i + 2
        self.index_to_phone = {v:k for k,v in self.word_to_index.items()}
    def phones_to_indexes(self,phones):
        return [self.phone_to_index[fono] for fono in phones]
    def word_to_indexes(self,word):
        return [self.word_to_index[letra] for letra in word]
vocab =  Vocab()
print(vocab.phone_to_index)
print("*--------*")
print(vocab.index_to_phone)
print("-----")
print(vocab.phones)
     