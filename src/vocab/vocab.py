class Vocab:
    def __init__(self):
        self.word_to_index = {'PAD': 0, 'UNK': 1}
        self.index_to_word = {}
        self.next_index = 0
        self.phones  = self.read_phones()
        self.read_symbols()
    def read_phones(self):
        phones = {}
        filename="cmudict.phones"
        file  = open(filename,'r',encoding='utf-8')
        for line in file:
            linea = line.strip().split("\t")
            print(linea)
            tipo_fono = linea[1]
            fono= linea[0]
            print(tipo_fono,fono)
            if tipo_fono not in phones:
                phones[tipo_fono] = []
            phones[tipo_fono].append(fono)
        return phones
    def read_symbols(self):
        filename="cmudict.symbols"
        file = open(filename,'r',encoding='utf-8')
        for i,line in enumerate(file):
            symbol = line.strip()
            if symbol not in self.word_to_index:
                self.word_to_index[symbol] = i + 2
        self.index_to_word = {v:k for k,v in self.word_to_index.items()}

vocab =  Vocab()
print(vocab.word_to_index)
print("*--------*")
print(vocab.index_to_word)
     