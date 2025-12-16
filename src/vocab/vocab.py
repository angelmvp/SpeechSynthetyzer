class VocabMVP:
    def __init__(self):
        self.phone_to_index = {'PAD': 0, 'UNK': 1,'EOS': 2}
        self.index_to_phone = {}
        self.word_to_index= {'PAD': 0, 'UNK': 1}
        self.index_to_word = {}
        self.phones  = self.read_phones()
        self.read_symbols()
        self.creat_vocab_words()
    def read_phones(self):
        """
        Lee el archivo de fonemas y devuelve un diccionario con los tipos de fonemas 
        como claves y listas de fonemas como valores.
        """
        phones = {}
        filename="../data/cmudict.phones"
        file  = open(filename,'r',encoding='utf-8')
        for line in file:
            linea = line.strip().split("\t")
            tipo_fono = linea[1]
            fono= linea[0]
            # print(tipo_fono,fono)
            if tipo_fono not in phones:
                phones[tipo_fono] = []
            phones[tipo_fono].append(fono)
        return phones
    def read_symbols(self):
        filename="../data/cmudict.symbols"
        file = open(filename,'r',encoding='utf-8')
        for i,line in enumerate(file):
            symbol = line.strip()
            if symbol not in self.phone_to_index:
                self.phone_to_index[symbol] = i + 3
        self.index_to_phone = {v:k for k,v in self.phone_to_index.items()}
    def creat_vocab_words(self):
        vocabulario = "abcdefghijklmnopqrstuvwxyz'.-"
        for i,char in enumerate(vocabulario):
            if char not in self.word_to_index:
                self.word_to_index[char] = i + 2
        self.index_to_word = {v:k for k,v in self.word_to_index.items()}
    def phones_to_indexes(self,phones):
        return [self.phone_to_index[fono] for fono in phones]
    def word_to_indexes(self,word):
        return [self.word_to_index[letra] for letra in word]
    def indexes_to_phones(self,indices):
        return [self.index_to_phone[indice] for indice in indices]
    def indexes_to_word(self,indices):
        return [self.index_to_word[indice] for indice in indices]
    def normalizeFono(self,fonos):
        """
        Al leer los fonemas de un archivo, algunos pueden tener caracteres no deseados.
        Esta función elimina esos caracteres no deseados de la lista de fonemas.
        """
        fonosAEvitar = '#'
        index=0
        for fono in fonos:
            if fono in fonosAEvitar:
                return fonos[:index]
            index+=1
        return fonos
    def normalizeWord(self,word):
        """
        Al leer las palabras de un archivo, algunas pueden tener caracteres no deseados.
        Esta función elimina esos caracteres no deseados de la palabra.
        """
        letras_a_evitar = "#()"
        index = 0
        for letra in word:
            if letra in letras_a_evitar:
                return word[:index]
            index+=1
        return word