from vocab.vocab import VocabMVP
import torch
from utils.normalizerMVP import NormalizerMVP
import logging
class PredictorPhones():
    def __init__(self,model,vocab:VocabMVP):
        self.model=model
        self.vocab=vocab
        pass
    def load(self,path):
        state = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state)
        self.model.eval()
    def obtener_fonos_palabra(self,word):
        EOS_INDEX=2
        indexes = self.vocab.word_to_indexes(word)
        x = torch.tensor([indexes])
        with torch.no_grad():
            logits = self.model(x)

        prediccion_indices = torch.argmax(logits,dim=-1).squeeze(0).tolist()
        if EOS_INDEX in prediccion_indices:
            eos_position = prediccion_indices.index(EOS_INDEX)
            prediccion_indices = prediccion_indices[:eos_position]
        fonos =self.vocab.indexes_to_phones(prediccion_indices)
        return fonos
    def obtener_fonos_oracion(self,oracion):
        normalizer = NormalizerMVP()
        palabras = normalizer.normalize(oracion)
        todosFonos=[]
        todos_tokens=[]
        PAUSAS={
            ",": "pau",
            ".": "pau pau",
            ";": "pau",
            ":": "pau",
            "?": "pau up",
            "!": "pau up"
        }

        for token in palabras:
            if token in PAUSAS:
                todosFonos.append(PAUSAS[token])
            else:
                #logging.info(f"Processing word: {token}")
                fonos = self.obtener_fonos_palabra(token)

                todosFonos.append(fonos)
            todos_tokens.append(token)
        return todos_tokens,todosFonos
