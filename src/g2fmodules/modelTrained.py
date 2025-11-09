from speechbrain.inference.text import GraphemeToPhoneme
g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")
text = "To be or not to be, that is the question"
phonemes = g2p(text)


class modelTrained():
    def __init__(self):
        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")
    def get_phonemes_from_text(self,text:str):
        return self.g2p(text)
    
modelTrainedInstance = modelTrained()
print(modelTrainedInstance.get_phonemes_from_text("Hello world"))