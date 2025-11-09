from g2fmodules.mvpg2f import modelMVPG2F
from g2fmodules.classifierMVP import classifierMVPG2F
from g2fmodules.modelTrained import modelTrained
class G2F():
    def __init__(self):
        self.modelTrained = modelTrained()
        self.classifier = classifierMVPG2F()
        self.mvpModel = modelMVPG2F()
        pass
    def get_phonemes_from_text_model(self,text:str):
        return self.modelTrained.get_phonemes_from_text(text)
    def get_phonemes_from_text_classifier(self,text:str):
        return self.classifier.get_phonemes_from_text(text)
    def get_phonemes_from_text_mvp(self,text:str):
        return self.mvpModel.get_phonemes_from_text(text)