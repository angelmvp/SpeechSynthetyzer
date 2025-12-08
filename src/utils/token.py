class Token:
    def __init__(self,token:str,token_text:str,fono:str):
        self.token=token
        self.token_text = token_text
        self.fono = fono
        self.lema=None
        self.stress = None  
        self.pausa = False
    def set_lema(self,lema:str):
        self.lema=lema
    def set_stress(self,stress:int):
        self.stress=stress
    def set_pausa(self,pausa:bool):
        self.pausa=pausa