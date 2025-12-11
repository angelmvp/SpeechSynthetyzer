from typing import List
class Token:
    def __init__(self,token:str,token_text:str,fonos:List= None,prosodia:str=None):
        self.token=token
        self.token_text = token_text
        self.fonos = fonos
        self.stress_fono = self.get_stress_fono() if fonos else None
        self.stress_prosodia = prosodia
        self.signo = False
        self.pausa = False 
    def set_fono(self,fono:str):
        self.fono=fono
    def get_stress_fono(self):
        if not self.fonos:
            return None
        max_fono = '0'
        for fono in self.fonos:
            if  '2' in fono:
                return '2'
            elif '1' in fono:
                max_fono = '1'
        return max_fono
    def set_stress_prosodia(self,stress_prosodia:str):
        self.stress_prosodia=stress_prosodia
    def set_signo(self,signo:bool):
        self.signo=signo
    def set_pausa(self,pausa:bool):
        self.pausa=pausa
    def get_token(self):
        return self.token
    
    def print(self):
        return f'Token: {self.token}, Token_text: {self.token_text}, Fonos: {self.fonos}, Stress_fono: {self.stress_fono}, Stress_prosodia: {self.stress_prosodia}, Signo: {self.signo}, Pausa: {self.pausa}'