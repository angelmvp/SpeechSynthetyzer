from typing import List
class Token:
    def __init__(self,token:str,token_text:str,fonos:List= None,prosodia:str=None):
        self.token=token
        self.token_text = token_text
        self.fonos = fonos
        self.stress_fono = self.get_stress_fono() if fonos else None
        self.stress_prosodia = prosodia
        self.fonos_prosodia = self.set_fonos_prosodia() if prosodia!='NA' else None
        self.signo = True if token in ["," ,"." , ";" ,":" ,"?" ,"!" ] else False
        if self.signo:
            self.reset_parameters()
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
    def set_pausa_level(self):
        if self.token in ["," ,";" ]:
            return 1
        elif self.token in ["." ,":" ,"?" ,"!" ]:
            return 2
        else:
            return 0
    def set_fonos_prosodia(self):
        """
        retorna la lista de fonos con el nivel de prominencia asignado 
        """
        ## revisamos si la prosodia no es la misma, por loque hay que cambiarla
        if self.stress_prosodia == self.stress_fono:
            return self.fonos
        # si no esta asignada la prosodia a nuestros fonos, la asignamos en todos los caracteres
        fonos_prosodia = []
        for f in self.fonos:
            if len(f) <= 2: # No es vocal
                fonos_prosodia.append(f)
                continue
            if f[-1].isdigit(): # reemplazamos por la nueva prosodi
                fono_prosodia = f[:-1] + str(self.stress_prosodia)
            else:
                # otherwise append the stress digit
                fono_prosodia = f + str(self.stress_prosodia)
            fonos_prosodia.append(fono_prosodia)
        return fonos_prosodia
    def get_token(self):
        return self.token
    def reset_parameters(self):
        self.fonos = None
        self.stress_fono = None
        self.stress_prosodia = None
        self.fonos_prosodia = None
    def to_string(self):
        print( f'Token: {self.token} \n St_m: {self.stress_fono} Fonos: {self.fonos} \n St_prosodia: {self.stress_prosodia} Fonos_prosodia: {self.fonos_prosodia}')