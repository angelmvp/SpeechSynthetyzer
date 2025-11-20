import num2words as n2w
import string
class NormalizerMVP:
    def __init__(self):
        pass
    def normalize(self,oracion):
        print("NORMALIZANDO")
        oracion = self.convertir_abreviaturas(oracion)
        oracion = self.convertir_numeros(oracion)
        oracion = self.eliminar_puntuacion(oracion)
        oracion = self.convertir_minusculas(oracion)
        return oracion.split()
    def convertir_numeros(self, oracion):
        oracion_normalizada = []
        for palabra in oracion.split():
            if palabra.isdigit():
                palabra_normalizada = n2w.num2words(palabra)
                oracion_normalizada.append(palabra_normalizada)
            else:
                oracion_normalizada.append(palabra)
        return ' '.join(oracion_normalizada)
    def convertir_abreviaturas(self, oracion):
        abreviaturas = {
            "Dr.": "Doctor",
            "Mr.":"Mister",
            "Ms.": "Mizz",
            "etc.": "etcetera",
        }
        for abreviatura, expansion in abreviaturas.items():
            oracion = oracion.replace(abreviatura, expansion)
        return oracion
    def eliminar_puntuacion(self, oracion):
        translator = str.maketrans('', '', string.punctuation)
        return oracion.translate(translator)
    def convertir_minusculas(self, oracion):
        return oracion.lower()