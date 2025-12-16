import num2words as n2w
import string
import re
class NormalizerMVP:
    def __init__(self):
        pass
    def normalize(self,oracion):
        print("NORMALIZANDO oracion")
        oracion = self.convertir_abreviaturas(oracion)
        oracion=  self.convertir_fechas(oracion)
        oracion = self.convertir_monedas(oracion)
        oracion = self.convertir_numeros(oracion)
        oracion = self.separar_puntuacion(oracion)
        oracion = self.convertir_minusculas(oracion)
        oracion = oracion.split()
        return oracion
    def convertir_fechas(self, oracion):
        """
        Convierte fechas en formato DD/MM/YYYY  a una representación textual.
        """
        
        months = {
            "01": 'january', "1": 'january',
            "02": 'february', "2": 'february',
            "03": 'march', "3": 'march',
            "04": 'april', "4": 'april',
            "05": 'may', "5": 'may',
            "06": 'june', "6": 'june',
            "07": 'july', "7": 'july',
            "08": 'august', "8": 'august',
            "09": 'september', "9": 'september',
            "10": 'october',
            "11": 'november',
            "12": 'december'
        }
        
        def replace_date(match):
            day = match.group(1)
            month = match.group(2)
            year = match.group(3)
            month_name = months.get(month, month)

            return f"{month_name} {day} of {year}"
        
        # Aplicar la conversión de fechas
        oracion = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', replace_date, oracion)
        return oracion
    def convertir_monedas(self,oracion):
        """
        Convierte representaciones monetarias como $12.50 a '12 dollars with 50 cents'
        """
        oracion = re.sub(r'\$(\d+)\.(\d{2})', r'\1 dollars with \2 cents', oracion) # regex
        oracion = re.sub(r'\$(\d+)', r'\1 dollars', oracion) # regex 
        return oracion
    def convertir_numeros(self, oracion):
        """
        Por medio de num2words convierte numeros a palabras
        """
        oracion_normalizada = []
        for palabra in oracion.split():
            if palabra.isdigit():
                numero = n2w.num2words(palabra)
                numero = numero.replace('-', ' ')
                oracion_normalizada.append(numero)
            else:
                oracion_normalizada.append(palabra)
        return ' '.join(oracion_normalizada)
    def convertir_abreviaturas(self, oracion):
        """
        Convierte abreviaturas comunes a su forma expandida.
        """
        abreviaturas = {
            "Dr.": "Doctor",
            "Mr.":"Mister",
            "Ms.": "Mizz",
            "etc.": "etcetera",
            "RB" : "RunningBack",
            "PG" : "PointGuard",
            "QB" : "QuarterBack",
            "TE" : "Tight End",
            "Km": "Kilometers",
            "Hz": "Hertz"
        }
        for abreviatura, expansion in abreviaturas.items():
            oracion = oracion.replace(abreviatura, expansion)
        return oracion
    def separar_puntuacion(self, oracion):
        SIGNOS = ["," ,"." , ";" ,":" ,"?" ,"!"]
        for simbolo in SIGNOS:
            oracion = oracion.replace(simbolo, f' {simbolo} ')
        return oracion
    def convertir_minusculas(self, oracion):
        return oracion.lower()