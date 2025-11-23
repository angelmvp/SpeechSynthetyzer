import os
from pydub import AudioSegment

class Reproductor():
    def __init__(self):
        self.tmp_dir = "./tmp_fonemas"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def limpiar_fonema(self, fonema):
        # Quitar números (AE1 → AE)
        return ''.join([c for c in fonema if not c.isdigit()])

    def generar_fonema(self, fonema):
        """
        Genera un .wav para un fonema individual usando espeak-ng.
        """
        if not isinstance(fonema, str):
            raise ValueError(f"Se esperaba string pero recibí: {fonema}")

        fonema_limpio = self.limpiar_fonema(fonema)
        output_path = f"{self.tmp_dir}/{fonema_limpio}.wav"

        cmd = f'espeak-ng -v en -w "{output_path}" "[[{fonema_limpio}]]"'
        print(cmd)
        os.system(cmd)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"No se generó el archivo del fonema: {output_path}")

        return output_path

    def generar_palabra_completa(self, lista_fonemas, nombre_archivo=None):
        """
        Genera un .wav para una palabra completa con todos sus fonemas en un solo comando.
        Formato: [[FONEMA1]][[FONEMA2]][[FONEMA3]]...
        """
        if not isinstance(lista_fonemas, list):
            raise ValueError("lista_fonemas debe ser una lista")

        # Crear el formato de comando: [[FONEMA1]][[FONEMA2]][[FONEMA3]]...
        fonemas_formateados = "".join([f"[[{self.limpiar_fonema(f)}]]" for f in lista_fonemas])
        
        # Si no se proporciona nombre de archivo, crear uno basado en los fonemas
        if nombre_archivo is None:
            nombre_archivo = "".join([self.limpiar_fonema(f) for f in lista_fonemas])
        
        output_path = f"{self.tmp_dir}/{nombre_archivo}.wav"

        cmd = f'espeak-ng -v en -w "{output_path}" "{fonemas_formateados}"'
        os.system(cmd)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"No se generó el archivo de la palabra: {output_path}")

        return output_path

    def unir_audios(self, archivos_wav, output="output.wav"):
        """
        Une múltiples archivos wav en uno solo.
        """
        audio_final = AudioSegment.silent(duration=50)  # Pequeño silencio inicial

        for archivo in archivos_wav:
            if os.path.exists(archivo):
                segmento = AudioSegment.from_wav(archivo)
                audio_final += segmento
            else:
                print(f"Advertencia: Archivo {archivo} no encontrado")

        audio_final.export(output, format="wav")
        return output

    def reproducir_fonemas_palabra(self, lista_fonemas, output="palabra.wav"):
        """
        Convierte una lista de fonemas en un audio de palabra completa.
        """
        return self.generar_palabra_completa(lista_fonemas, "palabra_temp")

    def reproducir_fonemas_oracion(self, oracion, output="oracion.wav", pausa_entre_palabras=150):
        """
        Reproduce una oración completa donde cada elemento es una palabra (lista de fonemas)
        Genera cada palabra completa en un solo comando y luego las une con pausas.
        
        Args:
            oracion: Lista de palabras, donde cada palabra es una lista de fonemas
            output: Nombre del archivo de salida
            pausa_entre_palabras: Duración de la pausa entre palabras en milisegundos
        """
        if not isinstance(oracion, list):
            raise ValueError("oracion debe ser una lista de palabras")
        
        archivos_palabras = []
        
        # Generar cada palabra completa
        for i, palabra in enumerate(oracion):
            if not isinstance(palabra, list):
                raise ValueError("Cada elemento de la oración debe ser una lista de fonemas")
            
            # Generar archivo para esta palabra
            archivo_palabra = self.generar_palabra_completa(palabra, f"palabra_{i}")
            archivos_palabras.append(archivo_palabra)
        
        # Unir todas las palabras con pausas
        audio_final = AudioSegment.silent(duration=50)  # Silencio inicial
        
        for i, archivo_palabra in enumerate(archivos_palabras):
            # Agregar la palabra
            segmento_palabra = AudioSegment.from_wav(archivo_palabra)
            audio_final += segmento_palabra
            
            # Agregar pausa después de cada palabra (excepto la última)
            if i < len(archivos_palabras) - 1:
                audio_final += AudioSegment.silent(duration=pausa_entre_palabras)
        
        audio_final.export(output, format="wav")
        print(f"Oración guardada en: {output}")
        return output

    def limpiar_temporales(self):
        """
        Limpia los archivos temporales en la carpeta tmp_fonemas
        """
        for archivo in os.listdir(self.tmp_dir):
            ruta_archivo = os.path.join(self.tmp_dir, archivo)
            if os.path.isfile(ruta_archivo):
                os.remove(ruta_archivo)
        print("Archivos temporales limpiados")