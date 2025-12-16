from pydub import AudioSegment
import os
import logging
from typing import List
from utils.token import Token
# --- CLASE REPRODUCTOR (Mantenida) ---

class Reproductor():
    """
    Clase para cargar fonemas individuales y concatenarlos para formar palabras o frases.
    """
    def __init__(self, path_audios="./reproductor/fonemas/", output_path="./output_synthesis"):
        self.path_audios = path_audios
        self.output_path = output_path
        
        # Crear el directorio de salida si no existe
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
    def generar_audios(self,Tokens:List[Token]):
        fonemas_modelo=[]
        fonos_prosodia=[]
        for token in Tokens:
            if token.signo:
                fonemas_modelo.append(token.token)
                fonos_prosodia.append(token.token)
            else:
                fonemas_modelo.extend(token.fonos)
                fonos_prosodia.extend(token.fonos_prosodia)
        print(fonemas_modelo)
        self.concatenar_fonemas(fonemas_modelo,nombre_archivo_salida="audio_modelo.wav")
        self.concatenar_fonemas(fonos_prosodia,nombre_archivo_salida="audio_prosodia.wav")
    def concatenar_fonemas(self, lista_fonemas: list, nombre_archivo_salida: str = "frase_sintetizada.wav"):
        """
        Concatena una lista secuencial de fonemas para crear un archivo de audio WAV.
        """
        audio_final = AudioSegment.empty()
        fonemas_encontrados = 0
        print(f"\n--- Concatenando {len(lista_fonemas)} segmentos ---")
        logging.info(f"(lista_fonemas) {lista_fonemas}")
        for fono in lista_fonemas:
            # Si el fonema es una pausa, usar un segmento silencioso
            if fono in [",", ".", ";", ":", "?", "!"]:
                pausa_duracion = 300 if fono in [".", ":", "?", "!"] else 150
                fono_audio = AudioSegment.silent(duration=pausa_duracion)
            else:
                fono_filename = fono.upper() + ".wav"
                full_path = os.path.join(self.path_audios, fono_filename)
                
                try:
                    fono_audio = AudioSegment.from_file(full_path, format="wav")
                    fonemas_encontrados += 1
                except FileNotFoundError:
                    print(f"   [ERROR] Fonema no encontrado: {fono_filename}. Usando una pausa corta.")
                    # Si no encuentra el archivo, inserta una pausa en su lugar para evitar que el proceso se detenga.
                    fono_audio = AudioSegment.silent(duration=50) 
                except Exception as e:
                    print(f"   [ERROR] No se pudo procesar {fono_filename}: {e}")
                    fono_audio = AudioSegment.silent(duration=50) 
            
            # Concatenar el segmento (fono o pausa)
            audio_final += fono_audio

        if fonemas_encontrados == 0:
            print("Error: No se pudo encontrar ningún fonema de voz.")
            return False

        output_full_path = os.path.join(self.output_path, nombre_archivo_salida)
        try:
            audio_final.export(output_full_path, format="wav")
            print(f"\n Síntesis completada. Archivo guardado en: {output_full_path}")
            return True
        except Exception as e:
            print(f"\nError al exportar el archivo: {e}")
            return False


def aplanar_lista_fonemas(fonema_input_estructurado: list) -> list:
    """
    Toma la lista estructurada de fonemas (lista de listas y cadenas) y la aplana 
    en una única lista secuencial de fonemas.

    :param fonema_input_estructurado: La segunda parte de tu input con listas de fonemas por palabra.
    :return: Una lista simple con todos los fonemas y pausas en orden.
    """
    lista_plana = []
    
    for elemento in fonema_input_estructurado:
        if isinstance(elemento, list):
            # Si es una lista (una palabra), añadir sus fonemas uno por uno
            lista_plana.extend(elemento)
        elif isinstance(elemento, str):
            # Si es una cadena (ej. 'pau', ',', '!', etc.), añadirla directamente
            lista_plana.append(elemento)
            
    return lista_plana

# ----------------------------------------------------------------------
# --- EJEMPLO DE EJECUCIÓN CON TU INPUT ---
# ----------------------------------------------------------------------

# if __name__ == "__main__":
    
#     # Tu Input Estructurado (La segunda lista que me proporcionaste)
#     # Esta es la lista que contiene listas de fonemas y las cadenas 'pau'.
#     fonemas_para_procesar = [
#         ['B', 'IH1', 'JH', 'AH0', 'N'], ['R', 'AA1', 'B', 'AH0', 'N', 'S', 'AH0', 'N'], 'pau', 
#         ['IH1', 'Z'], ['OW1', 'N', 'N'], ['AO0', 'F'], ['DH', 'EY0'], 
#         ['B', 'EH1', 'S', 'T'], ['R', 'AH1', 'N', 'IH0', 'NG', 'B', 'AE2'], 
#         ['IH0', 'N'], ['DH', 'EY0'], ['N', 'N', 'L'], 'pau', 'pau', 
#         ['HH'], ['W', 'AA1', 'Z'], ['B', 'AO1', 'R', 'N'], ['AO1', 'N'], 
#         ['S', 'EH0', 'P', 'T', 'EH1', 'B', 'B', 'ER0'], ['T', 'W', 'EH1', 'N', 'T', 'IY0', 'W', 'AH2', 'N'], 
#         ['AO0', 'F'], ['T', 'W', 'UW1'], ['TH', 'AW1', 'Z', 'AH0', 'N', 'D'], 
#         ['AE1', 'N', 'D'], ['T', 'W', 'UW1'], ['AE1', 'N', 'D'], 
#         ['HH', 'IH1', 'Z'], ['K', 'AA1', 'N', 'T', 'R', 'AE2', 'K', 'T'], 
#         ['IH1', 'Z'], ['W', 'ER1', 'TH', 'TH'], ['T', 'W', 'EH1', 'L', 'V'], 
#         ['D', 'AA1', 'L', 'ER0', 'Z'], 'pau', 'pau', 
#         ['F', 'IH1', 'F', 'T', 'IY0'], ['M', 'IH1', 'L', 'Y', 'AH0', 'N'], 
#         ['P', 'ER0'], ['Y', 'IH1', 'R'], 'pau', 'pau'
#     ]

#     # 1. Aplanar la lista de fonemas
#     fonemas_secuenciales = aplanar_lista_fonemas(fonemas_para_procesar)
    
#     # 2. Inicializar el reproductor
#     tts_engine = Reproductor(
#         path_audios="./fonemas/", 
#         output_path="./output_synthesis"
#     )
    
#     # 3. Concatenar y guardar el archivo final
#     tts_engine.concatenar_fonemas(
#         lista_fonemas=fonemas_secuenciales,
#         nombre_archivo_salida="bijan_robinson_bio.wav"
#     )
    
#     # Resultado de la lista aplanada (ejemplo de los primeros fonemas)
#     print("\nPrimeros 10 fonemas a concatenar (Aplanados):")
#     print(fonemas_secuenciales[:10])