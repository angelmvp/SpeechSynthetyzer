from pydub import AudioSegment
import os
import logging
from typing import List
from utils.token import Token

class Reproductor():
    """
    Clase para cargar fonemas individuales y concatenarlos para formar palabras o frases.
    """
    def __init__(self, path_audios="./reproductor/fonemas/", output_path="./output_synthesis"):
        self.path_audios = path_audios
        self.output_path = output_path
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
    def generar_audios(self,Tokens:List[Token]):
        """
        Generamos los audios de las palabras o frases concatenando los fonemas individuales.
        """
        fonemas_modelo=[]
        fonos_prosodia=[]
        for token in Tokens:
            if token.signo:
                fonemas_modelo.append(token.token)
                fonos_prosodia.append(token.token)
            else:
                fonemas_modelo.extend(token.fonos)
                fonos_prosodia.extend(token.fonos_prosodia)
            fonemas_modelo.append(' ')
            fonos_prosodia.append(' ')
        
        self.concatenar_fonemas(fonemas_modelo,nombre_archivo_salida="output.wav")
        self.concatenar_fonemas(fonos_prosodia,nombre_archivo_salida="output_prosodia.wav")
    def concatenar_fonemas(self, lista_fonemas: list, nombre_archivo_salida: str = "frase_sintetizada.wav"):
        """
        Concatena una lista secuencial de fonemas para crear el audio
        """
        # Configuramos los parámetros de mejora
        TRIM_INICIAL_MS = 8  # Milisegundos a eliminar del inicio de cada fonema
        TRIM_FINAL_MS = 5    # Milisegundos a eliminar del final de cada fonema (opcional)
        FADE_IN_MS = 5       # Fade in suave
        FADE_OUT_MS = 8      # Fade out suave
        CROSSFADE_MS = 10    # Overlap entre fonemas consecutivos
        FRAME_RATE = 22050   # Frecuencia de muestreo estándar
        NORMALIZE_HEADROOM = 0.95  # evitar clipping 
        
        audio_final = AudioSegment.empty()
        fonemas_encontrados = 0
        logging.info(f"Concatenando los fonemas: {lista_fonemas}")
        
        for idx, fono in enumerate(lista_fonemas):
            ### PAusas de signos y espacios entre palabras 
            if fono in ['pau', ' ']:
                fono_audio = AudioSegment.silent(duration=100)  
                fono_audio = fono_audio.fade_in(10).fade_out(10) 
            elif fono in [",", ";"]:
                fono_audio = AudioSegment.silent(duration=150)
                fono_audio = fono_audio.fade_in(15).fade_out(15)
            elif fono in [".", "!", "?"]:
                fono_audio = AudioSegment.silent(duration=300)
                fono_audio = fono_audio.fade_in(20).fade_out(20)
            else:
                ### Fonemas individuales
                fono_filename = fono.upper() + ".wav"
                full_path = os.path.join(self.path_audios, fono_filename)
                
                try:
                    fono_audio = AudioSegment.from_file(full_path, format="wav")
                    
                    # Estandarizar: mismo frame rate y canales
                    fono_audio = fono_audio.set_frame_rate(FRAME_RATE).set_channels(1)
                    
                    # Eliminar partes iniciales y finales de los fonos para la concatenacion
                    if len(fono_audio) > TRIM_INICIAL_MS + TRIM_FINAL_MS:
                        fono_audio = fono_audio[TRIM_INICIAL_MS:-TRIM_FINAL_MS] if TRIM_FINAL_MS > 0 else fono_audio[TRIM_INICIAL_MS:]
                    
                    # Normalizar el volumen de cada fonema individualmente
                    fono_audio = fono_audio.normalize()
                    
                    # Los fades son para que los fonos se concatenen de forma suave
                    if len(fono_audio) > FADE_IN_MS + FADE_OUT_MS:
                        fono_audio = fono_audio.fade_in(FADE_IN_MS).fade_out(FADE_OUT_MS)
                    elif len(fono_audio) > FADE_IN_MS:
                        fono_audio = fono_audio.fade_in(FADE_IN_MS)
                    
                    fonemas_encontrados += 1

                except Exception as e:
                    print(f"   [ERROR] Error al procesar fonema {fono_filename}: {e}. Usando pausa.")
                    fono_audio = AudioSegment.silent(duration=50).fade_in(5).fade_out(5)
            
            # Concatenar con crossfade para transiciones más suaves
            # Solo aplicar crossfade si hay audio previo y ambos segmentos son lo suficientemente largos
            if len(audio_final) > 0 and len(fono_audio) > CROSSFADE_MS:
                # Usar append con crossfade en lugar de concatenación simple
                # crossfade para concatenar palabras
                audio_final = audio_final.append(fono_audio, crossfade=CROSSFADE_MS)
            else:
                # Para el primer fonema o si el segmento es corto concatenar normalmente
                audio_final += fono_audio

        if fonemas_encontrados == 0:
            print("Error: No se pudo encontrar ningún fonema de voz.")
            return False

        # Normalizamos final del audio completo para volumen consistente
        audio_final = audio_final.normalize(headroom=NORMALIZE_HEADROOM)
        
        # Aseguramos frame rate y canales consistentes
        audio_final = audio_final.set_frame_rate(FRAME_RATE).set_channels(1)

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
    Junta todos los fonemas de los tokens para tener un array de 1D
    """
    lista_plana = []
    
    for elemento in fonema_input_estructurado:
        if isinstance(elemento, list):
            lista_plana.extend(elemento)
        elif isinstance(elemento, str):
            lista_plana.append(elemento)
            
    return lista_plana
