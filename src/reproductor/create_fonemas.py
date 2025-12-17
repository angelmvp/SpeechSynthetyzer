import pyworld as pw
import numpy as np
import soundfile as sf
import os
from scipy.interpolate import interp1d

# Lista de fonemas
FONOS_VOCALES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 
    'EH', 'ER', 'EY', 'IH', 'IY', 
    'OW', 'OY', 'UH', 'UW'
]

## Modificaciones en F0  y duracion dependiendo prominencia
prominencia_config = {
    '0': {'pitch': 0.90, 'dur': 0.9}, 
    '1': {'pitch': 1.18, 'dur': 1.3}, 
    '2': {'pitch': 1.11, 'dur': 1.1}
}

INPUT_DIR = "."  
OUTPUT_DIR = "output_phonemes"

    
def generar_fonemas_prominencia():
    
    # Crear el directorio de salida si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    
    for fonema in FONOS_VOCALES: # GENERAR LOS FONEMAS CON DIFERENTES NIVELES DE PROMINENCIA
        input_filename = os.path.join(INPUT_DIR, f"{fonema}.wav")

        x, fs = sf.read(input_filename, dtype=np.float64)
        if x.ndim > 1:
            x = x[:, 0]
        
        f0, sp, ap = pw.wav2world(x, fs) # estafuncino ayuda aconvertir el habla en caracteristicas     
        # f0 = frecuencia fundamental
        # sp = espectro
        # ap = envolvente espectral aperiodica
        
        # generar los niveles de prominencia
        for nivel_prominencia in prominencia_config:
            config = prominencia_config[nivel_prominencia]
            modified_f0 = f0 * config['pitch']

            # Cambiamos la duración del audio
            num_frames = len(f0)
            new_num_frames = int(num_frames * config['dur'])
            old_indices = np.arange(num_frames)
            new_indices = np.linspace(0, num_frames - 1, new_num_frames)

            # Interpolar con los NUEVOS índices  para que tengan el mismo tiempo de duración
            stretched_f0 = interp1d(old_indices, modified_f0, kind='linear')(new_indices)
            stretched_sp = interp1d(old_indices, sp, axis=0, kind='linear')(new_indices)
            stretched_ap = interp1d(old_indices, ap, axis=0, kind='linear')(new_indices)

            # sintetizamos nuestro fono con los nuevos features
            nuevo_audio = pw.synthesize(stretched_f0, stretched_sp, stretched_ap, fs)

            # Normalizar para evitar errores de clipping
            nuevo_audio = nuevo_audio / np.max(np.abs(nuevo_audio)) * 0.9
    print("Generados los fonemas con prominencia")

if __name__ == "__main__":
    generar_fonemas_prominencia()