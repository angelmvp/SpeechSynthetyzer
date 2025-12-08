import pyworld as pw
import numpy as np
import soundfile as sf
import os
from scipy.interpolate import interp1d

# --- Configuración ---

# Lista de fonemas vocálicos a procesar (basado en tu solicitud)
# NOTA: Solo se procesarán los archivos .wav que coincidan con estos nombres.
VOWEL_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 
    'EH', 'ER', 'EY', 'IH', 'IY', 
    'OW', 'OY', 'UH', 'UW'
]

# Parámetros de Modificación para cada nivel de Stress: (Factor de F0, Factor de Duración)
# El timbre (Spectral Envelope) se mantiene constante para conservar tu voz.
STRESS_FACTORS = {
    # 0: Sin Acento (Reduced)
    '0': (0.90, 0.85), # Reducción del 10% en F0 y 15% en duración
    
    # 1: Acento Primario (Primary)
    '1': (1.15, 1.15), # Aumento del 15% en F0 y 15% en duración
    
    # 2: Acento Secundario (Secondary)
    '2': (1.08, 1.05)  # Aumento del 8% en F0 y 5% en duración (intermedio)
}

INPUT_DIR = "."  # Directorio donde están tus archivos base (por defecto: el mismo donde está el script)
OUTPUT_DIR = "output_phonemes" # Directorio donde se guardarán los resultados

# --- Función Principal ---

def generate_stressed_phonemes():
    
    # Crear el directorio de salida si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Buscando archivos base en: {os.path.abspath(INPUT_DIR)}")
    print(f"Guardando resultados en: {os.path.abspath(OUTPUT_DIR)}")
    print("-" * 30)
    
    for base_phoneme in VOWEL_PHONEMES:
        input_filename = os.path.join(INPUT_DIR, f"{base_phoneme}.wav")
        print(input_filename)
        # Verificar si el archivo base existe
        if not os.path.exists(input_filename):
            print(f"⚠️ Archivo base no encontrado: {base_phoneme}.wav. ¡Saltando!")
            continue

        try:
            # 1. Cargar el archivo WAV (en formato float64 para pyworld)
            x, fs = sf.read(input_filename, dtype=np.float64)

            # Si el audio es estéreo, tomar un solo canal
            if x.ndim > 1:
                x = x[:, 0]
            
            # 2. ANÁLISIS: Descomponer el audio usando WORLD vocoder (extraer F0, espectro, aperiodicidad)
            f0, sp, ap = pw.wav2world(x, fs)
            
            # --- Iterar sobre cada nivel de stress (0, 1, 2) ---
            for stress_level, (pitch_factor, duration_factor) in STRESS_FACTORS.items():
                
                new_phoneme_name = f"{base_phoneme}{stress_level}"
                print(f"-> Generando {new_phoneme_name} (F0: x{pitch_factor}, Duración: x{duration_factor})")

                # 3. MODIFICAR F0 (Pitch)
                # Aplicar el factor de pitch directamente a todos los valores de F0
                modified_f0 = f0 * pitch_factor

                # 4. MODIFICAR DURACIÓN (Escalar las características)
                # La duración se modifica cambiando la longitud de los arrays de características.
                original_num_frames = len(f0)
                new_num_frames = int(original_num_frames * duration_factor)

                # Crear índices para la interpolación (reescalado)
                original_indices = np.arange(original_num_frames)
                new_indices = np.linspace(0, original_num_frames - 1, new_num_frames)

                # Interpolación: Escalar F0, Espectro (sp) y Aperiodicidad (ap) a la nueva duración
                
                # Interpolar F0
                f0_interpolator = interp1d(original_indices, modified_f0, kind='linear')
                stretched_f0 = f0_interpolator(new_indices)

                # Interpolar SP (Espectro)
                sp_interpolator = interp1d(original_indices, sp, axis=0, kind='linear')
                stretched_sp = sp_interpolator(new_indices)
                
                # Interpolar AP (Aperiodicidad)
                ap_interpolator = interp1d(original_indices, ap, axis=0, kind='linear')
                stretched_ap = ap_interpolator(new_indices)

                # 5. SÍNTESIS: Reconstruir el nuevo archivo WAV
                synthesized_wav = pw.synthesize(stretched_f0, stretched_sp, stretched_ap, fs)
                
                # 6. GUARDAR el archivo WAV generado
                output_filename = os.path.join(OUTPUT_DIR, f"{new_phoneme_name}.wav")
                # soundfile guarda automáticamente en el rango correcto
                sf.write(output_filename, synthesized_wav, fs)

            print("-" * 30)

        except Exception as e:
            print(f"❌ Error al procesar . Posiblemente el archivo no es válido o está corrupto: {e}")

    print("\n✨ ¡Proceso de generación de fonemas con estrés completado!")
    print(f"Revisa los archivos generados en la carpeta: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_stressed_phonemes()