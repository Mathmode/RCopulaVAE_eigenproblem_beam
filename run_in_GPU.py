# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 12:15:56 2026

@author: anafd
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import runpy
import sys


def configurar_gpu():
    """Configura la GPU para evitar errores de memoria en Windows."""
    
    # Asegurar que TensorFlow vea la GPU (0 es la primera tarjeta)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Listar dispositivos físicos
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # IMPORTANTE: Habilitar el crecimiento de memoria.
            # Esto evita que TF acapare toda la VRAM y crashee el sistema.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            detalles = tf.config.experimental.get_device_details(gpus[0])
            nombre_gpu = detalles.get('device_name', 'GPU desconocida')
            print(f"✅ GPU Detectada y Configurada: {nombre_gpu}")
            print(f"ℹ️  Modo Memory Growth: ACTIVADO")
            
        except RuntimeError as e:
            # El crecimiento de memoria debe configurarse antes de inicializar tensores
            print(f"❌ Error configurando GPU: {e}")
    else:
        print("⚠️ No se detectó GPU. TensorFlow usará la CPU.")

def ejecutar_servidor():
    script_objetivo = 'main_GCopula_GMM.py'
    
    if not os.path.exists(script_objetivo):
        print(f"❌ Error: No encuentro el archivo '{script_objetivo}' en este directorio.")
        return

    print(f"\n🚀 Iniciando {script_objetivo}...\n" + "-"*30)
    
    # run_path ejecuta el script como si lo hubieras llamado directamente
    try:
        runpy.run_path(script_objetivo, run_name="__main__")
    except KeyboardInterrupt:
        print("\n🛑 Ejecución detenida por el usuario.")
    except Exception as e:
        print(f"\n❌ Ocurrió un error en la ejecución: {e}")

if __name__ == "__main__":
    configurar_gpu()
    ejecutar_servidor()