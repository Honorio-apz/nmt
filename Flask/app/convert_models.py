import tensorflow as tf
import os

# --- Configuración de rutas ---
# Asegúrate de que estas rutas sean correctas relativas a donde ejecutes este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Asume que el script está en la raíz 'app'
TRANSFORMER_SAVED_MODEL_PATH = os.path.join(BASE_DIR, "transformers_translator")
SEQ2SEQ_SAVED_MODEL_PATH = os.path.join(BASE_DIR, "seq2seq_translator")

# Rutas de salida para los modelos TFLite cuantizados
TRANSFORMER_TFLITE_OUTPUT_PATH = os.path.join(TRANSFORMER_SAVED_MODEL_PATH, "transformers_translator_quantized.tflite")
SEQ2SEQ_TFLITE_OUTPUT_PATH = os.path.join(SEQ2SEQ_SAVED_MODEL_PATH, "seq2seq_translator_quantized.tflite")

print(f"Ruta de Transformer SavedModel: {TRANSFORMER_SAVED_MODEL_PATH}")
print(f"Ruta de Seq2Seq SavedModel: {SEQ2SEQ_SAVED_MODEL_PATH}\n")

# --- Función de conversión genérica ---
def convert_to_tflite_quantized(saved_model_path, tflite_output_path):
    """
    Convierte un SavedModel a un modelo TFLite cuantizado (rango dinámico).
    """
    print(f"Iniciando conversión para: {saved_model_path}")
    try:
        # Cargar el SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

        # Aplicar optimización: cuantización de rango dinámico
        # Esto reduce el tamaño del modelo y acelera la inferencia en CPU.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Si tu SavedModel tiene múltiples firmas, especifica cuál usar si no es 'serving_default'
        # converter.target_spec.supported_signatures = ['your_signature_name']

        # Convertir el modelo
        tflite_model = converter.convert()

        # Guardar el modelo TFLite
        with open(tflite_output_path, "wb") as f:
            f.write(tflite_model)

        print(f"Conversión exitosa y guardado en: {tflite_output_path}")
    except Exception as e:
        print(f"Error durante la conversión de {saved_model_path}: {e}")

# --- Ejecutar la conversión para ambos modelos ---
if __name__ == "__main__":
    # Convertir el modelo Transformer
    convert_to_tflite_quantized(TRANSFORMER_SAVED_MODEL_PATH, TRANSFORMER_TFLITE_OUTPUT_PATH)
    print("-" * 50)

    # Convertir el modelo Seq2Seq
    convert_to_tflite_quantized(SEQ2SEQ_SAVED_MODEL_PATH, SEQ2SEQ_TFLITE_OUTPUT_PATH)
    print("\nProceso de cuantización completado.")