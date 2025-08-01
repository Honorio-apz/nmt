import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text
import numpy as np
import os

# Rutas a los modelos (asegúrate de que estas rutas sean correctas
# con respecto a la ubicación de utils.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_MODEL_PATH = os.path.join(BASE_DIR, "transformers_translator")
SEQ2SEQ_MODEL_PATH = os.path.join(BASE_DIR, "seq2seq_translator")

# Variables globales para almacenar los modelos cargados
# Se inicializan a None y se cargan solo una vez.
_transformer_model = None
_transformer_infer = None
_seq2seq_model = None
_seq2seq_infer = None

def _load_transformer_model():
    """Carga el modelo Transformer si aún no está cargado."""
    global _transformer_model, _transformer_infer
    if _transformer_model is None:
        print(f"Cargando modelo Transformer desde: {TRANSFORMER_MODEL_PATH}")
        _transformer_model = tf.saved_model.load(TRANSFORMER_MODEL_PATH)
        _transformer_infer = _transformer_model.signatures["serving_default"]
        print("Modelo Transformer cargado.")
    return _transformer_infer

def _load_seq2seq_model():
    """Carga el modelo Seq2Seq si aún no está cargado."""
    global _seq2seq_model, _seq2seq_infer
    if _seq2seq_model is None:
        print(f"Cargando modelo Seq2Seq desde: {SEQ2SEQ_MODEL_PATH}")
        _seq2seq_model = tf.saved_model.load(SEQ2SEQ_MODEL_PATH)
        _seq2seq_infer = _seq2seq_model.signatures["serving_default"]
        print("Modelo Seq2Seq cargado.")
    return _seq2seq_infer

def translate(text, model_type='transformer'):
    """
    Traduce el texto dado utilizando el modelo especificado.
    Los modelos se cargan una sola vez al primer uso.
    """
    try:
        if not text.strip():
            return "Texto vacío, no se puede traducir."

        input_text = tf.constant([text])
        traduccion = ""

        if model_type == 'transformer':
            infer_fn = _load_transformer_model()
            result = infer_fn(input_text)
            output_key = list(result.keys())[0]
            # Decodificar el resultado, asumiendo que es un tensor de bytes
            traduccion = result[output_key].numpy().decode('utf-8')
            print(f"Traducción (Transformer): {traduccion}")

        elif model_type == 'seq2seq':
            infer_fn = _load_seq2seq_model()
            result = infer_fn(input_text)
            output_key = list(result.keys())[0]
            # Decodificar el resultado, asumiendo que es un tensor de bytes
            # y que el resultado es una lista de un solo elemento
            traduccion = result[output_key].numpy()[0].decode('utf-8')
            print(f"Traducción (Seq2Seq): {traduccion}")
        else:
            return "Modelo no válido."

        return traduccion

    except Exception as e:
        print(f"Error en la traducción con {model_type}: {e}")
        # Retorna un mensaje de error más amigable para el usuario final
        return "Error al traducir el texto. Por favor, inténtalo de nuevo."

# Opcional: Cargar los modelos al inicio de la aplicación
# Si quieres que los modelos estén listos desde el primer request,
# puedes llamar a estas funciones aquí. Sin embargo, si los modelos son muy grandes,
# esto puede ralentizar el inicio de Flask.
_load_transformer_model()
_load_seq2seq_model()