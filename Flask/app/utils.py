import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as tf_text

def trans(tesxt):
    # Load the model only once to improve performance
    model_path = '/media/honorio/sd/files/CSTHESIS/24/transformer/transformers_translator'
    reloaded = tf.saved_model.load(model_path)
    try:
        # Create a TensorFlow constant from the input text
        three_input_text = tf.constant(tesxt)
        result = reloaded(three_input_text)

        # Decode the result to a UTF-8 string
        texttranslate = result.numpy().decode()
        return texttranslate
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return ""


