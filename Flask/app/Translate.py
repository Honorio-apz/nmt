import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow_hub as hub

def trans(tesxt):

    global texttranslate
    three_input_text = tf.constant([
        tesxt
    ])
    #print(tesxt)
    reloaded = tf.saved_model.load('/../dynamic_translator')
    result = reloaded.translate(tf.constant([tesxt]))
    texttranslate=result[0].numpy().decode()
    print(texttranslate)
    return (texttranslate)
def ejecutar():
    tesxt = "Sutimax Kunasa?"
    trans(tesxt)
ejecutar()


