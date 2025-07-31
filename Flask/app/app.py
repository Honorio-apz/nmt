# app.py
from flask import Flask, render_template, request, jsonify
import utils # Importa tu módulo utils

app = Flask(__name__)

# Ruta principal para renderizar la página HTML
@app.route('/', methods=['GET'])
def index():
    """
    Renderiza la página principal del traductor.
    """
    return render_template("index.html", model="transformer") # Valor por defecto para el modelo

# Nueva ruta API para manejar las solicitudes de traducción de forma asíncrona
@app.route('/api/translate', methods=['POST'])
def api_translate():
    """
    Endpoint API para traducir texto.
    Recibe JSON con 'text' y 'model', y devuelve JSON con 'translation' o 'error'.
    """
    if request.is_json:
        data = request.get_json()
        text_to_translate = data.get('text', '').strip()
        model_type = data.get('model', 'transformer') # Valor por defecto 'transformer'

        if not text_to_translate:
            return jsonify({'error': 'El texto a traducir no puede estar vacío.'}), 400

        try:
            # Llama a la función de traducción de utils.py
            translated_text = utils.translate(text_to_translate, model_type)
            return jsonify({'translation': translated_text})
        except Exception as e:
            # Captura errores durante la traducción y los devuelve como JSON
            print(f"Error en la API de traducción: {e}")
            return jsonify({'error': 'Error interno del servidor al traducir el texto.'}), 500
    else:
        # Si la solicitud no es JSON, devuelve un error 400
        return jsonify({'error': 'Tipo de contenido no soportado. Se espera JSON.'}), 400

if __name__ == '__main__':
    # Opcional: precargar modelos al inicio de la aplicación
    # Descomenta las siguientes líneas si quieres que los modelos estén listos
    # desde el primer request, aunque esto puede ralentizar el inicio del servidor.
    # print("Precargando modelos...")
    # utils._load_transformer_model()
    # utils._load_seq2seq_model()
    # print("Modelos precargados.")

    app.run(debug=True, port=8090)
