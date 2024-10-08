from flask import Flask, render_template, request, redirect
import tensorflow as tf
import utils

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def traduccion():
    
    if request.method == 'GET':
        return render_template("index.html")

    if request.method == 'POST':
        text = request.form['text1']
        
        sim_msg = utils.trans(text)
        
        return render_template("index.html", espanol=sim_msg, aymara=text)

if __name__ == '__main__':
    app.run(debug=True, port=8090)
