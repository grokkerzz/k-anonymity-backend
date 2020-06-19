from flask import Flask, flash, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import pandas as pd

UPLOAD_FOLDER = "/tmp"
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources = {r"/api/*": {"origins": "*"}})


@app.route("/showschema", methods=['GET'])
def show_schema():
    pass


@app.route('/api/uploadfile', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        file = request.files['input']
        print(file.name)
        file.save(file.name + ".csv")
        return redirect(request.url)
    return 'THIS IS BACKEND! THERE IS NOTHING HERE!'

@app.route('/showinput', methods=['GET'])
def show_input():
    df = pd.read_csv('input.csv')
    return str(df.shape[0])