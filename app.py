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

@app.route('/getinput', methods=['GET'])
def get_input():
    df = pd.read_csv('input.csv')
    list_columns = []
    result = {
        'num_rows': df.shape[0],
        'num_cols': df.shape[1]
    }
    for item in df.columns:
        if "Unnamed" not in item:
            list_columns.append(item)
    result.update({'list_columns':list_columns})
    return result

@app.route('/inputtable', methods=['GET'])
def show_table():
    df = pd.read_csv('input.csv')
    return df.to_html(header='true', index=False)