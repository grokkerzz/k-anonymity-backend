from flask import Flask, flash, request, redirect, url_for, send_file
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import pandas as pd
from src.algorithm_function import anonymizeDataFrame, classifyNumCtg

LIMIT_RECORD = 100
UPLOAD_FOLDER = "/tmp"
ALLOWED_EXTENSIONS = {'csv'}
LIST_QI = []
k = int(0)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources = {r"/api/*": {"origins": "*"}})


@app.route('/api/uploadfile', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        file = request.files['input']
        file.save(file.name + ".csv")
        df = pd.read_csv('input.csv', index_col=0)
        df = df.sample(LIMIT_RECORD)
        df.to_csv('input2.csv', index=False)
        return redirect(request.url)
    return 'THIS IS BACKEND! THERE IS NOTHING HERE!'

@app.route('/getinput', methods=['GET'])
def get_input():
    df = pd.read_csv('input2.csv')
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

@app.route('/getQI', methods=['POST', 'GET'])
def get_qi():
    if request.method == 'POST':
        global LIST_QI
        LIST_QI = request.get_json()
        LIST_QI = map(lambda x: list(x.values())[0], LIST_QI)
        LIST_QI = list(LIST_QI)
        return 'oke'
    return str(LIST_QI)

@app.route('/getk', methods=['POST','GET'])
def get_k():
    if request.method == 'POST':
        global k
        k = request.get_data()
        return 'oke'
    return k

@app.route('/inputtable', methods=['GET'])
def show_table():
    df = pd.read_csv('input2.csv')
    return df.to_html(header='true', index=False)

@app.route('/getoutput', methods=['GET'])
def get_output():
    dfin = pd.read_csv('input2.csv')
    
    qiClass = classifyNumCtg(dfin, LIST_QI)
    dfout = anonymizeDataFrame(dfin, LIST_QI, True, int(k))
    dfout.to_csv('output.csv', index=False)
    return qiClass

@app.route('/outputtable', methods=['GET'])
def show_output_table():
    df = pd.read_csv('output.csv')
    return df.to_html(header='true', index=False)

@app.route('/download')
def downloadFile():
    path = './output.csv'
    return send_file(path, as_attachment=True)