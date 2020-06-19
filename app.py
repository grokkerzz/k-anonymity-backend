from flask import Flask, flash, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "/tmp"
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/show_schema", methods=['GET'])
def show_schema():
    pass


@app.route("/process/<file_id>")
def algorithm():
    df = pd.read_csv("/tmp" + filename)

