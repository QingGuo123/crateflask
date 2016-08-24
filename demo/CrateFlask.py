# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, render_template
from crate.client import connect
import xgboost as xgb
import json
import datetime
import numpy as np
from urllib.parse import unquote
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')





if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8002)