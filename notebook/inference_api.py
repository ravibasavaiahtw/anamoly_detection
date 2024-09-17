from flask import Flask
from flask import request
import keras
from tensorflow.keras import layers, losses
import tensorflow as tf
from tensorflow.keras.models import Model
from functions import Autoencoder, load_data, merge_dataframes
from pickle import load

#LOADING DATA ON MEMORY - WILL BE CHANGED AS A DL REQUEST
# Loading dataframes
df_sensors, df_metadata, df_maintenance, df_failures, df_errors = load_data()
# Extracting base feature list from table
base_features = list(df_sensors.drop(["datetime", "machineID"], axis=1).columns)
# Merging dataframes in one.
df = merge_dataframes(df_sensors, df_failures, df_errors, df_metadata)
time_step = 24
threshold = 0.8523336721336958

#Load scaler model
scaler = load(open('../model/scaler.pkl', 'rb'))

#Load Autoencoder model
model = keras.models.load_model("../model/autoencoder.keras")



app = Flask(__name__)

@app.route("/")
def index():
    return "<p>Anomaly Detection API!</p>"

@app.route('/inference', methods=['POST'])
def inference():
    if request.method == 'POST':
        # LOAD MODEL


        # LOAD DATA
        print(request.json)
        return "<p>Inference</p>"