from flask import Flask
from flask import request
import keras
from tensorflow.keras import layers, losses
import tensorflow as tf
from tensorflow.keras.models import Model
from functions import Autoencoder, load_data, merge_dataframes, transform_values_inference,result_autoencoder
from pickle import load
import pandas as pd
from flask import json

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
autoencoder = keras.models.load_model("../model/autoencoder.keras")

app = Flask(__name__)

@app.route("/")
def index():
    return "<p>Anomaly Detection API!</p>"

@app.route('/inference', methods=['POST'])
def inference():
    if request.method == 'POST':
        # Getting new value from body
        machine_id = request.json['machineID']
        datetime = pd.to_datetime(request.json['datetime'])
        volt = request.json['volt']
        rotate = request.json['rotate']
        pressure = request.json['pressure']
        vibration = request.json['vibration']
        # Getting history values from datatable
        mask = (df['machineID'] == machine_id) & (df['datetime'] < datetime)
        df_history = df[mask][-23:]
        # Concat historical with new data
        df_new_data = pd.DataFrame([[machine_id, datetime, volt, rotate, pressure, vibration]], columns=['machineID','datetime','volt','rotate','pressure','vibration'])
        df_new_data = merge_dataframes(df_new_data, df_failures, df_errors, df_metadata)
        df_inference = pd.concat([df_history,df_new_data]).reset_index(drop=True)

        # Scaling and transforming values
        inference_data = transform_values_inference(df_inference, time_step=time_step, scaler=scaler, base_features=base_features)

        # Make inference
        mse = result_autoencoder(autoencoder, inference_data)
        anomaly_scores = pd.Series(mse.numpy(), name="anomaly_scores")
        is_anomaly = anomaly_scores[0] > threshold

        # Response
        res_data = {
            'anomaly_score':anomaly_scores[0],
            'is_anomaly':str(is_anomaly),
            'threshold':threshold
        }
        response = app.response_class(
            response=json.dumps(res_data),
            mimetype='application/json'
        )
        # LOAD DATA
        return response