import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import keras
import tensorflow as tf




# ------------------------------------------------------
#               DATA RELATED FUNCTIONS
# ------------------------------------------------------
def load_data(path: str = "../data/") -> list[pd.DataFrame]:
    """Function to load all dataframes from the dataset"""
    df_sensors = pd.read_csv(path + "PdM_telemetry.csv")
    df_metadata = pd.read_csv(path + "PdM_machines.csv")
    df_maintenance = pd.read_csv(path + "PdM_maint.csv")
    df_failures = pd.read_csv(path + "PdM_failures.csv")
    df_errors = pd.read_csv(path + "PdM_errors.csv")

    # Format date & time. Sort based on date for better readability
    tables = [df_sensors, df_maintenance, df_failures, df_errors]
    for df in tables:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
        df.sort_values(["datetime", "machineID"], inplace=True, ignore_index=True)

    return df_sensors, df_metadata, df_maintenance, df_failures, df_errors


def merge_dataframes(
    df_sensors: pd.DataFrame,
    df_failures: pd.DataFrame,
    df_errors: pd.DataFrame,
    df_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Function to merge all dataframes."""
    # df = df_sensors.merge(df_failures, how='left').merge(df_errors, how='left').merge(df_metadata, how='left').query('machineID <= 20').reset_index(drop=True)
    df = (
        df_sensors.merge(df_failures, how="left")
        .merge(df_errors, how="left")
        .merge(df_metadata, how="left")
        .reset_index(drop=True)
    )
    df = df.replace("NaN", np.nan)
    df["is_error"] = ~((df["errorID"].isnull()))
    df["is_anomaly"] = ~((df["failure"].isnull()))
    df["model"] = df["model"].str.slice(-1).astype("int")
    return df


def create_sequences(values: list, time_step: int) -> np.array:
    """Function to create sequences windows."""
    output = []
    for i in range(len(values) - time_step + 1):
        output.append(values[i : (i + time_step)])
    return np.stack(output)


def transform_values(
    dataframe: pd.DataFrame, time_step: int, scaler: StandardScaler
) -> list[np.array]:
    """Function to transform grouped dataframe into data and labels numpy arrays"""
    # Reseting infex
    dataframe = dataframe.reset_index(drop=True)
    # Transforming values
    np_scaled = scaler.transform(
        dataframe.drop(
            [
                "machineID",
                "datetime",
                "failure",
                "errorID",
                "is_anomaly",
                "is_error",
                "model",
                "age",
            ],
            axis=1,
        )
    )
    # Adding model value
    np_scaled = np.hstack(
        (
            np_scaled,
            dataframe["model"].to_numpy().reshape((dataframe["model"].shape[0]), 1),
        )
    )
    # Create sequeces for each point
    data = create_sequences(np_scaled, time_step=time_step)
    # Set labels in each sequence
    labels = dataframe.loc[time_step - 1 :, "is_anomaly"].astype("int").values
    return data, labels


# ------------------------------------------------------
#                   AUTOENCODER MODEL
# ------------------------------------------------------
# Autoencoder
@keras.saving.register_keras_serializable()
class Autoencoder(Model):
    def __init__(
        self, input_shape: tuple, hidden_layers: int, n_units_latent_space: int
    ):
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.n_units_latent_space = n_units_latent_space
        if (n_units_latent_space % 2) != 0:
            raise Exception("Number of units in latent space must be an even value.")
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            # [
            #     layers.Dense(64, activation="relu"),
            #     layers.Dense(32, activation="relu"),
            #     layers.Dense(16, activation="relu"),
            # ]
            [
                layers.Dense(
                    n_units_latent_space * (2 ** (layer - 1)), activation="relu"
                )
                for layer in range(hidden_layers, 0, -1)
            ]
        )
        self.decoder = tf.keras.Sequential(
            # [
            #     layers.Dense(32, activation="relu"),
            #     layers.Dense(64, activation="relu"),
            #     layers.Dense(data.shape[2], activation="relu"),
            # ]
            [
                layers.Dense(
                    n_units_latent_space * (2 ** (layer - 1)), activation="relu"
                )
                for layer in range(2, hidden_layers + 1)
            ]
            + [layers.Dense(input_shape[2], activation="relu")]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        return {
            "input_shape": self.input_shape,
            "hidden_layers": self.hidden_layers,
            "n_units_latent_space": self.n_units_latent_space,
        }


def result_autoencoder(model: object, x_train: np.array) -> tf.Tensor:
    # Calculate the reconstruction error for each data point
    reconstructions_deep = model.predict(x_train)
    mse = tf.reduce_mean(tf.square(x_train - reconstructions_deep), axis=[1, 2])
    return mse


# ------------------------------------------------------
#                   INFERENCE FUNCTIONS
# ------------------------------------------------------
def identifying_anomalies(
    data: pd.DataFrame,
    time_step: int,
    start_index: int,
    anomaly_deep_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Function to set anomalies"""
    data = data.reset_index(drop=True)
    rows = data.shape[0]
    data.loc[time_step - 1 :, "anomaly_score"] = anomaly_deep_scores[
        start_index : start_index + rows - time_step + 1
    ].to_numpy()
    start_index += rows - time_step + 1
    data["is_anomaly"] = False
    for idx in data[~(data["failure"].isnull())].index:
        start_idx = max(0, idx - time_step)  # Index error
        data.loc[start_idx:idx, "is_anomaly"] = True
    return data, start_index


# ------------------------------------------------------
#                      PLOT FUNCTIONS
# ------------------------------------------------------
def plot_loss_accuracy(history) -> None:
    """Function to plot the loss and the accuracy of the model training."""
    rcParams["figure.figsize"] = 20, 3
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(history.history["loss"])
    ax[0].plot(history.history["val_loss"])
    ax[0].title.set_text("model loss")
    ax.flat[0].set(ylabel="loss", xlabel="epoch")
    ax[0].legend(["train", "val"], loc="upper right")

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    ax[1].plot(history.history["accuracy"])
    ax[1].plot(history.history["val_accuracy"])
    ax[0].title.set_text("model accuracy")
    ax.flat[1].set(ylabel="accuracy", xlabel="epoch")
    ax[1].legend(["train", "val"], loc="upper left")
    plt.show()


def plot_values(df, base_features):
    rcParams["figure.figsize"] = 20, 3
    # for feature in base_features:
    feature = "vibration"
    sns.lineplot(df, x="datetime", y=feature)
    sns.scatterplot(
        df[df["pred_anomaly"] == True],
        x="datetime",
        y=feature,
        color="red",
        label="Prediction",
        zorder=7,
    )
    # for error_date in df.loc[df['pred_anomaly'] == True, 'datetime']:
    #     plt.axvline(x=error_date, color='orange', linestyle='--', linewidth=1, label='Error')
    for error_date in df.loc[df["is_anomaly"] == True, "datetime"]:
        plt.axvline(
            x=error_date,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Error",
            alpha=0.05,
        )
    for error_date in df.loc[df["is_error"] == True, "datetime"]:
        plt.axvline(
            x=error_date,
            color="orange",
            linestyle="--",
            linewidth=1,
            label="Possible Error",
            alpha=0.5,
        )
    plt.show()
