import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath):
    """
    Load weather dataset.
    """
    df = pd.read_csv(filepath)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def scale_data(data):
    """
    Scale temperature values between 0 and 1.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(
        data.reshape(-1, 1)
    )

    return scaled_data, scaler


def create_sequences(data, seq_length=30):
    """
    Convert time-series data into LSTM sequences.

    Example:
    Input:
    [1,2,3,4,5]

    seq_length=3

    X = [1,2,3]
    y = [4]
    """

    X = []
    y = []

    for i in range(seq_length, len(data)):
        X.append(
            data[i - seq_length:i, 0]
        )

        y.append(
            data[i, 0]
        )

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y