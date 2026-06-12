import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

TEST_FILE = "dataset/DailyDelhiClimateTest.csv"

# Load model and scaler
model = load_model("models/weather_rnn.h5")

scaler = joblib.load(
    "models/scaler.pkl"
)

# Load test dataset
df = pd.read_csv(TEST_FILE)

temperature = df["meantemp"].values

# Scale test data
scaled_data = scaler.transform(
    temperature.reshape(-1, 1)
)

SEQ_LENGTH = 30

X_test = []
y_test = []

for i in range(SEQ_LENGTH, len(scaled_data)):
    X_test.append(
        scaled_data[i - SEQ_LENGTH:i, 0]
    )
    y_test.append(
        scaled_data[i, 0]
    )

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

# Predict
predictions = model.predict(X_test)

# Convert back to original scale
predictions = scaler.inverse_transform(
    predictions
)

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# Metrics
mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

r2 = r2_score(
    actual,
    predictions
)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")