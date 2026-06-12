
import os
import joblib
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping

from utils.preprocessing import scale_data, create_sequences
from utils.forecasting import build_rnn_model

TRAIN_FILE = "dataset/DailyDelhiClimateTrain.csv"

# Load training data
df = pd.read_csv(TRAIN_FILE)

temperature = df["meantemp"].values

# Scale data
scaled_data, scaler = scale_data(temperature)

# Create sequences
X_train, y_train = create_sequences(
    scaled_data,
    seq_length=30
)

# Build model
model = build_rnn_model(
    (X_train.shape[1], 1)
)

# Early stopping
early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Save model and scaler
os.makedirs("models", exist_ok=True)

model.save("models/weather_rnn.h5")
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Model saved successfully!")