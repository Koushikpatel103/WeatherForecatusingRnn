import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Weather Forecasting Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = load_model("models/weather_lstm.h5")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()

# --------------------------------------------------
# WEATHER CONDITION FUNCTION
# --------------------------------------------------

def get_weather_condition(temp):

    if temp >= 35:
        return "☀️ Sunny"

    elif temp >= 25:
        return "⛅ Partly Cloudy"

    elif temp >= 15:
        return "☁️ Cloudy"

    else:
        return "🌧️ Rainy / Cold"


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙ Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=1,
    max_value=30,
    value=7
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Weather Dataset",
    type=["csv"]
)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(
        "dataset/DailyDelhiClimateTrain.csv"
    )

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🌦 Weather Forecasting Using RNN")
st.markdown(
    "Forecast future weather conditions using a Recurrent Neural Network."
)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("📄 View Dataset"):
    st.dataframe(df.head())

# --------------------------------------------------
# METRICS
# --------------------------------------------------

st.subheader("📊 Weather Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Temp",
    f"{df['meantemp'].mean():.2f}°C"
)

col2.metric(
    "Maximum Temp",
    f"{df['meantemp'].max():.2f}°C"
)

col3.metric(
    "Minimum Temp",
    f"{df['meantemp'].min():.2f}°C"
)

col4.metric(
    "Humidity",
    f"{df['humidity'].mean():.2f}%"
)

# --------------------------------------------------
# HISTORICAL TEMPERATURE CHART
# --------------------------------------------------

st.subheader("📈 Historical Temperature Trend")

historical_fig = px.line(
    df,
    x="date",
    y="meantemp",
    title="Historical Temperature"
)

st.plotly_chart(
    historical_fig,
    use_container_width=True
)

# --------------------------------------------------
# FORECASTING
# --------------------------------------------------

temperature = df["meantemp"].values

scaled_data = scaler.transform(
    temperature.reshape(-1, 1)
)

last_sequence = scaled_data[-30:]

future_predictions = []

current_batch = last_sequence.copy()

for _ in range(forecast_days):

    current_input = current_batch.reshape(
        1,
        30,
        1
    )

    prediction = model.predict(
        current_input,
        verbose=0
    )

    future_predictions.append(
        prediction[0][0]
    )

    current_batch = np.append(
        current_batch[1:],
        prediction
    )

future_predictions = np.array(
    future_predictions
).reshape(-1, 1)

future_predictions = scaler.inverse_transform(
    future_predictions
)

# --------------------------------------------------
# FORECAST DATAFRAME
# --------------------------------------------------

forecast_df = pd.DataFrame({
    "Day": np.arange(
        1,
        forecast_days + 1
    ),
    "Predicted Temperature":
        future_predictions.flatten()
})

forecast_df["Weather"] = (
    forecast_df["Predicted Temperature"]
    .apply(get_weather_condition)
)

# --------------------------------------------------
# FORECAST SUMMARY
# --------------------------------------------------

st.subheader("🌍 Forecast Summary")

avg_forecast = (
    forecast_df["Predicted Temperature"]
    .mean()
)

st.success(
    f"Average Predicted Temperature: "
    f"{avg_forecast:.2f}°C"
)

# --------------------------------------------------
# WEATHER CARDS
# --------------------------------------------------

st.subheader("🌤 Predicted Weather")

cols = st.columns(
    min(forecast_days, 5)
)

for i in range(
    min(forecast_days, 5)
):

    cols[i].metric(
        f"Day {i+1}",
        f"{forecast_df.iloc[i]['Predicted Temperature']:.1f}°C"
    )

    cols[i].write(
        forecast_df.iloc[i]["Weather"]
    )

# --------------------------------------------------
# FORECAST TABLE
# --------------------------------------------------

st.subheader("📋 Forecast Table")

st.dataframe(
    forecast_df,
    use_container_width=True
)

# --------------------------------------------------
# FORECAST CHART
# --------------------------------------------------

st.subheader("📉 Future Forecast")

forecast_fig = px.line(
    forecast_df,
    x="Day",
    y="Predicted Temperature",
    markers=True,
    title="Future Temperature Forecast"
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)

# --------------------------------------------------
# DOWNLOAD CSV
# --------------------------------------------------

csv = forecast_df.to_csv(
    index=False
)

st.download_button(
    label="⬇ Download Forecast CSV",
    data=csv,
    file_name="weather_forecast.csv",
    mime="text/csv"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.caption(
    "Weather Forecasting Dashboard | RNN + Streamlit"
)