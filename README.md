# 🌦 Weather Forecasting Using Recurrent Neural Networks (RNN)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview

Weather forecasting plays a critical role in agriculture, transportation, disaster management, and everyday decision-making.

This project leverages a **Recurrent Neural Network (RNN)** to analyze historical weather patterns and predict future temperatures. The model learns temporal dependencies from past observations and forecasts upcoming weather conditions through an interactive Streamlit dashboard.

---

## 🚀 Features

✅ Historical Weather Data Analysis

✅ Temperature Trend Visualization

✅ Future Temperature Prediction

✅ Weather Condition Classification

✅ Interactive Streamlit Dashboard

✅ Forecast Download as CSV

✅ Deep Learning-Based Time Series Forecasting

---

## 🧠 Problem Statement

Traditional machine learning models struggle to capture sequential dependencies in weather data.

This project uses a **Simple RNN architecture** that processes weather observations as a sequence, enabling the model to learn patterns across time and generate future temperature predictions.

---

## 📊 Dataset

### Daily Delhi Climate Dataset

Dataset contains:

* Date
* Mean Temperature
* Humidity
* Wind Speed
* Mean Pressure

Files:

```text
DailyDelhiClimateTrain.csv
DailyDelhiClimateTest.csv
```

### Dataset Statistics

| Dataset | Records |
| ------- | ------- |
| Train   | ~1462   |
| Test    | ~114    |

---

## 🏗 Project Architecture

```text
Weather Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Normalization
        │
        ▼
Sequence Generation
        │
        ▼
RNN Model
        │
        ▼
Temperature Prediction
        │
        ▼
Weather Classification
        │
        ▼
Streamlit Dashboard
```

---

## 🔍 Model Workflow

### Step 1: Data Preprocessing

* Missing value handling
* Feature extraction
* MinMax scaling

### Step 2: Sequence Creation

Example:

```text
Previous 30 Days Temperature
          ↓
Predict Next Day Temperature
```

```text
[25,26,27,28,...,30]
              ↓
             31
```

### Step 3: RNN Training

The model learns weather trends using recurrent hidden states.

### Step 4: Forecast Generation

Future temperatures are generated recursively using previous predictions.

---

## 🧠 RNN Architecture

```text
Input Layer (30 Days)
         │
         ▼
SimpleRNN (64 Units)
         │
         ▼
Dropout (20%)
         │
         ▼
SimpleRNN (64 Units)
         │
         ▼
Dropout (20%)
         │
         ▼
Dense Layer (25 Units)
         │
         ▼
Output Layer (1 Unit)
         │
         ▼
Predicted Temperature
```

---

## 📁 Project Structure

```text
Weather-Forecasting-RNN/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
│
├── dataset/
│   ├── DailyDelhiClimateTrain.csv
│   └── DailyDelhiClimateTest.csv
│
├── models/
│   ├── weather_rnn.h5
│   └── scaler.pkl
│
├── utils/
│   ├── preprocessing.py
│   └── forecasting.py
│
└── README.md
```

---

## ⚙ Installation

Clone Repository

```bash
git clone https://github.com/yourusername/weather-forecasting-rnn.git
```

Move into project directory

```bash
cd weather-forecasting-rnn
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋 Train the Model

```bash
python train.py
```

This generates:

```text
models/weather_rnn.h5
models/scaler.pkl
```

---

## 📈 Evaluate Model

```bash
python predict.py
```

Metrics displayed:

```text
MAE
RMSE
R² Score
```

---

## 🌐 Run Streamlit Application

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

## 📊 Dashboard Features

### Historical Analysis

* Temperature Trends
* Weather Statistics
* Dataset Exploration

### Forecasting

* Future Temperature Prediction
* Weather Condition Detection
* Interactive Charts

### Export

* Download forecast results as CSV

---

## 🌤 Weather Classification Logic

| Temperature | Weather          |
| ----------- | ---------------- |
| ≥ 35°C      | ☀️ Sunny         |
| 25°C - 34°C | ⛅ Partly Cloudy  |
| 15°C - 24°C | ☁️ Cloudy        |
| < 15°C      | 🌧️ Rainy / Cold |

---

## 📷 Application Preview

```text
Dashboard
│
├── Dataset Statistics
├── Historical Trend Chart
├── Future Forecast Chart
├── Weather Cards
├── Forecast Table
└── CSV Download
```

---

## 💡 Skills Demonstrated

* Python
* Deep Learning
* Recurrent Neural Networks (RNN)
* TensorFlow / Keras
* Time Series Forecasting
* Data Visualization
* Streamlit
* Model Deployment
* Feature Engineering
* Sequence Modeling

---

## 🎯 Future Improvements

* Multi-City Forecasting
* Rainfall Prediction
* Humidity Forecasting
* GRU Implementation
* LSTM Comparison
* Transformer-Based Forecasting
* Live Weather API Integration

---

## 📝 Resume Description

Developed an end-to-end Weather Forecasting System using Recurrent Neural Networks (RNN) and Streamlit. Implemented sequence-based time series forecasting to predict future temperatures from historical climate data. Built an interactive dashboard featuring weather analytics, forecasting visualizations, weather condition classification, and downloadable prediction reports.

---

## 👨‍💻 Author

**Koushik Patel**

If you found this project useful, consider giving it a ⭐ on GitHub.
