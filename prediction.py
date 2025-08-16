import pandas as pd
import joblib
import datetime
import os
import glob

# --- CONFIGURATION ---
url = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/refs/heads/main/data/processed/final_features.csv"
MODEL_PATH = "models"
PREDICTION_OUTPUT = "data/predictions/predicted_aqi.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(PREDICTION_OUTPUT), exist_ok=True)

# --- Step 1: Load Latest Feature Data ---
df = pd.read_csv(url, parse_dates=["datetime"])
latest = df.sort_values("datetime").iloc[-1:]

features = [
    'pm2_5', 'pm10', 'o3', 'no2', 'co', 'so2',
    'pressure', 'wind_speed', 'temp', 'aqi_change_rate',
    'humidity', 'wind_direction'
]
X = latest[features]

# --- Helper to load model by horizon ---
def load_model_for_horizon(horizon):
    # find any file that ends with _t+XX.pkl
    pattern = os.path.join(MODEL_PATH, f"*_{horizon}.pkl")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No model found for horizon {horizon}")
    return joblib.load(matches[0])  # pick first match (only one expected)

# --- Load models dynamically ---
models = {
    "t+24": load_model_for_horizon("t+24"),
    "t+48": load_model_for_horizon("t+48"),
    "t+72": load_model_for_horizon("t+72"),
}

# --- Make Predictions ---
preds = {
    "date": datetime.date.today().isoformat(),
    "predicted_aqi_t+24": round(models["t+24"].predict(X)[0], 2),
    "predicted_aqi_t+48": round(models["t+48"].predict(X)[0], 2),
    "predicted_aqi_t+72": round(models["t+72"].predict(X)[0], 2),
}

# --- Save Predictions ---
pd.DataFrame([preds]).to_csv(PREDICTION_OUTPUT, index=False)
print("✅ Saved latest prediction to", PREDICTION_OUTPUT)