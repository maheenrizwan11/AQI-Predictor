import pandas as pd
import joblib
import datetime
import os

# --- CONFIGURATION ---
url = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/refs/heads/main/data/processed/final_features.csv"
MODEL_PATH = "models"
PREDICTION_OUTPUT = "data/predictions/predicted_aqi.csv"

# Make sure output directory exists
os.makedirs(os.path.dirname(PREDICTION_OUTPUT), exist_ok=True)

# --- Step 1: Load Latest Feature Data ---
df = pd.read_csv(url, parse_dates=["datetime"])
latest = df.sort_values("datetime").iloc[-1:]

features = [
    'pm2_5', 'pm10', 'o3', 'no2', 'co', 'so2',
    'pressure', 'wind_speed', 'aqi_change_rate',
    'temp', 'humidity', 'wind_direction'
]
X = latest[features]

# --- Step 2: Load Trained Models ---
model_t24 = joblib.load(os.path.join(MODEL_PATH, "RandomForest_t+24.pkl"))
model_t48 = joblib.load(os.path.join(MODEL_PATH, "RandomForest_t+48.pkl"))
model_t72 = joblib.load(os.path.join(MODEL_PATH, "XGBoost_t+72.pkl"))

# --- Step 3: Make Predictions ---
pred_t24 = model_t24.predict(X)[0]
pred_t48 = model_t48.predict(X)[0]
pred_t72 = model_t72.predict(X)[0]

# --- Step 4: Save to File ---
today = datetime.datetime.now()
preds = {
    "date": today.isoformat(),
    "predicted_aqi_t+24": round(pred_t24, 2),
    "predicted_aqi_t+48": round(pred_t48, 2),
    "predicted_aqi_t+72": round(pred_t72, 2)
}

pd.DataFrame([preds]).to_csv(PREDICTION_OUTPUT, index=False)
print("✅ Saved predictions to", PREDICTION_OUTPUT)
