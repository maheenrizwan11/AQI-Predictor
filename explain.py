import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- CONFIG ---
FEATURE_FILE = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/refs/heads/main/data/processed/final_features.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "data/explainability"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features used in training
FEATURES = [
    'pm2_5', 'pm10', 'o3', 'no2', 'co', 'so2',
    'pressure', 'wind_speed', 'aqi_change_rate',
    'temp', 'humidity', 'wind_direction'
]

# Horizons we care about
HORIZONS = ["t+24", "t+48", "t+72"]

# --- LOAD DATA ---
df = pd.read_csv(FEATURE_FILE, parse_dates=["datetime"])
X = df[FEATURES].tail(100)  # Use last 100 rows for faster SHAP

# --- HELPER: find model dynamically ---
def load_model_for_horizon(horizon):
    pattern = os.path.join(MODEL_DIR, f"*_{horizon}.pkl")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No model found for horizon {horizon}")
    return joblib.load(matches[0])  # take first (latest) match

# --- GENERATE EXPLANATIONS ---
for horizon in HORIZONS:
    print(f"🔍 Generating SHAP values for {horizon}...")

    # Load model
    model = load_model_for_horizon(horizon)

    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # --- GLOBAL FEATURE IMPORTANCE ---
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"Global Feature Importance - AQI {horizon}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_summary_{horizon}.png"), bbox_inches="tight")
    plt.close()

    # --- LOCAL EXPLANATION FOR LATEST ROW ---
    latest_X = X.tail(1)
    latest_shap_values = explainer(latest_X)

print(f"✅ SHAP explanations saved in '{OUTPUT_DIR}'")
