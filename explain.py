import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

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

# Model names
MODELS = {
    "t+24": "Randomforest_t+24.pkl",
    "t+48": "XGBoost_t+48.pkl",
    "t+72": "XGBoost_t+72.pkl"
}

# --- LOAD DATA ---
df = pd.read_csv(FEATURE_FILE, parse_dates=["datetime"])
X = df[FEATURES].tail(100)  # Use last 100 rows for faster SHAP

# --- GENERATE EXPLANATIONS ---
for horizon, model_file in MODELS.items():
    print(f"🔍 Generating SHAP values for {horizon}...")

    # Load model
    model_path = os.path.join(MODEL_DIR, model_file)
    model = joblib.load(model_path)

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

    plt.figure()
    shap.plots.waterfall(latest_shap_values[0], show=False)
    plt.title(f"Prediction Breakdown - Latest AQI {horizon}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_waterfall_{horizon}.png"), bbox_inches="tight")
    plt.close()

print(f"✅ SHAP explanations saved in '{OUTPUT_DIR}'")
