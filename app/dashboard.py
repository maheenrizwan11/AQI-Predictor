import streamlit as st
import pandas as pd
import datetime

# --- CONFIG ---
PREDICTION_FILE = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/refs/heads/main/data/predictions/predicted_aqi.csv"
FINAL_FEATURES_FILE = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/refs/heads/main/data/processed/features.csv"

# AQI category mapping (EPA standard)
AQI_CATEGORIES = [
    (0, 50, "🟢 Good", "Air quality is satisfactory."),
    (51, 100, "🟡 Moderate", "Air quality is acceptable."),
    (101, 150, "🟠 Unhealthy for Sensitive Groups", "May affect sensitive individuals."),
    (151, 200, "🔴 Unhealthy", "Everyone may begin to feel effects."),
    (201, 300, "🟣 Very Unhealthy", "Health alert: everyone affected."),
    (301, 500, "⚫ Hazardous", "Serious health effects for entire population.")
]

def get_aqi_category(aqi_value):
    aqi_value = round(aqi_value)
    for low, high, label, description in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return label, description
    return "❓ Unknown", "No category found."

# --- PAGE CONFIG ---
st.set_page_config(page_title="🌫️ AQI Prediction Dashboard", layout="wide")
st.title("🌫️ Air Quality Prediction Dashboard")

# --- LOAD PREDICTIONS ---
try:
    pred_df = pd.read_csv(PREDICTION_FILE)
except FileNotFoundError:
    st.error("Prediction file not found. Please run prediction.py first.")
    st.stop()

if pred_df.empty:
    st.warning("No prediction data available.")
    st.stop()

# --- LOAD TODAY'S ACTUAL AQI ---
try:
    features_df = pd.read_csv(FINAL_FEATURES_FILE, parse_dates=["datetime"])
    today_date = datetime.date.today()
    today_data = features_df[features_df["datetime"].dt.date == today_date]
    if not today_data.empty:
        today_aqi = today_data["aqi_us"].mean()
    else:
        today_aqi = None
except Exception as e:
    st.error(f"Error loading today's AQI: {e}")
    today_aqi = None

# Get latest prediction row
pred = pred_df.iloc[-1]

# --- DISPLAY 4 COLUMNS (Today + 3 Future Days) ---
cols = st.columns(4)

# Generate dates for today + t+24, t+48, t+72
dates = [
    datetime.date.today(),
    datetime.date.fromisoformat(pred["date"]) + datetime.timedelta(days=1),
    datetime.date.fromisoformat(pred["date"]) + datetime.timedelta(days=2),
    datetime.date.fromisoformat(pred["date"]) + datetime.timedelta(days=3)
]

aqi_values = [
    today_aqi,
    pred["predicted_aqi_t+24"],
    pred["predicted_aqi_t+48"],
    pred["predicted_aqi_t+72"]
]

# --- DISPLAY WITH CUSTOM STYLE ---
for i, col in enumerate(cols):
    day_name = dates[i].strftime("%A")
    aqi_value = aqi_values[i]

    if aqi_value is None or pd.isna(aqi_value):
        col.markdown(f"### {day_name}")
        col.markdown("_No data available_")
        continue

    aqi_value = round(aqi_value)
    category, description = get_aqi_category(aqi_value)

    # Choose background color
    bg_color = {
        "🟢 Good": "#4CAF50",
        "🟡 Moderate": "#FFEB3B",
        "🟠 Unhealthy for Sensitive Groups": "#FF9800",
        "🔴 Unhealthy": "#F44336",
        "🟣 Very Unhealthy": "#9C27B0",
        "⚫ Hazardous": "#212121"
    }.get(category, "#607D8B")

    with col:
        st.markdown(f"### {day_name}")
        st.markdown(
            f"""
            <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center;'>
                <h2 style='color:white; margin:0;'>{aqi_value}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f"**Category:** {category}")
        st.caption(description)
