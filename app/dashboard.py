import streamlit as st
import pandas as pd
import datetime

# --- CONFIG ---
PREDICTION_FILE = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/refs/heads/main/data/predictions/predicted_aqi.csv"

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
    aqi_value = round(aqi_value)  # fix float comparison issue
    for low, high, label, description in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return label, description
    return "❓ Unknown", "No category found."

# --- PAGE CONFIG ---
st.set_page_config(page_title="🌫️ AQI Prediction Dashboard", layout="wide")
st.title("🌫️ Air Quality Prediction Dashboard")

# --- LOAD DATA ---
try:
    pred_df = pd.read_csv(PREDICTION_FILE)
except FileNotFoundError:
    st.error("Prediction file not found. Please run prediction.py first.")
    st.stop()

if pred_df.empty:
    st.warning("No prediction data available.")
    st.stop()

# Get latest prediction row
pred = pred_df.iloc[-1]

# --- DISPLAY 3 COLUMNS ---
cols = st.columns(3)

# Generate dates for t+24, t+48, t+72
base_date = datetime.date.fromisoformat(pred["date"])
future_dates = [
    base_date + datetime.timedelta(days=1),
    base_date + datetime.timedelta(days=2),
    base_date + datetime.timedelta(days=3)
]

aqi_values = [
    pred["predicted_aqi_t+24"],
    pred["predicted_aqi_t+48"],
    pred["predicted_aqi_t+72"]
]

# --- DISPLAY WITH CUSTOM STYLE ---
for i, col in enumerate(cols):
    day_name = future_dates[i].strftime("%A")
    aqi_value = round(aqi_values[i])
    category, description = get_aqi_category(aqi_value)

    # Choose background color based on category
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
