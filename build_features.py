import os
import json
import pandas as pd
from datetime import datetime

RAW_DIR = "data/raw"

def load_data():
    rows = []

    for filename in sorted(os.listdir(RAW_DIR)):
        if filename.startswith("iqair_") or filename.startswith("openweather_"):
            with open(os.path.join(RAW_DIR, filename)) as f:
                data = json.load(f)
                timestamp_str = filename.replace("iqair_", "").replace("openweather_", "").replace(".json", "").replace("T", " ").replace("Z", "")
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H-%M-%S")
                rows.append((filename, dt, data))

    return rows

def process_rows(rows):
    records = []
    paired = {}

    for fname, dt, data in rows:
        key = dt.strftime("%Y-%m-%d %H:%M")

        if fname.startswith("iqair_"):
            current = data.get("data", {}).get("current", {})
            weather = current.get("weather", {})
            pollution = current.get("pollution", {})
            paired.setdefault(key, {}).update({
                "aqi_us": pollution.get("aqius"),
                "aqi_cn": pollution.get("aqicn"),
                "main_pollutant": pollution.get("mainus"),
                "temp": weather.get("tp"),
                "humidity": weather.get("hu"),
                "pressure": weather.get("pr"),
                "wind_speed": weather.get("ws"),
                "wind_direction": weather.get("wd"),
            })

        elif fname.startswith("openweather_"):
            try:
                comp = data["list"][0]["components"]
                main = data["list"][0]["main"]
                paired.setdefault(key, {}).update({
                    "openweather_aqi": main.get("aqi"),
                    "co": comp.get("co"),
                    "no": comp.get("no"),
                    "no2": comp.get("no2"),
                    "o3": comp.get("o3"),
                    "so2": comp.get("so2"),
                    "pm2_5": comp.get("pm2_5"),
                    "pm10": comp.get("pm10"),
                    "nh3": comp.get("nh3"),
                })
            except (KeyError, IndexError):
                print(f"⚠️ Skipped openweather data in {fname} — invalid structure.")

    for timestamp, vals in paired.items():
        if "aqi_us" in vals or "openweather_aqi" in vals:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
            record = {
                "datetime": dt,
                "hour": dt.hour,
                "day": dt.day,
                "month": dt.month,
                "weekday": dt.weekday(),
                **vals
            }
            records.append(record)

    return pd.DataFrame(records)

if __name__ == "__main__":
    rows = load_data()
    df = process_rows(rows)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    print(f"✅ Saved features.csv with {len(df)} rows and {df.shape[1]} columns.")
