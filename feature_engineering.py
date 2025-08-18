import pandas as pd
import numpy as np
import os

BREAKPOINTS = {
    "PM2.5": [
        [0.0, 12.0, 0, 50],
        [12.1, 35.4, 51, 100],
        [35.5, 55.4, 101, 150],
        [55.5, 150.4, 151, 200],
        [150.5, 250.4, 201, 300],
        [250.5, 350.4, 301, 400],
        [350.5, 500.4, 401, 500]
    ],
    "PM10": [
        [0, 54, 0, 50],
        [55, 154, 51, 100],
        [155, 254, 101, 150],
        [255, 354, 151, 200],
        [355, 424, 201, 300],
        [425, 504, 301, 400],
        [505, 604, 401, 500]
    ],
    "O3": [
        [0, 108, 0, 50],
        [109, 140, 51, 100],
        [141, 170, 101, 150],
        [171, 210, 151, 200],
        [211, 400, 201, 300],
        [401, 600, 301, 400],
        [601, 1000, 401, 500]
    ],
    "NO2": [
        [0, 100, 0, 50],
        [101, 188, 51, 100],
        [189, 677, 101, 150],
        [678, 1221, 151, 200],
        [1222, 2349, 201, 300],
        [2350, 3102, 301, 400],
        [3103, 3853, 401, 500]
    ],
    "SO2": [
        [0, 92, 0, 50],
        [93, 196, 51, 100],
        [197, 484, 101, 150],
        [485, 797, 151, 200],
        [798, 1582, 201, 300],
        [1583, 2104, 301, 400],
        [2105, 2627, 401, 500]
    ],
    "CO": [
        [0, 4400, 0, 50],
        [4401, 9400, 51, 100],
        [9401, 12400, 101, 150],
        [12401, 15400, 151, 200],
        [15401, 30400, 201, 300],
        [30401, 40400, 301, 400],
        [40401, 50400, 401, 500]
    ]
}

def calculate_aqi(pollutant, concentration):
    if concentration is None or pd.isna(concentration) or concentration < 0:
        return None
    bps = BREAKPOINTS.get(pollutant)
    if not bps:
        return None
    for bp_lo, bp_hi, i_lo, i_hi in bps:
        if bp_lo <= concentration <= bp_hi:
            return round(((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo)
    if concentration > bps[-1][1]:
        return 500
    return None

def calculate_nowcast(values):
    values = [v for v in values if pd.notnull(v)]
    if len(values) < 3:
        return None
    c_min, c_max = min(values), max(values)
    if c_max == 0:
        return 0
    weight = min(max(1 - (c_max - c_min) / c_max, 0.5), 1)
    num, denom = 0, 0
    for i, v in enumerate(reversed(values[-12:])):  # last 12 hours
        w = weight ** i
        num += v * w
        denom += w
    return num / denom if denom > 0 else None

def add_features(df):
    df = df.sort_values("datetime").reset_index(drop=True)
    df["computed_aqi"] = None
    df["main_pollutant_calc"] = None
    df["pm2_5_nowcast"] = None
    df["pm10_nowcast"] = None

    for i in range(len(df)):
        pm25_nc = calculate_nowcast(df.loc[max(0, i-11):i, "pm2_5"].tolist())
        pm10_nc = calculate_nowcast(df.loc[max(0, i-11):i, "pm10"].tolist())
        df.loc[i, "pm2_5_nowcast"] = pm25_nc
        df.loc[i, "pm10_nowcast"] = pm10_nc

        sub_indices = {}
        if pd.notnull(pm25_nc):
            sub_indices["PM2.5"] = calculate_aqi("PM2.5", pm25_nc)
        if pd.notnull(pm10_nc):
            sub_indices["PM10"] = calculate_aqi("PM10", pm10_nc)
        if pd.notnull(df.loc[i, "o3"]):
            sub_indices["O3"] = calculate_aqi("O3", df.loc[i, "o3"])
        if pd.notnull(df.loc[i, "no2"]):
            sub_indices["NO2"] = calculate_aqi("NO2", df.loc[i, "no2"])
        if pd.notnull(df.loc[i, "so2"]):
            sub_indices["SO2"] = calculate_aqi("SO2", df.loc[i, "so2"])
        if pd.notnull(df.loc[i, "co"]):
            sub_indices["CO"] = calculate_aqi("CO", df.loc[i, "co"])

        sub_indices = {k: v for k, v in sub_indices.items() if v is not None}
        if sub_indices:
            main_pollutant = max(sub_indices, key=sub_indices.get)
            df.loc[i, "computed_aqi"] = sub_indices[main_pollutant]
            df.loc[i, "main_pollutant_calc"] = main_pollutant

    df["aqi_change_rate"] = df["computed_aqi"].pct_change()

    # Target variables for next 3 days
    df["computed_aqi_t+24"] = df["computed_aqi"].shift(-24)
    df["computed_aqi_t+48"] = df["computed_aqi"].shift(-48)
    df["computed_aqi_t+72"] = df["computed_aqi"].shift(-72)

    # Calculate MAE between aqi_us and computed_aqi
    if "aqi_us" in df.columns:
        valid_rows = df.dropna(subset=["aqi_us", "computed_aqi"])
        mae = np.mean(np.abs(valid_rows["aqi_us"] - valid_rows["computed_aqi"]))
        print(f"📏 Mean Absolute Error (aqi_us vs computed_aqi): {mae:.2f}")

    # Only drop rows where core pollutant values or computed_aqi are missing
    essential = ["computed_aqi", "pm2_5", "pm10", "o3", "no2", "so2", "co"]
    df = df.dropna(subset=essential).reset_index(drop=True)
    return df

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/refs/heads/main/data/processed/features.csv"
    df = pd.read_csv(url, parse_dates=["datetime"])
    df = add_features(df)
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/final_features.csv", index=False)
    print(f"✅ Saved final_features.csv with {len(df)} rows and {df.shape[1]} columns.")
