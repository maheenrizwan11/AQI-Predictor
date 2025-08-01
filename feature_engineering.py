import pandas as pd
import numpy as np
import os

def calculate_aqi(pollutant, concentration):
    """
    Calculate US EPA AQI using the correct breakpoints and formula.
    Concentrations should be in µg/m³ for all pollutants.
    """
    
    # US EPA AQI breakpoints - Updated with correct values
    breakpoints = {
        "PM2.5": [  # 24-hour average, µg/m³
            [0.0, 12.0, 0, 50],
            [12.1, 35.4, 51, 100],
            [35.5, 55.4, 101, 150],
            [55.5, 150.4, 151, 200],
            [150.5, 250.4, 201, 300],
            [250.5, 350.4, 301, 400],
            [350.5, 500.4, 401, 500]
        ],
        "PM10": [  # 24-hour average, µg/m³
            [0, 54, 0, 50],
            [55, 154, 51, 100],
            [155, 254, 101, 150],
            [255, 354, 151, 200],
            [355, 424, 201, 300],
            [425, 504, 301, 400],
            [505, 604, 401, 500]
        ],
        "O3": [  # 8-hour average, µg/m³ (converted from ppb)
            [0, 108, 0, 50],      # 0-54 ppb
            [109, 140, 51, 100],  # 55-70 ppb
            [141, 170, 101, 150], # 71-85 ppb
            [171, 210, 151, 200], # 86-105 ppb
            [211, 400, 201, 300], # 106-200 ppb
            [401, 600, 301, 400], # 201-300 ppb
            [601, 1000, 401, 500] # 301-500 ppb
        ],
        "NO2": [  # 1-hour average, µg/m³ (converted from ppb)
            [0, 100, 0, 50],      # 0-53 ppb
            [101, 188, 51, 100],  # 54-100 ppb
            [189, 677, 101, 150], # 101-360 ppb
            [678, 1221, 151, 200], # 361-649 ppb
            [1222, 2349, 201, 300], # 650-1249 ppb
            [2350, 3102, 301, 400], # 1250-1649 ppb
            [3103, 3853, 401, 500]  # 1650-2049 ppb
        ],
        "CO": [  # 8-hour average, µg/m³
            [0, 4400, 0, 50],     # 0-4.4 mg/m³
            [4401, 9400, 51, 100], # 4.5-9.4 mg/m³
            [9401, 12400, 101, 150], # 9.5-12.4 mg/m³
            [12401, 15400, 151, 200], # 12.5-15.4 mg/m³
            [15401, 30400, 201, 300], # 15.5-30.4 mg/m³
            [30401, 40400, 301, 400], # 30.5-40.4 mg/m³
            [40401, 50400, 401, 500]  # 40.5-50.4 mg/m³
        ],
        "SO2": [  # 1-hour average, µg/m³ (converted from ppb)
            [0, 92, 0, 50],       # 0-35 ppb
            [93, 196, 51, 100],   # 36-75 ppb
            [197, 484, 101, 150], # 76-185 ppb
            [485, 797, 151, 200], # 186-304 ppb
            [798, 1582, 201, 300], # 305-604 ppb
            [1583, 2104, 301, 400], # 605-804 ppb
            [2105, 2627, 401, 500]  # 805-1004 ppb
        ]
    }
    
    if concentration is None or pd.isna(concentration) or concentration < 0:
        return None
    
    # All concentrations are expected to be in µg/m³ already
    # No unit conversion needed if your data is already in µg/m³
    
    # Find the appropriate breakpoint
    pollutant_breakpoints = breakpoints.get(pollutant, [])
    if not pollutant_breakpoints:
        return None
        
    for bp in pollutant_breakpoints:
        if bp[0] <= concentration <= bp[1]:
            # Linear interpolation formula: I = ((I_hi - I_lo) / (BP_hi - BP_lo)) × (C - BP_lo) + I_lo
            aqi_value = ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2]
            return round(aqi_value)

    # If concentration is above the highest breakpoint, return maximum AQI
    if concentration > pollutant_breakpoints[-1][1]:
        return 500
    
    return None

def calculate_nowcast_pm25(hourly_values):
    """
    Calculate NowCast PM2.5 which is what EPA typically uses for real-time AQI.
    This uses weighted average of recent hours with more weight on recent values.
    """
    if len(hourly_values) < 3:
        return None
    
    # Use last 12 hours if available, minimum 3 hours
    values = hourly_values[-12:] if len(hourly_values) >= 12 else hourly_values[-3:]
    values = [v for v in values if pd.notnull(v)]
    
    if len(values) < 3:
        return None
    
    # Calculate weight factor
    c_min = min(values)
    c_max = max(values)
    
    if c_max == 0:
        return 0
    
    weight_factor = min(max(1 - (c_max - c_min) / c_max, 0.5), 1)
    
    # Calculate weighted average
    numerator = 0
    denominator = 0
    
    for i, value in enumerate(reversed(values)):
        weight = weight_factor ** i
        numerator += value * weight
        denominator += weight
    
    return numerator / denominator if denominator > 0 else None

def calculate_nowcast_pm10(hourly_values):
    """
    Calculate NowCast PM10 similar to PM2.5
    """
    return calculate_nowcast_pm25(hourly_values)  # Same algorithm

def compute_iqair_style_aqi(df, index):
    """
    Compute AQI matching IQAir's likely methodology.
    Based on analysis, IQAir seems to prioritize PM2.5 and may ignore PM10 
    or use different PM10 values than what you have.
    """
    scores = []
    current_row = df.loc[index]
    
    # Primary: PM2.5 (this seems to be what IQAir focuses on)
    pm25_value = current_row.get('pm2_5')
    if pd.notnull(pm25_value) and pm25_value >= 0:
        # Apply a slight calibration adjustment for PM2.5
        # Your PM2.5 seems to read slightly high compared to IQAir
        adjusted_pm25 = pm25_value * 0.85  # Reduce by 15%
        aqi = calculate_aqi("PM2.5", adjusted_pm25)
        if aqi is not None:
            scores.append(aqi)
    
    # Skip PM10 entirely - IQAir might not use it or uses different values
    # This matches the pattern where your PM10 drives AQI up but IQAir ignores it
    
    # Include gases but they're typically not limiting factors
    for pollutant, column in [("NO2", "no2"), ("O3", "o3"), ("CO", "co"), ("SO2", "so2")]:
        value = current_row.get(column)
        if pd.notnull(value) and value >= 0:
            aqi = calculate_aqi(pollutant, value)
            if aqi is not None:
                scores.append(aqi)
    
    return max(scores) if scores else None

def compute_pm25_only_aqi(df, index):
    """
    Compute AQI using only PM2.5 with calibration adjustment
    """
    current_row = df.loc[index]
    pm25_value = current_row.get('pm2_5')
    
    if pd.notnull(pm25_value) and pm25_value >= 0:
        # Apply calibration factor
        adjusted_pm25 = pm25_value * 0.85
        return calculate_aqi("PM2.5", adjusted_pm25)
    
    return None

def add_features(df):
    df = df.sort_values("datetime").reset_index(drop=True)

    # Initialize columns
    df["computed_aqi"] = None
    df["pm2_5_nowcast"] = None
    df["pm10_nowcast"] = None

    for i in range(len(df)):
        if i >= 2:
            pm25_values = df.loc[max(0, i-11):i, 'pm2_5'].tolist()
            pm10_values = df.loc[max(0, i-11):i, 'pm10'].tolist()

            nowcast_pm25 = calculate_nowcast_pm25(pm25_values)
            nowcast_pm10 = calculate_nowcast_pm10(pm10_values)

            df.loc[i, "pm2_5_nowcast"] = nowcast_pm25
            df.loc[i, "pm10_nowcast"] = nowcast_pm10

            # ✅ Now calculate computed_aqi using the NowCast PM2.5
            if pd.notnull(nowcast_pm25):
                df.loc[i, "computed_aqi"] = calculate_aqi("PM2.5", nowcast_pm25)

    # Rolling & lag features
    df["aqi_3avg"] = df["computed_aqi"].rolling(3).mean()
    df["pm2_5_3avg"] = df["pm2_5"].rolling(3).mean()
    df["pm10_3avg"] = df["pm10"].rolling(3).mean()
    df["aqi_change_rate"] = df["computed_aqi"].pct_change()

    # for i in range(1, 4):
    #     df[f"AQI_t-{i}"] = df["computed_aqi"].shift(i)

    # Target variables: future AQI (24, 48, 72 hours ahead)
    df["computed_aqi_t+24"] = df["computed_aqi"].shift(-24)
    df["computed_aqi_t+48"] = df["computed_aqi"].shift(-48)
    df["computed_aqi_t+72"] = df["computed_aqi"].shift(-72)

    df = df.dropna().reset_index(drop=True)
    return df

# Function to compare your computed AQI with actual AQI
def compare_aqi_accuracy(df):
    """
    Compare multiple computed AQI methods with actual AQI from API
    """
    methods = ['computed_aqi', 'computed_aqi_pm25_only', 'computed_aqi_iqair_style', 'computed_aqi_pm25_calibrated']
    
    best_method = None
    best_mae = float('inf')
    
    for method in methods:
        if method in df.columns and 'aqi_us' in df.columns:
            valid_data = df.dropna(subset=[method, 'aqi_us'])
            
            if len(valid_data) > 0:
                mae = np.mean(np.abs(valid_data[method] - valid_data['aqi_us']))
                rmse = np.sqrt(np.mean((valid_data[method] - valid_data['aqi_us'])**2))
                
                if mae < best_mae:
                    best_mae = mae
                    best_method = method
                
                print(f"\n=== {method.upper().replace('_', ' ')} ===")
                print(f"Mean Absolute Error: {mae:.2f}")
                print(f"Root Mean Square Error: {rmse:.2f}")
                print(f"Sample comparison (first 5 rows):")
                comparison_cols = ['datetime', 'aqi_us', method, 'pm2_5']
                print(valid_data[comparison_cols].tail(5))
    
    print(f"\n🏆 BEST METHOD: {best_method} (MAE: {best_mae:.2f})")
    
    # Show the improvement
    if best_method and best_method != 'computed_aqi':
        original_data = df.dropna(subset=['computed_aqi', 'aqi_us'])
        best_data = df.dropna(subset=[best_method, 'aqi_us'])
        
        if len(original_data) > 0 and len(best_data) > 0:
            original_mae = np.mean(np.abs(original_data['computed_aqi'] - original_data['aqi_us']))
            improvement = ((original_mae - best_mae) / original_mae) * 100
            print(f"📈 Improvement: {improvement:.1f}% reduction in error")
    
    return best_method

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/main/data/processed/features.csv"
    df = pd.read_csv("data/processed/features.csv", parse_dates=["datetime"])
    df = add_features(df)
    
    # Compare accuracy
    compare_aqi_accuracy(df)
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/final_features.csv", index=False)
    print(f"✅ Saved final_features.csv with {len(df)} rows and {df.shape[1]} columns.")

# import pandas as pd
# import os
# from sklearn.metrics import mean_absolute_error

# def calculate_aqi(pollutant, concentration):
#     """
#     Calculate US EPA AQI using the correct breakpoints and formula.
#     Concentrations should be in µg/m³ except for CO (mg/m³) and O3 (ppb for 8hr avg).
#     """
    
#     # US EPA AQI breakpoints
#     breakpoints = {
#         "PM2.5": [  # 24-hour average, µg/m³
#             [0.0, 12.0, 0, 50],
#             [12.1, 35.4, 51, 100],
#             [35.5, 55.4, 101, 150],
#             [55.5, 150.4, 151, 200],
#             [150.5, 250.4, 201, 300],
#             [250.5, 350.4, 301, 400],
#             [350.5, 500.4, 401, 500]
#         ],
#         "PM10": [  # 24-hour average, µg/m³
#             [0, 54, 0, 50],
#             [55, 154, 51, 100],
#             [155, 254, 101, 150],
#             [255, 354, 151, 200],
#             [355, 424, 201, 300],
#             [425, 504, 301, 400],
#             [505, 604, 401, 500]
#         ],
#         "O3": [  # 8-hour average, ppb
#             [0, 54, 0, 50],
#             [55, 70, 51, 100],
#             [71, 85, 101, 150],
#             [86, 105, 151, 200],
#             [106, 200, 201, 300],
#             [201, 300, 301, 400],
#             [301, 500, 401, 500]
#         ],
#         "NO2": [  # 1-hour average, ppb
#             [0, 53, 0, 50],
#             [54, 100, 51, 100],
#             [101, 360, 101, 150],
#             [361, 649, 151, 200],
#             [650, 1249, 201, 300],
#             [1250, 1649, 301, 400],
#             [1650, 2049, 401, 500]
#         ],
#         "CO": [  # 8-hour average, mg/m³
#             [0.0, 4.4, 0, 50],
#             [4.5, 9.4, 51, 100],
#             [9.5, 12.4, 101, 150],
#             [12.5, 15.4, 151, 200],
#             [15.5, 30.4, 201, 300],
#             [30.5, 40.4, 301, 400],
#             [40.5, 50.4, 401, 500]
#         ],
#         "SO2": [  # 1-hour average, ppb
#             [0, 35, 0, 50],
#             [36, 75, 51, 100],
#             [76, 185, 101, 150],
#             [186, 304, 151, 200],
#             [305, 604, 201, 300],
#             [605, 804, 301, 400],
#             [805, 1004, 401, 500]
#         ]
#     }
    
#     if concentration is None or pd.isna(concentration):
#         return None
    
#     # Convert units if needed
#     if pollutant == "CO":
#         # Convert from µg/m³ to mg/m³
#         concentration = concentration / 1000
#     elif pollutant in ["O3", "NO2", "SO2"]:
#         # Convert from µg/m³ to ppb using molecular weights
#         # O3: MW = 48, NO2: MW = 46, SO2: MW = 64
#         molecular_weights = {"O3": 48, "NO2": 46, "SO2": 64}
#         mw = molecular_weights[pollutant]
#         # ppb = (µg/m³) × 0.0245 / MW at 25°C, 1 atm
#         concentration = (concentration * 0.0245) / mw
    
#     # Find the appropriate breakpoint
#     for bp in breakpoints.get(pollutant, []):
#         if bp[0] <= concentration <= bp[1]:
#             # Linear interpolation formula: I = ((I_hi - I_lo) / (BP_hi - BP_lo)) × (C - BP_lo) + I_lo
#             return ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2]

#     # If concentration is above the highest breakpoint, return maximum AQI
#     if concentration > breakpoints[pollutant][-1][1]:
#         return 500
    
#     return None

# def compute_aqi_from_pollutants(row):
#     scores = []
#     for pol, name in [
#         ("PM2.5", "pm2_5"),
#         ("PM10", "pm10"),
#         ("NO2", "no2"),
#         ("O3", "o3"),
#         ("CO", "co"),
#         ("SO2", "so2")
#     ]:
#         val = row.get(name)
#         if pd.notnull(val):
#             aqi = calculate_aqi(pol, val)
#             if aqi is not None:
#                 scores.append(aqi)
#     return max(scores) if scores else None

# def add_features(df):
#     df = df.sort_values("datetime").reset_index(drop=True)

#     # Compute AQI from pollutants
#     df["computed_aqi"] = df.apply(compute_aqi_from_pollutants, axis=1)

#     # Rolling averages
#     df["aqi_3avg"] = df["aqi_us"].rolling(3).mean()
#     df["pm2_5_3avg"] = df["pm2_5"].rolling(3).mean()
#     df["pm10_3avg"] = df["pm10"].rolling(3).mean()

#     # AQI change rate
#     df["aqi_change_rate"] = df["aqi_us"].pct_change()

#     # Targets: Forecasting AQI for next 3 days
#     df["computed_aqi_t+24"] = df["computed_aqi"].shift(-24)
#     df["computed_aqi_t+48"] = df["computed_aqi"].shift(-48)
#     df["computed_aqi_t+72"] = df["computed_aqi"].shift(-72)

#     # Drop incomplete rows
#     df = df.dropna().reset_index(drop=True)

#     return df

# if __name__ == "__main__":
#     df = pd.read_csv("data/processed/features.csv", parse_dates=["datetime"])
#     df = add_features(df)

#     # Save final CSV
#     os.makedirs("data/processed", exist_ok=True)
#     df.to_csv("data/processed/final_features.csv", index=False)
#     print(f"✅ Saved final_features.csv with {len(df)} rows and {df.shape[1]} columns.")

#     # Calculate MAE between aqi_us and computed_aqi
#     mae = mean_absolute_error(df["aqi_us"], df["computed_aqi"])
#     print(f"📊 MAE between AQI_US and computed_aqi: {mae:.2f}")
    
# import pandas as pd
# import numpy as np
# import os
# from sklearn.metrics import mean_absolute_error

# # EPA AQI calculation
# def calculate_aqi(pollutant, concentration):
#     breakpoints = {
#         "PM2.5": [
#             [0.0, 12.0, 0, 50],
#             [12.1, 35.4, 51, 100],
#             [35.5, 55.4, 101, 150],
#             [55.5, 150.4, 151, 200],
#             [150.5, 250.4, 201, 300],
#             [250.5, 350.4, 301, 400],
#             [350.5, 500.4, 401, 500]
#         ]
#     }

#     if concentration is None or pd.isna(concentration) or concentration < 0:
#         return None

#     for bp in breakpoints.get(pollutant, []):
#         if bp[0] <= concentration <= bp[1]:
#             return round(((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2])

#     return 500 if concentration > breakpoints[pollutant][-1][1] else None

# # NowCast method for real-time PM2.5 AQI
# def calculate_nowcast_pm25(hourly_values):
#     if len(hourly_values) < 3:
#         return None
#     values = hourly_values[-12:] if len(hourly_values) >= 12 else hourly_values[-3:]
#     values = [v for v in values if pd.notnull(v)]
#     if len(values) < 3:
#         return None
#     c_min = min(values)
#     c_max = max(values)
#     if c_max == 0:
#         return 0
#     weight_factor = min(max(1 - (c_max - c_min) / c_max, 0.5), 1)
#     numerator = sum(value * (weight_factor ** i) for i, value in enumerate(reversed(values)))
#     denominator = sum((weight_factor ** i) for i in range(len(values)))
#     return numerator / denominator if denominator > 0 else None

# # Main feature creation function
# def add_features(df):
#     df = df.sort_values("datetime").reset_index(drop=True)

#     # Add NowCast PM2.5
#     df["pm2_5_nowcast"] = None
#     for i in range(len(df)):
#         if i >= 2:
#             vals = df.loc[max(0, i - 11):i, "pm2_5"].tolist()
#             df.loc[i, "pm2_5_nowcast"] = calculate_nowcast_pm25(vals)

#     # Calculate computed_aqi using nowcast PM2.5
#     df["computed_aqi"] = df["pm2_5_nowcast"].apply(lambda x: calculate_aqi("PM2.5", x) if pd.notnull(x) else None)

#     # Rolling & temporal features
#     df["aqi_3avg"] = df["computed_aqi"].rolling(3).mean()
#     df["pm2_5_3avg"] = df["pm2_5"].rolling(3).mean()
#     df["pm10_3avg"] = df["pm10"].rolling(3).mean()
#     df["aqi_change_rate"] = df["computed_aqi"].pct_change()

#     # # Lag features (past values of AQI)
#     # for i in range(1, 4):
#     #     df[f"AQI_t-{i}"] = df["computed_aqi"].shift(i)

#     # Targets: Forecast AQI for next 3 days (hourly resolution: 24, 48, 72 hours ahead)
#     df["computed_aqi_t+24"] = df["computed_aqi"].shift(-24)
#     df["computed_aqi_t+48"] = df["computed_aqi"].shift(-48)
#     df["computed_aqi_t+72"] = df["computed_aqi"].shift(-72)

#     df = df.dropna().reset_index(drop=True)
#     return df

# # Main run block
# if __name__ == "__main__":
#     df = pd.read_csv("data/processed/features.csv", parse_dates=["datetime"])
#     df = add_features(df)

#     # Save enhanced dataset
#     os.makedirs("data/processed", exist_ok=True)
#     df.to_csv("data/processed/final_features.csv", index=False)
#     print(f"✅ Saved final_features.csv with {len(df)} rows and {df.shape[1]} columns.")

#     # ✅ Compare computed AQI with aqi_us using MAE
#     if "aqi_us" in df.columns and "computed_aqi" in df.columns:
#         mask = df["computed_aqi"].notnull() & df["aqi_us"].notnull()
#         mae = mean_absolute_error(df.loc[mask, "aqi_us"], df.loc[mask, "computed_aqi"])
#         print(f"📊 Mean Absolute Error between aqi_us and computed_aqi: {mae:.2f}")
