# import pandas as pd
# df = pd.read_csv("../data/processed/final_features.csv", parse_dates=["datetime"])

# # Features you’ll use to predict
# features = [
#     'pm2_5', 'pm10', 'o3', 'no2', 'co', 'so2',
#     'pressure', 'wind_speed', 'aqi_change_rate', 
#     'temp', 'humidity', 'wind_direction'
# ]

# # Targets for 1-day, 2-day, 3-day forecasts
# targets = {
#     "t+24": "computed_aqi_t+24",
#     "t+48": "computed_aqi_t+48",
#     "t+72": "computed_aqi_t+72"
# }

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import joblib
# import numpy as np

# models = {
#     "LinearRegression": LinearRegression(),
#     "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42),
#     "XGBoost": XGBRegressor(random_state=42),
#     "RidgeRegression": Ridge(alpha=10),
#     "GradientBoost": GradientBoostingRegressor()
# }

# for horizon, target_col in targets.items():
#     print(f"\nTraining model for AQI {horizon} ahead (target: {target_col})")

#     X = df[features]
#     y = df[target_col]

#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )

#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#         mse = mean_squared_error(y_test, preds)
#         rmse = np.sqrt(mse)  # Manually calculate RMSE
#         mae = mean_absolute_error(y_test, preds)
#         r2 = r2_score(y_test, preds)

#         print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

#         # Save best model per horizon
#         # joblib.dump(model, f"models/{name}_{horizon}.pkl")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv("data/processed/final_features.csv", parse_dates=["datetime"])

# Features to use
features = [
    'pm2_5', 'pm10', 'o3', 'no2', 'co', 'so2',
    'pressure', 'wind_speed', 'aqi_change_rate',
    'temp', 'humidity', 'wind_direction'
]

# Targets
targets = {
    "t+24": "computed_aqi_t+24",
    "t+48": "computed_aqi_t+48",
    "t+72": "computed_aqi_t+72"
}

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "RidgeRegression": Ridge(alpha=10),
    "GradientBoost": GradientBoostingRegressor()
}

# Fit scaler on full dataset
scaler = StandardScaler()
X_all = df[features]
X_scaled_all = scaler.fit_transform(X_all)

# # Save scaler for inference
# joblib.dump(scaler, "models/aqi_scaler.pkl")

for horizon, target_col in targets.items():
    print(f"\nTraining model for AQI {horizon} ahead (target: {target_col})")

    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_all, y, test_size=0.2, random_state=42
    )

    best_model = None
    best_score = -np.inf
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

        if r2 > best_score:
            best_model = model
            best_score = r2
            best_name = name

    print(f"✅ Best model for {horizon}: {best_name} (R²={best_score:.2f})")
    
    # Save best model
    joblib.dump(best_model, f"models/{best_name}_{horizon}.pkl")

    # Example prediction using latest row
    latest_row = df[features].iloc[-1:]
    latest_scaled = scaler.transform(latest_row)
    pred_value = best_model.predict(latest_scaled)[0]
    print(f"🔮 Predicted AQI for {horizon}: {pred_value:.2f}")

