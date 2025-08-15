import hopsworks
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def upload_features():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    # Load final engineered data
    url = "https://raw.githubusercontent.com/maheenrizwan11/AQI-Predictor/refs/heads/main/data/processed/final_features.csv"
    df = pd.read_csv(url)
    df["datetime"] = pd.to_datetime(df["datetime"])

    df.rename(columns={
        "computed_aqi_t+24": "computed_aqi_t24",
        "computed_aqi_t+48": "computed_aqi_t48",
        "computed_aqi_t+72": "computed_aqi_t72"
    }, inplace=True)

    df["record_id"] = df["datetime"].dt.strftime("%Y%m%d%H%M")
    df.columns = [col.lower().replace("-", "_") for col in df.columns]

    float_cols = df.select_dtypes("float").columns
    df[float_cols] = df[float_cols].round(4)

    feature_group = fs.get_or_create_feature_group(
        name="karachi_aqi_features",
        version=5,
        primary_key=["record_id"],
        description="Engineered AQI features for Karachi",
        online_enabled=True
    )

    feature_group.insert(df)
    print("✅ Uploaded final_features.csv to Hopsworks")

if __name__ == "__main__":
    upload_features()
