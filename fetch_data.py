import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = "/mnt/c/users/dell/aqi_predictor1" 
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))  

IQ_AIR = os.getenv("IQAIR_API_KEY")
OPENWEATHER_API = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 24.8607, 67.0011 

def fetch_airvisual():
    url = f"http://api.airvisual.com/v2/city?city=Karachi&state=Sindh&country=Pakistan&key={IQ_AIR}"
    r = requests.get(url)
    return r.json()

def fetch_openweather():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_API}"
    return requests.get(url).json()

def save_json(data, prefix):
    now = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")  
    os.makedirs(raw_dir, exist_ok=True)
    filename = os.path.join(raw_dir, f"{prefix}_{now}.json")  
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {prefix} data to {filename}")

if __name__ == "__main__":
    iq_data = fetch_airvisual()
    openweather_data = fetch_openweather()
    save_json(iq_data, "iqair")
    save_json(openweather_data, "openweather")