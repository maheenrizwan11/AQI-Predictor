# import os
# import requests
# import json
# from datetime import datetime
# from dotenv import load_dotenv

# # Load keys
# load_dotenv()
# IQ_AIR = os.getenv("IQAIR_API_KEY")
# AQICN_TOKEN = os.getenv("AQICN_API_TOKEN")

# LAT, LON = 24.8607, 67.0011 

# def fetch_aqicn():
#     url = f"https://api.waqi.info/feed/karachi/?token={AQICN_TOKEN}"
#     return requests.get(url).json()

# def fetch_airvisual():
#     url = f"http://api.airvisual.com/v2/city?city=Karachi&state=Sindh&country=Pakistan&key={IQ_AIR}"
#     r = requests.get(url)
#     return r.json()

# def save_json(data, prefix):
#     now = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
#     filename = f"data/raw/{prefix}_{now}.json"
#     os.makedirs("data/raw", exist_ok=True)
#     with open(filename, "w") as f:
#         json.dump(data, f, indent=2)
#     print(f"Saved {prefix} data to {filename}")

# if __name__ == "__main__":
#     aqicn_data = fetch_aqicn()
#     iq_data = fetch_airvisual()

#     save_json(aqicn_data, "aqicn")
#     save_json(iq_data, "iqair")
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# Load keys (absolute path to .env)
PROJECT_ROOT = "/mnt/c/users/dell/aqi_predictor"  # Update this!
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))  # Fix: Specify .env path

IQ_AIR = os.getenv("IQAIR_API_KEY")
# AQICN_TOKEN = os.getenv("AQICN_API_TOKEN")
OPENWEATHER_API = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 24.8607, 67.0011 

# def fetch_aqicn():
#     url = f"https://api.waqi.info/feed/karachi/?token={AQICN_TOKEN}"
#     return requests.get(url).json()

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
    # aqicn_data = fetch_aqicn()
    iq_data = fetch_airvisual()
    openweather_data = fetch_openweather()
    # save_json(aqicn_data, "aqicn")
    save_json(iq_data, "iqair")
    save_json(openweather_data, "openweather")