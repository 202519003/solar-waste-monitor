import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key safely
MAP_KEY = os.getenv("NASA_FIRMS_KEY")

if not MAP_KEY:
    raise ValueError("❌ NASA_FIRMS_KEY not found in .env file")


# -------------------------------
# NASA POWER API (Solar Data)
# -------------------------------
def get_solar_potential(lat, lon, year):
    url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&start={year}&end={year}&format=JSON"

    response = requests.get(url)
    data = response.json()

    try:
        return data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    except Exception:
        print("⚠️ Error fetching solar data")
        return {}


# -------------------------------
# NASA FIRMS API (Fire Data)
# -------------------------------
def get_fire_hotspots(days=1):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/world/{days}"

    try:
        df = pd.read_csv(url)

        # Keep important columns only
        df = df[["latitude", "longitude", "brightness", "confidence"]]

        return df

    except Exception as e:
        print("⚠️ Error fetching fire data:", e)
        return pd.DataFrame()
