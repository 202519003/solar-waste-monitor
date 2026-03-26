import requests
import pandas as pd

# Source: NASA FIRMS - Free Map Key for fire hotspot data
MAP_KEY = '607ea5aeb4ae9ca8d5bdb4052426d7a5'


def get_solar_potential(lat, lon, year):
    """
    Calls NASA POWER API and returns monthly GHI values for a given location.
    GHI = Global Horizontal Irradiance (kWh/m2/day) — how much sunlight hits that location.
    This is the POTENTIAL solar energy — what should have been generated.
    """
    url = 'https://power.larc.nasa.gov/api/temporal/monthly/point'
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN',
        'community': 'RE',
        'longitude': lon,
        'latitude': lat,
        'start': year,
        'end': year,
        'format': 'JSON'
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    raw = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']

    # Filter out -999 (missing/unavailable data) and annual summary key (length 6 = YYYYMM)
    ghi = {k: v for k, v in raw.items() if v != -999.0 and len(k) == 6}
    return ghi
    # Example return: {'202501': 4.05, '202502': 4.97, '202503': 6.12, ...}


def get_fire_hotspots(days=7):
    """
    Calls NASA FIRMS API and returns DataFrame of fire hotspots across India.
    Data is updated every 3 hours from VIIRS satellite thermal sensors.
    India bounding box: min_lon=68, min_lat=7, max_lon=97, max_lat=37
    """
    bbox = '68,7,97,37'  # India bounding box
    url = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_NOAA20_NRT/{bbox}/{days}'

    try:
        df = pd.read_csv(url)
        return df[['latitude', 'longitude', 'brightness', 'confidence']]
    except Exception as e:
        print(f'FIRMS API error: {e}')
        return pd.DataFrame(columns=['latitude', 'longitude', 'brightness', 'confidence'])


def load_cea_data(filepath='data/india_generation.csv'):
    """
    Loads the CEA solar generation CSV downloaded from Robbie Andrew's site.
    This is the ACTUAL recorded generation — what was really produced.
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df
    # Returns DataFrame with columns: date, state, solar_mw, wind_mw, coal_mw, demand_mw