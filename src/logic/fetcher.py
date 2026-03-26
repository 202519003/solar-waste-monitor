# fetcher.py — SolarWaste Monitor
import requests
import pandas as pd
import os

# NASA FIRMS API key — load from environment or use default
MAP_KEY = os.environ.get('NASA_FIRMS_KEY', '607ea5aeb4ae9ca8d5bdb4052426d7a5')


def get_solar_potential(lat, lon, year):
    """
    Calls NASA POWER API and returns monthly GHI values for a given location.

    GHI = Global Horizontal Irradiance (kWh/m²/day)
    This is the available solar energy per square metre per day at that location.

    Args:
        lat  : float — latitude of the representative point for the region
        lon  : float — longitude
        year : int   — year to fetch (e.g. 2025)

    Returns:
        dict : {YYYYMM: GHI_value}  e.g. {'202501': 4.05, '202502': 4.97, ...}
        Returns {} on failure (caller should use a fallback value).
    """
    url = 'https://power.larc.nasa.gov/api/temporal/monthly/point'
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN',
        'community':  'RE',
        'longitude':  lon,
        'latitude':   lat,
        'start':      year,
        'end':        year,
        'format':     'JSON',
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        raw  = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
        # Filter: remove -999 (missing) and the 13th entry which is annual average
        ghi = {k: v for k, v in raw.items() if v != -999.0 and len(k) == 6}
        return ghi
    except Exception as e:
        print(f'NASA POWER API error: {e}')
        return {}


def get_fire_hotspots(days=7):
    """
    Calls NASA FIRMS API and returns a DataFrame of fire hotspots across India.
    Data is updated every 3 hours from VIIRS satellite thermal sensors.

    Args:
        days : int — look-back window (1, 3, 7, or 14 days)

    Returns:
        DataFrame with columns: latitude, longitude, brightness, confidence
        Returns empty DataFrame on failure.
    """
    bbox = '68,7,97,37'   # India bounding box: lon_min, lat_min, lon_max, lat_max
    url  = (
        f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/'
        f'{MAP_KEY}/VIIRS_NOAA20_NRT/{bbox}/{days}'
    )
    try:
        df = pd.read_csv(url)
        if df.empty:
            return pd.DataFrame(columns=['latitude', 'longitude', 'brightness', 'confidence'])
        # Keep only the columns we need; ignore others gracefully
        cols = [c for c in ['latitude', 'longitude', 'brightness', 'confidence'] if c in df.columns]
        return df[cols]
    except Exception as e:
        print(f'NASA FIRMS API error: {e}')
        return pd.DataFrame(columns=['latitude', 'longitude', 'brightness', 'confidence'])


def load_cea_data(filepath='data/india_generation_clean.csv'):
    """
    Loads the cleaned CEA solar generation CSV.
    This contains ACTUAL recorded generation per region per day.

    Args:
        filepath : str — path to india_generation_clean.csv

    Returns:
        DataFrame with columns:
            date, region, region_name, state, lat, lon,
            solar_mwh, demand_mwh, coal_mwh

    NOTE:
        solar_mwh  = actual daily solar generation for the region (MWh/day)
        demand_mwh = total energy demand for the region (MWh/day)
        coal_mwh   = coal generation for the region (MWh/day)
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    return df
