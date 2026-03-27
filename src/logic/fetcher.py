"""
fetcher.py  —  Data Acquisition Layer
======================================
Responsibilities (Yash Daslaniya):
  1. load_generation()   → loads india_generation.csv into a DataFrame
  2. fetch_ghi()         → calls NASA POWER API for monthly GHI per state
  3. load_fire_data()    → loads NASA FIRMS local CSV (fire hotspots)
  4. get_installed_mw()  → MNRE installed capacity lookup table per state
"""

import os
import pandas as pd
import requests
from geopy.geocoders import Nominatim


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# MNRE installed solar capacity (MW) per state — 2025 figures
INSTALLED_CAPACITY_MW = {
    "Rajasthan":        18700,
    "Gujarat":          10600,
    "Karnataka":         9800,
    "Tamil Nadu":        7800,
    "Andhra Pradesh":    6500,
    "Telangana":         4600,
    "Maharashtra":       3600,
    "Madhya Pradesh":    3200,
    "Uttar Pradesh":     2800,
    "Punjab":            1200,
    "Haryana":            800,
    "Odisha":             700,
    "Chhattisgarh":       600,
    "Kerala":             200,
    "Bihar":              180,
    "West Bengal":        150,
    "Assam":              120,
    "Himachal Pradesh":    80,
    "Uttarakhand":         70,
    "Jharkhand":           60,
    "Goa":                 30,
    "Manipur":             15,
    "Meghalaya":           10,
    "Tripura":              8,
    "Nagaland":             5,
    "Mizoram":              4,
    "Arunachal Pradesh":    3,
    "Sikkim":               2,
}

# GeoJSON name mismatches → fix to match CSV names
GEOJSON_TO_CSV = {
    "Orissa":      "Odisha",
    "Uttaranchal": "Uttarakhand",
}

# Geocode cache — avoids repeated API calls
_geocache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CEA Generation Data
# ─────────────────────────────────────────────────────────────────────────────

def load_generation(path: str = None) -> pd.DataFrame:
    """
    Loads india_generation.csv.

    Returns DataFrame with columns:
        State (str), Month (int 1-12), Year (int), Generation_MW (float)
    """
    if path is None:
        path = os.path.join(DATA_DIR, "india_generation.csv")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Year"]  = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Generation_MW"] = pd.to_numeric(df["Generation_MW"], errors="coerce").fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. NASA POWER API  —  Monthly GHI
# ─────────────────────────────────────────────────────────────────────────────

def _get_lat_lon(state_name: str):
    """Returns (lat, lon) for a given Indian state using Nominatim geocoder."""
    if state_name in _geocache:
        return _geocache[state_name]

    geolocator = Nominatim(user_agent="solarwaste_monitor_msc2026")
    try:
        loc = geolocator.geocode(f"{state_name}, India", timeout=10)
        if loc:
            _geocache[state_name] = (loc.latitude, loc.longitude)
            return loc.latitude, loc.longitude
    except Exception as e:
        print(f"[fetcher] Geocoding error for {state_name}: {e}")
    return None, None


def fetch_ghi(state_name: str, year: int) -> dict:
    """
    Fetches monthly Global Horizontal Irradiance (GHI) from NASA POWER API.

    Parameters
    ----------
    state_name : str   e.g. "Rajasthan"
    year       : int   e.g. 2024

    Returns
    -------
    dict  {month_int: GHI_kWh_per_m2_per_day}
    e.g.  {1: 5.2, 2: 5.8, 3: 6.1, ...}
    Returns empty dict on failure.
    """
    lat, lon = _get_lat_lon(state_name)
    if lat is None:
        print(f"[fetcher] Could not geocode: {state_name}")
        return {}

    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",  # GHI in kWh/m²/day
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      str(year),
        "end":        str(year),
        "format":     "JSON",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        monthly_raw = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        result = {}
        for k, v in monthly_raw.items():
            if k.endswith("13"):      # "13" = annual average key — skip
                continue
            month = int(k[4:])        # "202304" → 4
            result[month] = v if v != -999 else None   # -999 = missing data
        return result
    except Exception as e:
        print(f"[fetcher] NASA POWER API error for {state_name} {year}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. NASA FIRMS  —  Fire Hotspot Data
# ─────────────────────────────────────────────────────────────────────────────

def load_fire_data(path: str = None, year: int = None) -> pd.DataFrame:
    """
    Loads NASA FIRMS VIIRS fire archive CSV.

    Parameters
    ----------
    path : str   Full path to fire_archive_SV-C2_*.csv
                 If None, auto-detects any matching file inside data/
    year : int   Optional — filter to one year only

    Returns
    -------
    DataFrame with columns: latitude, longitude, acq_date, frp, confidence
    Returns empty DataFrame if file not found (app still runs without it).
    """
    if path is None:
        for fname in os.listdir(DATA_DIR):
            if fname.startswith("fire_archive") and fname.endswith(".csv"):
                path = os.path.join(DATA_DIR, fname)
                break

    if path is None or not os.path.exists(str(path)):
        print("[fetcher] WARNING: FIRMS fire CSV not found.")
        print("          Place fire_archive_SV-C2_*.csv inside the data/ folder.")
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "frp", "confidence"])

    df = pd.read_csv(path, parse_dates=["acq_date"])

    if year is not None:
        df = df[df["acq_date"].dt.year == year]

    # Drop low-confidence fire detections (noise reduction)
    conf_map = {"l": 0, "n": 1, "h": 2}
    df["conf_level"] = df["confidence"].map(conf_map).fillna(0)
    df = df[df["conf_level"] >= 1]

    return df[["latitude", "longitude", "acq_date", "frp", "confidence"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Installed Capacity Lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_installed_mw(state_name: str) -> float:
    """Returns MNRE installed solar capacity (MW) for a state. Returns 0 if unknown."""
    return INSTALLED_CAPACITY_MW.get(state_name, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing fetcher.py ===\n")

    gen = load_generation()
    print(f"[1] Generation CSV loaded: {gen.shape[0]} rows, {gen['State'].nunique()} states")
    print(gen[gen["State"] == "Rajasthan"].head(3).to_string(index=False))

    print("\n[2] Fetching GHI for Rajasthan 2024 from NASA POWER ...")
    ghi = fetch_ghi("Rajasthan", 2024)
    print("    GHI result:", ghi)

    print("\n[3] Loading fire data ...")
    fires = load_fire_data(year=2024)
    print(f"    Rows loaded: {len(fires)}")

    print("\n[4] Installed capacity Rajasthan:", get_installed_mw("Rajasthan"), "MW")
