"""
data_cleaner.py — SolarWaste Monitor
Run this ONCE to clean india_generation.csv before using it in the app.
Saves cleaned file as: data/india_generation_clean.csv

Issues fixed:
1. Date format — YYYYMMDD integer → proper datetime
2. Missing solar values — 1365 rows missing (all pre-2016, not needed)
3. Negative solar values — 7 rows in 2017, replaced with 0
4. Column structure — reshape from wide (regions) to long (one row per region per day)
5. Units — MU (GWh) converted to MWh for easier calculation
6. Keep only 2020 onwards — solar data before 2020 is sparse and less relevant
"""

import pandas as pd
import os

def clean_cea_data(input_path='data/india_generation.csv',
                   output_path='data/india_generation_clean.csv'):

    print('Loading raw CSV...')
    df = pd.read_csv(input_path)
    print(f'Raw shape: {df.shape}')

    # ── FIX 1: Parse date ───────────────────────────────────────────────────
    df['date'] = pd.to_datetime(df['yyyymmdd'], format='%Y%m%d')
    df = df.drop(columns=['yyyymmdd'])
    print('Date parsed.')

    # ── FIX 2: Keep only 2020 onwards ───────────────────────────────────────
    # Solar data before 2020 is incomplete (1365 missing rows are all pre-2016)
    df = df[df['date'] >= '2020-01-01'].copy()
    print(f'After filtering to 2020+: {len(df)} rows')

    # ── FIX 3: Extract region-level solar generation columns ────────────────
    # Map: region code → readable name → approximate main state
    region_map = {
        'NR': 'North India',   # Rajasthan, Punjab, Haryana, UP, Delhi
        'WR': 'West India',    # Gujarat, Maharashtra, MP
        'SR': 'South India',   # Tamil Nadu, Karnataka, Andhra Pradesh, Telangana
        'ER': 'East India',    # Odisha, West Bengal, Jharkhand
        'NER': 'NE India',     # Assam and NE states
    }

    # Representative state coords for NASA POWER API calls
    region_coords = {
        'NR':  {'state': 'Rajasthan',        'lat': 26.91, 'lon': 74.22},
        'WR':  {'state': 'Gujarat',           'lat': 23.02, 'lon': 72.57},
        'SR':  {'state': 'Tamil Nadu',        'lat': 11.12, 'lon': 78.66},
        'ER':  {'state': 'Odisha',            'lat': 20.95, 'lon': 85.09},
        'NER': {'state': 'Assam',             'lat': 26.20, 'lon': 92.93},
    }

    rows = []
    for region_code, region_name in region_map.items():
        solar_col   = f'{region_code}: SolarGen'
        demand_col  = f'{region_code}: EnergyMet'
        coal_col    = f'{region_code}: Coal'

        if solar_col not in df.columns:
            continue

        temp = pd.DataFrame()
        temp['date']        = df['date'].values
        temp['region']      = region_code
        temp['region_name'] = region_name
        temp['state']       = region_coords[region_code]['state']
        temp['lat']         = region_coords[region_code]['lat']
        temp['lon']         = region_coords[region_code]['lon']

        # Solar generation — MU (GWh) → MWh
        solar_vals = pd.to_numeric(df[solar_col], errors='coerce') * 1000
        temp['solar_mwh']   = solar_vals.clip(lower=0).values

        # Demand — MU → MWh
        temp['demand_mwh']  = df[demand_col].values * 1000 if demand_col in df.columns else None

        # Coal — MU → MWh
        temp['coal_mwh']    = df[coal_col].values * 1000 if coal_col in df.columns else None

        rows.append(temp)

    clean_df = pd.concat(rows, ignore_index=True)

    # ── FIX 4: Handle remaining missing values ───────────────────────────────
    before = clean_df['solar_mwh'].isnull().sum()
    clean_df['solar_mwh']  = clean_df['solar_mwh'].fillna(0)
    clean_df['demand_mwh'] = clean_df['demand_mwh'].fillna(0)
    clean_df['coal_mwh']   = clean_df['coal_mwh'].fillna(0)
    print(f'Filled {before} missing solar values with 0')

    # ── FIX 5: Sort by date ──────────────────────────────────────────────────
    clean_df = clean_df.sort_values(['date', 'region']).reset_index(drop=True)

    # ── SAVE ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)

    print(f'\nCleaned shape: {clean_df.shape}')
    print(f'Regions: {clean_df["region"].unique()}')
    print(f'Date range: {clean_df["date"].min()} to {clean_df["date"].max()}')
    print(f'\nSample output:')
    print(clean_df.head(5).to_string())
    print(f'\nSaved to: {output_path}')
    return clean_df


if __name__ == '__main__':
    clean_cea_data()
