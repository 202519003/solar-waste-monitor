import sys
import os
sys.path.append(os.path.dirname(**file**))

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.logic.fetcher    import get_solar_potential, get_fire_hotspots, load_cea_data
from src.logic.calculator import calculate_curtailment, calculate_losses, calculate_curtailment_percent
from src.logic.classifier import prepare_features, train_model, predict_risk, get_risk_color

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
page_title='SolarWaste Monitor',
page_icon='☀️',
layout='wide',
initial_sidebar_state='expanded',
)

# ── Region metadata ───────────────────────────────────────────────────────────

REGIONS = {
'North India (NR)': {'code': 'NR',  'state': 'Rajasthan',  'lat': 26.91, 'lon': 74.22, 'capacity_mw': 35000},
'West India (WR)':  {'code': 'WR',  'state': 'Gujarat',    'lat': 23.02, 'lon': 72.57, 'capacity_mw': 22000},
'South India (SR)': {'code': 'SR',  'state': 'Tamil Nadu', 'lat': 11.12, 'lon': 78.66, 'capacity_mw': 32000},
'East India (ER)':  {'code': 'ER',  'state': 'Odisha',     'lat': 20.95, 'lon': 85.09, 'capacity_mw': 8000},
'NE India (NER)':   {'code': 'NER', 'state': 'Assam',      'lat': 26.20, 'lon': 92.93, 'capacity_mw': 1000},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
st.title("☀️ SolarWaste Monitor")
selected_region = st.selectbox('Select Region', list(REGIONS.keys()))
selected_year   = st.slider('Year', 2022, 2025, 2025)
fire_days       = st.selectbox('Fire data window', [1, 3, 7, 14], index=2)
run_btn = st.button('Run Analysis')

# ── Before button click ───────────────────────────────────────────────────────

if not run_btn:
m = folium.Map(location=[22, 80], zoom_start=5)
st_folium(m, width=None, height=500)
st.stop()

# ── Run Analysis ──────────────────────────────────────────────────────────────

region_info = REGIONS[selected_region]
lat         = region_info['lat']
lon         = region_info['lon']
capacity_mw = region_info['capacity_mw']
region_code = region_info['code']
state_name  = region_info['state']

# ── Fetch Data ────────────────────────────────────────────────────────────────

ghi_data = get_solar_potential(lat, lon, selected_year)
avg_ghi  = round(np.mean(list(ghi_data.values())), 3) if ghi_data else 5.0

cea_df    = load_cea_data('data/india_generation_clean.csv')
region_df = cea_df[
(cea_df['region'] == region_code) &
(cea_df['date'].dt.year == selected_year)
]
actual_mwh_day = region_df['solar_mwh'].mean() if len(region_df) > 0 else 0.0

fires_df = get_fire_hotspots(days=fire_days)

# ── Correct Calculations ──────────────────────────────────────────────────────

potential_mwh_day, wasted_mwh_day = calculate_curtailment(
avg_ghi, capacity_mw, actual_mwh_day
)

losses = calculate_losses(wasted_mwh_day)

curtail_pct = calculate_curtailment_percent(
wasted_mwh_day, potential_mwh_day
)

# ── ML Risk ───────────────────────────────────────────────────────────────────

full_df         = load_cea_data('data/india_generation_clean.csv')
features        = prepare_features(full_df)
clf, scaler, labelled = train_model(features)
row             = labelled[labelled['region'] == region_code]
solar_share     = float(row['solar_share_pct'].values[0]) if len(row) > 0 else 20.0
coal_share      = float(row['coal_share_pct'].values[0])  if len(row) > 0 else 50.0
risk_label      = predict_risk(clf, scaler, curtail_pct, solar_share, coal_share)
risk_color      = get_risk_color(risk_label)

# ── Metrics ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Avg GHI',          f'{avg_ghi} kWh/m²/day')
c2.metric('Curtailed Energy', f'{losses["wasted_kwh"]:,.0f} kWh/day')
c3.metric('Revenue Loss',     f'₹{losses["money_rs"]:,.0f}/day')
c4.metric('CO₂ Emissions',    f'{losses["co2_kg"]:,.0f} kg/day')
c5.metric('Curtailment',      f'{curtail_pct:.1f}%')

st.markdown(f"### ML Risk Classification: {risk_label} Risk")

# ── Map ───────────────────────────────────────────────────────────────────────

m = folium.Map(location=[22, 80], zoom_start=5)

for rname, info in REGIONS.items():
is_sel = (info['code'] == region_code)
folium.CircleMarker(
location=[info['lat'], info['lon']],
radius=15 if is_sel else 8,
color=risk_color if is_sel else 'blue',
fill=True,
fill_color=risk_color if is_sel else 'blue',
fill_opacity=0.7,
tooltip=f"{info['state']} - {info['capacity_mw']} MW"
).add_to(m)

if len(fires_df) > 0:
for _, row in fires_df.iterrows():
folium.CircleMarker(
location=[row['latitude'], row['longitude']],
radius=2,
color='red',
fill=True,
fill_color='red',
fill_opacity=0.6
).add_to(m)

st_folium(m, width=None, height=500)

# ── Summary Table ─────────────────────────────────────────────────────────────

st.subheader("Summary")
st.dataframe(pd.DataFrame({
'Region':             [selected_region],
'State':              [state_name],
'Year':               [selected_year],
'Capacity (MW)':      [capacity_mw],
'GHI (kWh/m²/day)':   [avg_ghi],
'Potential (MWh/day)':[potential_mwh_day],
'Actual (MWh/day)':   [actual_mwh_day],
'Curtailed (kWh/day)':[losses["wasted_kwh"]],
'Revenue Loss (₹)':   [losses["money_rs"]],
'CO2 (kg)':           [losses["co2_kg"]],
'Curtailment %':      [curtail_pct],
'Risk':               [risk_label],
}))
