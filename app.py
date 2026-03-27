import sys
import os
sys.path.append(os.path.dirname(__file__))

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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b8fa8 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #13151f;
        border-right: 1px solid #2a2d3a;
    }
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .main-subtitle {
        font-size: 14px;
        color: #8b8fa8;
        margin-bottom: 24px;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
    }
    .risk-high   { background:#3d1515; color:#e24b4a; border:1px solid #e24b4a; }
    .risk-medium { background:#3d2e0a; color:#ef9f27; border:1px solid #ef9f27; }
    .risk-low    { background:#0a2e1e; color:#1d9e75; border:1px solid #1d9e75; }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #8b8fa8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px;
        border-bottom: 1px solid #2a2d3a;
        padding-bottom: 8px;
    }
    .info-box {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-left: 3px solid #378add;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 13px;
        color: #8b8fa8;
        margin: 12px 0;
    }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Region metadata ───────────────────────────────────────────────────────────
# capacity_mw = installed solar capacity per region (CEA, March 2025 estimates)
# Formula used: potential_MWh/day = GHI × capacity_mw  (GHI already in kWh/m²/day)
REGIONS = {
    'North India (NR)': {'code': 'NR',  'state': 'Rajasthan',  'lat': 26.91, 'lon': 74.22, 'capacity_mw': 35000},
    'West India (WR)':  {'code': 'WR',  'state': 'Gujarat',    'lat': 23.02, 'lon': 72.57, 'capacity_mw': 22000},
    'South India (SR)': {'code': 'SR',  'state': 'Tamil Nadu', 'lat': 11.12, 'lon': 78.66, 'capacity_mw': 32000},
    'East India (ER)':  {'code': 'ER',  'state': 'Odisha',     'lat': 20.95, 'lon': 85.09, 'capacity_mw': 8000},
    'NE India (NER)':   {'code': 'NER', 'state': 'Assam',      'lat': 26.20, 'lon': 92.93, 'capacity_mw': 1000},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ☀️ SolarWaste Monitor")
    st.markdown("---")
    selected_region = st.selectbox('Select Region', list(REGIONS.keys()))
    selected_year   = st.slider('Year', 2022, 2025, 2025)
    fire_days       = st.selectbox('Fire data window', [1, 3, 7, 14], index=2,
                                   format_func=lambda x: f'Last {x} days')
    run_btn = st.button('Run Analysis', use_container_width=True, type='primary')
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px;color:#555870;'>
    <b style='color:#8b8fa8;'>Data sources</b><br><br>
    CEA / GRID-INDIA — actual generation<br>
    NASA POWER — solar potential (GHI)<br>
    NASA FIRMS — fire hotspots<br><br>
    <b style='color:#8b8fa8;'>Formula</b><br><br>
    Potential = GHI × Capacity (MWh/day)<br>
    Curtailed = Potential − Actual (min 0)<br>
    CO₂ = Wasted kWh × 0.727 kg
    </div>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">☀️ SolarWaste Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">India Solar Curtailment Tracker — Real-time wasted energy, CO₂ loss & fire risk</div>', unsafe_allow_html=True)

# ── Before button click ───────────────────────────────────────────────────────
if not run_btn:
    st.markdown(
        '<div class="info-box">Select a region and year in the sidebar, '
        'then click <b>Run Analysis</b> to begin.</div>',
        unsafe_allow_html=True,
    )
    m = folium.Map(location=[22, 80], zoom_start=5, tiles='CartoDB dark_matter')
    st_folium(m, width=None, height=480, returned_objects=[])
    st.stop()

# ── Run Analysis ──────────────────────────────────────────────────────────────
region_info = REGIONS[selected_region]
lat         = region_info['lat']
lon         = region_info['lon']
capacity_mw = region_info['capacity_mw']
region_code = region_info['code']
state_name  = region_info['state']

with st.spinner(f'Fetching data for {selected_region}...'):

    # ── NASA POWER — monthly GHI ──────────────────────────────────────────────
    try:
        ghi_data = get_solar_potential(lat, lon, selected_year)
        avg_ghi  = round(np.mean(list(ghi_data.values())), 3) if ghi_data else 5.0
    except Exception as e:
        st.warning(f'NASA POWER error: {e}')
        ghi_data, avg_ghi = {}, 5.0

    # ── CEA data — actual generation ──────────────────────────────────────────
    try:
        cea_df    = load_cea_data('data/india_generation_clean.csv')
        region_df = cea_df[
            (cea_df['region'] == region_code) &
            (cea_df['date'].dt.year == selected_year)
        ]
        # solar_mwh is already MWh per day — just take the mean across days
        actual_mwh_day = region_df['solar_mwh'].mean() if len(region_df) > 0 else 0.0
    except Exception as e:
        st.warning(f'CEA data error: {e}')
        actual_mwh_day = 0.0

    # ── NASA FIRMS — fire hotspots ────────────────────────────────────────────
    try:
        fires_df = get_fire_hotspots(days=fire_days)
    except Exception as e:
        st.warning(f'FIRMS error: {e}')
        fires_df = pd.DataFrame(columns=['latitude', 'longitude', 'brightness', 'confidence'])

    # ── Calculations (all in MWh/day — no unit conversion errors) ────────────
    potential_mwh_day = avg_ghi * capacity_mw                       # MWh/day
    wasted_mwh_day    = calculate_curtailment(avg_ghi, capacity_mw, actual_mwh_day)
    losses            = calculate_losses(wasted_mwh_day)            # kWh, Rs, kg CO2
    curtail_pct       = calculate_curtailment_percent(potential_mwh_day, actual_mwh_day)

    # ── ML classifier ─────────────────────────────────────────────────────────
    try:
        full_df         = load_cea_data('data/india_generation_clean.csv')
        features        = prepare_features(full_df)
        clf, scaler, labelled = train_model(features)
        row             = labelled[labelled['region'] == region_code]
        solar_share     = float(row['solar_share_pct'].values[0]) if len(row) > 0 else 20.0
        coal_share      = float(row['coal_share_pct'].values[0])  if len(row) > 0 else 50.0
        risk_label      = predict_risk(clf, scaler, curtail_pct, solar_share, coal_share)
    except Exception as e:
        st.warning(f'ML error: {e}')
        risk_label = 'Medium'

    risk_color = get_risk_color(risk_label)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Avg GHI',       f'{avg_ghi} kWh/m²/d')
c2.metric('Wasted Energy', f'{losses["wasted_kwh"]:,.0f} kWh/day')
c3.metric('Money Lost',    f'₹{losses["money_rs"]:,.0f}/day')
c4.metric('CO₂ Released',  f'{losses["co2_kg"]:,.0f} kg/day')
c5.metric('Curtailment',   f'{curtail_pct:.1f}%')

risk_class = f'risk-{risk_label.lower()}'
st.markdown(
    f'ML Risk Classification: <span class="risk-badge {risk_class}">{risk_label} Risk</span>',
    unsafe_allow_html=True,
)

# ── Map ───────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Interactive Map — Fire Hotspots & Solar Regions</div>',
    unsafe_allow_html=True,
)

m = folium.Map(location=[22, 80], zoom_start=5, tiles='CartoDB dark_matter')

for rname, info in REGIONS.items():
    is_sel = (info['code'] == region_code)
    folium.CircleMarker(
        location     = [info['lat'], info['lon']],
        radius       = 18 if is_sel else 10,
        color        = risk_color if is_sel else '#378add',
        fill         = True,
        fill_color   = risk_color if is_sel else '#185fa5',
        fill_opacity = 0.8 if is_sel else 0.4,
        tooltip=folium.Tooltip(
            f"<b>{info['state']}</b><br>Region: {info['code']}<br>"
            f"Capacity: {info['capacity_mw']:,} MW"
            + (f"<br><b>Risk: {risk_label}</b>" if is_sel else "")
        ),
    ).add_to(m)

folium.Marker(
    location=[lat + 1.5, lon],
    icon=folium.DivIcon(
        html=(
            f'<div style="font-size:12px;font-weight:700;color:{risk_color};'
            f'background:#0f1117;padding:3px 8px;border-radius:4px;'
            f'border:1px solid {risk_color};white-space:nowrap;">'
            f'{state_name} — {risk_label} Risk</div>'
        ),
        icon_size=(180, 30),
        icon_anchor=(90, 0),
    ),
).add_to(m)

if len(fires_df) > 0:
    for _, row in fires_df.iterrows():
        try:
            conf = str(row.get('confidence', 'n')).lower()
            if conf in ['high', 'h', 'nominal', 'n']:
                folium.CircleMarker(
                    location     = [float(row['latitude']), float(row['longitude'])],
                    radius       = 2,
                    color        = '#ff6b35',
                    fill         = True,
                    fill_color   = '#ff4500',
                    fill_opacity = 0.7,
                    tooltip      = 'Fire hotspot',
                ).add_to(m)
        except Exception:
            continue

legend_html = f"""
<div style='position:fixed;bottom:30px;left:30px;z-index:1000;
     background:#0f1117;border:1px solid #2a2d3a;border-radius:10px;
     padding:12px 16px;font-size:12px;color:#c0c3d4;'>
  <b style='color:#fff;'>Legend</b><br><br>
  <span style='color:{risk_color};'>●</span> {state_name} ({risk_label} Risk)<br>
  <span style='color:#378add;'>●</span> Other regions<br>
  <span style='color:#ff6b35;'>●</span> Fire hotspots
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))
st_folium(m, width=None, height=520, returned_objects=[])

# ── GHI Monthly Chart ─────────────────────────────────────────────────────────
if ghi_data:
    st.markdown(
        '<div class="section-header">Monthly Solar Potential (GHI) — NASA POWER</div>',
        unsafe_allow_html=True,
    )
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    months = sorted(ghi_data.keys())
    values = [ghi_data[m] for m in months]
    xlbls  = [month_labels[int(m[4:6]) - 1] for m in months]

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')
    bars = ax.bar(xlbls, values, color='#378add', alpha=0.7, width=0.6, edgecolor='none')
    bars[values.index(max(values))].set_color('#EF9F27')
    ax.set_ylabel('kWh/m²/day', color='#8b8fa8', fontsize=10)
    ax.tick_params(colors='#8b8fa8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#2a2d3a')
    ax.axhline(np.mean(values), color='#1d9e75', linestyle='--', alpha=0.6, linewidth=1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Summary Table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
st.dataframe(
    pd.DataFrame({
        'Region':             [selected_region],
        'State':              [state_name],
        'Year':               [selected_year],
        'Capacity (MW)':      [f'{capacity_mw:,}'],
        'GHI (kWh/m²/d)':    [avg_ghi],
        'Potential (MWh/d)':  [f'{potential_mwh_day:,.0f}'],
        'Actual (MWh/d)':     [f'{actual_mwh_day:,.0f}'],
        'Wasted (kWh/day)':   [f'{losses["wasted_kwh"]:,.0f}'],
        'Money Lost (₹/day)': [f'{losses["money_rs"]:,.0f}'],
        'CO₂ (kg/day)':       [f'{losses["co2_kg"]:,.0f}'],
        'Curtailment %':      [f'{curtail_pct:.1f}%'],
        'Risk':               [risk_label],
    }),
    use_container_width=True,
    hide_index=True,
)

# ── Fire Hotspot Stats ────────────────────────────────────────────────────────
if len(fires_df) > 0:
    st.markdown('<div class="section-header">Fire Hotspot Summary</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    f1.metric('Total Hotspots', f'{len(fires_df):,}')
    high_conf = fires_df[
        fires_df['confidence'].astype(str).str.lower().isin(['high', 'h'])
    ].shape[0]
    f2.metric('High Confidence', f'{high_conf:,}')
    if 'brightness' in fires_df.columns:
        f3.metric('Avg Brightness (K)', f'{fires_df["brightness"].mean():.1f}')
