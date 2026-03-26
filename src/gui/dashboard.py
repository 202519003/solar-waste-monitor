import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.logic.fetcher import get_solar_potential, get_fire_hotspots, load_cea_data
from src.logic.calculator import calculate_curtailment, calculate_losses, calculate_curtailment_percent
from src.logic.classifier import prepare_features, train_model, predict_risk, get_risk_color

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='SolarWaste Monitor',
    page_icon='☀️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
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

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #13151f;
        border-right: 1px solid #2a2d3a;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] p {
        color: #c0c3d4 !important;
    }

    /* Title */
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

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
    }
    .risk-high   { background: #3d1515; color: #e24b4a; border: 1px solid #e24b4a; }
    .risk-medium { background: #3d2e0a; color: #ef9f27; border: 1px solid #ef9f27; }
    .risk-low    { background: #0a2e1e; color: #1d9e75; border: 1px solid #1d9e75; }

    /* Section headers */
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

    /* Info box */
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

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Region metadata ───────────────────────────────────────────────────────────
REGIONS = {
    'North India (NR)':  {'code': 'NR', 'state': 'Rajasthan',  'lat': 26.91, 'lon': 74.22, 'capacity_mw': 18000},
    'West India (WR)':   {'code': 'WR', 'state': 'Gujarat',    'lat': 23.02, 'lon': 72.57, 'capacity_mw': 12000},
    'South India (SR)':  {'code': 'SR', 'state': 'Tamil Nadu', 'lat': 11.12, 'lon': 78.66, 'capacity_mw': 16000},
    'East India (ER)':   {'code': 'ER', 'state': 'Odisha',     'lat': 20.95, 'lon': 85.09, 'capacity_mw': 4000},
    'NE India (NER)':    {'code': 'NER','state': 'Assam',      'lat': 26.20, 'lon': 92.93, 'capacity_mw': 500},
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ☀️ SolarWaste Monitor")
    st.markdown("---")

    selected_region = st.selectbox(
        'Select Region',
        list(REGIONS.keys()),
        index=0
    )

    year = st.slider(
    "Select Year",
    min_value=2023,
    max_value=2025,
    value=2025
    )

    fire_days = st.selectbox(
        'Fire data window',
        [1, 3, 7, 14],
        index=2,
        format_func=lambda x: f'Last {x} days'
    )

    run_btn = st.button('Run Analysis', use_container_width=True, type='primary')

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#555870;'>
    <strong style='color:#8b8fa8;'>Data sources</strong><br><br>
    CEA / GRID-INDIA — actual generation<br>
    NASA POWER — solar potential (GHI)<br>
    NASA FIRMS — fire hotspots<br><br>
    <strong style='color:#8b8fa8;'>Formula</strong><br><br>
    Curtailed = GHI × Capacity × 0.15 − Actual<br>
    CO₂ = Wasted kWh × 0.727 kg
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">☀️ SolarWaste Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">India Solar Curtailment Tracker — Real-time wasted energy, CO₂ loss & fire risk</div>', unsafe_allow_html=True)

# ── Default state (before button click) ──────────────────────────────────────
if not run_btn:
    st.markdown('<div class="info-box">Select a region and year in the sidebar, then click <strong>Run Analysis</strong> to begin.</div>', unsafe_allow_html=True)

    # Show empty map of India
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

    # 1. NASA POWER — solar potential
    try:
        ghi_data = get_solar_potential(lat, lon, selected_year)
        avg_ghi  = round(np.mean(list(ghi_data.values())), 3) if ghi_data else 5.0
    except Exception as e:
        st.warning(f'NASA POWER API error: {e}. Using fallback GHI = 5.0')
        avg_ghi = 5.0

    # 2. CEA data — actual generation
    try:
        cea_df = load_cea_data('data/india_generation_clean.csv')
        region_df = cea_df[
            (cea_df['region'] == region_code) &
            (cea_df['date'].dt.year == selected_year)
        ]
        actual_mwh_daily = region_df['solar_mwh'].mean() if len(region_df) > 0 else 0
        actual_mw        = actual_mwh_daily / 24  # MWh/day → average MW
    except Exception as e:
        st.warning(f'CEA data error: {e}. Using fallback actual = 0')
        actual_mw = 0

    # 3. NASA FIRMS — fire hotspots
    try:
        fires_df = get_fire_hotspots(days=fire_days)
    except Exception as e:
        st.warning(f'FIRMS API error: {e}. Fire data unavailable.')
        fires_df = pd.DataFrame(columns=['latitude', 'longitude', 'brightness', 'confidence'])

    # 4. Calculations
    wasted_mw      = calculate_curtailment(avg_ghi, capacity_mw, actual_mw)
    losses         = calculate_losses(wasted_mw, hours=24)
    curtail_pct    = calculate_curtailment_percent(avg_ghi * capacity_mw * 0.15, actual_mw)

    # 5. ML Risk classification
    try:
        full_df   = load_cea_data('data/india_generation_clean.csv')
        features  = prepare_features(full_df)
        clf, scaler, labelled = train_model(features)
        region_row = labelled[labelled['region'] == region_code]
        if len(region_row) > 0:
            solar_share = float(region_row['solar_share_pct'].values[0])
            coal_share  = float(region_row['coal_share_pct'].values[0])
        else:
            solar_share, coal_share = 20.0, 50.0
        risk_label = predict_risk(clf, scaler, curtail_pct, solar_share, coal_share)
    except Exception as e:
        st.warning(f'ML classifier error: {e}')
        risk_label = 'Medium'

    risk_color = get_risk_color(risk_label)

# ── Metrics row ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric('Avg GHI', f'{avg_ghi} kWh/m²/d', help='NASA satellite solar potential')
with col2:
    st.metric('Wasted Energy', f'{losses["wasted_kwh"]:,.0f} kWh', help='Energy curtailed per day')
with col3:
    st.metric('Money Lost', f'₹{losses["money_rs"]:,.0f}', help='At ₹3/kWh compensation rate')
with col4:
    st.metric('CO₂ Released', f'{losses["co2_kg"]:,.0f} kg', help='At 0.727 kg CO₂ per kWh (CEA)')
with col5:
    st.metric('Curtailment', f'{curtail_pct:.1f}%', help='% of potential solar that was wasted')

# Risk badge
risk_class = f'risk-{risk_label.lower()}'
st.markdown(
    f'ML Risk Classification: <span class="risk-badge {risk_class}">{risk_label} Risk</span>',
    unsafe_allow_html=True
)

# ── Map ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Interactive Map — Fire Hotspots & Solar Regions</div>', unsafe_allow_html=True)

m = folium.Map(location=[22, 80], zoom_start=5, tiles='CartoDB dark_matter')

# Add all region markers
for region_name, info in REGIONS.items():
    is_selected = (info['code'] == region_code)
    folium.CircleMarker(
        location=[info['lat'], info['lon']],
        radius=18 if is_selected else 10,
        color=risk_color if is_selected else '#378add',
        fill=True,
        fill_color=risk_color if is_selected else '#185fa5',
        fill_opacity=0.8 if is_selected else 0.4,
        tooltip=folium.Tooltip(
            f"<b>{info['state']}</b><br>"
            f"Region: {info['code']}<br>"
            f"Capacity: {info['capacity_mw']:,} MW"
            + (f"<br><b>Risk: {risk_label}</b>" if is_selected else "")
        )
    ).add_to(m)

# Add selected region label
folium.Marker(
    location=[lat + 1.5, lon],
    icon=folium.DivIcon(
        html=f'<div style="font-size:12px;font-weight:700;color:{risk_color};'
             f'background:#0f1117;padding:3px 8px;border-radius:4px;'
             f'border:1px solid {risk_color};white-space:nowrap;">'
             f'{state_name} — {risk_label} Risk</div>',
        icon_size=(180, 30),
        icon_anchor=(90, 0)
    )
).add_to(m)

# Add fire hotspot dots
if len(fires_df) > 0:
    fire_count = 0
    for _, row in fires_df.iterrows():
        try:
            conf = str(row.get('confidence', 'n')).lower()
            if conf in ['high', 'h', 'nominal', 'n']:
                folium.CircleMarker(
                    location=[float(row['latitude']), float(row['longitude'])],
                    radius=2,
                    color='#ff6b35',
                    fill=True,
                    fill_color='#ff4500',
                    fill_opacity=0.7,
                    tooltip='Fire hotspot'
                ).add_to(m)
                fire_count += 1
        except Exception:
            continue

# Legend
legend_html = f"""
<div style='position:fixed;bottom:30px;left:30px;z-index:1000;
     background:#0f1117;border:1px solid #2a2d3a;border-radius:10px;
     padding:12px 16px;font-size:12px;color:#c0c3d4;'>
  <div style='font-weight:700;margin-bottom:8px;color:#fff;'>Legend</div>
  <div><span style='color:{risk_color};font-size:16px;'>●</span> {state_name} ({risk_label} Risk)</div>
  <div><span style='color:#378add;font-size:16px;'>●</span> Other regions</div>
  <div><span style='color:#ff6b35;font-size:16px;'>●</span> Fire hotspots (NASA FIRMS)</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=None, height=520, returned_objects=[])

# ── Monthly GHI chart ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Monthly Solar Potential (GHI) — NASA POWER</div>', unsafe_allow_html=True)

if ghi_data:
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    months = sorted(ghi_data.keys())
    values = [ghi_data[m] for m in months]
    x_labels = [month_labels[int(m[4:6]) - 1] for m in months]

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')

    bars = ax.bar(x_labels, values, color='#378add', alpha=0.7, width=0.6, edgecolor='none')
    # Highlight peak month
    peak_idx = values.index(max(values))
    bars[peak_idx].set_color('#EF9F27')
    bars[peak_idx].set_alpha(1.0)

    ax.set_ylabel('kWh/m²/day', color='#8b8fa8', fontsize=10)
    ax.tick_params(colors='#8b8fa8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#2a2d3a')
    ax.yaxis.set_tick_params(labelcolor='#8b8fa8')
    ax.xaxis.set_tick_params(labelcolor='#8b8fa8')
    ax.axhline(y=np.mean(values), color='#1d9e75', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(len(values) - 0.5, np.mean(values) + 0.1,
            f'avg {np.mean(values):.1f}', color='#1d9e75', fontsize=9, ha='right')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Region Summary</div>', unsafe_allow_html=True)

summary_data = {
    'Region':          [selected_region],
    'Representative State': [state_name],
    'Year':            [selected_year],
    'Avg GHI (kWh/m²/d)': [avg_ghi],
    'Wasted (kWh/day)':    [f'{losses["wasted_kwh"]:,.0f}'],
    'Money Lost (₹/day)':  [f'{losses["money_rs"]:,.0f}'],
    'CO₂ (kg/day)':        [f'{losses["co2_kg"]:,.0f}'],
    'Curtailment %':       [f'{curtail_pct:.1f}%'],
    'Risk':                [risk_label],
}
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ── Fire stats ────────────────────────────────────────────────────────────────
if len(fires_df) > 0:
    st.markdown('<div class="section-header">Fire Hotspot Summary (NASA FIRMS)</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Hotspots', f'{len(fires_df):,}', help=f'Last {fire_days} days across India')
    with col2:
        high_conf = fires_df[fires_df['confidence'].astype(str).str.lower().isin(['high','h'])].shape[0]
        st.metric('High Confidence', f'{high_conf:,}')
    with col3:
        if 'brightness' in fires_df.columns:
            avg_bright = fires_df['brightness'].mean()
            st.metric('Avg Brightness (K)', f'{avg_bright:.1f}')
