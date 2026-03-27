"""
dashboard.py  —  Streamlit GUI
================================
Responsibilities (Mehul Chaudhary):
  render_sidebar()     → state + year dropdowns + run button
  render_metrics()     → three st.metric cards
  render_map()         → Folium choropleth + fire overlay
  render_bar_chart()   → monthly curtailment bar chart
  render_risk_badge()  → ML risk prediction badge
  render_fire_panel()  → solar-fire connection side panel
  run_dashboard()      → main function — wires all panels together
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd

from src.logic.fetcher    import load_generation, load_fire_data, GEOJSON_TO_CSV
from src.logic.calculator import compute_curtailment, compute_all_states, _fallback_ghi
from src.logic.classifier import train_model, predict_risk, get_risk_color


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")
GEOJSON    = os.path.join(DATA_DIR, "india_states.geojson")

SOLAR_STATES = [
    "Rajasthan", "Gujarat", "Karnataka", "Tamil Nadu",
    "Andhra Pradesh", "Telangana", "Maharashtra", "Madhya Pradesh",
    "Uttar Pradesh", "Punjab", "Haryana", "Odisha",
    "Chhattisgarh", "Kerala", "Bihar", "West Bengal",
    "Assam", "Himachal Pradesh", "Uttarakhand", "Jharkhand",
]

MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
}


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

def set_page_config():
    st.set_page_config(
        page_title  = "SolarWaste Monitor",
        page_icon   = "☀️",
        layout      = "wide",
        initial_sidebar_state = "expanded",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Renders sidebar controls. Returns (selected_state, selected_year, run_clicked)."""
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Solar_panel.jpg/320px-Solar_panel.jpg",
                 use_column_width=True)
        st.title("☀️ SolarWaste Monitor")
        st.caption("India Solar Curtailment & Forest Fire Risk")
        st.divider()

        state = st.selectbox(
            "Select State",
            options=SOLAR_STATES,
            index=0,
            help="Choose an Indian state to analyse solar curtailment.",
        )

        year = st.selectbox(
            "Select Year",
            options=[2023, 2024, 2025],
            index=1,
            help="CEA generation data is available for 2023–2025.",
        )

        use_api = st.checkbox(
            "Fetch live GHI from NASA POWER",
            value=False,
            help="Uncheck to use offline GHI estimates (faster for testing).",
        )

        run = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

        st.divider()
        st.caption("Data sources: CEA, NASA POWER, NASA FIRMS VIIRS, MNRE")

    return state, year, use_api, run


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────

def render_metrics(result: dict):
    """Renders three metric cards: Wasted MWh, Money Lost, CO2."""
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="⚡ Energy Wasted",
            value=f"{result['total_curtailed_mwh']:,.0f} MWh",
            delta=f"{result['curtailment_pct']}% curtailment",
            delta_color="inverse",
        )

    with c2:
        st.metric(
            label="💸 Money Lost",
            value=f"₹{result['money_lost_cr']:.2f} Cr",
            help="Based on avg. SECI solar tariff ₹2.50/kWh",
        )

    with c3:
        st.metric(
            label="🏭 CO₂ Released",
            value=f"{result['co2_released_tons']:,.0f} tonnes",
            help="CEA Grid Emission Factor: 0.727 kg/kWh",
        )


# ─────────────────────────────────────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────────────────────────────────────

def render_map(summary_df: pd.DataFrame, fire_df: pd.DataFrame,
               selected_state: str):
    """
    Renders Folium choropleth map with:
      - State fill colour by curtailed MWh (green → dark red)
      - NASA FIRMS fire hotspots as orange dots
      - Popup on each state showing key metrics
    """
    st.subheader("🗺️ India Curtailment Map + Fire Hotspots")

    # Load GeoJSON
    gdf = gpd.read_file(GEOJSON)

    # Fix name mismatches
    gdf["NAME_1"] = gdf["NAME_1"].replace(GEOJSON_TO_CSV)

    # Merge with summary data
    gdf = gdf.merge(summary_df, left_on="NAME_1", right_on="State", how="left")

    # Build Folium map centred on India
    m = folium.Map(
        location=[22.5, 80.0],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # Choropleth layer
    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=summary_df,
        columns=["State", "total_curtailed_mwh"],
        key_on="feature.properties.NAME_1",
        fill_color="RdYlGn_r",      # Green = low waste, Red = high waste
        fill_opacity=0.75,
        line_opacity=0.4,
        legend_name="Curtailed Energy (MWh)",
        nan_fill_color="lightgrey",
    ).add_to(m)

    # State popups
    for _, row in gdf.iterrows():
        if pd.isna(row.get("curtailment_pct")):
            continue
        popup_html = f"""
        <b>{row['NAME_1']}</b><br>
        Curtailment: {row.get('curtailment_pct', 'N/A')}%<br>
        Wasted: {row.get('total_curtailed_mwh', 0):,.0f} MWh<br>
        Money lost: ₹{row.get('money_lost_cr', 0):.1f} Cr<br>
        CO₂: {row.get('co2_released_tons', 0):,.0f} t
        """
        # Place a transparent marker at state centroid for popup
        centroid = row.geometry.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            popup=folium.Popup(popup_html, max_width=220),
            icon=folium.DivIcon(html="", icon_size=(0, 0)),
        ).add_to(m)

    # Fire hotspot overlay
    if not fire_df.empty:
        fire_sample = fire_df.sample(min(len(fire_df), 1000), random_state=42)
        for _, frow in fire_sample.iterrows():
            folium.CircleMarker(
                location=[frow["latitude"], frow["longitude"]],
                radius=2,
                color="#FF6600",
                fill=True,
                fill_color="#FF6600",
                fill_opacity=0.6,
                popup=f"FRP: {frow.get('frp', 'N/A')} MW | {frow.get('acq_date', '')}",
            ).add_to(m)

    # Highlight selected state
    folium.GeoJson(
        gdf[gdf["NAME_1"] == selected_state].__geo_interface__,
        style_function=lambda x: {
            "fillColor":   "none",
            "color":       "#0047AB",
            "weight":       3,
            "dashArray":   "5 5",
        },
    ).add_to(m)

    st_folium(m, width="100%", height=520)


# ─────────────────────────────────────────────────────────────────────────────
# BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def render_bar_chart(monthly_df: pd.DataFrame, state: str, year: int):
    """Renders a monthly curtailment bar chart using Matplotlib."""
    st.subheader(f"📊 Monthly Curtailment — {state} ({year})")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    months     = monthly_df["Month"].apply(lambda m: MONTH_NAMES[m])
    actual     = monthly_df["Actual_MWh"]
    curtailed  = monthly_df["Curtailed_MWh"]

    x = np.arange(len(months))
    width = 0.4

    bars1 = ax.bar(x - width/2, actual / 1000,    width, label="Actual (GWh)",    color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, curtailed / 1000, width, label="Curtailed (GWh)", color="#F44336", alpha=0.85)

    ax.set_xlabel("Month", color="white")
    ax.set_ylabel("Energy (GWh)", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(months, color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1E1E1E", labelcolor="white")

    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ML RISK BADGE
# ─────────────────────────────────────────────────────────────────────────────

def render_risk_badge(risk_label: str, curtailment_pct: float):
    """Renders the ML risk prediction badge."""
    st.subheader("🤖 ML Curtailment Risk Prediction")
    color = get_risk_color(risk_label)

    messages = {
        "Low":    f"✅ **Low Risk** — Curtailment at {curtailment_pct:.1f}%. Grid absorption is adequate for this state.",
        "Medium": f"⚠️ **Medium Risk** — Curtailment at {curtailment_pct:.1f}%. Grid upgrades recommended in peak months.",
        "High":   f"🔴 **High Risk** — Curtailment at {curtailment_pct:.1f}%. Urgent grid investment and storage needed.",
    }

    msg = messages.get(risk_label, f"Risk: {risk_label} ({curtailment_pct:.1f}%)")

    if color == "success":
        st.success(msg)
    elif color == "warning":
        st.warning(msg)
    else:
        st.error(msg)


# ─────────────────────────────────────────────────────────────────────────────
# SOLAR-FIRE CONNECTION PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_fire_panel(fire_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Shows how high-curtailment states correlate with fire hotspot counts."""
    st.subheader("🔥 Solar Curtailment ↔ Forest Fire Connection")

    if fire_df.empty:
        st.info("Fire data not loaded. Place the NASA FIRMS CSV in the data/ folder.")
        return

    # Rough state bounding boxes for fire-to-state assignment
    STATE_BOUNDS = {
        "Rajasthan":    (23, 30, 69, 78),
        "Gujarat":      (20, 25, 68, 74),
        "Karnataka":    (11, 18, 74, 78),
        "Tamil Nadu":   (8,  13, 77, 80),
        "Andhra Pradesh": (13, 20, 77, 84),
        "Telangana":    (16, 20, 77, 81),
        "Maharashtra":  (15, 22, 72, 80),
        "Madhya Pradesh": (21, 27, 74, 82),
        "Uttar Pradesh":  (24, 30, 77, 84),
    }

    fire_counts = []
    for state, (lat_min, lat_max, lon_min, lon_max) in STATE_BOUNDS.items():
        mask = (
            (fire_df["latitude"]  >= lat_min) & (fire_df["latitude"]  <= lat_max) &
            (fire_df["longitude"] >= lon_min) & (fire_df["longitude"] <= lon_max)
        )
        count = mask.sum()
        curtailment = summary_df.loc[summary_df["State"] == state, "curtailment_pct"]
        curtailment_val = curtailment.values[0] if len(curtailment) > 0 else 0
        fire_counts.append({"State": state, "Fire Hotspots": count, "Curtailment %": curtailment_val})

    fc_df = pd.DataFrame(fire_counts).sort_values("Fire Hotspots", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.scatter(fc_df["Curtailment %"], fc_df["Fire Hotspots"],
                   color="#FF6600", s=80, alpha=0.85)
        for _, r in fc_df.iterrows():
            ax.annotate(r["State"][:3], (r["Curtailment %"], r["Fire Hotspots"]),
                        fontsize=7, color="white", ha="left", va="bottom")
        ax.set_xlabel("Curtailment %", color="white")
        ax.set_ylabel("Fire Hotspots", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#555")
        ax.spines["left"].set_color("#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_dashboard():
    """Entry point — wires all panels together."""
    set_page_config()

    st.title("☀️ SolarWaste Monitor")
    st.caption("India State-Level Solar Curtailment, Energy Loss & Forest Fire Risk Mapper")

    state, year, use_api, run = render_sidebar()

    # Load base data once (cached by Streamlit)
    @st.cache_data
    def cached_gen():
        return load_generation()

    gen_df = cached_gen()

    if not run:
        st.info("👈 Select a state and year in the sidebar, then click **Run Analysis**.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Solar_panel.jpg/640px-Solar_panel.jpg",
                 caption="India generated over 2.3 TWh of wasted solar energy in 2025.")
        return

    # ── Run Analysis ─────────────────────────────────────────────────────────
    with st.spinner("Fetching data and computing curtailment ..."):

        # 1. GHI data
        if use_api:
            ghi_data = fetch_ghi(state, year)
        else:
            from src.logic.calculator import _fallback_ghi
            ghi_data = _fallback_ghi(state)

        # 2. Curtailment for selected state
        result = compute_curtailment(state, year, gen_df=gen_df, ghi_data=ghi_data)

        # 3. All-states summary for map and classifier
        summary_df = compute_all_states(year=year, gen_df=gen_df, use_api=use_api)

        # 4. Train / load ML model
        model_bundle = train_model(year=year, use_api=use_api)
        risk = predict_risk(result["curtailment_pct"], state, model_bundle=model_bundle)

        # 5. Fire data
        fire_df = load_fire_data(year=year)

    # ── Render panels ─────────────────────────────────────────────────────────
    st.success(f"Analysis complete for **{state}** ({year})")
    st.divider()

    # Metric cards
    render_metrics(result)
    st.divider()

    # Risk badge
    render_risk_badge(risk, result["curtailment_pct"])
    st.divider()

    # Map
    render_map(summary_df, fire_df, state)
    st.divider()

    # Bar chart
    render_bar_chart(result["monthly_df"], state, year)
    st.divider()

    # Fire connection panel
    render_fire_panel(fire_df, summary_df)

    # Footer
    st.divider()
    st.caption(
        "Data: CEA Solar Generation Records | NASA POWER API | NASA FIRMS VIIRS | "
        "India States GeoJSON | MNRE Installed Capacity | CEA CO₂ Factor v21.0"
    )
