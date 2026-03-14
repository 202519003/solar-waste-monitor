# SolarWaste Monitor 🌞

> A real-time Python dashboard that tracks wasted solar energy across Indian states — calculating lost revenue, excess CO₂ emissions, and correlating with fire hotspot data.

**MSc (AA) / PGD (SDS) — Python Project 2026**

---

## The Problem

Every day in India, solar plants are switched off because the electricity grid cannot absorb the energy they generate. This is called **solar curtailment** — free, clean energy is wasted while coal plants keep running unnecessarily, releasing CO₂ that contributes to forest fires and climate damage.

SolarWaste Monitor makes this invisible problem visible: it compares how much solar energy *should* have been generated (from NASA satellite data) against how much *was actually* generated (from official CEA reports), and shows the difference on an interactive map of India.

---

## Features

- 🗺️ **Interactive choropleth map** — colour-coded by curtailment level per state
- 🔥 **Live fire hotspot overlay** — NASA FIRMS data updated every 3 hours
- 📊 **Loss calculator** — wasted kWh, money lost (₹), and CO₂ released (kg)
- 🤖 **ML risk classifier** — KMeans + Decision Tree labels each state Low / Medium / High
- 📡 **Live data** — pulls from NASA POWER API and NASA FIRMS API on every run

---

## Screenshots

> *(Add screenshots of the running app here after completing Phase 3)*

---

## Data Sources

| Source | What it provides | Link |
|---|---|---|
| CEA / Robbie Andrew | Actual state-wise solar generation (CSV) | https://robbieandrew.github.io/india/ |
| NASA POWER API | Theoretical solar potential (GHI) | https://power.larc.nasa.gov |
| NASA FIRMS API | Real-time fire hotspot coordinates | https://firms.modaps.eosdis.nasa.gov |
| DataMeet GeoJSON | India state boundary shapes | https://github.com/Subhash9325/GeoJson-Data-of-Indian-States |
| CEA CO₂ Database v21.0 | Official emission factor (0.727 kg/kWh) | https://cea.nic.in/cdm-co2-baseline-database/ |

---

## Project Structure

```
solar-waste-monitor/
├── main.py                  # Entry point — run this to launch the app
├── requirements.txt         # All dependencies
├── README.md
├── PROPOSAL.md
├── data/
│   ├── india_generation.csv # CEA solar generation data
│   └── india_states.geojson # India state boundary shapes
├── docs/
│   └── writeup.pdf          # Project report
└── src/
    ├── gui/
    │   └── dashboard.py     # Streamlit dashboard (Member C)
    └── logic/
        ├── fetcher.py       # NASA API + CEA data loader (Member A)
        ├── calculator.py    # Curtailment & CO₂ calculations (Member B)
        └── classifier.py   # KMeans + Decision Tree ML model (Member B)
```

---

## Installation

**Prerequisites:** Python 3.11 or higher

**Step 1 — Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/solar-waste-monitor.git
cd solar-waste-monitor
```

**Step 2 — Install all dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 — Add your NASA FIRMS Map Key**

Get a free key from https://firms.modaps.eosdis.nasa.gov/api/map_key/ (takes 2 minutes).

Open `src/logic/fetcher.py` and replace:
```python
MAP_KEY = 'your_key_here'
```
with your actual key.

---

## How to Run

```bash
python main.py
```

Then open your browser at: **http://localhost:8501**

1. Select a state from the sidebar
2. Choose a year
3. Click **Run Analysis**
4. View the map, fire hotspots, and calculated losses

---

## How It Works

```
NASA POWER API  ──►  fetcher.py  ──►  calculator.py  ──►  dashboard.py
CEA CSV         ──►               ──►  classifier.py  ──►  (Streamlit GUI)
NASA FIRMS API  ──►               ──►                 ──►  (Folium map)
```

**Core formula:**

```
curtailed_mw  = (GHI × capacity_mw × 0.15) − actual_mw
co2_lost_kg   = curtailed_kwh × 0.727
money_lost_rs = curtailed_kwh × 3.0
```

The emission factor `0.727 kg CO₂/kWh` is the official India grid value from CEA CO₂ Baseline Database Version 21.0 (December 2025).

---

## ML Model

The classifier uses a two-step approach:

1. **KMeans clustering** (k=3) — finds natural groupings in curtailment % and temperature data without needing pre-labelled data
2. **Decision Tree** (max_depth=3) — trained on the KMeans-generated labels to predict risk level for new inputs

Output labels: `Low` / `Medium` / `High` curtailment risk per state.

---

## Team Contributions

| Member | Responsibility |
|---|---|
| Member A | `fetcher.py` — NASA POWER API, NASA FIRMS API, CEA CSV loader |
| Member B | `calculator.py` — curtailment formula, CO₂ and money loss calculations; `classifier.py` — KMeans + Decision Tree ML model |
| Member C | `dashboard.py` — Streamlit GUI, Folium map, metrics display |
| Member D | `main.py`, `requirements.txt`, `README.md`, GitHub setup, integration testing |

---

## Dependencies

```
streamlit>=1.32.0
requests>=2.31.0
pandas>=2.0.0
geopandas>=0.14.0
scikit-learn>=1.4.0
folium>=0.16.0
matplotlib>=3.8.0
streamlit-folium>=0.18.0
geopy>=2.4.0
scipy>=1.12.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Common Errors & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `API timeout` | Add `timeout=10` to the `requests.get()` call in `fetcher.py` |
| `CSV file not found` | Check the file is saved as `data/india_generation.csv` |
| `Map not showing` | Run `pip install streamlit-folium` separately |
| `geopandas install fails` (Windows) | Run `pip install geopandas` on its own first |

---

## License

This project was built for academic submission. Data is sourced from NASA (public domain) and the Government of India (CEA — open data). All data sources are free and openly available.

---

*SolarWaste Monitor — Python Project 2026*
