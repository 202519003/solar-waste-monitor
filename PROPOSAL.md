# Project Proposal — SolarWaste Monitor

**MSc (AA) / PGD (SDS) — Python Project 2026**
**Submission Date:** 15 March 2026
**GitHub Repository:** https://github.com/202519003/solar-waste-monitor

---

## Team Members

| Member | Student ID | Role |
|---|---|---|
| Member A | 202519003 | Data fetching — fetcher.py (NASA POWER, NASA FIRMS, CEA CSV) |
| Member B | — | Calculations & ML — calculator.py, classifier.py |
| Member C | — | GUI — dashboard.py (Streamlit, Folium map) |
| Member D | — | Integration, GitHub, README, main.py |

> Fill in full names and remaining student IDs before pushing to GitHub.

---

## Project Title

**SolarWaste Monitor — India Solar Curtailment Tracker**

---

## Problem Statement

India is one of the world's fastest-growing solar energy markets, with over 80 GW of installed solar capacity as of 2025. However, a significant and largely invisible problem exists: **solar curtailment** — the forced shutdown of solar plants when the electricity grid cannot absorb the energy they are generating.

When solar plants are curtailed:
- Clean, free energy is permanently wasted
- Coal-fired plants continue running unnecessarily because they cannot ramp down quickly
- Extra CO₂ is released into the atmosphere for no productive purpose
- Solar plant operators are paid compensation (curtailment fees) for energy they never actually delivered

This problem is not well visualised. Grid operators and policymakers lack an easy-to-use tool that shows, at a glance, which states are wasting the most solar energy, how much money is being lost, and what the environmental cost is.

---

## Proposed Solution

SolarWaste Monitor is a Python-based dashboard that:

1. **Calculates curtailed solar energy** by comparing NASA satellite-measured solar potential against CEA-recorded actual generation for each Indian state
2. **Quantifies financial loss** using the official curtailment compensation rate (₹3/kWh)
3. **Quantifies CO₂ waste** using the official CEA emission factor (0.727 kg CO₂ per kWh)
4. **Classifies risk** using a KMeans + Decision Tree ML model that labels each state as Low / Medium / High curtailment risk
5. **Visualises everything** on an interactive Folium map of India with fire hotspot overlay from NASA FIRMS

---

## Data Sources

| Source | Data Provided | Access Method |
|---|---|---|
| CEA / GRID-INDIA via Robbie Andrew | State-wise actual solar generation (2013–present) | Free CSV download |
| NASA POWER API | Monthly GHI (solar potential) for any location | Free REST API, no key needed |
| NASA FIRMS API | Real-time fire hotspot GPS coordinates across India | Free API, Map Key required |
| DataMeet / GitHub GeoJSON | India state boundary shapes for Folium map | Free file download |
| CEA CO₂ Baseline Database v21.0 | Official grid emission factor: 0.727 kg CO₂/kWh | Hardcoded constant from PDF |

---

## Technical Architecture

```
Data Layer          Logic Layer           Presentation Layer
──────────          ───────────           ──────────────────
NASA POWER   ──►
CEA CSV      ──►   fetcher.py   ──►   calculator.py   ──►   dashboard.py
NASA FIRMS   ──►                  ──►   classifier.py  ──►   (Streamlit GUI)
GeoJSON      ──────────────────────────────────────────►   (Folium Map)
```

### Module breakdown

**`src/logic/fetcher.py`** (Member A)
Responsible for all external data retrieval. Three functions: `get_solar_potential()` calls NASA POWER API, `get_fire_hotspots()` calls NASA FIRMS API, `load_cea_data()` reads the CEA CSV. No calculation logic — pure data fetching only.

**`src/logic/calculator.py`** (Member B)
Responsible for all numerical calculations. Takes GHI from NASA and actual generation from CEA, computes curtailed energy in kWh, money lost in ₹, and CO₂ released in kg. Uses CEA official emission factor 0.727 kg/kWh.

**`src/logic/classifier.py`** (Member B)
Responsible for ML risk classification. Uses KMeans (k=3) to discover natural clusters in curtailment data without pre-labelled training data, then trains a Decision Tree (max_depth=3) on those cluster labels to predict Low / Medium / High risk for any state.

**`src/gui/dashboard.py`** (Member C)
Responsible for the entire user interface. Built with Streamlit. Shows three metric cards (wasted kWh, money lost, CO₂), an interactive Folium choropleth map of India coloured by curtailment level, and orange fire dots from NASA FIRMS.

**`main.py`** (Member D)
Single entry point. Run `python main.py` to launch the entire application.

---

## Machine Learning Approach

The classifier uses a two-step pipeline:

**Step 1 — KMeans Clustering**
We have no pre-labelled training data (no dataset that says "Rajasthan = High risk"). KMeans automatically finds 3 natural groups in the feature space (curtailment %, average temperature) without needing labels. This is an unsupervised learning approach.

**Step 2 — Decision Tree Classifier**
Once KMeans assigns cluster labels, we sort clusters by mean curtailment and name them Low, Medium, High. We then train a Decision Tree on these labels. The Decision Tree creates human-interpretable decision rules (e.g. "if curtailment > 18% AND temperature > 35°C → High risk").

**Why this approach?**
KMeans alone cannot predict new inputs. Decision Tree alone requires pre-labelled data we don't have. The combination solves both problems.

---

## Project Folder Structure

```
solar-waste-monitor/
├── main.py
├── requirements.txt
├── README.md
├── PROPOSAL.md
├── data/
│   ├── india_generation.csv
│   └── india_states.geojson
├── docs/
│   └── writeup.pdf
└── src/
    ├── gui/
    │   └── dashboard.py
    └── logic/
        ├── fetcher.py
        ├── calculator.py
        └── classifier.py
```

---

## Timeline

| Dates | Phase | Deliverable |
|---|---|---|
| Mar 13–14 | Phase 1: Setup | Python, libraries, GitHub, folder structure |
| Mar 15–16 | Phase 2: Data | CEA CSV, GeoJSON, NASA APIs tested, PROPOSAL.md submitted |
| Mar 17–22 | Phase 3A: Code | fetcher.py, calculator.py complete |
| Mar 23–27 | Phase 3B: Code | classifier.py, dashboard.py complete |
| Mar 28–30 | Phase 4: Integration | Full app runs end to end, bugs fixed |
| Mar 31–Apr 1 | Phase 5: Submission | README, writeup, GitHub pushed, submitted by Apr 1 23:59 |
| Apr 1–3 | Phase 6: Viva | All members read all code, demo rehearsal, Viva Apr 3 09:30 |

---

## Expected Outcomes

By the end of the project, SolarWaste Monitor will:

- Display a colour-coded interactive map of India showing curtailment levels by state
- Show real-time fire hotspot data overlaid on the same map
- Calculate and display exact figures for wasted energy (kWh), money lost (₹), and CO₂ released (kg) for any selected state and year
- Classify each state as Low, Medium, or High curtailment risk using a trained ML model
- Run with a single command: `python main.py`

---

## Why This Problem Matters

Solar curtailment is a direct barrier to India's renewable energy targets. Every megawatt-hour wasted is a megawatt-hour that had to be replaced by coal. By making curtailment data visible and interactive, SolarWaste Monitor gives grid planners and energy policymakers a tool to identify where grid infrastructure investment is most urgently needed.

---

*Proposal submitted to GitHub repository on 15 March 2026.*
