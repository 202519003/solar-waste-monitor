"""
calculator.py  —  Energy Loss Computation
==========================================
Responsibilities (Yash Daslaniya):
  compute_curtailment()  → calculates wasted MWh, money lost, CO2 for one state+year
  compute_all_states()   → runs computation for every state and returns a summary DataFrame
  monthly_breakdown()    → returns month-by-month curtailment for bar chart

Curtailment Formula:
  potential_MWh = GHI (kWh/m²/day) × days_in_month × installed_MW × 1000 m²/kW × panel_efficiency
  curtailed_MWh = max(0,  potential_MWh  −  actual_MWh)
  curtailment_% = curtailed_MWh / potential_MWh × 100

Official constants used:
  Panel efficiency : 0.18  (18% — standard mono-PERC modules, CEA assumption)
  CO2 factor       : 0.727 kg/kWh  (CEA Grid Emission Factor v21.0)
  Solar tariff     : ₹2.50/kWh     (average SECI/MNRE PPA rate 2024)
"""

import calendar
import pandas as pd
import numpy as np

from src.logic.fetcher import (
    load_generation,
    fetch_ghi,
    get_installed_mw,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PANEL_EFFICIENCY   = 0.18           # 18%
CO2_FACTOR_KG_KWH  = 0.727          # kg CO2 per kWh (CEA v21.0)
SOLAR_TARIFF_RS    = 2.50           # ₹ per kWh (average PPA rate)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _days(month: int, year: int) -> int:
    """Returns number of days in a given month/year."""
    return calendar.monthrange(year, month)[1]


def _potential_mwh(ghi_kwh_m2_day: float, installed_mw: float,
                   month: int, year: int) -> float:
    """
    Calculates theoretical maximum solar generation for a state in one month.

    Formula:
        potential_MWh = GHI × days × installed_MW × 1000 (m²/kW) × efficiency / 1000 (kW→MW)
                      = GHI × days × installed_MW × efficiency
    """
    if ghi_kwh_m2_day is None or ghi_kwh_m2_day <= 0:
        return 0.0
    days = _days(month, year)
    # installed_MW × 1000 kW/MW × 1000 m² per kW (1 kWp needs ~5.5 m² but we use 1 m² GHI directly)
    # Simplified: potential_kWh = GHI × days × installed_MW × 1000 × efficiency
    potential_kwh = ghi_kwh_m2_day * days * installed_mw * 1000 * PANEL_EFFICIENCY
    return potential_kwh / 1000   # kWh → MWh


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_curtailment(state: str, year: int,
                        gen_df: pd.DataFrame = None,
                        ghi_data: dict = None) -> dict:
    """
    Computes curtailment metrics for a single state and year.

    Parameters
    ----------
    state    : str   e.g. "Rajasthan"
    year     : int   e.g. 2024
    gen_df   : DataFrame (optional — pass to avoid reloading CSV every call)
    ghi_data : dict  (optional — pass pre-fetched GHI to avoid API call)

    Returns
    -------
    dict with keys:
        state, year,
        total_actual_mwh, total_potential_mwh, total_curtailed_mwh,
        curtailment_pct, money_lost_cr, co2_released_tons,
        monthly_df (DataFrame with per-month breakdown)
    """
    if gen_df is None:
        gen_df = load_generation()

    if ghi_data is None:
        ghi_data = fetch_ghi(state, year)

    installed_mw = get_installed_mw(state)

    # Filter generation data for this state + year
    state_gen = gen_df[(gen_df["State"] == state) & (gen_df["Year"] == year)].copy()

    rows = []
    for _, row in state_gen.iterrows():
        month  = int(row["Month"])
        actual = float(row["Generation_MW"]) * _days(month, year) * 24 / 1000
        # actual_MWh = Generation_MW × hours_in_month / 1000 ... wait:
        # Generation_MW is already average MW for the month from CEA
        # So actual_MWh = Generation_MW × days × 24
        actual_mwh = float(row["Generation_MW"]) * _days(month, year) * 24

        ghi = ghi_data.get(month, None)
        potential_mwh = _potential_mwh(ghi, installed_mw, month, year)
        curtailed_mwh = max(0.0, potential_mwh - actual_mwh)

        rows.append({
            "Month":          month,
            "Actual_MWh":     round(actual_mwh, 1),
            "Potential_MWh":  round(potential_mwh, 1),
            "Curtailed_MWh":  round(curtailed_mwh, 1),
            "GHI":            ghi,
        })

    monthly_df = pd.DataFrame(rows).sort_values("Month").reset_index(drop=True)

    total_actual    = monthly_df["Actual_MWh"].sum()
    total_potential = monthly_df["Potential_MWh"].sum()
    total_curtailed = monthly_df["Curtailed_MWh"].sum()

    curtailment_pct = (total_curtailed / total_potential * 100) if total_potential > 0 else 0.0

    # Money lost:  curtailed_MWh × 1000 kWh/MWh × ₹2.50/kWh → ₹, then ÷ 1 crore
    money_lost_cr   = total_curtailed * 1000 * SOLAR_TARIFF_RS / 1e7

    # CO2 that kept burning:  curtailed_MWh × 1000 × 0.727 kg/kWh → tonnes
    co2_tons        = total_curtailed * 1000 * CO2_FACTOR_KG_KWH / 1000

    return {
        "state":                state,
        "year":                 year,
        "total_actual_mwh":     round(total_actual, 1),
        "total_potential_mwh":  round(total_potential, 1),
        "total_curtailed_mwh":  round(total_curtailed, 1),
        "curtailment_pct":      round(curtailment_pct, 2),
        "money_lost_cr":        round(money_lost_cr, 2),
        "co2_released_tons":    round(co2_tons, 1),
        "monthly_df":           monthly_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ALL-STATES SUMMARY  (used by classifier and choropleth map)
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_states(year: int, gen_df: pd.DataFrame = None,
                       use_api: bool = True) -> pd.DataFrame:
    """
    Runs curtailment calculation for every state for a given year.

    Parameters
    ----------
    year    : int    e.g. 2024
    gen_df  : DataFrame (optional)
    use_api : bool   If False, skips NASA POWER API and uses GHI estimates
                     (faster for testing — uses hardcoded average GHI values)

    Returns
    -------
    DataFrame with one row per state:
        State, curtailment_pct, total_curtailed_mwh,
        money_lost_cr, co2_released_tons, installed_mw
    """
    if gen_df is None:
        gen_df = load_generation()

    states = gen_df["State"].unique()
    summary_rows = []

    for state in states:
        ghi_data = None
        if not use_api:
            # Fallback: use rough average GHI values (avoids API during testing)
            ghi_data = _fallback_ghi(state)

        try:
            result = compute_curtailment(state, year, gen_df=gen_df, ghi_data=ghi_data)
            summary_rows.append({
                "State":               result["state"],
                "curtailment_pct":     result["curtailment_pct"],
                "total_curtailed_mwh": result["total_curtailed_mwh"],
                "money_lost_cr":       result["money_lost_cr"],
                "co2_released_tons":   result["co2_released_tons"],
                "installed_mw":        get_installed_mw(state),
            })
        except Exception as e:
            print(f"[calculator] Error for {state}: {e}")

    return pd.DataFrame(summary_rows)


def _fallback_ghi(state: str) -> dict:
    """
    Rough average annual GHI (kWh/m²/day) per state for offline/testing use.
    Values from NASA POWER historical averages.
    """
    avg = {
        "Rajasthan": 6.0, "Gujarat": 5.8, "Karnataka": 5.5,
        "Tamil Nadu": 5.4, "Andhra Pradesh": 5.3, "Telangana": 5.3,
        "Maharashtra": 5.2, "Madhya Pradesh": 5.1, "Uttar Pradesh": 4.8,
        "Punjab": 4.9, "Haryana": 5.0, "Odisha": 5.0,
        "Chhattisgarh": 4.9, "Kerala": 4.8, "Bihar": 4.7,
        "West Bengal": 4.6, "Assam": 4.4, "Himachal Pradesh": 4.5,
        "Uttarakhand": 4.5, "Jharkhand": 4.8, "Goa": 5.0,
        "Manipur": 4.3, "Meghalaya": 4.2, "Tripura": 4.4,
        "Nagaland": 4.2, "Mizoram": 4.3, "Arunachal Pradesh": 4.1,
        "Sikkim": 4.0,
    }
    g = avg.get(state, 4.5)
    return {m: g for m in range(1, 13)}


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing calculator.py (offline mode) ===\n")

    gen = load_generation()
    result = compute_curtailment("Rajasthan", 2024, gen_df=gen,
                                  ghi_data=_fallback_ghi("Rajasthan"))

    print(f"State            : {result['state']}")
    print(f"Year             : {result['year']}")
    print(f"Actual MWh       : {result['total_actual_mwh']:,.0f}")
    print(f"Potential MWh    : {result['total_potential_mwh']:,.0f}")
    print(f"Curtailed MWh    : {result['total_curtailed_mwh']:,.0f}")
    print(f"Curtailment %    : {result['curtailment_pct']}%")
    print(f"Money Lost       : ₹{result['money_lost_cr']} Crore")
    print(f"CO2 Released     : {result['co2_released_tons']} tonnes")
    print("\nMonthly breakdown:")
    print(result["monthly_df"].to_string(index=False))
