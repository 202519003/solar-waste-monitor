# calculator.py — SolarWaste Monitor
# Source: CEA CO2 Baseline Database Version 21.0, December 2025
# Official India grid emission factor
CO2_PER_KWH = 0.727  # kg of CO2 released per kWh of coal electricity

# Official curtailment compensation rate (Rs per kWh paid to solar plants)
COMPENSATION_RATE = 3.0  # Rs per kWh

# NOTE: The correct formula for solar potential is:
#   potential_MWh_per_day = GHI (kWh/m²/day) × capacity_MW
#
# Explanation:
#   GHI is measured in kWh/m²/day — this already represents the usable
#   solar energy per unit area per day (it is NOT raw irradiance that needs
#   efficiency applied). When multiplied by capacity_MW, it gives the
#   expected output in MWh/day directly.
#
#   The old formula (GHI × capacity × 0.15) was WRONG because it applied
#   a panel efficiency factor on top of GHI, which double-discounts the
#   energy. GHI from NASA POWER is already the surface-level energy — the
#   capacity rating of the panels accounts for their own efficiency.


def calculate_curtailment(potential_ghi, capacity_mw, actual_mwh_day):
    """
    Calculates how much solar energy was wasted (curtailed) in a region per day.

    Args:
        potential_ghi   : float — GHI from NASA POWER (kWh/m²/day)
        capacity_mw     : float — installed solar capacity in the region (MW)
        actual_mwh_day  : float — actual daily solar generation from CEA (MWh/day)

    Returns:
        wasted_mwh_day  : float — MWh wasted per day (minimum 0)

    Formula:
        potential_mwh = GHI × capacity_mw        ← correct (no efficiency factor)
        wasted_mwh    = potential_mwh - actual_mwh (minimum 0)
    """
    potential_mwh_day = potential_ghi * capacity_mw
    wasted_mwh_day = max(0.0, potential_mwh_day - actual_mwh_day)
    return round(wasted_mwh_day, 2)


def calculate_losses(wasted_mwh_day):
    """
    Converts wasted MWh/day into kWh, money lost (Rs), and CO2 released (kg).

    Args:
        wasted_mwh_day : float — curtailed energy in MWh per day

    Returns:
        dict with keys:
            wasted_kwh  — energy wasted in kilowatt-hours per day
            money_rs    — money lost in Indian Rupees per day
            co2_kg      — CO2 released in kilograms per day
    """
    wasted_kwh = wasted_mwh_day * 1000          # MWh → kWh

    money_lost_rs = wasted_kwh * COMPENSATION_RATE
    co2_lost_kg   = wasted_kwh * CO2_PER_KWH

    return {
        'wasted_kwh': round(wasted_kwh, 2),
        'money_rs':   round(money_lost_rs, 2),
        'co2_kg':     round(co2_lost_kg, 2),
    }


def calculate_curtailment_percent(potential_mwh_day, actual_mwh_day):
    """
    Calculates curtailment as a percentage of potential generation.
    Used by the ML classifier as the main input feature.

    Args:
        potential_mwh_day : float — what could have been generated (MWh/day)
        actual_mwh_day    : float — what was actually generated (MWh/day)

    Returns:
        curtailment_pct : float — percentage of potential energy that was wasted
    """
    if potential_mwh_day <= 0:
        return 0.0
    curtailment_pct = ((potential_mwh_day - actual_mwh_day) / potential_mwh_day) * 100
    return round(max(0.0, curtailment_pct), 2)
