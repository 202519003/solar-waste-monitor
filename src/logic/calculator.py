# calculator.py — SolarWaste Monitor
# Source: CEA CO2 Baseline Database Version 21.0, December 2025

# India grid emission factor (official CEA value)
CO2_PER_KWH = 0.727      # kg CO2 per kWh

# Solar curtailment compensation rate
# Updated to reflect 2024–25 SECI/MNRE average discovered tariff
COMPENSATION_RATE = 2.55  # Rs per kWh


def calculate_curtailment(potential_ghi, capacity_mw, actual_mwh_day):
    """
    Calculates solar potential and curtailed energy per day.

    Args:
        potential_ghi   : GHI from NASA POWER (kWh/m²/day)
        capacity_mw     : Installed solar capacity (MW)
        actual_mwh_day  : Actual generation (MWh/day)

    Returns:
        potential_mwh_day : Total possible solar generation (MWh/day)
        wasted_mwh_day    : Curtailed energy (MWh/day)
    """
    # Solar potential: GHI (kWh/m²/day) × Capacity (MW) × efficiency factor
    # 0.15 = ~15% panel efficiency (standard for utility-scale in India)
    potential_mwh_day = potential_ghi * capacity_mw * 0.15

    # Curtailment cannot be negative
    wasted_mwh_day = max(0.0, potential_mwh_day - actual_mwh_day)

    return round(potential_mwh_day, 2), round(wasted_mwh_day, 2)


def calculate_losses(wasted_mwh_day):
    """
    Converts curtailed energy into:
    - Energy wasted (kWh)
    - Revenue loss (Rs)
    - CO2 emissions (kg)
    """
    # Convert MWh → kWh
    wasted_kwh = wasted_mwh_day * 1000

    # Money loss
    money_lost_rs = wasted_kwh * COMPENSATION_RATE

    # CO2 emissions
    co2_lost_kg = wasted_kwh * CO2_PER_KWH

    return {
        'wasted_kwh': round(wasted_kwh, 2),
        'money_rs':   round(money_lost_rs, 2),
        'co2_kg':     round(co2_lost_kg, 2),
    }


def calculate_curtailment_percent(wasted_mwh_day, potential_mwh_day):
    """
    Curtailment percentage calculation.

    Formula:
        Curtailment % = (Wasted Energy / Potential Energy) × 100

    Args:
        wasted_mwh_day    : curtailed energy (MWh/day)
        potential_mwh_day : total possible generation (MWh/day)
    """
    if potential_mwh_day <= 0:
        return 0.0

    curtailment_pct = (wasted_mwh_day / potential_mwh_day) * 100

    return round(max(0.0, min(curtailment_pct, 100.0)), 2)
