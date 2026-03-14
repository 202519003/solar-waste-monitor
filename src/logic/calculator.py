# Source: CEA CO2 Baseline Database Version 21.0, December 2025
# Official India grid emission factor
CO2_PER_KWH = 0.727  # kg of CO2 released per kWh of coal electricity

# Official curtailment compensation rate (Rs per kWh paid to solar plants)
COMPENSATION_RATE = 3.0  # Rs per kWh

# Average solar panel efficiency (industry standard for utility-scale plants)
PANEL_EFFICIENCY = 0.15  # 15%


def calculate_curtailment(potential_ghi, capacity_mw, actual_mw):
    """
    Calculates how much solar energy was wasted (curtailed) in a state.

    Args:
        potential_ghi  : float — sunlight value from NASA POWER (kWh/m2/day)
        capacity_mw    : float — installed solar capacity in the state (MW)
        actual_mw      : float — what CEA recorded as actually generated (MW)

    Returns:
        wasted_mw      : float — difference between potential and actual (MW)

    Formula:
        potential_mw = GHI x capacity_mw x panel_efficiency
        wasted_mw    = potential_mw - actual_mw  (minimum 0, never negative)
    """
    potential_mw = potential_ghi * capacity_mw * PANEL_EFFICIENCY
    wasted_mw = max(0, potential_mw - actual_mw)
    return round(wasted_mw, 2)


def calculate_losses(wasted_mw, hours=24):
    """
    Converts wasted MW into kWh, money lost (Rs), and CO2 released (kg).

    Args:
        wasted_mw  : float — curtailed energy in MW (from calculate_curtailment)
        hours      : int   — number of hours in the period (default 24 = one day)

    Returns:
        dict with keys:
            wasted_kwh  — energy wasted in kilowatt-hours
            money_rs    — money lost in Indian Rupees
            co2_kg      — CO2 released in kilograms
    """
    wasted_kwh = wasted_mw * 1000 * hours  # MW → kW (×1000) × hours = kWh

    money_lost_rs = wasted_kwh * COMPENSATION_RATE
    co2_lost_kg = wasted_kwh * CO2_PER_KWH

    return {
        'wasted_kwh': round(wasted_kwh, 2),
        'money_rs':   round(money_lost_rs, 2),
        'co2_kg':     round(co2_lost_kg, 2)
    }


def calculate_curtailment_percent(potential_mw, actual_mw):
    """
    Calculates curtailment as a percentage of potential generation.
    Used by the ML classifier as the main input feature.

    Args:
        potential_mw : float — what could have been generated
        actual_mw    : float — what was actually generated

    Returns:
        curtailment_pct : float — percentage of potential energy that was wasted
    """
    if potential_mw == 0:
        return 0.0
    curtailment_pct = ((potential_mw - actual_mw) / potential_mw) * 100
    return round(max(0, curtailment_pct), 2)