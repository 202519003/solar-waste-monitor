# calculator.py — SolarWaste Monitor
# Source: CEA CO2 Baseline Database Version 21.0, December 2025

# India grid emission factor (official CEA value)
CO2_PER_KWH = 0.727          # kg CO2 per kWh

# Solar curtailment compensation rate (SECI/MNRE 2024-25 average discovered tariff)
COMPENSATION_RATE = 2.55      # Rs per kWh

# Average peak sun hours for India (used to convert GHI into capacity factor)
PEAK_SUN_HOURS = 5.5          # hours/day (India average across states)

# Base capacity factor for utility-scale solar in India
BASE_CAPACITY_FACTOR = 0.20   # 20% — standard for Indian utility-scale plants


def calculate_curtailment(potential_ghi, capacity_mw, actual_mwh_day):
    """
    Calculates solar potential and curtailed energy per day.

    Args:
        potential_ghi   : GHI from NASA POWER (kWh/m²/day)
        capacity_mw     : Installed solar capacity (MW)
        actual_mwh_day  : Actual generation from CEA data (MWh/day)

    Returns:
        potential_mwh_day : Total possible solar generation (MWh/day)
        wasted_mwh_day    : Curtailed energy (MWh/day)

    Formula:
        capacity_factor = (GHI / PEAK_SUN_HOURS) × BASE_CAPACITY_FACTOR
            — GHI scales the base CF: more sun = higher fraction of rated capacity used
        potential_mwh_day = capacity_mw × 24 hours × capacity_factor
            — total energy the plant could have produced in a full day

    Why the old formula was wrong:
        OLD: potential = GHI × capacity_mw × 0.15
             GHI is per m², capacity_mw is total plant MW — units don't match.
             Result was ~34,000 MWh for 35,000 MW plant, far below real actual
             generation, so wasted = max(0, negative) = 0 always.

    Example (Rajasthan):
        GHI=6.5, capacity=18000 MW, actual=50000 MWh/day
        capacity_factor = (6.5/5.5) × 0.20 = 0.236
        potential = 18000 × 24 × 0.236 = 101,890 MWh
        wasted   = 101,890 − 50,000    =  51,890 MWh  ✓
    """
    # Scale capacity factor by today's GHI vs average peak sun hours
    capacity_factor = (potential_ghi / PEAK_SUN_HOURS) * BASE_CAPACITY_FACTOR

    # Total energy the installed capacity could generate in 24 hours
    potential_mwh_day = capacity_mw * 24 * capacity_factor

    # Curtailment cannot be negative
    wasted_mwh_day = max(0.0, potential_mwh_day - actual_mwh_day)

    return round(potential_mwh_day, 2), round(wasted_mwh_day, 2)


def calculate_losses(wasted_mwh_day):
    """
    Converts curtailed energy (MWh/day) into money and CO2 losses.

    Args:
        wasted_mwh_day : float — curtailed energy in MWh/day

    Returns:
        dict with keys:
            wasted_kwh  : energy wasted (kWh/day)
            money_rs    : revenue lost (Rs/day)
            co2_kg      : CO2 emitted unnecessarily (kg/day)
    """
    # MWh → kWh
    wasted_kwh = wasted_mwh_day * 1000

    money_lost_rs = wasted_kwh * COMPENSATION_RATE
    co2_lost_kg   = wasted_kwh * CO2_PER_KWH

    return {
        'wasted_kwh': round(wasted_kwh, 2),
        'money_rs':   round(money_lost_rs, 2),
        'co2_kg':     round(co2_lost_kg, 2),
    }


def calculate_curtailment_percent(wasted_mwh_day, potential_mwh_day):
    """
    Curtailment percentage.

    Args:
        wasted_mwh_day    : curtailed energy (MWh/day)
        potential_mwh_day : total possible energy (MWh/day)

    Formula:
        Curtailment % = (Wasted / Potential) × 100
    """
    if potential_mwh_day <= 0:
        return 0.0

    curtailment_pct = (wasted_mwh_day / potential_mwh_day) * 100

    # Cap at 100% — cannot curtail more than potential
    return round(max(0.0, min(curtailment_pct, 100.0)), 2)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Rajasthan example values
    GHI          = 6.5      # kWh/m²/day  (from NASA POWER)
    CAPACITY     = 18_000   # MW installed
    ACTUAL       = 50_000   # MWh/day     (from CEA)

    potential, wasted = calculate_curtailment(GHI, CAPACITY, ACTUAL)
    losses            = calculate_losses(wasted)
    pct               = calculate_curtailment_percent(wasted, potential)

    print(f"Potential   : {potential:>12,.2f} MWh/day")
    print(f"Wasted      : {wasted:>12,.2f} MWh/day")
    print(f"Curtailment : {pct:>11.2f} %")
    print(f"Energy lost : {losses['wasted_kwh']:>12,.2f} kWh/day")
    print(f"Money lost  : Rs {losses['money_rs']:>10,.2f} /day")
    print(f"CO2 released: {losses['co2_kg']:>12,.2f} kg/day")

    # Expected output (approximately):
    # Potential   :   101,890.91 MWh/day
    # Wasted      :    51,890.91 MWh/day
    # Curtailment :        50.92 %
    # Energy lost :  51,890,909.09 kWh/day
    # Money lost  : Rs 1,32,321,818.18 /day
    # CO2 released: 37,724,590.91 kg/day
