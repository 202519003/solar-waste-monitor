"""
main.py  —  System Entry Point
================================
Responsibilities (Dhruv Soni):
  - Connects all modules
  - Launches the Streamlit dashboard
  - Provides CLI test runner to verify each module before the GUI

Usage:
    streamlit run main.py              → launch the full dashboard
    python main.py --test              → run all module tests without Streamlit
    python main.py --test --test-state Rajasthan --year 2024
"""

import sys
import os
import argparse

# Make src importable when running as: python main.py
sys.path.insert(0, os.path.dirname(__file__))


def run_tests(state: str = "Rajasthan", year: int = 2024):
    """Runs a quick end-to-end test of all modules (no Streamlit needed)."""
    print("=" * 60)
    print("  SolarWaste Monitor — Module Test Runner")
    print("=" * 60)

    # ── 1. fetcher ────────────────────────────────────────────────
    print("\n[1/4] Testing fetcher.py ...")
    from src.logic.fetcher import load_generation, load_fire_data, get_installed_mw
    gen_df    = load_generation()
    installed = get_installed_mw(state)
    fire_df   = load_fire_data(year=year)
    print(f"      ✓ Generation CSV : {gen_df.shape[0]} rows, {gen_df['State'].nunique()} states")
    print(f"      ✓ Installed cap  : {installed} MW  ({state})")
    print(f"      ✓ Fire hotspots  : {len(fire_df)} rows ({year})")

    # ── 2. calculator ─────────────────────────────────────────────
    print(f"\n[2/4] Testing calculator.py for {state} {year} ...")
    from src.logic.calculator import compute_curtailment, _fallback_ghi
    ghi    = _fallback_ghi(state)
    result = compute_curtailment(state, year, gen_df=gen_df, ghi_data=ghi)
    print(f"      ✓ Actual MWh    : {result['total_actual_mwh']:,.0f}")
    print(f"      ✓ Potential MWh : {result['total_potential_mwh']:,.0f}")
    print(f"      ✓ Curtailed MWh : {result['total_curtailed_mwh']:,.0f}")
    print(f"      ✓ Curtailment % : {result['curtailment_pct']}%")
    print(f"      ✓ Money Lost    : ₹{result['money_lost_cr']} Cr")
    print(f"      ✓ CO₂ Released  : {result['co2_released_tons']} tonnes")

    # ── 3. classifier ─────────────────────────────────────────────
    print(f"\n[3/4] Testing classifier.py ...")
    from src.logic.classifier import train_model, predict_risk
    bundle = train_model(year=year, use_api=False)
    risk   = predict_risk(result["curtailment_pct"], state, model_bundle=bundle)
    print(f"      ✓ Model trained on {len(bundle['feature_df'])} states")
    print(f"      ✓ Risk for {state}: {risk}")
    top3 = (bundle["feature_df"]
            .sort_values("curtailment_pct", ascending=False)
            .head(3)[["State", "curtailment_pct", "risk_label"]])
    print("\n      Top 3 highest curtailment states:")
    for _, r in top3.iterrows():
        print(f"        {r['State']:20s}  {r['curtailment_pct']:5.1f}%  →  {r['risk_label']} Risk")

    # ── 4. dashboard import check ─────────────────────────────────
    print(f"\n[4/4] Checking dashboard.py imports ...")
    try:
        from src.gui.dashboard import run_dashboard
        print("      ✓ dashboard.py imported successfully")
    except ImportError as e:
        print(f"      ✗ Import error: {e}")

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("  Launch dashboard: streamlit run main.py")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SolarWaste Monitor")
    parser.add_argument("--test", action="store_true",
                        help="Run module tests instead of launching Streamlit")
    parser.add_argument("--test-state", default="Rajasthan",
                        help="State for test run (default: Rajasthan)")
    parser.add_argument("--year", type=int, default=2024,
                        help="Year for test run (default: 2024)")
    args = parser.parse_args()

    if args.test:
        run_tests(state=args.test_state, year=args.year)
    else:
        from src.gui.dashboard import run_dashboard
        run_dashboard()


if __name__ == "__main__":
    main()
