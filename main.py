# main.py — SolarWaste Monitor
# Entry point for the entire application
# Run this file to launch the app: python main.py

import subprocess
import sys
import os


def check_dependencies():
    """
    Checks that all required libraries are installed before launching.
    If any are missing, installs them automatically.
    """
    required = [
        'streamlit',
        'requests',
        'pandas',
        'geopandas',
        'scikit-learn',
        'folium',
        'matplotlib',
        'streamlit_folium',
        'geopy',
        'scipy',
        'numpy',
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f'Installing missing packages: {missing}')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install'] + missing,
            check=True
        )
        print('All packages installed.\n')
    else:
        print('All dependencies satisfied.\n')


def check_data_files():
    """
    Checks that required data files exist before launching.
    Warns the user if any are missing.
    """
    required_files = {
        'data/india_generation.csv':       'Download from https://robbieandrew.github.io/india/',
        'data/india_generation_clean.csv': 'Run: python data_cleaner.py',
        'data/india_states.geojson':       'Download from https://github.com/Subhash9325/GeoJson-Data-of-Indian-States',
    }

    all_ok = True
    for filepath, instruction in required_files.items():
        if os.path.exists(filepath):
            print(f'  [OK] {filepath}')
        else:
            print(f'  [MISSING] {filepath}')
            print(f'           → {instruction}')
            all_ok = False

    return all_ok


def main():
    print('=' * 55)
    print('  SolarWaste Monitor — India Solar Curtailment Tracker')
    print('  MSc (AA) / PGD (SDS) — Python Project 2026')
    print('=' * 55)
    print()

    # Step 1 — Check libraries
    print('Checking dependencies...')
    check_dependencies()

    # Step 2 — Check data files
    print('Checking data files...')
    data_ok = check_data_files()
    print()

    if not data_ok:
        print('WARNING: Some data files are missing.')
        print('The app will launch but some features may not work.')
        print('Follow the instructions above to fix missing files.\n')

    # Step 3 — Launch Streamlit dashboard
    dashboard_path = os.path.join('src', 'gui', 'dashboard.py')

    if not os.path.exists(dashboard_path):
        print(f'ERROR: Dashboard not found at {dashboard_path}')
        print('Make sure src/gui/dashboard.py exists.')
        sys.exit(1)

    print('Launching SolarWaste Monitor...')
    print('Open your browser at: http://localhost:8501')
    print('Press Ctrl+C to stop the app.\n')

    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
        '--server.headless', 'false',
        '--browser.gatherUsageStats', 'false',
    ])


if __name__ == '__main__':
    main()
