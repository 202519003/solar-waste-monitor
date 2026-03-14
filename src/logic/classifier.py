import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def prepare_features(clean_df):
    """
    Builds the feature table needed to train the ML model.
    Groups cleaned CEA data by region and computes:
        - avg_solar_mwh        : average daily solar generation
        - avg_demand_mwh       : average daily demand
        - avg_curtailment_pct  : estimated curtailment percentage
        - avg_coal_mwh         : average coal generation (proxy for grid inflexibility)

    Args:
        clean_df : DataFrame — output of data_cleaner.py (india_generation_clean.csv)

    Returns:
        features_df : DataFrame — one row per region with computed features
    """
    # Average daily values per region
    grouped = clean_df.groupby('region').agg(
        state            = ('state', 'first'),
        lat              = ('lat', 'first'),
        lon              = ('lon', 'first'),
        avg_solar_mwh    = ('solar_mwh', 'mean'),
        avg_demand_mwh   = ('demand_mwh', 'mean'),
        avg_coal_mwh     = ('coal_mwh', 'mean'),
    ).reset_index()

    # Curtailment % = how much of demand is already met by solar
    # Higher solar share + high coal = more likely curtailment
    grouped['solar_share_pct'] = (
        grouped['avg_solar_mwh'] / grouped['avg_demand_mwh'] * 100
    ).clip(0, 100).fillna(0)

    # Coal dependency % = how much demand is still covered by coal
    grouped['coal_share_pct'] = (
        grouped['avg_coal_mwh'] / grouped['avg_demand_mwh'] * 100
    ).clip(0, 100).fillna(0)

    # Curtailment risk proxy: high solar share + high coal = high risk of curtailment
    grouped['avg_curtailment_pct'] = (
        grouped['solar_share_pct'] * 0.6 + grouped['coal_share_pct'] * 0.4
    ).round(2)

    return grouped


def train_model(features_df):
    """
    Two-step ML pipeline:

    Step 1 — KMeans (unsupervised):
        We have no pre-labelled data saying which region is "High risk".
        KMeans finds 3 natural clusters automatically based on curtailment features.

    Step 2 — Decision Tree (supervised):
        We label the KMeans clusters as Low / Medium / High by their mean curtailment.
        Then we train a Decision Tree on those labels so we can predict new inputs.

    Why KMeans first?
        Decision Tree needs labels to train on. We don't have them.
        KMeans creates the labels for us. Then Decision Tree learns the rules.

    Args:
        features_df : DataFrame — output of prepare_features()

    Returns:
        clf     : trained DecisionTreeClassifier
        scaler  : fitted StandardScaler (needed to scale new inputs before predicting)
        df      : features_df with added columns: cluster, risk_label
    """
    df = features_df.copy()

    feature_cols = ['avg_curtailment_pct', 'solar_share_pct', 'coal_share_pct']
    X = df[feature_cols].values

    # ── STEP 1: Scale features ───────────────────────────────────────────────
    # KMeans is distance-based — features must be on same scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── STEP 2: KMeans clustering ────────────────────────────────────────────
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # ── STEP 3: Label clusters Low / Medium / High ───────────────────────────
    # Sort clusters by their mean curtailment percentage
    cluster_means = df.groupby('cluster')['avg_curtailment_pct'].mean().sort_values()
    label_map = {
        cluster_means.index[0]: 'Low',
        cluster_means.index[1]: 'Medium',
        cluster_means.index[2]: 'High'
    }
    df['risk_label'] = df['cluster'].map(label_map)

    # ── STEP 4: Train Decision Tree on KMeans labels ─────────────────────────
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_scaled, df['risk_label'])

    print('Model trained successfully.')
    print(f'Regions and their risk labels:')
    print(df[['region', 'state', 'avg_curtailment_pct', 'risk_label']].to_string(index=False))
    print()
    print('Decision Tree rules:')
    print(classification_report(df['risk_label'], clf.predict(X_scaled)))

    return clf, scaler, df


def predict_risk(clf, scaler, curtailment_pct, solar_share_pct, coal_share_pct):
    """
    Predicts curtailment risk for a new region/state.

    Args:
        clf              : trained DecisionTreeClassifier (from train_model)
        scaler           : fitted StandardScaler (from train_model)
        curtailment_pct  : float — estimated curtailment percentage
        solar_share_pct  : float — solar as % of total demand
        coal_share_pct   : float — coal as % of total demand

    Returns:
        risk : str — 'Low', 'Medium', or 'High'
    """
    features = np.array([[curtailment_pct, solar_share_pct, coal_share_pct]])
    features_scaled = scaler.transform(features)
    risk = clf.predict(features_scaled)
    return risk[0]


def get_risk_color(risk_label):
    """
    Returns a Folium-compatible colour for the choropleth map.

    Args:
        risk_label : str — 'Low', 'Medium', or 'High'

    Returns:
        color : str — hex colour code
    """
    colors = {
        'Low':    '#1D9E75',   # green
        'Medium': '#EF9F27',   # amber
        'High':   '#E24B4A',   # red
    }
    return colors.get(risk_label, '#888780')  # gray as fallback


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == '__main__':
    import os

    clean_path = 'data/india_generation_clean.csv'

    if not os.path.exists(clean_path):
        print(f'File not found: {clean_path}')
        print('Run data_cleaner.py first to generate the cleaned CSV.')
    else:
        df = pd.read_csv(clean_path, parse_dates=['date'])
        features = prepare_features(df)
        clf, scaler, labelled = train_model(features)

        print('\nTest prediction:')
        test_risk = predict_risk(clf, scaler,
                                 curtailment_pct=45.0,
                                 solar_share_pct=30.0,
                                 coal_share_pct=60.0)
        print(f'Input: curtailment=45%, solar_share=30%, coal_share=60%')
        print(f'Predicted risk: {test_risk}')
        print(f'Map colour: {get_risk_color(test_risk)}')