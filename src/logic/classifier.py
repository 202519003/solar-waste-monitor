"""
classifier.py  —  ML Risk Classification
=========================================
Responsibilities (Dhruv Soni):
  train_model()    → KMeans k=3 clustering + Decision Tree training
  predict_risk()   → predicts Low / Medium / High curtailment risk for a state
  get_risk_color() → returns Streamlit status type for the risk badge

Pipeline:
  Step 1 — Collect features: curtailment_pct + avg_temperature per state
  Step 2 — StandardScaler normalises both features
  Step 3 — KMeans (k=3) clusters states into 3 groups (unsupervised)
  Step 4 — Clusters are labelled Low/Medium/High by curtailment centroid
  Step 5 — Decision Tree trained on cluster labels (supervised, for prediction)
  Step 6 — predict_risk() uses the trained tree to classify new inputs
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

from src.logic.calculator import compute_all_states, _fallback_ghi
from src.logic.fetcher import fetch_ghi


# ─────────────────────────────────────────────────────────────────────────────
# AVERAGE TEMPERATURE LOOKUP  (NASA POWER T2M annual avg, °C per state)
# ─────────────────────────────────────────────────────────────────────────────

AVG_TEMP_C = {
    "Rajasthan": 28.5, "Gujarat": 27.2, "Karnataka": 26.5,
    "Tamil Nadu": 28.3, "Andhra Pradesh": 27.8, "Telangana": 28.0,
    "Maharashtra": 26.8, "Madhya Pradesh": 25.5, "Uttar Pradesh": 24.0,
    "Punjab": 22.5, "Haryana": 23.5, "Odisha": 26.5,
    "Chhattisgarh": 25.8, "Kerala": 27.5, "Bihar": 24.5,
    "West Bengal": 26.0, "Assam": 23.8, "Himachal Pradesh": 12.5,
    "Uttarakhand": 14.0, "Jharkhand": 24.5, "Goa": 27.8,
    "Manipur": 20.0, "Meghalaya": 18.5, "Tripura": 24.5,
    "Nagaland": 19.0, "Mizoram": 21.0, "Arunachal Pradesh": 17.0,
    "Sikkim": 10.5,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(year: int = 2024, use_api: bool = False) -> dict:
    """
    Trains the KMeans + Decision Tree pipeline.

    Parameters
    ----------
    year    : int   Year to base training data on
    use_api : bool  If True, fetches GHI from NASA POWER API (slower)
                    If False, uses fallback GHI (fast, good for testing)

    Returns
    -------
    dict with keys:
        scaler, kmeans, tree, label_map, summary_df, feature_df
    """
    print("[classifier] Computing curtailment for all states ...")
    summary_df = compute_all_states(year=year, use_api=use_api)

    # Build feature matrix
    states = summary_df["State"].tolist()
    curtailment = summary_df["curtailment_pct"].values
    temperature = np.array([AVG_TEMP_C.get(s, 25.0) for s in states])

    X = np.column_stack([curtailment, temperature])

    # Step 1: Normalise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: KMeans clustering (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Step 3: Label clusters by curtailment centroid
    # Cluster with highest curtailment centroid → "High Risk"
    centroids = kmeans.cluster_centers_
    # centroids[:,0] is normalised curtailment — rank them
    centroid_curtailment = [centroids[i][0] for i in range(3)]
    order = np.argsort(centroid_curtailment)  # low → high
    label_map = {
        order[0]: "Low",
        order[1]: "Medium",
        order[2]: "High",
    }

    risk_labels = [label_map[c] for c in cluster_labels]

    # Step 4: Train Decision Tree on cluster labels
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X_scaled, risk_labels)

    # Save model artifacts
    model_bundle = {
        "scaler":     scaler,
        "kmeans":     kmeans,
        "tree":       tree,
        "label_map":  label_map,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"[classifier] Model saved to {MODEL_PATH}")

    # Add risk labels to summary for map colouring
    feature_df = pd.DataFrame({
        "State":      states,
        "curtailment_pct": curtailment,
        "temperature": temperature,
        "risk_label": risk_labels,
    })

    summary_df = summary_df.merge(feature_df[["State", "risk_label"]], on="State", how="left")

    return {
        "scaler":      scaler,
        "kmeans":      kmeans,
        "tree":        tree,
        "label_map":   label_map,
        "summary_df":  summary_df,
        "feature_df":  feature_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_risk(curtailment_pct: float, state: str,
                 model_bundle: dict = None) -> str:
    """
    Predicts curtailment risk for a given state + curtailment percentage.

    Parameters
    ----------
    curtailment_pct : float   e.g. 35.5
    state           : str     e.g. "Rajasthan"
    model_bundle    : dict    Output of train_model(). If None, loads from disk.

    Returns
    -------
    str   "Low", "Medium", or "High"
    """
    if model_bundle is None:
        if os.path.exists(MODEL_PATH):
            model_bundle = joblib.load(MODEL_PATH)
        else:
            # Train fresh if no saved model
            print("[classifier] No saved model found — training now ...")
            model_bundle = train_model(use_api=False)

    scaler = model_bundle["scaler"]
    tree   = model_bundle["tree"]

    temperature = AVG_TEMP_C.get(state, 25.0)
    X = np.array([[curtailment_pct, temperature]])
    X_scaled = scaler.transform(X)

    prediction = tree.predict(X_scaled)[0]
    return prediction


def get_risk_color(risk_label: str) -> str:
    """Returns Streamlit status type string for st.success / st.warning / st.error."""
    return {
        "Low":    "success",
        "Medium": "warning",
        "High":   "error",
    }.get(risk_label, "info")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing classifier.py (offline mode) ===\n")

    bundle = train_model(year=2024, use_api=False)

    print("\nRisk labels per state:")
    print(bundle["feature_df"][["State", "curtailment_pct", "risk_label"]]
          .sort_values("curtailment_pct", ascending=False)
          .to_string(index=False))

    print("\nPredicting risk for Rajasthan with 40% curtailment:")
    risk = predict_risk(40.0, "Rajasthan", model_bundle=bundle)
    print(f"  → {risk} Risk")

    print("\nPredicting risk for Sikkim with 5% curtailment:")
    risk = predict_risk(5.0, "Sikkim", model_bundle=bundle)
    print(f"  → {risk} Risk")
