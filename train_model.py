"""
train_model.py
---------------
Trains a customer churn classifier and serializes it to disk.
If you already have a churn dataset from your customer-churn-predictor
project, point CSV_PATH at it. Otherwise this generates a realistic
synthetic dataset so the pipeline runs end-to-end out of the box.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

CSV_PATH = os.environ.get("CHURN_CSV_PATH", "")
MODEL_OUT = "model/churn_model.joblib"
METRICS_OUT = "model/metrics.json"


def load_or_generate_data(n=4000, seed=42):
    if CSV_PATH and os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly_charges = rng.normal(65, 25, n).clip(15, 150)
    contract_month_to_month = rng.integers(0, 2, n)
    support_calls = rng.poisson(1.5, n)
    has_addon = rng.integers(0, 2, n)

    churn_score = (
        -0.04 * tenure
        + 0.015 * monthly_charges
        + 0.9 * contract_month_to_month
        + 0.25 * support_calls
        - 0.3 * has_addon
        + rng.normal(0, 1, n)
    )
    churn = (churn_score > np.percentile(churn_score, 75)).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "month_to_month_contract": contract_month_to_month,
            "support_calls_last_year": support_calls,
            "has_addon_service": has_addon,
            "churned": churn,
        }
    )


def main():
    os.makedirs("model", exist_ok=True)
    df = load_or_generate_data()

    X = df.drop(columns=["churned"])
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "feature_names": list(X.columns),
    }

    print(classification_report(y_test, preds))
    print("Metrics:", metrics)

    joblib.dump({"model": clf, "feature_names": list(X.columns)}, MODEL_OUT)

    import json
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {MODEL_OUT}")
    print(f"Metrics saved to {METRICS_OUT}")


if __name__ == "__main__":
    main()
