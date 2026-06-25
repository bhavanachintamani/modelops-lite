"""
Basic API tests - run with: pytest tests/ -v
Assumes train_model.py has already produced model/churn_model.joblib
(the CI workflow does this automatically before running tests).
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    payload = {
        "tenure_months": 12,
        "monthly_charges": 70.5,
        "month_to_month_contract": 1,
        "support_calls_last_year": 3,
        "has_addon_service": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_predict_invalid_input_rejected():
    payload = {
        "tenure_months": -5,  # invalid: below allowed range
        "monthly_charges": 70.5,
        "month_to_month_contract": 1,
        "support_calls_last_year": 3,
        "has_addon_service": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint_after_predictions():
    client.post(
        "/predict",
        json={
            "tenure_months": 24,
            "monthly_charges": 55.0,
            "month_to_month_contract": 0,
            "support_calls_last_year": 1,
            "has_addon_service": 1,
        },
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["prediction_count"] >= 1
