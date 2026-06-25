"""
app.py
------
FastAPI service that serves the churn model, with:
- input validation (pydantic)
- structured prediction logging (for monitoring/drift checks)
- a /health endpoint for uptime checks
- a /metrics endpoint exposing basic running stats

Run locally:
    uvicorn app:app --reload --port 8000

Run via Docker:
    docker build -t modelops-lite .
    docker run -p 8000:8000 modelops-lite
"""

import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("modelops-lite")

MODEL_PATH = "model/churn_model.joblib"
LOG_FILE = "logs/predictions.jsonl"
os.makedirs("logs", exist_ok=True)

app = FastAPI(title="ModelOps-Lite: Churn Prediction API", version="1.0.0")

_model_bundle = None
_recent_predictions = deque(maxlen=500)  # in-memory ring buffer for /metrics


class ChurnRequest(BaseModel):
    tenure_months: float = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0, le=500)
    month_to_month_contract: int = Field(..., ge=0, le=1)
    support_calls_last_year: int = Field(..., ge=0, le=50)
    has_addon_service: int = Field(..., ge=0, le=1)


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    model_version: str
    latency_ms: float


def get_model():
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
            )
        _model_bundle = joblib.load(MODEL_PATH)
        logger.info("Model loaded into memory.")
    return _model_bundle


@app.on_event("startup")
def startup_event():
    get_model()


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    start = time.perf_counter()
    bundle = get_model()
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    row = np.array([[getattr(request, f) for f in feature_names]])

    try:
        prob = float(model.predict_proba(row)[0, 1])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    pred = int(prob >= 0.5)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request.dict(),
        "churn_probability": prob,
        "churn_prediction": pred,
        "latency_ms": latency_ms,
    }
    _recent_predictions.append(log_entry)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.info(f"Prediction made: prob={prob:.3f} pred={pred} latency={latency_ms}ms")

    return ChurnResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        model_version="rf-v1",
        latency_ms=latency_ms,
    )


@app.get("/metrics")
def metrics():
    """Lightweight monitoring endpoint: volume + simple drift signal.
    In a real system this would feed Prometheus/Grafana; here it's
    enough to demonstrate the concept of post-deployment monitoring."""
    if not _recent_predictions:
        return {"prediction_count": 0}

    probs = [p["churn_probability"] for p in _recent_predictions]
    latencies = [p["latency_ms"] for p in _recent_predictions]

    return {
        "prediction_count": len(_recent_predictions),
        "avg_churn_probability": round(float(np.mean(probs)), 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "positive_rate": round(float(np.mean([p["churn_prediction"] for p in _recent_predictions])), 4),
    }
