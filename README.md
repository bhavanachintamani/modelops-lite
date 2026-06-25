# ⚙️ ModelOps-Lite — End-to-End MLOps Pipeline for a Churn Model

Most fresher portfolios stop at "I trained a model in a notebook." This project goes further: it takes a churn classifier all the way to a **served, containerized, tested, and CI/CD-automated API** — the part of the ML lifecycle that actually matters in industry.

## Why this project

Companies don't hire AI engineers to train models in isolation — they hire people who can ship models that other systems can call reliably. This project demonstrates the full path: train → serve → containerize → test → automate → monitor.

## What's included

| Stage | Tool | What it shows |
|---|---|---|
| Training | scikit-learn (RandomForest) | Reproducible training script, versioned metrics output |
| Serving | FastAPI | Input validation, structured JSON responses, `/health` check |
| Packaging | Docker | Model + API bundled into a portable, runnable image |
| Testing | pytest | Automated tests for valid/invalid inputs and live endpoints |
| CI/CD | GitHub Actions | Auto-runs tests and builds the Docker image on every push |
| Monitoring | Custom `/metrics` endpoint + JSONL prediction logs | Tracks prediction volume, latency, and a simple drift signal (avg probability over time) |

## Architecture

```
train_model.py → model/churn_model.joblib
                       ↓
                 FastAPI (app.py) → /predict, /health, /metrics
                       ↓
                 Dockerfile → containerized image
                       ↓
            GitHub Actions → test on every push → build image
                       ↓
            logs/predictions.jsonl → drift/monitoring signal
```

## Setup

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python train_model.py          # trains and saves the model
uvicorn app:app --reload       # serves it on http://localhost:8000
```

Test it:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"tenure_months\":12,\"monthly_charges\":70.5,\"month_to_month_contract\":1,\"support_calls_last_year\":3,\"has_addon_service\":0}"
```

Run with Docker instead:
```bash
docker build -t modelops-lite .
docker run -p 8000:8000 modelops-lite
```

Run tests:
```bash
pytest tests/ -v
```

## Results / Metrics (fill in after running train_model.py)

- Accuracy: __  |  F1: __  |  ROC-AUC: __
- Avg prediction latency: __ ms
- CI pipeline run time: __

## Future improvements

- Push built images to Docker Hub / GHCR automatically
- Add real drift detection (e.g., population stability index) instead of the simple average
- Swap synthetic data for your real `customer-churn-predictor` dataset
