"""Task 6/8 - FastAPI model-serving app with request logging and Prometheus metrics."""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# ---------- logging (stdout + optional file) ----------
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
handlers = [logging.StreamHandler()]
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(LOG_DIR / "api.log"))
except OSError:  # read-only filesystem etc.
    pass
logging.basicConfig(level=logging.INFO, handlers=handlers,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart-api")

# ---------- Prometheus metrics ----------
REQUESTS = Counter("api_requests_total", "Total HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["label"])
CONFIDENCE = Histogram("model_prediction_confidence", "Prediction confidence",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
MODEL_LOADED = Gauge("model_loaded", "1 if model is loaded")

state = {"model": None, "meta": {}}


class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, description="1=male, 0=female", examples=[1])
    cp: int = Field(..., ge=1, le=4, description="chest pain type", examples=[1])
    trestbps: float = Field(..., ge=50, le=300, description="resting blood pressure",
                            examples=[145])
    chol: float = Field(..., ge=50, le=700, description="serum cholesterol mg/dl", examples=[233])
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120", examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[2])
    thalach: float = Field(..., ge=40, le=260, description="max heart rate", examples=[150])
    exang: int = Field(..., ge=0, le=1, description="exercise induced angina", examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[2.3])
    slope: int = Field(..., ge=1, le=3, examples=[3])
    ca: int = Field(..., ge=0, le=4, description="major vessels coloured", examples=[0])
    thal: int = Field(..., ge=3, le=7, description="3=normal, 6=fixed, 7=reversible", examples=[6])


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    try:
        state["model"] = joblib.load(model_path)
        meta_path = model_path.with_name("metadata.json")
        if meta_path.exists():
            state["meta"] = json.loads(meta_path.read_text())
        MODEL_LOADED.set(1)
        logger.info("Model loaded from %s", model_path)
    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("Could not load model from %s", model_path)
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("req=%s unhandled error on %s", req_id, request.url.path)
        raise
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUESTS.labels(request.method, path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    logger.info("req=%s %s %s -> %s in %.1fms", req_id, request.method, path,
                response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok" if state["model"] is not None else "model_not_loaded",
            "model": state["meta"].get("model_name")}


@app.get("/model-info")
def model_info():
    return state["meta"]


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientData):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = pd.DataFrame([patient.model_dump()])[FEATURES]
    p = float(state["model"].predict_proba(X)[0, 1])
    pred = int(p >= 0.5)
    conf = p if pred else 1 - p
    label = "heart disease" if pred else "no heart disease"
    PREDICTIONS.labels(label).inc()
    CONFIDENCE.observe(conf)
    logger.info("prediction=%s confidence=%.3f input=%s", pred, conf, patient.model_dump())
    return Prediction(prediction=pred, label=label, confidence=round(conf, 4),
                      probability_disease=round(p, 4))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
