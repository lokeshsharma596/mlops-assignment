import joblib
import pytest
from fastapi.testclient import TestClient

VALID = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
         "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
         "ca": 0, "thal": 6}


@pytest.fixture()
def client(tmp_path, monkeypatch, trained_pipeline):
    model_path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_out_of_range(client):
    assert client.post("/predict", json={**VALID, "sex": 5}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=VALID)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
