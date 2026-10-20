# Heart Disease Risk - End-to-End MLOps Project

Pipeline: UCI data -> EDA -> sklearn pipelines -> MLflow tracking -> packaged model -> FastAPI -> Docker -> Kubernetes -> Prometheus/Grafana, with GitHub Actions CI.

## 0. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Data + EDA  (Task 1)
```bash
python scripts/download_data.py      # -> data/raw/heart_disease_raw.csv
python -m src.data                   # -> data/processed/heart_disease_clean.csv
python -m src.eda                    # -> reports/figures/*.png
```

## 2-4. Train, track, package  (Tasks 2-4)
```bash
python -m src.train                  # 3 models, GridSearchCV (5-fold), logs to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
python -m src.predict                # CLI inference with a sample row
```
Outputs: `models/model.joblib` (full preprocessing + model pipeline), `models/metadata.json`.

## 5. Tests + CI  (Task 5)
```bash
flake8 src app tests scripts
pytest -v
```
`.github/workflows/ci.yml`: lint -> test -> train (uploads model, MLflow DB, figures) -> docker build + smoke test. Any failure stops the pipeline.

## 6. Docker  (Task 6)
```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Swagger UI: http://localhost:8000/docs

## 7. Kubernetes  (Task 7)
```bash
minikube start
eval $(minikube docker-env) && docker build -t heart-api:latest .   # Docker Desktop: skip the eval
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
minikube tunnel        # separate terminal; then use the EXTERNAL-IP of svc/heart-api
```
Optional ingress: `minikube addons enable ingress && kubectl apply -f k8s/ingress.yaml`.

## 8. Monitoring  (Task 8)
```bash
docker compose up --build            # API :8000, Prometheus :9090, Grafana :3000 (admin/admin)
python scripts/send_traffic.py       # generate load
```
Request logs go to stdout and `logs/api.log`. Metrics at `/metrics`; panel queries in `monitoring/grafana_queries.md`.

## Screenshots to capture for the report
MLflow runs table + comparison, pytest output, GitHub Actions green run, `docker run` + curl, `kubectl get pods,svc`, Prometheus targets, Grafana dashboard.
