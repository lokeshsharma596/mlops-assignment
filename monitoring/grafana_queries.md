# Grafana panels to create (Dashboards -> New -> Add visualization)

| Panel | PromQL |
|---|---|
| Request rate | `sum(rate(api_requests_total[1m])) by (path)` |
| p95 latency | `histogram_quantile(0.95, sum(rate(api_request_latency_seconds_bucket[1m])) by (le, path))` |
| Error rate (5xx/4xx) | `sum(rate(api_requests_total{status=~"4..\|5.."}[1m]))` |
| Predictions by class | `sum(model_predictions_total) by (label)` |
| Avg confidence | `rate(model_prediction_confidence_sum[5m]) / rate(model_prediction_confidence_count[5m])` |
| Model loaded | `model_loaded` |
