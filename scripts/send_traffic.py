"""Generate sample traffic so Grafana/Prometheus have data to show."""
import random
import time

import urllib.request
import json

URL = "http://localhost:8001/predict"
for i in range(200):
    body = {"age": random.randint(35, 75), "sex": random.randint(0, 1),
            "cp": random.randint(1, 4), "trestbps": random.randint(100, 180),
            "chol": random.randint(150, 400), "fbs": random.randint(0, 1),
            "restecg": random.randint(0, 2), "thalach": random.randint(90, 190),
            "exang": random.randint(0, 1), "oldpeak": round(random.uniform(0, 4), 1),
            "slope": random.randint(1, 3), "ca": random.randint(0, 3),
            "thal": random.choice([3, 6, 7])}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    print(i, urllib.request.urlopen(req).read().decode())
    time.sleep(0.2)
