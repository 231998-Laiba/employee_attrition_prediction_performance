"""Smoke test against a RUNNING Flask server (start it first: python backend/app.py)."""
import json
from pathlib import Path

import requests

payload = json.loads((Path(__file__).resolve().parent / "test_payload.json").read_text(encoding="utf-8"))
with requests.Session() as session:
    session.trust_env = False  # ignore proxies for localhost
    response = session.post("http://127.0.0.1:5000/predict", json=payload, timeout=60)
print("HTTP Status:", response.status_code)
print("Response:", response.text)
response.raise_for_status()
print("PASS: API request")
