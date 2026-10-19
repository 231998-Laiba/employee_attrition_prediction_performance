"""Employee Attrition Prediction API (Flask).

Endpoints
  GET  /health   liveness + model info
  GET  /schema   input schema (names, types, ranges, allowed categories)
  GET  /report   training metrics and top features
  POST /predict  JSON object with all raw features -> prediction, probability, risk level, top drivers
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models" / "model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
FEATURES: list[str] = BUNDLE["features"]
TASK = BUNDLE.get("task", "classification")
SCHEMA = {f["name"]: f for f in BUNDLE["schema"]}
REFERENCE = BUNDLE["reference"]
REPORT_PATH = BASE.parent / "docs" / "model_report.json"

LOW, HIGH = 0.35, 0.60  # risk bands on the predicted probability of leaving
DISCLAIMER = "Educational prototype. Use as decision support, never as the sole basis for HR decisions."

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def validate(data: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems = []
    for name, spec in SCHEMA.items():
        value = data[name]
        if spec["kind"] == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{name}: must be a number")
            elif not (spec["min"] <= value <= spec["max"]):
                problems.append(f"{name}: {value} outside allowed range {spec['min']} to {spec['max']}")
        else:
            allowed = [o["value"] for o in spec["options"]]
            if value not in allowed:
                problems.append(f"{name}: '{value}' not one of {allowed}")
    return problems


def risk_level(p: float) -> str:
    return "Low" if p < LOW else "Medium" if p < HIGH else "High"


def top_drivers(frame: pd.DataFrame, base_prob: float, k: int = 5) -> list[dict]:
    """Model-agnostic local explanation: reset one feature at a time to a typical value
    and measure how much the leaving-probability changes."""
    variants = []
    for f in FEATURES:
        row = frame.iloc[0].copy()
        row[f] = REFERENCE[f]
        variants.append(row)
    probs = MODEL.predict_proba(pd.DataFrame(variants, columns=FEATURES))[:, 1]
    impacts = [(f, base_prob - p) for f, p in zip(FEATURES, probs)]
    impacts.sort(key=lambda t: abs(t[1]), reverse=True)
    return [{"feature": f, "label": SCHEMA[f]["label"], "value": frame.iloc[0][f].item() if hasattr(frame.iloc[0][f], "item") else frame.iloc[0][f],
             "impact": round(float(d), 4), "direction": "raises risk" if d > 0 else "lowers risk"}
            for f, d in impacts[:k] if abs(d) > 1e-4]


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK, model=BUNDLE.get("model_name"),
                   model_version=BUNDLE.get("model_version"))


@app.get("/schema")
def schema():
    return jsonify(features=BUNDLE["schema"], class_labels=BUNDLE["class_labels"])


@app.get("/report")
def report():
    if not REPORT_PATH.exists():
        return jsonify(error="Report not generated yet. Run python ml/train.py"), 404
    return jsonify(json.loads(REPORT_PATH.read_text(encoding="utf-8")))


@app.post("/predict")
def predict():
    start = time.perf_counter()
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400
    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch", missing=missing, extra=extra), 400
    problems = validate(data)
    if problems:
        return jsonify(error="Invalid input", details=problems), 400

    frame = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    try:
        # --- inference block: replace for custom model architectures ---
        prob = float(MODEL.predict_proba(frame)[0, 1])
        label = int(MODEL.predict(frame)[0])
        result = {
            "prediction": BUNDLE["class_labels"][label],
            "prediction_code": label,
            "task": TASK,
            "probability_leave": round(prob, 4),
            "probabilities": [round(1 - prob, 4), round(prob, 4)],
            "risk_level": risk_level(prob),
            "drivers": top_drivers(frame, prob),
            "model": BUNDLE.get("model_name"),
            "model_version": BUNDLE.get("model_version"),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "disclaimer": DISCLAIMER,
        }
        app.logger.info("predict -> %s (p=%.3f, %s)", result["prediction"], prob, result["risk_level"])
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:  # noqa: BLE001
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
