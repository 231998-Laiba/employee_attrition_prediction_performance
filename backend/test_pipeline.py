"""Direct (no-API) inference check against the saved bundle."""
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
payload = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))
features = bundle["features"]
assert set(features) == set(payload), "Payload keys must match features"
frame = pd.DataFrame([[payload[f] for f in features]], columns=features)
model = bundle["pipeline"]
print("Prediction:", model.predict(frame).tolist())
if hasattr(model, "predict_proba"):
    print("Probabilities:", model.predict_proba(frame).tolist())
print("PASS: local inference")
