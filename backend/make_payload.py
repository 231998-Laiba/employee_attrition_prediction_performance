"""Create backend/test_payload.json from data/sample_inputs.csv (one non-sensitive record)."""
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
features = bundle["features"]
df = pd.read_csv(ROOT / "data/sample_inputs.csv", nrows=1)
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Sample CSV missing columns: {missing}")
row = df.loc[0, features]
payload = {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in row.items()}
out = ROOT / "backend/test_payload.json"
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Created {out} with {len(payload)} features")
