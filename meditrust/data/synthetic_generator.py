"""Deterministic, non-identifying synthetic triage data."""
import numpy as np
import pandas as pd
from meditrust.config import FEATURES

def generate_hospital_data(hospital_id: str, shift: float = 0.0, n: int = 240, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed + ord(hospital_id[0]) * 31)
    age = np.clip(rng.normal(52 + shift * 8, 17, n), 18, 96)
    heart_rate = np.clip(rng.normal(78 + shift * 5, 13, n), 42, 170)
    systolic = np.clip(rng.normal(123 + shift * 9, 18, n), 82, 220)
    diastolic = np.clip(rng.normal(78 + shift * 5, 11, n), 45, 135)
    temperature = np.clip(rng.normal(36.9 + shift * .12, .55, n), 34.5, 41.5)
    oxygen = np.clip(rng.normal(96.5 - shift * .6, 2.0, n), 75, 100)
    glucose = np.clip(rng.normal(112 + shift * 12, 28, n), 55, 330)
    linear = ((age - 50)/18 + (heart_rate - 80)/16 + (systolic - 125)/22 +
              (temperature - 37)/.65 + (94 - oxygen)/2.2 + (glucose - 115)/42 - 1.0)
    risk = (linear + rng.normal(0, .65, n) > 0).astype(int)
    df = pd.DataFrame(dict(patient_id=[f"SYN-{hospital_id}-{i:04d}" for i in range(n)], age=age.round(1),
        heart_rate=heart_rate.round(1), systolic_bp=systolic.round(1), diastolic_bp=diastolic.round(1),
        temperature=temperature.round(2), oxygen_saturation=oxygen.round(1), glucose=glucose.round(1), triage_risk=risk))
    return df

def generate_validation_set(n: int = 500, seed: int = 3000) -> pd.DataFrame:
    return generate_hospital_data("V", 0.08, n, seed)

def add_malformed_rows(df: pd.DataFrame) -> pd.DataFrame:
    bad = df.head(3).copy()
    bad.loc[bad.index[0], "age"] = 150
    bad.loc[bad.index[1], "oxygen_saturation"] = -5
    bad.loc[bad.index[2], "heart_rate"] = np.nan
    return pd.concat([df, bad], ignore_index=True)
