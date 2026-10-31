import pandas as pd
from meditrust.config import FEATURES

RANGES = {"age": (0, 120), "heart_rate": (20, 250), "systolic_bp": (50, 280), "diastolic_bp": (30, 180), "temperature": (30, 45), "oxygen_saturation": (50, 100), "glucose": (20, 700)}

def validate_dataset(df: pd.DataFrame):
    reasons, statuses = [], []
    duplicates = df.duplicated("patient_id", keep=False)
    for i, row in df.iterrows():
        item = []
        missing = [c for c in FEATURES + ["triage_risk", "patient_id"] if pd.isna(row.get(c))]
        if missing: item.append("missing:" + ",".join(missing))
        for col, (lo, hi) in RANGES.items():
            val = row.get(col)
            if pd.notna(val) and (not isinstance(val, (int, float)) or val < lo or val > hi): item.append(f"invalid_{col}")
        if row.get("triage_risk") not in (0, 1): item.append("invalid_target")
        if duplicates.loc[i]: item.append("duplicate_patient_id")
        # Extreme-but-physiologically possible readings are review, malformed values rejected.
        rejected = any(x.startswith(("missing", "invalid_")) for x in item)
        statuses.append("REJECTED" if rejected else ("NEEDS REVIEW" if item else "VALID"))
        reasons.append("; ".join(item))
    result = df.copy(); result["validation_status"] = statuses; result["validation_reasons"] = reasons
    summary = {"total": len(result), "valid": statuses.count("VALID"), "review": statuses.count("NEEDS REVIEW"), "rejected": statuses.count("REJECTED")}
    return result, summary
