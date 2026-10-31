from pathlib import Path

SEED = 2026
FEATURES = ["age", "heart_rate", "systolic_bp", "diastolic_bp", "temperature", "oxygen_saturation", "glucose"]
HOSPITALS = [
    ("A", "General Hospital", 0.00),
    ("B", "Urban Medical Center", 0.25),
    ("C", "Regional Hospital", -0.20),
    ("D", "Community Hospital", 0.10),
    ("E", "Specialty Medical Center", 0.45),
]
DB_PATH = Path("meditrust.db")
TOP_K = 8
TRIM_FRACTION = 0.2
MAX_F1_DROP = 0.08
