from pathlib import Path

SEED = 2026
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FEATURES = ["age", "bp_systolic", "bp_diastolic", "glucose", "troponin", "heart_rate", "spo2", "crp"]
TARGET = "target"
HOSPITALS = [
    ("A", "General Hospital", "General medicine", "node-1_general_hospital.csv"),
    ("B", "Urban Medical Center", "Emergency care", "node-2_urban_medical_center.csv"),
    ("C", "Regional Hospital", "Regional care", "node-3_regional_hospital.csv"),
    ("D", "Community Hospital", "Community care", "node-4_community_hospital.csv"),
    ("E", "Specialty Medical Center", "Cardiology", "node-5_specialty_medical_center.csv"),
]
HELDOUT_FILE = "heldout_validation_cohort.csv"
DB_PATH = ROOT / "meditrust.db"
TOP_K = 6
TRIM_FRACTION = .2
MAX_F1_DROP = .08
