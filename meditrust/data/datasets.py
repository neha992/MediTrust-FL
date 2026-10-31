"""Local-only dataset discovery; missing source data is always surfaced explicitly."""
from pathlib import Path
import pandas as pd
from meditrust.config import DATA_DIR, FEATURES, HOSPITALS, HELDOUT_FILE, TARGET

REQUIRED = ["patient_id", *FEATURES, TARGET]
class DatasetUnavailable(FileNotFoundError): pass
def find_data_dir(base=None): return Path(base) if base else DATA_DIR
def missing_files(base=None):
    folder=find_data_dir(base); names=[x[3] for x in HOSPITALS]+[HELDOUT_FILE]
    return [name for name in names if not (folder/name).is_file()]
def load_csv(filename, base=None):
    path=find_data_dir(base)/filename
    if not path.is_file(): raise DatasetUnavailable(f"Required dataset not found: {path}")
    df=pd.read_csv(path); missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"{filename} is missing required columns: {', '.join(missing)}")
    return df
def load_network(base=None):
    absent=missing_files(base)
    if absent: raise DatasetUnavailable("Missing required datasets: "+", ".join(absent))
    return {hid:load_csv(filename,base) for hid,_,_,filename in HOSPITALS}
def load_heldout(base=None): return load_csv(HELDOUT_FILE,base)
