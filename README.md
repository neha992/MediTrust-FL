# MediTrust FL

MediTrust FL is a local Streamlit demonstration of privacy-aware federated healthcare model security. It is not clinical software, does not use patient-identifying records, and does not guarantee detection of poisoning attacks.

## Dataset contract

Place these CSVs in `data/`: `node-1_general_hospital.csv`, `node-2_urban_medical_center.csv`, `node-3_regional_hospital.csv`, `node-4_community_hospital.csv`, `node-5_specialty_medical_center.csv`, and `heldout_validation_cohort.csv`.

Each needs `patient_id, age, bp_systolic, bp_diastolic, glucose, troponin, heart_rate, spo2, crp, target`. The held-out cohort is loaded only for evaluation. If any source file is absent, the dashboard blocks training and clearly lists it; it never manufactures substitute records.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider
streamlit run app.py
```

## Implementation

Five local simulated hospitals validate their own data, use one compatible fixed preprocessing transform, train logistic regression locally, and submit Top-K sparse parameter updates. The coordinator reconstructs updates, evaluates norms and cosine similarity, records `ACCEPTED`, `NEEDS REVIEW`, or `QUARANTINED` decisions, and aggregates only accepted updates with coordinate-wise trimmed mean. Candidate models are evaluated on the independent held-out cohort and roll back if F1 falls by more than 0.08.

The attack simulator offers sign flip, Gaussian noise, label flip, model replacement, and stealth drift demos. Isolation Forest marks unusual records for review, not as proof of an attack. SQLite stores aggregate validation results, update/security metadata, trust history, rounds, metrics, approved model vectors, and audit events; raw patient rows stay local/in memory.
