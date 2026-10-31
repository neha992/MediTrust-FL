# MediTrust FL

Local Streamlit hackathon demonstration of privacy-aware, Byzantine-resilient federated learning for **synthetic** healthcare triage data. It is not clinical software and must never be used for medical decisions.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

## What it demonstrates

- Five deterministic, non-IID synthetic hospitals. The risk label is a noisy rule over simulated vital signs and glucose.
- Local validation rejects malformed values and duplicates before local logistic-regression training.
- All sites use one documented fixed transform (shared mean/scale) before parameter aggregation, avoiding incompatible locally fitted scalers.
- Top-K sparsification stores int32 positions and float64 values; the dashboard reports bytes including index cost.
- Submitted updates are reconstructed then inspected by norm ratio and cohort cosine similarity. `ACCEPTED` enters aggregation; `NEEDS REVIEW` is held for an explicit reviewer; `QUARANTINED` never enters automatic aggregation.
- Coordinate-wise trimmed mean aggregates accepted vectors. The evaluated candidate is persisted only if its F1 does not fall by more than 0.08 from the current approved model; otherwise the model rolls back.
- SQLite persists hospitals, validation summaries, rounds, decisions, trust history, metrics, approved parameter vectors, and audit metadata.

## Data boundaries and limitations

Raw synthetic patient rows live only in the one-process hospital simulation and are neither saved centrally nor shown in central audit tables. The central SQLite database holds aggregate validation counts, update metadata, and model metadata only. This demonstrates the protocol inside one local process; it is not five separately deployed servers. Top-K compression lowers payload size, not poisoning risk. The simple detection thresholds and trimmed mean do not protect against all coordinated attacks.

## Demo sequence

1. Open **Overview** and run a normal federated round.
2. Inspect measured compression, decisions, and held-out metrics.
3. In **Attack Simulator**, select Hospital C, an attack, and a strength, then run a round.
4. Inspect the actual anomaly measurements, contributors, metrics, and audit log. Use **Security Monitoring** to resolve any review-status update.
