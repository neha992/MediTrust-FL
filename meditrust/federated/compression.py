import numpy as np
def compress(update, k):
    arr = np.asarray(update, dtype=float).ravel()
    if not np.isfinite(arr).all(): raise ValueError("Update must be finite")
    if k < 0: raise ValueError("k must be non-negative")
    k = min(k, arr.size)
    idx = np.array([], dtype=np.int32) if k == 0 else np.argpartition(np.abs(arr), -k)[-k:].astype(np.int32)
    return {"shape": list(np.asarray(update).shape), "size": int(arr.size), "indices": idx.tolist(), "values": arr[idx].tolist()}
def reconstruct(payload):
    size, indices, values = int(payload["size"]), payload["indices"], payload["values"]
    if size < 0 or len(indices) != len(values) or len(set(indices)) != len(indices) or any(i < 0 or i >= size for i in indices): raise ValueError("Invalid sparse payload")
    out = np.zeros(size); out[np.asarray(indices, dtype=int)] = np.asarray(values, dtype=float)
    if not np.isfinite(out).all(): raise ValueError("Non-finite sparse values")
    return out.reshape(payload["shape"])
def payload_stats(update, payload):
    dense = np.asarray(update, dtype=np.float64).nbytes
    compressed = 8 + len(payload["indices"]) * (4 + 8) # length header + int32 index + float64
    return {"dense_bytes": dense, "compressed_bytes": compressed, "ratio": dense/compressed if compressed else 0, "saved_pct": 100*(1-compressed/dense) if dense else 0}
