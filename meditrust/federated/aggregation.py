import numpy as np
def trimmed_mean(updates, trim_fraction=.2):
    if not updates: raise ValueError("No updates to aggregate")
    arr = np.asarray(updates, dtype=float)
    if arr.ndim != 2 or not np.isfinite(arr).all(): raise ValueError("Updates must be finite compatible vectors")
    if not 0 <= trim_fraction < .5: raise ValueError("trim_fraction must be in [0, .5)")
    trim = int(np.floor(len(arr) * trim_fraction))
    if 2*trim >= len(arr): trim = 0
    ordered = np.sort(arr, axis=0)
    return ordered[trim:len(arr)-trim].mean(axis=0), trim
