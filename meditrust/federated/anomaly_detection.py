import numpy as np
def inspect_updates(updates, round_id, version, expected_version):
    vectors = {h: np.asarray(v, float) for h,v in updates.items()}
    norms = {h: float(np.linalg.norm(v)) for h,v in vectors.items()}
    median = np.median(list(norms.values())) or 1e-9
    out = {}
    for h,v in vectors.items():
        reasons=[]; norm_ratio=norms[h]/median
        others=[x for key,x in vectors.items() if key != h]
        cosine = float(np.mean([np.dot(v,x)/(np.linalg.norm(v)*np.linalg.norm(x)+1e-12) for x in others])) if others else 1.0
        if not np.isfinite(v).all(): decision, reasons = "QUARANTINED", ["non_finite"]
        elif norm_ratio > 4.0: decision, reasons = "QUARANTINED", ["excessive_update_norm"]
        elif norm_ratio > 2.0 or cosine < -0.25: decision, reasons = "NEEDS REVIEW", (["unusual_norm"] if norm_ratio>2 else []) + (["low_cosine_similarity"] if cosine<-.25 else [])
        else: decision = "ACCEPTED"
        out[h] = {"hospital_id":h,"round_id":round_id,"model_version":version,"decision":decision,"norm":norms[h],"norm_ratio":norm_ratio,"cosine":cosine,"score":max(0., norm_ratio-1)+max(0., -cosine),"reasons":reasons,"explanation":"; ".join(reasons) or "Update is structurally valid and within configured cohort thresholds."}
    return out
