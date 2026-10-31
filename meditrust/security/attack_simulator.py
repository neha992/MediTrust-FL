import numpy as np
def attack(update,attack_type,strength,seed=2026):
    v=np.asarray(update,dtype=float); s=float(strength); rng=np.random.default_rng(seed)
    if attack_type in {"sign_flip","sign_reversal"}: return -v*s
    if attack_type in {"scaled_update","model_replacement"}: return v*max(1.,s)
    if attack_type=="gaussian_noise": return v+rng.normal(0,max(np.linalg.norm(v),1e-6)*s,v.shape)
    if attack_type=="stealth_drift": return v+np.sign(v+1e-9)*np.linalg.norm(v)*min(s,.25)/max(1,len(v))
    raise ValueError("Unknown update attack")
def flip_labels(df,target="target"):
    copy=df.copy(); copy[target]=1-copy[target].astype(int); return copy
