import numpy as np
from sklearn.linear_model import LogisticRegression
from meditrust.config import FEATURES

MEAN = np.array([52, 78, 123, 78, 36.9, 96.5, 112.0]); SCALE = np.array([18, 16, 22, 11, .65, 2.2, 42.0])
def transform(df): return (df[FEATURES].to_numpy(float) - MEAN) / SCALE
def baseline_params(): return np.zeros(len(FEATURES) + 1)
def train_update(df, baseline):
    X, y = transform(df), df["triage_risk"].to_numpy()
    if len(np.unique(y)) < 2: return np.zeros_like(baseline)
    model = LogisticRegression(C=1.0, max_iter=300, random_state=17, solver="lbfgs")
    model.fit(X, y)
    params = np.r_[model.coef_.ravel(), model.intercept_]
    return params - baseline
def predict(params, df):
    z = transform(df) @ params[:-1] + params[-1]
    return (1/(1+np.exp(-np.clip(z, -30, 30))) >= .5).astype(int)
