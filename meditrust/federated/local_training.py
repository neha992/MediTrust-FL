import numpy as np
from sklearn.linear_model import LogisticRegression
from meditrust.config import FEATURES,TARGET

# Fixed shared transformation makes every local parameter vector compatible.
MEAN=np.array([52,125,78,110,.05,78,96,8.]); SCALE=np.array([18,22,11,40,.08,16,2.5,10.])
def transform(df): return (df[FEATURES].astype(float).to_numpy()-MEAN)/SCALE
def baseline_params(): return np.zeros(len(FEATURES)+1)
def train_update(df,baseline):
    X,y=transform(df),df[TARGET].astype(int).to_numpy()
    if len(np.unique(y))<2:return np.zeros_like(baseline)
    model=LogisticRegression(C=1,max_iter=300,random_state=17,solver="lbfgs").fit(X,y)
    return np.r_[model.coef_.ravel(),model.intercept_]-np.asarray(baseline)
def predict(params,df):
    z=transform(df)@np.asarray(params)[:-1]+np.asarray(params)[-1]
    return (1/(1+np.exp(-np.clip(z,-30,30)))>=.5).astype(int)
