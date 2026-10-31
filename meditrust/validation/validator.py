import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from meditrust.config import FEATURES, TARGET

RANGES={"age":(0,120),"bp_systolic":(50,280),"bp_diastolic":(30,180),"glucose":(20,700),"troponin":(0,100),"heart_rate":(20,250),"spo2":(50,100),"crp":(0,500)}
def validate_dataset(df):
    """Reject malformed rows; place Isolation Forest outliers under review, not blame."""
    result=df.copy(); required=["patient_id",*FEATURES,TARGET]
    duplicate=result.duplicated("patient_id",keep=False) if "patient_id" in result else pd.Series(True,index=result.index)
    reasons=[]; statuses=[]
    for idx,row in result.iterrows():
        item=[]; missing=[c for c in required if c not in result or pd.isna(row.get(c))]
        if missing: item.append("missing:"+",".join(missing))
        for col,(low,high) in RANGES.items():
            value=row.get(col)
            if pd.notna(value) and (not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value<low or value>high): item.append(f"invalid_{col}")
        if row.get(TARGET) not in (0,1): item.append("invalid_target")
        if duplicate.loc[idx]: item.append("duplicate_patient_id")
        statuses.append("REJECTED" if any(x.startswith(("missing","invalid_","duplicate")) for x in item) else "VALID"); reasons.append("; ".join(item))
    valid=pd.Series(statuses,index=result.index).eq("VALID")
    if valid.sum()>=10:
        flags=IsolationForest(contamination=.03,random_state=2026).fit_predict(result.loc[valid,FEATURES].astype(float))==-1
        for idx,flag in zip(result.index[valid],flags):
            if flag: statuses[result.index.get_loc(idx)]="NEEDS REVIEW"; reasons[result.index.get_loc(idx)]="unusual_record_isolation_forest"
    result["validation_status"]=statuses; result["validation_reasons"]=reasons
    return result,{"total":len(result),"valid":statuses.count("VALID"),"review":statuses.count("NEEDS REVIEW"),"rejected":statuses.count("REJECTED"),"missing":sum("missing:" in x for x in reasons),"duplicates":sum("duplicate" in x for x in reasons),"invalid":sum("invalid_" in x for x in reasons),"unusual":sum("isolation" in x for x in reasons)}
