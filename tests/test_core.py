import numpy as np
import pandas as pd
import pytest
from meditrust.config import FEATURES,HOSPITALS,HELDOUT_FILE
from meditrust.validation.validator import validate_dataset
from meditrust.federated.local_training import baseline_params,train_update
from meditrust.federated.compression import compress,reconstruct
from meditrust.federated.aggregation import trimmed_mean
from meditrust.federated.anomaly_detection import inspect_updates
from meditrust.security.attack_simulator import attack,flip_labels
from meditrust.data.datasets import missing_files,DatasetUnavailable
from meditrust.services.simulation_service import SimulationService

def records(n=30,seed=1):
    rng=np.random.default_rng(seed); data={"patient_id":[f"p{i}" for i in range(n)],"age":rng.integers(20,80,n),"bp_systolic":rng.normal(125,12,n),"bp_diastolic":rng.normal(78,8,n),"glucose":rng.normal(110,20,n),"troponin":rng.uniform(.001,.08,n),"heart_rate":rng.normal(78,10,n),"spo2":rng.normal(96,1,n),"crp":rng.normal(7,3,n)}; data['target']=(data['age']+data['bp_systolic']>185).astype(int); return pd.DataFrame(data)
def write_network(folder):
    folder.mkdir()
    for i,(_,_,_,name) in enumerate(HOSPITALS): records(35,i).to_csv(folder/name,index=False)
    records(50,99).to_csv(folder/HELDOUT_FILE,index=False)
def test_validation_detects_duplicate_invalid_and_outlier():
    df=records(); df.loc[0,'age']=150; df.loc[1,'patient_id']=df.loc[2,'patient_id']; checked,s=validate_dataset(df)
    assert s['rejected']>=3 and 'invalid_age' in checked.loc[0,'validation_reasons']
def test_local_training_and_compression_roundtrip():
    update=train_update(records(),baseline_params()); p=compress(update,3)
    assert update.shape==(len(FEATURES)+1,) and np.count_nonzero(reconstruct(p))==3
@pytest.mark.parametrize('kind',["sign_flip","gaussian_noise","model_replacement","stealth_drift"])
def test_update_attacks_modify_update(kind): assert not np.allclose(attack(np.array([1.,2.,3.]),kind,2),[1,2,3])
def test_label_flip_changes_labels(): assert not flip_labels(records())['target'].equals(records()['target'])
def test_anomaly_quarantine_and_trimmed_mean():
    d=inspect_updates({'A':np.ones(3),'B':np.ones(3)*1.1,'C':np.ones(3)*20},1,1,1); out,trim=trimmed_mean([np.array([1.]),np.array([2.]),np.array([100.]),np.array([3.]),np.array([4.])],.2)
    assert d['C']['decision']=='QUARANTINED' and trim==1 and out[0]==3
def test_missing_files_is_explicit(tmp_path):
    assert missing_files(tmp_path)
    with pytest.raises(DatasetUnavailable): SimulationService(tmp_path/'x.db',tmp_path).run_round()
def test_end_to_end_sqlite_and_quarantine(tmp_path):
    data=tmp_path/'data'; write_network(data); service=SimulationService(tmp_path/'state.db',data); normal=service.run_round(); attacked=service.run_round({'hospital_id':'C','type':'model_replacement','strength':12})
    assert normal['metrics'] and attacked['decisions']['C']['decision']=='QUARANTINED' and 'C' not in attacked['contributors'] and service.state()['rounds']
