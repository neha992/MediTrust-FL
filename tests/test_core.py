import numpy as np
from meditrust.data.synthetic_generator import generate_hospital_data, add_malformed_rows
from meditrust.validation.validator import validate_dataset
from meditrust.federated.compression import compress, reconstruct, payload_stats
from meditrust.federated.aggregation import trimmed_mean
from meditrust.federated.anomaly_detection import inspect_updates
from meditrust.services.simulation_service import SimulationService

def test_validation_rejects_malformed():
    _, s=validate_dataset(add_malformed_rows(generate_hospital_data("A",n=15)))
    assert s['rejected'] >= 3
def test_sparse_roundtrip_and_bytes():
    x=np.array([1.,-4.,2.,.5]); p=compress(x,2)
    assert np.allclose(reconstruct(p),[0,-4,2,0]) and payload_stats(x,p)['compressed_bytes'] > 0
def test_trimmed_mean_hand_calculated():
    val,trim=trimmed_mean([np.array([1.,2.]),np.array([2.,3.]),np.array([100.,4.]),np.array([3.,5.]),np.array([4.,6.])],.2)
    assert trim==1 and np.allclose(val,[3,4])
def test_outlier_quarantined():
    d=inspect_updates({'A':np.ones(3),'B':np.ones(3)*1.1,'C':np.ones(3)*20},1,1,1)
    assert d['C']['decision']=='QUARANTINED'
def test_end_to_end_persists(tmp_path):
    service=SimulationService(tmp_path/'demo.db'); result=service.run_round()
    assert result['round_id']==1 and service.state()['rounds']
