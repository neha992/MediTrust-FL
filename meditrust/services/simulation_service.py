"""Single-process federated coordinator. Patient rows stay local in memory."""
import json
import numpy as np
from meditrust.config import HOSPITALS, TOP_K, TRIM_FRACTION, MAX_F1_DROP
from meditrust.data.datasets import load_network, load_heldout, missing_files
from meditrust.validation.validator import validate_dataset
from meditrust.federated.local_training import baseline_params, train_update
from meditrust.federated.compression import compress, reconstruct, payload_stats
from meditrust.federated.anomaly_detection import inspect_updates
from meditrust.federated.aggregation import trimmed_mean
from meditrust.federated.evaluation import evaluate
from meditrust.security.attack_simulator import attack, flip_labels
from meditrust.security.trust_engine import updated_trust
from meditrust.persistence.database import connect

class SimulationService:
  def __init__(self, db_path, data_dir=None): self.db_path, self.data_dir=db_path, data_dir
  def initialize(self):
    con=connect(self.db_path)
    for hid,name,_,_ in HOSPITALS:
      con.execute("INSERT OR IGNORE INTO hospitals(id,name,dataset_size) VALUES(?,?,?)",(hid,name,0)); con.execute("INSERT OR IGNORE INTO trust_history(hospital_id,score,reason) VALUES(?,?,?)",(hid,80,"Initial score"))
    con.commit(); con.close()
  def readiness(self): return {"ready":not missing_files(self.data_dir),"missing":missing_files(self.data_dir)}
  def state(self):
    self.initialize(); con=connect(self.db_path); rows=lambda q:[dict(x) for x in con.execute(q).fetchall()]
    result={"hospitals":rows("SELECT * FROM hospitals ORDER BY id"),"rounds":rows("SELECT * FROM rounds ORDER BY id DESC"),"decisions":rows("SELECT * FROM decisions ORDER BY id DESC"),"metrics":rows("SELECT * FROM metrics ORDER BY id DESC"),"audit":rows("SELECT * FROM audit_events ORDER BY id DESC LIMIT 200")}; con.close(); return result
  def local_dataset(self,hid): return load_network(self.data_dir)[hid]
  def run_round(self,attack_config=None):
    self.initialize(); data=load_network(self.data_dir); heldout=load_heldout(self.data_dir); con=connect(self.db_path)
    prev=con.execute("SELECT * FROM rounds ORDER BY id DESC LIMIT 1").fetchone(); rid=(prev["id"] if prev else 0)+1; version=(prev["model_version"] if prev else 0)+1
    model=con.execute("SELECT * FROM approved_models ORDER BY model_version DESC LIMIT 1").fetchone(); baseline=np.asarray(json.loads(model["params"]),float) if model else baseline_params()
    previous=con.execute("SELECT * FROM metrics WHERE status='APPROVED' ORDER BY id DESC LIMIT 1").fetchone(); updates={}; stats={}; summaries={}; attacks={}; local_metrics={}
    for hid,_,_,_ in HOSPITALS:
      checked,summary=validate_dataset(data[hid]); clean=checked[checked.validation_status!="REJECTED"].copy(); summaries[hid]=summary
      con.execute("INSERT INTO validation_runs(hospital_id,round_id,total,valid,review,rejected) VALUES(?,?,?,?,?,?)",(hid,rid,summary['total'],summary['valid'],summary['review'],summary['rejected']))
      attack_name=None; train=flip_labels(clean) if attack_config and attack_config.get("hospital_id")==hid and attack_config.get("type")=="label_flip" else clean
      raw=train_update(train,baseline)
      if attack_config and attack_config.get("hospital_id")==hid and attack_config.get("type")!="label_flip": raw=attack(raw,attack_config['type'],attack_config['strength']); attack_name=attack_config['type']
      elif attack_config and attack_config.get("hospital_id")==hid: attack_name="label_flip"
      payload=compress(raw,TOP_K); updates[hid]=reconstruct(payload); stats[hid]=payload_stats(raw,payload); attacks[hid]=attack_name; local_metrics[hid]=evaluate(baseline,clean)
      con.execute("UPDATE hospitals SET dataset_size=?,last_round=? WHERE id=?",(len(data[hid]),rid,hid))
    decisions=inspect_updates(updates,rid,version,version); accepted=[]; contributors=[]
    for hid,info in decisions.items():
      info['attack']=attacks[hid]; con.execute("INSERT INTO decisions(round_id,hospital_id,decision,norm,cosine,score,explanation,attack) VALUES(?,?,?,?,?,?,?,?)",(rid,hid,info['decision'],info['norm'],info['cosine'],info['score'],info['explanation'],attacks[hid]))
      if info['decision']=="ACCEPTED": accepted.append(updates[hid]); contributors.append(hid)
      old=con.execute("SELECT trust FROM hospitals WHERE id=?",(hid,)).fetchone()['trust']; score,delta=updated_trust(old,info['decision'],summaries[hid]); con.execute("UPDATE hospitals SET trust=? WHERE id=?",(score,hid)); con.execute("INSERT INTO trust_history(hospital_id,score,reason) VALUES(?,?,?)",(hid,score,f"Round {rid}: {info['decision']} ({delta:+.1f})")); con.execute("INSERT INTO audit_events(event_type,hospital_id,round_id,decision,explanation) VALUES(?,?,?,?,?)",("UPDATE_INSPECTED",hid,rid,info['decision'],info['explanation']))
    status="ROLLED_BACK"; metrics=None
    if accepted:
      aggregate,trim=trimmed_mean(accepted,TRIM_FRACTION); candidate=baseline+aggregate; metrics=evaluate(candidate,heldout); prior=previous['f1'] if previous else evaluate(baseline,heldout)['f1']
      if metrics['f1']>=prior-MAX_F1_DROP: status="APPROVED"; con.execute("INSERT INTO approved_models(model_version,round_id,params) VALUES(?,?,?)",(version,rid,json.dumps(candidate.tolist())))
      con.execute("INSERT INTO metrics(round_id,model_version,accuracy,precision,recall,f1,status) VALUES(?,?,?,?,?,?,?)",(rid,version,metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1'],status))
    con.execute("INSERT INTO rounds(id,model_version,status,contributors) VALUES(?,?,?,?)",(rid,version,status,json.dumps(contributors))); con.execute("INSERT INTO audit_events(event_type,round_id,decision,explanation) VALUES(?,?,?,?)",("ROUND_COMPLETED",rid,status,f"Contributors: {', '.join(contributors) or 'none'}")); con.commit(); con.close()
    return {"round_id":rid,"status":status,"decisions":decisions,"metrics":metrics,"compression":stats,"contributors":contributors,"local_metrics":local_metrics}
  def review(self,decision_id,approve):
    con=connect(self.db_path); d=con.execute("SELECT * FROM decisions WHERE id=?",(decision_id,)).fetchone()
    if not d or d['decision']!="NEEDS REVIEW" or d['review']: raise ValueError("Only pending review updates can be reviewed")
    result="APPROVED" if approve else "REJECTED"; con.execute("UPDATE decisions SET review=? WHERE id=?",(result,decision_id)); con.execute("INSERT INTO audit_events(event_type,hospital_id,round_id,decision,explanation) VALUES(?,?,?,?,?)",("HUMAN_REVIEW",d['hospital_id'],d['round_id'],result,"Authorised simulated reviewer decision")); con.commit(); con.close()
