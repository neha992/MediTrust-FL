"""Orchestrates a single-process simulation. Raw patient rows remain in memory locally."""
import json
import numpy as np
from meditrust.config import HOSPITALS, TOP_K, TRIM_FRACTION, MAX_F1_DROP
from meditrust.data.synthetic_generator import generate_hospital_data, generate_validation_set, add_malformed_rows
from meditrust.validation.validator import validate_dataset
from meditrust.federated.local_training import baseline_params, train_update
from meditrust.federated.compression import compress, reconstruct, payload_stats
from meditrust.federated.anomaly_detection import inspect_updates
from meditrust.federated.aggregation import trimmed_mean
from meditrust.federated.evaluation import evaluate
from meditrust.security.attack_simulator import attack
from meditrust.security.trust_engine import updated_trust
from meditrust.persistence.database import connect

class SimulationService:
  def __init__(self, db_path): self.db_path=db_path
  def initialize(self):
    con=connect(self.db_path)
    for hid,name,shift in HOSPITALS:
      con.execute("INSERT OR IGNORE INTO hospitals(id,name,dataset_size) VALUES(?,?,?)",(hid,name,240))
      con.execute("INSERT OR IGNORE INTO trust_history(hospital_id,score,reason) VALUES(?,?,?)",(hid,80,"Initial synthetic network score"))
    con.commit(); con.close()
  def state(self):
    con=connect(self.db_path)
    rows=lambda q,args=(): [dict(x) for x in con.execute(q,args).fetchall()]
    result={"hospitals":rows("SELECT * FROM hospitals ORDER BY id"),"rounds":rows("SELECT * FROM rounds ORDER BY id DESC"),"decisions":rows("SELECT * FROM decisions ORDER BY id DESC"),"metrics":rows("SELECT * FROM metrics ORDER BY id DESC"),"audit":rows("SELECT * FROM audit_events ORDER BY id DESC LIMIT 100")}
    con.close(); return result
  def run_round(self, attack_config=None, malformed_hospital=None):
    self.initialize(); con=connect(self.db_path)
    previous=con.execute("SELECT * FROM rounds ORDER BY id DESC LIMIT 1").fetchone(); rid=(previous["id"] if previous else 0)+1
    version=(previous["model_version"] if previous else 0)+1
    approved_model=con.execute("SELECT * FROM approved_models ORDER BY model_version DESC LIMIT 1").fetchone()
    baseline=np.asarray(json.loads(approved_model["params"]), dtype=float) if approved_model else baseline_params()
    validation=generate_validation_set()
    latest_metric=con.execute("SELECT * FROM metrics WHERE status='APPROVED' ORDER BY id DESC LIMIT 1").fetchone()
    datasets={}; summaries={}; updates={}; compressed={}; stats={}
    for hid,name,shift in HOSPITALS:
      data=generate_hospital_data(hid,shift)
      if malformed_hospital==hid: data=add_malformed_rows(data)
      checked, summary=validate_dataset(data); clean=checked[checked.validation_status!="REJECTED"]
      datasets[hid]=clean; summaries[hid]=summary
      con.execute("INSERT INTO validation_runs(hospital_id,round_id,total,valid,review,rejected) VALUES(?,?,?,?,?,?)",(hid,rid,summary["total"],summary["valid"],summary["review"],summary["rejected"]))
      raw=train_update(clean,baseline)
      attack_label=None
      if attack_config and attack_config.get("hospital_id")==hid:
        raw=attack(raw,attack_config["type"],attack_config["strength"]); attack_label=f"SIMULATED {attack_config['type']} x{attack_config['strength']}"
      payload=compress(raw,TOP_K); updates[hid]=reconstruct(payload); compressed[hid]=attack_label; stats[hid]=payload_stats(raw,payload)
    decisions=inspect_updates(updates,rid,version,version)
    approved=[]; contributors=[]
    for hid, info in decisions.items():
      label=compressed[hid]
      info["attack"]=label
      con.execute("INSERT INTO decisions(round_id,hospital_id,decision,norm,cosine,score,explanation,attack) VALUES(?,?,?,?,?,?,?,?)",(rid,hid,info["decision"],info["norm"],info["cosine"],info["score"],info["explanation"],label))
      if info["decision"]=="ACCEPTED": approved.append(updates[hid]); contributors.append(hid)
      hospital=con.execute("SELECT trust FROM hospitals WHERE id=?",(hid,)).fetchone(); score,delta=updated_trust(hospital["trust"],info["decision"],summaries[hid])
      con.execute("UPDATE hospitals SET trust=?,last_round=? WHERE id=?",(score,rid,hid)); con.execute("INSERT INTO trust_history(hospital_id,score,reason) VALUES(?,?,?)",(hid,score,f"Round {rid}: {info['decision']} ({delta:+.1f})"))
      con.execute("INSERT INTO audit_events(event_type,hospital_id,round_id,decision,explanation) VALUES(?,?,?,?,?)",("UPDATE_INSPECTED",hid,rid,info["decision"],info["explanation"]))
    status="ROLLED_BACK"; metrics=None
    if approved:
      agg, trim=trimmed_mean(approved,TRIM_FRACTION); candidate=baseline+agg; metrics=evaluate(candidate,validation)
      previous_f1=latest_metric["f1"] if latest_metric else evaluate(baseline,validation)["f1"]
      if metrics["f1"] >= previous_f1-MAX_F1_DROP:
        status="APPROVED"
        con.execute("INSERT INTO approved_models(model_version,round_id,params) VALUES(?,?,?)",(version,rid,json.dumps(candidate.tolist())))
      else: status="ROLLED_BACK"
      con.execute("INSERT INTO metrics(round_id,model_version,accuracy,precision,recall,f1,status) VALUES(?,?,?,?,?,?,?)",(rid,version,metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1'],status))
    con.execute("INSERT INTO rounds(id,model_version,status,contributors) VALUES(?,?,?,?)",(rid,version,status,json.dumps(contributors)))
    con.execute("INSERT INTO audit_events(event_type,round_id,decision,explanation) VALUES(?,?,?,?)",("ROUND_COMPLETED",rid,status,f"Contributors: {', '.join(contributors) or 'none'}"))
    con.commit(); con.close()
    return {"round_id":rid,"status":status,"decisions":decisions,"metrics":metrics,"compression":stats,"contributors":contributors}
  def review(self, decision_id, approve):
    con=connect(self.db_path); d=con.execute("SELECT * FROM decisions WHERE id=?",(decision_id,)).fetchone()
    if not d or d['decision'] != 'NEEDS REVIEW': raise ValueError("Only pending review updates can be reviewed")
    result="APPROVED" if approve else "REJECTED"; con.execute("UPDATE decisions SET review=? WHERE id=?",(result,decision_id)); con.execute("INSERT INTO audit_events(event_type,hospital_id,round_id,decision,explanation) VALUES(?,?,?,?,?)",("HUMAN_REVIEW",d['hospital_id'],d['round_id'],result,"Simulated reviewer decision")); con.commit(); con.close()
