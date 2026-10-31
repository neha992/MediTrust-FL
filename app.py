import pandas as pd
import plotly.express as px
import streamlit as st
from meditrust.config import DB_PATH,HOSPITALS
from meditrust.services.simulation_service import SimulationService
from meditrust.data.datasets import DatasetUnavailable

st.set_page_config("MediTrust FL","⚕",layout="wide")
st.markdown("""<style>.stApp{background:#071827;color:#e8f1f8}.block-container{padding-top:2rem}.hero{background:linear-gradient(110deg,#0a3145,#0c6071);padding:1.5rem;border-radius:16px}.status{color:#4dd4c7}</style>""",unsafe_allow_html=True)
svc=SimulationService(DB_PATH); state=svc.state(); ready=svc.readiness()
st.markdown("<div class='hero'><h1>MediTrust FL</h1><p>Federated healthcare model security command center · Local synthetic-data demonstration only</p></div>",unsafe_allow_html=True)
page=st.sidebar.radio("Navigation",["Command Center","Hospital Network","Data Integrity","Federated Learning","Attack Simulator","Security Monitoring","Model Performance","Audit Center"])
if not ready['ready']:
    st.error("Training is blocked: required local datasets are missing. Add the six CSVs to the project's `data/` directory.")
    st.code("\n".join(ready['missing']))
def run(config=None):
    try:
        with st.spinner("Local validation, training, compression, security inspection, aggregation, and held-out evaluation…"): result=svc.run_round(config)
        st.session_state['last']=result; st.success(f"Round {result['round_id']} {result['status'].lower()}."); st.rerun()
    except (DatasetUnavailable,ValueError) as exc: st.error(str(exc))
def frame(rows): return pd.DataFrame(rows) if rows else pd.DataFrame()
if page=="Command Center":
    latest=state['rounds'][0] if state['rounds'] else {}; decisions=state['decisions'][:5]
    a,b,c,d=st.columns(4); a.metric("Hospitals",len(state['hospitals'])); b.metric("Training records",sum(x['dataset_size'] for x in state['hospitals'])); c.metric("Federated round",latest.get('id','—')); d.metric("Quarantined updates",sum(x['decision']=='QUARANTINED' for x in decisions))
    if st.button("Start training round",type="primary",disabled=not ready['ready']): run()
    st.info("Workflow: local data integrity → shared preprocessing → local LR updates → Top-K payload → anomaly inspection → trimmed mean → independent held-out evaluation → approval or rollback.")
    st.subheader("Recent security activity"); st.dataframe(frame(state['audit']),use_container_width=True,hide_index=True)
elif page=="Hospital Network":
    for hid,name,specialty,_ in HOSPITALS:
        h=next(x for x in state['hospitals'] if x['id']==hid)
        with st.container(border=True):
            x,y,z,w=st.columns(4); x.write(f"**{name}**\n\n{specialty}"); y.metric("Records",h['dataset_size']); z.metric("Trust",f"{h['trust']:.1f}/100"); w.write(f"Local status: {'Ready' if ready['ready'] else 'Awaiting dataset'}")
            if ready['ready']:
                data=svc.local_dataset(hid); st.download_button("Download local CSV",data.to_csv(index=False),file_name=f"hospital_{hid}.csv",key=f"dl{hid}"); st.dataframe(data.head(5),use_container_width=True,hide_index=True)
elif page=="Data Integrity":
    st.caption("Malformed records are rejected. Isolation Forest flags statistically unusual records for review; it is not proof of maliciousness.")
    df=frame(state['audit']); st.dataframe(frame(state['rounds']),use_container_width=True,hide_index=True)
    if ready['ready']:
        hid=st.selectbox("Inspect hospital",[x[0] for x in HOSPITALS]); from meditrust.validation.validator import validate_dataset
        checked,summary=validate_dataset(svc.local_dataset(hid)); st.json(summary); st.dataframe(checked[checked.validation_status!='VALID'].head(30),use_container_width=True,hide_index=True)
elif page=="Federated Learning":
    if st.button("Start training round",type="primary",disabled=not ready['ready']): run()
    if 'last' in st.session_state: st.json(st.session_state['last'])
    else: st.info("Run a round to see actual updates, compression, accepted contributors, aggregation, and distribution status.")
elif page=="Attack Simulator":
    h=st.selectbox("Target hospital",[x[0] for x in HOSPITALS],index=2); typ=st.selectbox("Attack",["sign_flip","gaussian_noise","label_flip","model_replacement","stealth_drift"]); strength=st.slider("Strength",.1,12.,5.,.1)
    st.caption("Attacks are controlled demos. Label flip changes only the local training copy; other attacks modify only the submitted model update.")
    if st.button("Run attacked round",type="primary",disabled=not ready['ready']): run({"hospital_id":h,"type":typ,"strength":strength})
elif page=="Security Monitoring":
    df=frame(state['decisions'])
    if df.empty: st.info("No submitted updates yet.")
    else:
        st.plotly_chart(px.scatter(df,x='cosine',y='norm',color='decision',hover_name='hospital_id',size='score'),use_container_width=True); st.dataframe(df,use_container_width=True,hide_index=True)
        pending=df[(df.decision=='NEEDS REVIEW') & df.review.isna()]
        if not pending.empty:
            choice=st.selectbox("Pending decision",pending.id.tolist()); a,b=st.columns(2)
            if a.button("Approve review"): svc.review(choice,True); st.rerun()
            if b.button("Reject review"): svc.review(choice,False); st.rerun()
elif page=="Model Performance":
    df=frame(state['metrics'])
    if df.empty: st.info("No held-out evaluation yet.")
    else: st.dataframe(df,use_container_width=True,hide_index=True); st.plotly_chart(px.line(df.sort_values('round_id'),x='round_id',y=['accuracy','precision','recall','f1'],markers=True),use_container_width=True)
    st.caption("Candidate models roll back only when held-out F1 drops more than 0.08. This does not guarantee robustness against all attacks.")
else:
    df=frame(state['audit']); kind=st.selectbox("Event type",['All']+(sorted(df.event_type.unique().tolist()) if not df.empty else []));
    if kind!='All': df=df[df.event_type==kind]
    st.dataframe(df,use_container_width=True,hide_index=True); st.download_button("Export audit CSV",df.to_csv(index=False),"meditrust_audit.csv")
