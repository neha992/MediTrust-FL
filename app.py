import os
import pandas as pd
import plotly.express as px
import streamlit as st
from meditrust.config import DB_PATH, HOSPITALS
from meditrust.services.simulation_service import SimulationService

st.set_page_config(page_title="MediTrust FL", page_icon="M", layout="wide")
st.markdown("""<style>.stApp{background:#f6f9fb}.hero{background:#102a43;color:#fff;padding:1.4rem;border-radius:12px;margin-bottom:1rem}.accepted{color:#16803c}.review{color:#a86600}.quarantine{color:#b42318}</style>""",unsafe_allow_html=True)
svc=SimulationService(DB_PATH); svc.initialize(); state=svc.state()
st.markdown('<div class="hero"><h1>MediTrust FL</h1><p>Privacy-preserving, Byzantine-resilient federated healthcare intelligence — synthetic demonstration only; never for clinical decisions.</p></div>',unsafe_allow_html=True)
page=st.sidebar.radio("Navigation",["Overview","Hospital Network","Data Verification","Federated Learning","Security Monitoring","Attack Simulator","Model Performance","Audit Logs"])
def run(attack=None, malformed=None):
    with st.spinner("Validating local data, training locally, inspecting updates, and evaluating candidate model…"):
        result=svc.run_round(attack,malformed)
    st.success(f"Round {result['round_id']} {result['status'].lower()}.")
    st.session_state.last=result; st.rerun()
if page=="Overview":
    c=st.columns(4); latest=state['rounds'][0] if state['rounds'] else None
    c[0].metric("Registered hospitals",len(state['hospitals'])); c[1].metric("Latest round",latest['id'] if latest else "No rounds")
    c[2].metric("Accepted updates",sum(x['decision']=='ACCEPTED' for x in state['decisions'][:5])); c[3].metric("Quarantined",sum(x['decision']=='QUARANTINED' for x in state['decisions'][:5]))
    st.info("Local validation → local training → Top-K compression → central anomaly inspection → trimmed-mean aggregation → held-out evaluation → approval/rollback.")
    if st.button("Run Federated Round",type="primary"): run()
    st.subheader("Recent security events"); st.dataframe(pd.DataFrame(state['audit']),use_container_width=True,hide_index=True)
elif page=="Hospital Network":
    for h in state['hospitals']:
        with st.container(border=True):
            a,b,c,d=st.columns(4); a.write(f"**Hospital {h['id']} — {h['name']}**"); b.metric("Synthetic records",h['dataset_size']); c.metric("Trust",f"{h['trust']:.1f}/100"); d.write(f"Last training round: {h['last_round']}")
    st.caption("Trust starts at 80 and changes from observed validation and security outcomes; it prioritizes risk, not proof of maliciousness.")
elif page=="Data Verification":
    hospital=st.selectbox("Hospital",[x[0] for x in HOSPITALS]); malformed=st.checkbox("Introduce malformed synthetic records (age, oxygen, missing heart rate)")
    if st.button("Validate and run a round"): run(malformed=hospital if malformed else None)
    rows=[x for x in state['audit'] if x['event_type']=='ROUND_COMPLETED']; st.info("Rejected records are excluded locally before model training. Validation summaries are persisted without raw patient rows.")
    st.dataframe(pd.DataFrame([x for x in svc.state()['rounds']]),use_container_width=True,hide_index=True)
elif page=="Federated Learning":
    if st.button("Run Federated Round",type="primary"): run()
    if 'last' in st.session_state:
        r=st.session_state.last; st.write("Contributors:",", ".join(r['contributors']) or "none"); st.json({"status":r['status'],"compression":r['compression']})
    else: st.info("Run a round to view actual local training and aggregation results.")
elif page=="Security Monitoring":
    df=pd.DataFrame(state['decisions'])
    if df.empty: st.info("No updates inspected yet.")
    else:
        st.dataframe(df[["id","round_id","hospital_id","decision","norm","cosine","score","explanation","attack","review"]],use_container_width=True,hide_index=True)
        st.plotly_chart(px.scatter(df,x="cosine",y="norm",color="decision",hover_name="hospital_id"),use_container_width=True)
        pending=df[df.decision.eq('NEEDS REVIEW') & df.review.isna()]
        if not pending.empty:
            choice=st.selectbox("Pending update ID",pending.id.tolist())
            x,y=st.columns(2)
            if x.button("Approve after review"): svc.review(choice,True); st.rerun()
            if y.button("Reject after review"): svc.review(choice,False); st.rerun()
elif page=="Attack Simulator":
    h=st.selectbox("Target hospital",[x[0] for x in HOSPITALS],index=2); typ=st.selectbox("Attack type",["sign_reversal","scaled_update"]); strength=st.slider("Attack strength",1.0,12.0,6.0,.5)
    st.caption("This alters only a submitted model update. It does not modify any local synthetic dataset. Detection is measured, not guaranteed.")
    if st.button("Simulate attack and run round",type="primary"): run({"hospital_id":h,"type":typ,"strength":strength})
elif page=="Model Performance":
    df=pd.DataFrame(state['metrics'])
    if df.empty: st.info("No valid evaluation is available until a round runs.")
    else: st.dataframe(df,use_container_width=True,hide_index=True); st.plotly_chart(px.line(df.sort_values('round_id'),x='round_id',y=['accuracy','precision','recall','f1'],markers=True),use_container_width=True)
    st.caption("Candidates roll back only when F1 falls by more than the configured 0.08 tolerance. Trimmed mean is not a defense against every coordinated attack.")
else:
    df=pd.DataFrame(state['audit']);
    if df.empty: st.info("No audit events yet.")
    else:
        kind=st.selectbox("Event type",["All"]+sorted(df.event_type.dropna().unique().tolist()))
        if kind!="All": df=df[df.event_type==kind]
        st.dataframe(df,use_container_width=True,hide_index=True)
