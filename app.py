"""Live sustainability dashboard: streamlit run app.py"""
from pathlib import Path
import pandas as pd, streamlit as st, matplotlib.pyplot as plt
import analytics as an
from survey_schema import Q
st.set_page_config(page_title="Nagpur Rooftop Cafe Dashboard",layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Nunito:wght@400;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'Nunito',sans-serif;color:#243B2F}
.stApp{background:radial-gradient(circle at 92% 4%,#FFE3C4 0,transparent 30%),radial-gradient(circle at 4% 40%,#E3EEDC 0,transparent 32%),#FBF6EC}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2.2rem;max-width:1280px}
h1,h2,h3{font-family:'Fraunces',serif!important;color:#1F4D3A!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1F4D3A,#2F7D5B)}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"]{color:#FBF6EC}
[data-testid="stSidebar"] button,[data-testid="stSidebar"] button p,[data-testid="stSidebar"] small,[data-testid="stSidebar"] section *{color:#243B2F}
.hero{background:linear-gradient(120deg,#1F4D3A 0%,#2F7D5B 58%,#E8792F 130%);border-radius:26px;padding:34px 38px;margin-bottom:22px;box-shadow:0 16px 38px rgba(31,77,58,.24)}
.hero .tag{display:inline-block;background:rgba(255,255,255,.17);color:#FFF4E2;padding:5px 14px;border-radius:30px;font-size:.84rem;font-weight:700;letter-spacing:.04em}
.hero h1{color:#FFF9EE!important;font-size:2.65rem;line-height:1.12;margin:14px 0 8px;padding:0}
.hero p{color:#EAF1E4;margin:0;font-size:1.04rem}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #EFE6D4;border-radius:20px;padding:16px 18px;box-shadow:0 7px 20px rgba(92,74,40,.08);border-top:5px solid #E8792F}
[data-testid="stMetricLabel"] p{color:#60715F;font-weight:700;white-space:normal}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] div{font-family:'Fraunces',serif;color:#1F4D3A;font-size:1.3rem;white-space:normal!important;overflow:visible!important;text-overflow:clip!important;overflow-wrap:anywhere;line-height:1.2}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:#FFFFFFB8;padding:8px;border-radius:40px;margin:12px 0 20px;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{height:42px;padding:0 18px;border-radius:30px;font-weight:700;color:#2F5D49}
.stTabs [aria-selected="true"]{background:#E8792F!important;color:white!important}
.stTabs [aria-selected="true"] p{color:white!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
[data-testid="stImage"] img{border-radius:16px;box-shadow:0 6px 18px rgba(92,74,40,.09)}
[data-testid="stDataFrame"],[data-testid="stTable"]{border-radius:14px;overflow:hidden;background:white;box-shadow:0 6px 18px rgba(92,74,40,.07)}
</style>""",unsafe_allow_html=True)
inr=lambda x:f"₹{x:,.0f}"
@st.cache_resource(show_spinner="Running the four algorithms...")
def go(raw_bytes):
    import io
    R=an.run(pd.read_csv(io.BytesIO(raw_bytes)));return R,an.figs(R)
st.markdown('<div class="hero"><span class="tag">🌿 PLANT-FORWARD &nbsp;·&nbsp; 🍊 NAGPUR ORANGE &nbsp;·&nbsp; 🌇 ROOFTOP</span><h1>Rooftop Café, Manish Nagar</h1><p>70% vegan, 30% regular vegetarian. No meat, eggs or alcohol. Is it a sustainable business?</p></div>',unsafe_allow_html=True)
st.caption("The included survey is mock data with intentional noise and outliers. Upload real responses in the sidebar to rerun every model.")
up=st.sidebar.file_uploader("Upload real survey responses (CSV)",type="csv")
st.sidebar.download_button("Download mock CSV as a template",(Path(__file__).parent/"data"/"Harshita_Chopda_Survey_Data.csv").read_bytes(),"survey_template.csv")
src=up.getvalue() if up else (Path(__file__).parent/"data"/"Harshita_Chopda_Survey_Data.csv").read_bytes()
import io
miss=[c for c in an.required_columns() if c not in pd.read_csv(io.BytesIO(src),nrows=1).columns]
if miss: st.error("This file is missing columns: "+", ".join(miss[:12]));st.stop()
R,F=go(src);d,c,k,a,r,b=R["data"],R["clf"],R["clu"],R["arm"],R["reg"],R["biz"]
st.sidebar.write("Source:", "Uploaded data" if up else "Mock survey")
x=st.columns(4);x[0].metric("Verdict",b["verdict"]);x[1].metric("Stated visit intent",f"{c['positive_rate']:.0%}");x[2].metric("Year 2 monthly profit after market rent",inr(b["profit"]));x[3].metric("Payback",f"{b['payback']} months" if b["payback"] else "Over 60 months")
T=st.tabs(["📋 Survey data","🎯 Classification","🧩 Clustering","🍰 Association rules","📈 Regression","🌱 Sustainability"])
sp=lambda f,col=st:col.pyplot(f,clear_figure=False)
with T[0]:
    l,m=st.columns(2);l.subheader("Cleaning log");l.dataframe(pd.DataFrame(R["log"].items(),columns=["Step","Count"]),hide_index=True);sp(F["intent"],m)
    st.subheader("Descriptive statistics");st.dataframe(d[an.NUM+["q26_expected_spend","q27_expected_visits_month"]].describe().T.round(1))
    with st.expander("The 30 survey questions"): st.dataframe(pd.DataFrame(Q,columns=["Column","Question","Answer format"]),hide_index=True)
with T[1]:
    st.caption("Will this person visit in the first month?")
    th=st.slider("Decision threshold",.1,.9,.5,.01);q=(c["p"]>=th).astype(int);cm=an.confusion_matrix(c["yb"],q)
    x=st.columns(5);x[0].metric("Model",c["best"]);x[1].metric("Accuracy",f"{an.accuracy_score(c['yb'],q):.1%}");x[2].metric("Precision",f"{an.precision_score(c['yb'],q,zero_division=0):.1%}");x[3].metric("Recall",f"{an.recall_score(c['yb'],q):.1%}");x[4].metric("ROC-AUC",f"{c['auc']:.3f}")
    st.table(pd.DataFrame(cm,index=["Actual: won't visit","Actual: will visit"],columns=["Predicted: won't visit","Predicted: will visit"]))
    x=st.columns(3);sp(F["cm"],x[0]);sp(F["roc"],x[1]);sp(F["drivers"],x[2])
    st.subheader("Who to target: conversion probability tiers");st.dataframe(c["tiers"].round(2),hide_index=True)
with T[2]:
    x=st.columns(3);x[0].metric("Clusters",k["k"]);x[1].metric("Silhouette",f"{k['silhouette']:.3f}");x[2].metric("Respondents",len(d))
    x=st.columns(2);sp(F["k"],x[0]);sp(F["seg"],x[1]);sp(F["conv"]);st.subheader("Segment profiles");st.dataframe(k["profile"].drop(columns="cluster").round(2),hide_index=True)
with T[3]:
    x=st.columns(3);x[0].metric("Rules found",a["n_rules"]);x[1].metric("Rules holding on unseen responses",a["n_hold"]);x[2].metric("7-item menu covers",f"{a['coverage'][-1]:.0%}")
    st.subheader("Recommended launch menu");st.write(", ".join(i.replace("_"," ").title() for i in a["launch"]))
    st.subheader("If you launch A and B, people also expect C");st.dataframe(a["gap"].round(2),hide_index=True)
    x=st.columns(2);sp(F["cov"],x[0]);sp(F["rules"],x[1]);st.subheader("All rules");st.dataframe(a["rules"].drop(columns=["n_if","n_then"]).round(3),hide_index=True)
with T[4]:
    x=st.columns(4);x[0].metric("Model",r["best"]);x[1].metric("Average error",inr(r["mae"]));x[2].metric("R²",f"{r['r2']:.2f}");x[3].metric("Better than guessing the average",f"{r['improve']:.0%}")
    x=st.columns(2);sp(F["reg"],x[0]);sp(F["regd"],x[1]);st.dataframe(r["compare"].round(3),hide_index=True)
with T[5]:
    st.subheader("Go/no-go gates, fixed before seeing results");st.table(pd.DataFrame([{"Gate":g,"Result":round(float(v[0]),4),"Required":v[1],"Passed":"Yes" if v[2] else "No"} for g,v in b["gates"].items()]))
    x=st.columns(2);sp(F["pnl"],x[0]);sp(F["sim"],x[1])
    st.subheader("What would change the answer?");x=st.columns(4);A=an.A
    u=dict(market_rent=x[0].slider("Market rent",0,300000,A["market_rent"],10000),catchment_adults=x[1].slider("Target adults within reach",20000,120000,A["catchment_adults"],5000),reach=x[2].slider("Share reached",.1,.5,A["reach"],.01),food_cost=x[3].slider("Food cost",.25,.45,A["food_cost"],.01))
    w=an.business(d,u);x=st.columns(4);x[0].metric("Verdict",w["verdict"]);x[1].metric("Year 2 monthly profit",inr(w["profit"]));x[2].metric("Break-even customers per day",f"{w['breakeven_daily_cust']:.0f}");x[3].metric("Payback",f"{w['payback']} months" if w["payback"] else "Over 60 months")
    st.caption("Father's view: setting market rent to 0 shows cash profit, but the owned space could otherwise earn that rent.")
