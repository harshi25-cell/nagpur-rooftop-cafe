"""All cleaning, algorithms and business logic. Runs on the mock file or on uploaded real responses."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from survey_schema import ITEM_COLS, LIKERT, INTENT_ORDER, Q
SEED=42
BLUE,ORANGE,GREEN,GREY,INK="#2a78d6","#eb6834","#1baf7a","#9AA3AD","#0b0b0b"; CAT=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300"]
plt.rcParams.update({"figure.dpi":130,"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":GREY,"axes.titleweight":"bold","axes.titlelocation":"left","axes.grid":True,"grid.color":"#E7EAEE","axes.axisbelow":True})
NUM=["q01_age","q04_distance_km","q07_cafe_visits_month","q08_current_spend"]+LIKERT
CATS=["q02_gender","q03_occupation","q05_income","q06_diet","q09_companions","q10_time_slot","q12_discovery"]
INC=["Below 25k","25-50k","50k-1L","1-2L","Above 2L"]
# Rupees. Planning assumptions, to be replaced by real quotes.
A=dict(catchment_adults=60000,reach=.25,def_to_real=.35,prob_to_real=.12,unsure_to_real=.03,visit_haircut=.5,seats=60,turns_per_day=3.0,food_cost=.34,aggregator_share=.10,aggregator_fee=.25,market_rent=150000,staff=240000,utilities=70000,marketing=40000,other=50000,capex=4500000)
SEASON={1:1.1,2:1.1,3:.95,4:.7,5:.6,6:.7,7:.8,8:.8,9:.85,10:1.05,11:1.15,12:1.2}
GATES=dict(margin_of_safety=.20,prob_profit=.75,payback_months=48)

def required_columns():
    c=["respondent_id"]+NUM+CATS+ITEM_COLS+["q26_expected_spend","q27_expected_visits_month","q29_visit_intent"]
    return c

def clean(raw):
    d=raw.copy();log={"Responses received":len(d)}
    n=len(d);d=d.drop_duplicates("respondent_id");log["Duplicate responses removed"]=n-len(d)
    s=d[LIKERT];flat=(s.nunique(axis=1,dropna=True)==1)&(s.notna().sum(axis=1)>=8);log["Careless straight-line responses removed"]=int(flat.sum());d=d[~flat].copy()
    fix=0
    for c,m in {"q06_diet":{"veg":"Vegetarian","non veg":"Non-vegetarian","vegan":"Vegan"},"q03_occupation":{"student":"Student","working professional":"Working professional"},"q29_visit_intent":{"probably":"Probably","definitely":"Definitely"}}.items():
        o=d[c].astype("string");x=o.str.strip();l=x.str.lower();x=x.where(~l.isin(m),l.map(m));fix+=int((x!=o).fillna(False).sum());d[c]=x.astype(object)
    log["Inconsistent labels standardised"]=fix
    bounds={"q01_age":(15,80),"q04_distance_km":(0,60),"q07_cafe_visits_month":(0,40),"q08_current_spend":(50,3000),"q26_expected_spend":(50,3000)}
    out=0
    for c,(a,b) in bounds.items():
        v=pd.to_numeric(d[c],errors="coerce");bad=~v.between(a,b)&v.notna();out+=int(bad.sum());d[c]=v.where(~bad)
    log["Impossible outliers blanked"]=out
    d=d[d.q26_expected_spend.notna()&d.q29_visit_intent.isin(INTENT_ORDER)].copy()
    miss=int(d[NUM+CATS].isna().sum().sum());log["Missing answers filled"]=miss
    for c in NUM: d[c]=pd.to_numeric(d[c],errors="coerce");d[c]=d[c].fillna(d[c].median())
    for c in CATS: d[c]=d[c].fillna(d[c].mode().iloc[0])
    d["q27_expected_visits_month"]=pd.to_numeric(d.q27_expected_visits_month,errors="coerce").fillna(0).clip(0,20)
    d["visit"]=d.q29_visit_intent.isin(INTENT_ORDER[:2]).astype(int);log["Clean responses used"]=len(d)
    return d,log

def prep(num=NUM,cats=CATS):
    return ColumnTransformer([("n",StandardScaler(),num),("c",OneHotEncoder(handle_unknown="ignore"),cats)])

def classification(d):
    X,y=d[NUM+CATS],d.visit.values;Xa,Xb,ya,yb,ia,ib=train_test_split(X,y,d.index,test_size=.25,stratify=y,random_state=SEED);cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
    M={"Logistic Regression":make_pipeline(prep(),LogisticRegression(max_iter=2000,class_weight="balanced")),"Random Forest":make_pipeline(prep(),RandomForestClassifier(400,min_samples_leaf=4,class_weight="balanced",random_state=SEED,n_jobs=-1)),"Gradient Boosting":make_pipeline(prep(),__import__("sklearn.preprocessing",fromlist=["FunctionTransformer"]).FunctionTransformer(lambda x:x.toarray() if hasattr(x,"toarray") else x,accept_sparse=True),HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=150,random_state=SEED))}
    cmp=[{"model":n,"cv_roc_auc":cross_val_score(m,Xa,ya,cv=cv,scoring="roc_auc").mean()} for n,m in M.items()];cmp=pd.DataFrame(cmp);best=cmp.loc[cmp.cv_roc_auc.idxmax(),"model"];m=M[best].fit(Xa,ya);p=m.predict_proba(Xb)[:,1];q=(p>=.5).astype(int);cm=confusion_matrix(yb,q)
    lr=M["Logistic Regression"].fit(Xa,ya);names=lr[0].get_feature_names_out();co=pd.DataFrame({"feature":[n.split("__")[1] for n in names],"effect":lr[1].coef_[0]}).sort_values("effect",key=np.abs,ascending=False)
    oof=cross_val_predict(M[best],X,y,cv=cv,method="predict_proba")[:,1]
    tiers=d.assign(probability=oof,tier=pd.cut(oof,[-.01,.4,.7,1.01],labels=["Low","Medium","High"]).astype(str)).groupby("tier").agg(people=("visit","size"),avg_probability=("probability","mean"),actual_intent=("visit","mean"),expected_spend=("q26_expected_spend","mean"),age=("q01_age","mean")).reindex(["High","Medium","Low"]).reset_index()
    tiers["action"]=["Target first with launch offers","Nurture with reels and tasting events","No paid marketing"]
    M[best].fit(Xa,ya)
    return dict(tiers=tiers,best=best,compare=cmp,model=M[best],Xb=Xb,yb=yb,p=p,cm=cm,auc=roc_auc_score(yb,p),accuracy=accuracy_score(yb,q),precision=precision_score(yb,q),recall=recall_score(yb,q),f1=f1_score(yb,q),baseline=max(yb.mean(),1-yb.mean()),positive_rate=float(y.mean()),n_test=len(yb),drivers=co)

def clustering(d):
    F=["q01_age","q07_cafe_visits_month","q08_current_spend"]+LIKERT;X=d[F].copy();X["q08_current_spend"]=np.log1p(X.q08_current_spend);Z=StandardScaler().fit_transform(X);rows=[]
    for k in range(2,9):
        l=KMeans(k,n_init=20,random_state=SEED).fit_predict(Z);rows.append(dict(k=k,silhouette=silhouette_score(Z,l),inertia=KMeans(k,n_init=5,random_state=SEED).fit(Z).inertia_,min_share=np.bincount(l).min()/len(l)))
    t=pd.DataFrame(rows);u=t[t.k.between(3,6)&(t.min_share>=.08)];k=int((u if len(u) else t).sort_values("silhouette",ascending=False).k.iloc[0]);km=KMeans(k,n_init=30,random_state=SEED).fit(Z);d=d.assign(cluster=km.labels_)
    g=d.groupby("cluster");p=g.agg(people=("visit","size"),conversion=("visit","mean"),age=("q01_age","mean"),current_spend=("q08_current_spend","mean"),expected_spend=("q26_expected_spend","mean"),expected_visits=("q27_expected_visits_month","mean"),instagram=("q11_instagram_influence","mean"),ambience=("q13_ambience_importance","mean"),price=("q14_price_importance","mean"),health=("q15_health_conscious","mean"),vegan_open=("q16_vegan_openness","mean"),rooftop=("q17_rooftop_appeal","mean")).reset_index()
    p["tier"]=np.select([p.conversion>=.6,p.conversion>=.35],["High","Medium"],"Low")
    def nm(r):
        if r.vegan_open<2.8: return "Unconvinced Regulars"
        if r.health>=4.1: return "Health-First Professionals"
        if r.age>=38: return "Comfort-Seeking Families"
        if r.age<25 and r.price>=3.8: return "Budget Trend Students"
        return "Sunset Date Couples"
    p["segment"]=[nm(r) for r in p.itertuples()]
    p["share"]=p.people/p.people.sum();p["strategy"]=p.tier.map({"High":"Target first with launch offers and pre-booking","Medium":"Nurture with reels, tasting events and referral offers","Low":"Do not spend paid marketing here"});p=p.sort_values("conversion",ascending=False)
    pc=PCA(2,random_state=SEED).fit_transform(Z);return dict(k=k,table=t,profile=p,pc=pc,labels=km.labels_,silhouette=float(t.loc[t.k==k,"silhouette"].iloc[0]),best_any=int(t.sort_values("silhouette").k.iloc[-1]),data=d)

def association(d):
    t=d[d.visit==1];B=t[ITEM_COLS].fillna(0).astype(bool).rename(columns=lambda c:c.split("_",1)[1]);tr,te=train_test_split(B,test_size=.3,random_state=SEED);R=association_rules(apriori(tr,min_support=.06,use_colnames=True,max_len=3),metric="lift",min_threshold=1.2);R=R[R.confidence>=.45];rows=[]
    for r in R.itertuples():
        a,c=sorted(r.antecedents),sorted(r.consequents);x=te[a].all(axis=1);y=te[c].all(axis=1);cf=(x&y).sum()/max(x.sum(),1);rows.append(dict(if_interested_in=" + ".join(a),also_wants=" + ".join(c),support=r.support,confidence=r.confidence,lift=r.lift,test_confidence=cf,test_lift=cf/max(y.mean(),1e-9),n_if=len(a),n_then=len(c)))
    rules=pd.DataFrame(rows).sort_values(["lift","if_interested_in"],ascending=[False,True]);rules["holds_up"]=rules.test_lift>=1.2
    # Choose a small launch set: greedily maximise people who find at least two things they want.
    order=[];cols=list(B.columns);food=[c.split("_",1)[1] for c in ITEM_COLS if c[:3] in("q21","q22","q23")]
    for _ in range(7):
        rem=[c for c in food if c not in order];sc={c:(B[order+[c]].sum(axis=1)>=2).mean()+B[c].mean()/1000 for c in rem};best=max(sorted(sc),key=sc.get);order.append(best)
    cov=[(B[order[:i+1]].sum(axis=1)>=2).mean() for i in range(7)]
    feat=[c.split("_",1)[1] for c in ITEM_COLS if c[:3]=="q24"];top_feat=B[feat].mean().sort_values(ascending=False)
    miss=rules[(rules.n_then==1)&(rules.holds_up)&(~rules.also_wants.isin(order))&(rules.also_wants.isin(food))];good=[r for r in miss.itertuples() if r.test_confidence>=.5 and all(x in order for x in r.if_interested_in.split(" + "))]
    gap=pd.DataFrame([dict(launching=r.if_interested_in,people_also_expect=r.also_wants,confidence=r.test_confidence) for r in good]).drop_duplicates("people_also_expect") if good else pd.DataFrame(columns=["launching","people_also_expect","confidence"])
    return dict(rules=rules,launch=order,coverage=cov,features=top_feat,gap=gap,n=len(B),n_rules=len(rules),n_hold=int(rules.holds_up.sum()) if len(rules) else 0)

def regression(d):
    X,y=d[NUM+CATS],d.q26_expected_spend.values;Xa,Xb,ya,yb=train_test_split(X,y,test_size=.25,random_state=SEED);cv=KFold(5,shuffle=True,random_state=SEED);dense=__import__("sklearn.preprocessing",fromlist=["FunctionTransformer"]).FunctionTransformer(lambda x:x.toarray() if hasattr(x,"toarray") else x,accept_sparse=True)
    M={"Ridge Regression":make_pipeline(prep(),Ridge(3)),"Random Forest":make_pipeline(prep(),RandomForestRegressor(400,min_samples_leaf=4,random_state=SEED,n_jobs=-1)),"Gradient Boosting":make_pipeline(prep(),dense,HistGradientBoostingRegressor(max_depth=3,learning_rate=.05,max_iter=200,random_state=SEED))};rows=[]
    for n,m in M.items():
        cvm=-cross_val_score(m,Xa,ya,cv=cv,scoring="neg_mean_absolute_error").mean();z=m.fit(Xa,ya).predict(Xb);rows.append(dict(model=n,cv_mae=cvm,test_mae=mean_absolute_error(yb,z),rmse=mean_squared_error(yb,z)**.5,mape=mean_absolute_percentage_error(yb,z),r2=r2_score(yb,z)))
    t=pd.DataFrame(rows);best=t.loc[t.cv_mae.idxmin(),"model"];m=M[best].fit(Xa,ya);p=m.predict(Xb);bz=np.repeat(ya.mean(),len(yb));r=t[t.model==best].iloc[0]
    rg=M["Ridge Regression"].fit(Xa,ya);names=[n.split("__")[1] for n in rg[0].get_feature_names_out()];co=pd.DataFrame({"feature":names,"rupee_effect":rg[1].coef_}).sort_values("rupee_effect",key=np.abs,ascending=False);M[best].fit(Xa,ya)
    return dict(best=best,compare=t,pred=p,actual=yb,mae=r.test_mae,mape=r.mape,r2=r.r2,rmse=r.rmse,baseline_mae=mean_absolute_error(yb,bz),improve=1-r.test_mae/mean_absolute_error(yb,bz),drivers=co)

def business(d,a=None,seg=None):
    a={**A,**(a or {})};sh={k:(d.q29_visit_intent==k).mean() for k in INTENT_ORDER};conv=a["def_to_real"]*sh["Definitely"]+a["prob_to_real"]*sh["Probably"]+a["unsure_to_real"]*sh["Not sure"]
    inter=d[d.visit==1];spend=float(inter.q26_expected_spend.mean());vis=float(inter.q27_expected_visits_month.mean())*a["visit_haircut"]
    cust=a["catchment_adults"]*a["reach"]*conv;mv=cust*vis;cap=a["seats"]*a["turns_per_day"]*30.4
    fixed=a["market_rent"]+a["staff"]+a["utilities"]+a["marketing"]+a["other"];cm=1-a["food_cost"]-a["aggregator_share"]*a["aggregator_fee"]
    months=[];cum=-a["capex"];pay=None;start=1
    for i in range(60):
        mo=(start+i-1)%12+1;ramp=min(1,.45+.1*i);v=min(mv*ramp*SEASON[mo],cap);rev=v*spend;pr=rev*cm-fixed;cum+=pr;months.append(dict(month=i+1,calendar=mo,visits=v,revenue=rev,profit=pr,cumulative=cum))
        if pay is None and cum>=0: pay=i+1
    m=pd.DataFrame(months);yr=m[(m.month>12)&(m.month<=24)];rev=yr.revenue.sum()/12;pf=yr.profit.sum()/12;bep=fixed/cm;mos=1-bep/rev if rev else -1
    r=np.random.default_rng(SEED);n=20000;c=r.normal(1,.25,n).clip(.2);sp=r.normal(1,.1,n);fc=r.normal(0,.03,n);fx=r.normal(1,.08,n);v=np.minimum(mv*c*(sum(SEASON.values())/12),cap);sim=v*spend*sp*(cm-fc)-fixed*fx;pp=float((sim>0).mean())
    rent0=pf+a["market_rent"];gs={"Margin of safety":[mos,GATES["margin_of_safety"],mos>=GATES["margin_of_safety"]],"Probability of monthly profit":[pp,GATES["prob_profit"],pp>=GATES["prob_profit"]],"Payback months":[pay if pay else 999,GATES["payback_months"],pay is not None and pay<=GATES["payback_months"]]};ok=sum(x[2] for x in gs.values())
    return dict(assume=a,intent=sh,conv=conv,avg_spend=spend,visits_cust=vis,customers=cust,monthly_visits=mv,cap=cap,util=mv/cap,fixed=fixed,cm=cm,months=m,rev=rev,profit=pf,profit_no_rent=rent0,breakeven_month=bep,breakeven_daily_cust=bep/spend/30.4,capex=a["capex"],mos=mos,prob=pp,payback=pay,sim=sim,gates=gs,verdict="GO" if ok==3 else "CONDITIONAL GO" if ok==2 else "NOT YET")

def run(raw):
    d,log=clean(raw);c=classification(d);k=clustering(d);a=association(d);r=regression(d);b=business(d);return dict(data=d,log=log,clf=c,clu=k,arm=a,reg=r,biz=b)

def figs(R):
    F={}
    def new(w,h): return plt.subplots(figsize=(w,h))
    d=R["data"];b=R["biz"];c=R["clf"];k=R["clu"];a=R["arm"];r=R["reg"]
    v=[b["intent"][x] for x in INTENT_ORDER];f,ax=new(6.5,3);ax.bar(INTENT_ORDER,v,color=BLUE,width=.65);ax.set(title="Would you visit in the first month?",ylabel="Share of respondents");ax.grid(axis="x",visible=False);F["intent"]=f
    f,ax=new(4.8,4);ax.grid(False);ax.imshow(c["cm"],cmap="Blues");L=["Won't visit","Will visit"];ax.set_xticks([0,1],L);ax.set_yticks([0,1],L);ax.set(title="Confusion matrix",xlabel="Predicted",ylabel="Actual");[ax.text(j,i,c["cm"][i,j],ha="center",va="center",fontsize=16,color="white" if c["cm"][i,j]>c["cm"].max()/2 else INK) for i in range(2) for j in range(2)];ax.spines[:].set_visible(False);F["cm"]=f
    x,y,_=roc_curve(c["yb"],c["p"]);f,ax=new(4.8,4);ax.plot(x,y,color=BLUE,lw=2);ax.plot([0,1],[0,1],color=GREY,ls="--",lw=1);ax.set(title=f"ROC curve, AUC {c['auc']:.3f}",xlabel="False positive rate",ylabel="True positive rate");F["roc"]=f
    s=c["drivers"].head(8).iloc[::-1];f,ax=new(6,3.4);ax.barh(s.feature,s.effect,color=[BLUE if z>0 else ORANGE for z in s.effect],height=.65);ax.axvline(0,color=INK,lw=.8);ax.grid(axis="y",visible=False);ax.set(title="What pushes visit intent up (blue) or down (orange)",xlabel="Logistic regression effect");F["drivers"]=f
    t=k["table"];f,ax=new(6,3);ax.plot(t.k,t.silhouette,color=BLUE,lw=2,marker="o",ms=6);ax.axvline(k["k"],color=GREY,ls="--",lw=1);ax.set(title=f"Silhouette by number of clusters, chosen k = {k['k']}",xlabel="Clusters",ylabel="Silhouette");F["k"]=f
    f,ax=new(7,4)
    for i,x in enumerate(k["profile"].itertuples()):
        q=k["labels"]==x.cluster;ax.scatter(k["pc"][q,0],k["pc"][q,1],s=12,alpha=.6,linewidths=0,color=CAT[i],label=f"{x.segment} ({x.tier})")
    ax.set(title="Customer segments",xlabel="Component 1",ylabel="Component 2");ax.legend(frameon=False,fontsize=8,markerscale=2);F["seg"]=f
    p=k["profile"].iloc[::-1];f,ax=new(6.5,2.8);ax.barh(p.segment,p.conversion,color=BLUE,height=.6);ax.grid(axis="y",visible=False);ax.set(title="Stated visit intent by segment",xlabel="Share likely to visit");ax.set_xlim(0,1);F["conv"]=f
    f,ax=new(6.5,3);ax.plot(range(1,8),a["coverage"],color=BLUE,lw=2,marker="o",ms=6);ax.set_ylim(0,1);ax.set(title="How many items does the launch menu need?",xlabel="Menu items launched",ylabel="Interested people with 2+ picks");F["cov"]=f
    q=a["rules"].head(8).iloc[::-1];f,ax=new(7.5,3.6);ax.barh(q.if_interested_in+" → "+q.also_wants,q.lift,color=BLUE,height=.65);ax.axvline(1,color=GREY,ls="--",lw=1);ax.grid(axis="y",visible=False);ax.set(title="Strongest 'if A then B' rules",xlabel="Lift");F["rules"]=f
    f,ax=new(4.8,4);ax.scatter(r["actual"],r["pred"],s=18,color=BLUE,alpha=.65);l=[r["actual"].min(),r["actual"].max()];ax.plot(l,l,color=GREY,ls="--",lw=1);ax.set(title="Predicted vs stated spend",xlabel="Stated ₹",ylabel="Predicted ₹");F["reg"]=f
    z=r["drivers"].head(8).iloc[::-1];f,ax=new(6,3.4);ax.barh(z.feature,z.rupee_effect,color=[BLUE if w>0 else ORANGE for w in z.rupee_effect],height=.65);ax.axvline(0,color=INK,lw=.8);ax.grid(axis="y",visible=False);ax.set(title="What moves expected spend",xlabel="₹ effect per standardised unit");F["regd"]=f
    m=b["months"].head(36);f,ax=new(8,3.2);ax.bar(m.month,m.profit/1e5,color=[BLUE if x>=0 else ORANGE for x in m.profit],width=.75);ax.axhline(0,color=INK,lw=.8);ax.grid(axis="x",visible=False);ax.set(title="Monthly profit after market rent; orange = loss",xlabel="Month after opening",ylabel="₹ lakh");F["pnl"]=f
    f,ax=new(7,3.2);ax.hist(b["sim"]/1e5,bins=60,color=BLUE);ax.axvline(0,color=INK,lw=1);ax.grid(axis="x",visible=False);ax.set(title=f"Simulated monthly profit: {b['prob']:.0%} of scenarios are profitable",xlabel="₹ lakh per month",ylabel="Scenarios");F["sim"]=f
    return F
