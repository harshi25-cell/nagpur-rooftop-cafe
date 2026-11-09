"""Creates a mock survey with intentional noise and outliers. Replace its output with real responses when available."""
import numpy as np, pandas as pd, csv
from pathlib import Path
from survey_schema import *
ROOT=Path(__file__).parent; SEED=42; N=1000
P={ # share age occ_prob(St,Wk,Bus,Home,Oth) inc_probs diet(V,Veg,Egg,Non) visits spend ig amb price health vegan roof base
"Students":(.28,21,(.9,.06,.01,0,.03),(.45,.33,.15,.05,.02),(.02,.42,.16,.40),5,260,4.3,4.2,4.4,3,3.3,4.2,.35),
"Couples":(.24,27,(.1,.72,.12,.02,.04),(.05,.25,.42,.22,.06),(.03,.42,.15,.40),5,480,4.0,4.6,3.4,3.3,3.5,4.6,.75),
"Health":(.16,34,(.02,.65,.2,.1,.03),(.02,.1,.33,.38,.17),(.2,.55,.1,.15),4,620,2.9,3.9,2.9,4.6,4.5,3.4,.9),
"Families":(.17,44,(.01,.38,.3,.28,.03),(.02,.12,.32,.36,.18),(.02,.75,.12,.11),3,520,2.4,3.7,3.5,3.7,3.2,3.2,.3),
"NotFit":(.15,33,(.15,.55,.15,.1,.05),(.25,.35,.25,.12,.03),(.0,.08,.12,.8),2,240,2.5,2.5,4.5,2.3,1.8,2.6,-1.2)}
OCC=["Student","Working professional","Business owner","Homemaker","Other"];INC=["Below 25k","25-50k","50k-1L","1-2L","Above 2L"];D=["Vegan","Vegetarian","Eggetarian","Non-vegetarian"]
BASE={"Students":dict(orange_matcha=.45,cold_brew=.45,oat_latte=.25,filter_coffee=.25,orange_cooler=.45,kombucha=.12,vegan_burger=.42,cheese_pizza=.55,paneer_sandwich=.45,avocado_toast=.2,buddha_bowl=.12,thecha_tofu_wrap=.35,vegan_brownie=.5,regular_ice_cream=.5,vegan_cheesecake=.28,orange_tart=.25,board_games=.6,wifi_workspace=.45,live_acoustic=.45,sunset_seating=.45,plant_wall=.35,private_celebrations=.2),
"Couples":dict(orange_matcha=.5,oat_latte=.42,cold_brew=.35,filter_coffee=.25,orange_cooler=.4,kombucha=.2,avocado_toast=.42,buddha_bowl=.28,paneer_sandwich=.4,cheese_pizza=.45,vegan_burger=.3,thecha_tofu_wrap=.28,vegan_cheesecake=.55,orange_tart=.4,vegan_brownie=.35,regular_ice_cream=.3,sunset_seating=.8,plant_wall=.6,private_celebrations=.5,live_acoustic=.5,board_games=.2,wifi_workspace=.2),
"Health":dict(oat_latte=.55,orange_matcha=.4,kombucha=.45,cold_brew=.3,filter_coffee=.3,orange_cooler=.45,buddha_bowl=.7,avocado_toast=.6,thecha_tofu_wrap=.3,vegan_burger=.3,paneer_sandwich=.25,cheese_pizza=.1,vegan_cheesecake=.45,orange_tart=.25,vegan_brownie=.3,regular_ice_cream=.05,wifi_workspace=.5,plant_wall=.5,sunset_seating=.4,live_acoustic=.2,board_games=.1,private_celebrations=.2),
"Families":dict(filter_coffee=.55,orange_cooler=.55,oat_latte=.1,orange_matcha=.12,cold_brew=.15,kombucha=.05,paneer_sandwich=.6,cheese_pizza=.65,thecha_tofu_wrap=.3,vegan_burger=.15,avocado_toast=.12,buddha_bowl=.12,regular_ice_cream=.65,vegan_brownie=.25,orange_tart=.3,vegan_cheesecake=.15,private_celebrations=.6,sunset_seating=.35,board_games=.3,plant_wall=.2,live_acoustic=.25,wifi_workspace=.05),
"NotFit":dict(filter_coffee=.3,cold_brew=.2,orange_cooler=.2,oat_latte=.03,orange_matcha=.05,kombucha=.02,cheese_pizza=.4,paneer_sandwich=.25,vegan_burger=.08,avocado_toast=.05,buddha_bowl=.03,thecha_tofu_wrap=.15,regular_ice_cream=.3,vegan_brownie=.08,orange_tart=.08,vegan_cheesecake=.04,wifi_workspace=.2,board_games=.15,sunset_seating=.12,plant_wall=.05,live_acoustic=.1,private_celebrations=.08)}
def main():
    r=np.random.default_rng(SEED);names=list(P);rows=[];key=[]
    L=lambda m:int(np.clip(round(r.normal(m,.8)),1,5))
    for i in range(N):
        p=names[r.choice(5,p=[P[x][0] for x in names])];s,a,oc,ic,di,v,sp,ig,am,pr,he,ve,ro,base=P[p]
        age=int(np.clip(round(r.normal(a,3.5 if p=="Students" else 7)),16,68));inc=r.choice(5,p=ic);diet=D[r.choice(4,p=di)];dist=round(float(np.clip(r.gamma(2.2,2.3),.3,28)),1)
        vals=dict(q11_instagram_influence=L(ig),q13_ambience_importance=L(am),q14_price_importance=L(pr),q15_health_conscious=L(he),q16_vegan_openness=L(ve+(.5 if diet=="Vegan" else 0)),q17_rooftop_appeal=L(ro),q18_cooled_section_summer=L(3.6),q19_family_comfort=L(4.3 if diet in("Vegan","Vegetarian") else 3),q20_sustainability=L(3.2+.3*(he>4)))
        cur=max(80,sp*(.75+.12*inc)*r.lognormal(0,.25))
        z=base+.45*(vals["q16_vegan_openness"]-3)+.42*(vals["q17_rooftop_appeal"]-3)+.3*(vals["q13_ambience_importance"]-3)+.22*(vals["q11_instagram_influence"]-3)-.14*(dist-5)-.28*(vals["q14_price_importance"]-3)+.25*(vals["q19_family_comfort"]-3)+.6*(diet=="Vegan")-.5*(diet=="Non-vegetarian")+r.logistic(0,.75)
        intent=INTENT_ORDER[0 if z>1.7 else 1 if z>.55 else 2 if z>-.35 else 3 if z>-1.5 else 4]
        spend=max(100,(.55*cur+130+45*(vals["q13_ambience_importance"]-3)-35*(vals["q14_price_importance"]-3)+40*inc+45*(p=="Couples")+60*(p=="Families"))*r.lognormal(0,.15))
        rec=dict(respondent_id=f"R{i+1:05d}",q01_age=age,q02_gender=r.choice(["Female","Male","Prefer not to say"],p=[.52,.46,.02]),q03_occupation=OCC[r.choice(5,p=oc)],q04_distance_km=dist,q05_income=INC[inc],q06_diet=diet,q07_cafe_visits_month=int(r.poisson(v)),q08_current_spend=int(round(cur,-1)),q09_companions=r.choice(["Friends","Partner","Family","Alone","Colleagues"],p={"Students":[.75,.12,.05,.05,.03],"Couples":[.2,.65,.05,.05,.05],"Health":[.3,.2,.15,.2,.15],"Families":[.08,.1,.75,.02,.05],"NotFit":[.5,.15,.15,.1,.1]}[p]),q10_time_slot=r.choice(["Morning","Afternoon","Evening","Night"],p=[.12,.2,.53,.15]),q12_discovery=r.choice(["Instagram","Friends","Google Maps","Zomato-Swiggy","Walking past"],p=[.45,.25,.14,.1,.06] if ig>3.5 else [.1,.45,.2,.15,.1]))
        rec.update(vals)
        b=BASE[p];pick={k:r.random()<b[k] for k in b}
        if pick["orange_matcha"] and pick["vegan_cheesecake"]: pick["orange_tart"]=r.random()<.72
        if pick["sunset_seating"] and pick["plant_wall"]: pick["private_celebrations"]=r.random()<.65
        if pick["paneer_sandwich"] and pick["regular_ice_cream"]: pick["cheese_pizza"]=r.random()<.78
        if pick["buddha_bowl"] and pick["oat_latte"]: pick["kombucha"]=r.random()<.62
        for c in ITEM_COLS: rec[c]=int(pick[c.split("_",1)[1]])
        vis=max(0,round(r.gamma(1.6,1.1)*(1 if intent in INTENT_ORDER[:2] else .4)))
        rec.update(q25_signature_drink_price=int(round(max(80,.32*spend+r.normal(70,30)),-1)),q26_expected_spend=int(round(spend,-1)),q27_expected_visits_month=int(vis),q28_plant_milk_premium=int(max(0,round(r.normal(30+15*(he>4),15),-1))),q29_visit_intent=intent,q30_comment=r.choice(["Love the rooftop idea","Please keep prices fair","Would come for sunset views","Vegan food must taste good","Needs parking","Worried about summer heat","Hope there are regular veg options","Looks perfect for celebrations",""]))
        rows.append(rec);key.append(p)
    d=pd.DataFrame(rows)
    # intentional noise and outliers
    idx=lambda k:r.choice(N,k,replace=False)
    d.loc[idx(12),"q01_age"]=[120,3,150,999,0,200,131,5,8,250,140,175]
    d.loc[idx(10),"q26_expected_spend"]=[25000,50000,15000,9999,30000,40000,20000,12000,18000,75000]
    d.loc[idx(9),"q08_current_spend"]=[20000,0,15000,12000,50000,30000,25000,9999,1]
    d.loc[idx(8),"q04_distance_km"]=[999,250,400,120,800,600,500,300]
    d.loc[idx(6),"q07_cafe_visits_month"]=[300,200,150,90,120,250]
    for c,v in {"q06_diet":{"Vegetarian":"veg","Non-vegetarian":"non veg ","Vegan":"VEGAN"},"q03_occupation":{"Student":"student ","Working professional":"Working Professional"},"q29_visit_intent":{"Probably":"probably","Definitely":"DEFINITELY"}}.items():
        i=idx(35);d.loc[i,c]=d.loc[i,c].replace(v)
    d[LIKERT]=d[LIKERT].astype("Float64")
    for c in LIKERT+["q05_income","q08_current_spend","q04_distance_km","q09_companions"]: d.loc[idx(28),c]=pd.NA
    i=idx(18);d.loc[i,LIKERT]=3.0   # careless straight-liners
    for c in LIKERT: d[c]=d[c].astype(float)
    d=pd.concat([d,d.sample(20,random_state=SEED)],ignore_index=True).sample(frac=1,random_state=SEED).reset_index(drop=True)
    (ROOT/"data").mkdir(exist_ok=True);d.to_csv(ROOT/"data"/"Harshita_Chopda_Survey_Data.csv",index=False)
    (ROOT/"survey").mkdir(exist_ok=True)
    with open(ROOT/"survey"/"Harshita_Chopda_Survey_Questions.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["number","column_or_prefix","question","answer_format"]);[w.writerow([n+1,*q]) for n,q in enumerate(Q)]
    print(len(d),d.q29_visit_intent.str.title().value_counts().to_dict())
if __name__=="__main__":main()
