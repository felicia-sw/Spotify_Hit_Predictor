import pandas as pd, numpy as np, glob, os, time, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import shap
OUT="results/sensitivity.csv"
order=['60s','70s','80s','90s','00s','10s']
files={f.split('-')[-1].replace('.csv',''):f for f in glob.glob("spotify_data/dataset-of-*.csv")}
base=pd.concat([pd.read_csv(files[d]).assign(decade=d) for d in order],ignore_index=True)
feats=['danceability','energy','key','loudness','mode','speechiness','acousticness',
       'instrumentalness','liveness','valence','tempo','duration_ms','time_signature','chorus_hit','sections']
base=base[(base['tempo']>0)&(base['time_signature']>0)].reset_index(drop=True)
# schemes: list of (scheme, ordered_group_labels, decade->label map, dataframe)
alt2={'60s':'Pre-2000','70s':'Pre-2000','80s':'Pre-2000','90s':'Pre-2000','00s':'2000s+','10s':'2000s+'}
alt3={'60s':'Early','70s':'Early','80s':'Mid','90s':'Mid','00s':'Late','10s':'Late'}
main={'60s':'Analog','70s':'Analog','80s':'Analog','90s':'CD/Digital','00s':'CD/Digital','10s':'Streaming'}
ded=base.drop_duplicates('uri',keep='first').reset_index(drop=True)
schemes=[('alt2_2era',['Pre-2000','2000s+'],alt2,base),
         ('alt3_pairs',['Early','Mid','Late'],alt3,base),
         ('dedup_main',['Analog','CD/Digital','Streaming'],main,ded)]
cv3=StratifiedKFold(3,shuffle=True,random_state=42)
done=set()
if os.path.exists(OUT): done={(r.scheme,r.group) for r in pd.read_csv(OUT).itertuples()}
t0=time.time()
for scheme,labels,mp,df in schemes:
    d=df.assign(grp=df.decade.map(mp))
    for g in labels:
        if (scheme,g) in done: continue
        sub=d[d.grp==g]; X=sub[feats].values; y=sub['target'].values
        auc=cross_val_score(RandomForestClassifier(n_estimators=300,max_depth=20,random_state=42,n_jobs=-1),X,y,cv=cv3,scoring='roc_auc',n_jobs=1).mean()
        rf=RandomForestClassifier(n_estimators=150,max_depth=12,random_state=42,n_jobs=-1).fit(X,y)
        rng=np.random.RandomState(42); Xs=X[rng.choice(len(sub),min(200,len(sub)),replace=False)]
        sv=np.asarray(shap.TreeExplainer(rf).shap_values(Xs,check_additivity=False)); sv=sv[:,:,1] if sv.ndim==3 else sv
        ma=np.abs(sv).mean(0)
        recs=[{'scheme':scheme,'group':g,'n':len(sub),'metric':'roc_auc','feature':'','value':round(float(auc),4)}]
        recs+=[{'scheme':scheme,'group':g,'n':len(sub),'metric':'mean_abs_shap','feature':feats[i],'value':round(float(ma[i]),4)} for i in range(len(feats))]
        pd.DataFrame(recs).to_csv(OUT,mode='a',header=not os.path.exists(OUT),index=False)
        print(f"done {scheme}:{g} n={len(sub)} auc={auc:.3f} [{time.time()-t0:.0f}s]",flush=True)
        if time.time()-t0>22: print("...bail",flush=True); raise SystemExit
print("ALL_DONE",flush=True)
