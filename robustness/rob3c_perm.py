import pandas as pd, numpy as np, glob, os, time, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
OUT="results/perm_importance.csv"
order=['60s','70s','80s','90s','00s','10s']
files={f.split('-')[-1].replace('.csv',''):f for f in glob.glob("spotify_data/dataset-of-*.csv")}
era_map={'60s':'Analog (1960-89)','70s':'Analog (1960-89)','80s':'Analog (1960-89)',
         '90s':'CD/Digital (1990-2009)','00s':'CD/Digital (1990-2009)','10s':'Streaming (2010s)'}
big=pd.concat([pd.read_csv(files[d]).assign(era=era_map[d]) for d in order],ignore_index=True)
feats=['danceability','energy','key','loudness','mode','speechiness','acousticness',
       'instrumentalness','liveness','valence','tempo','duration_ms','time_signature','chorus_hit','sections']
big=big[(big['tempo']>0)&(big['time_signature']>0)].reset_index(drop=True)
eras=['Analog (1960-89)','CD/Digital (1990-2009)','Streaming (2010s)']
done=set()
if os.path.exists(OUT): done={r.era for r in pd.read_csv(OUT).itertuples()}
t0=time.time()
for era in eras:
    if era in done: continue
    sub=big[big.era==era]; X=sub[feats].values; y=sub['target'].values
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
    if len(Xte)>5000:
        idx=np.random.RandomState(42).choice(len(Xte),5000,replace=False); Xte,yte=Xte[idx],yte[idx]
    rf=RandomForestClassifier(n_estimators=300,max_depth=20,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    r=permutation_importance(rf,Xte,yte,scoring='roc_auc',n_repeats=5,random_state=42,n_jobs=1)
    recs=[{'era':era,'feature':feats[i],'perm_imp_mean':round(float(r.importances_mean[i]),5),'perm_imp_std':round(float(r.importances_std[i]),5)} for i in range(len(feats))]
    pd.DataFrame(recs).to_csv(OUT,mode='a',header=not os.path.exists(OUT),index=False)
    print(f"done {era} [{time.time()-t0:.0f}s]",flush=True)
    if time.time()-t0>22: print("...bail",flush=True); raise SystemExit
print("ALL_DONE",flush=True)
