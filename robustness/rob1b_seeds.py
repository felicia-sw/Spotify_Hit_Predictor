import pandas as pd, numpy as np, glob, os, time, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
import shap
OUT="results/shap_seed300.csv"   # matched config: 300 trees, depth 12, sample 250 (same as main interpretation)
order=['60s','70s','80s','90s','00s','10s']
files={f.split('-')[-1].replace('.csv',''):f for f in glob.glob("spotify_data/dataset-of-*.csv")}
era_map={'60s':'Analog (1960-89)','70s':'Analog (1960-89)','80s':'Analog (1960-89)',
         '90s':'CD/Digital (1990-2009)','00s':'CD/Digital (1990-2009)','10s':'Streaming (2010s)'}
big=pd.concat([pd.read_csv(files[d]).assign(era=era_map[d]) for d in order],ignore_index=True)
feats=['danceability','energy','key','loudness','mode','speechiness','acousticness',
       'instrumentalness','liveness','valence','tempo','duration_ms','time_signature','chorus_hit','sections']
big=big[(big['tempo']>0)&(big['time_signature']>0)].reset_index(drop=True)
eras=['Analog (1960-89)','CD/Digital (1990-2009)','Streaming (2010s)']
SEEDS=[42,1,2,3,4]   # 42 = the seed used in the main table, so the CI brackets Table 4
done=set()
if os.path.exists(OUT): done={(r.era,r.seed) for r in pd.read_csv(OUT).itertuples()}
t0=time.time()
for era in eras:
    sub=big[big.era==era]; X=sub[feats].values; y=sub['target'].values
    for s in SEEDS:
        if (era,s) in done: continue
        rf=RandomForestClassifier(n_estimators=300,max_depth=12,random_state=s,n_jobs=-1).fit(X,y)
        rng=np.random.RandomState(s); Xs=X[rng.choice(len(sub),min(250,len(sub)),replace=False)]
        sv=np.asarray(shap.TreeExplainer(rf).shap_values(Xs,check_additivity=False)); sv=sv[:,:,1] if sv.ndim==3 else sv
        ma=np.abs(sv).mean(0)
        rows=[{'era':era,'seed':s,'feature':feats[i],'mean_abs_shap':float(ma[i])} for i in range(len(feats))]
        pd.DataFrame(rows).to_csv(OUT,mode='a',header=not os.path.exists(OUT),index=False)
        print(f"done {era} seed{s} [{time.time()-t0:.0f}s]",flush=True)
        if time.time()-t0>30: print("...bail",flush=True); raise SystemExit
print("ALL_DONE",flush=True)
