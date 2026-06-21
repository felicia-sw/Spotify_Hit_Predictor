import pandas as pd, numpy as np, glob, os, json, time, warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
order=['60s','70s','80s','90s','00s','10s']
files={f.split('-')[-1].replace('.csv',''):f for f in glob.glob("spotify_data/dataset-of-*.csv")}
era_map={'60s':'Analog (1960-89)','70s':'Analog (1960-89)','80s':'Analog (1960-89)',
         '90s':'CD/Digital (1990-2009)','00s':'CD/Digital (1990-2009)','10s':'Streaming (2010s)'}
big=pd.concat([pd.read_csv(files[d]).assign(era=era_map[d]) for d in order],ignore_index=True)
feats=['danceability','energy','key','loudness','mode','speechiness','acousticness',
       'instrumentalness','liveness','valence','tempo','duration_ms','time_signature','chorus_hit','sections']
big=big[(big['tempo']>0)&(big['time_signature']>0)].reset_index(drop=True)
eras=['Analog (1960-89)','CD/Digital (1990-2009)','Streaming (2010s)']
P=json.load(open("results/grid_final_params.json"))
def mk(model,era):
    if model=='LogReg':
        p=P[f"{era}|LogReg"]; return Pipeline([('sc',StandardScaler()),('m',LogisticRegression(C=p['m__C'],penalty=p['m__penalty'],solver='liblinear',max_iter=4000))])
    if model=='RandomForest':
        return RandomForestClassifier(n_estimators=300,max_depth=20,random_state=42,n_jobs=-1)
    p=P[f"{era}|KNN"]; return Pipeline([('sc',StandardScaler()),('m',KNeighborsClassifier(n_neighbors=p['m__n_neighbors'],weights=p['m__weights'],n_jobs=-1))])
cv=StratifiedKFold(5,shuffle=True,random_state=42)
os.makedirs("results/oof",exist_ok=True)
def safe(e): return e.split(' ')[0].replace('/','_')
combos=[(e,m) for e in eras for m in ['LogReg','RandomForest','KNN']]
t0=time.time()
for era,model in combos:
    fn=f"results/oof/oof_{safe(era)}_{model}.npz"
    if os.path.exists(fn): continue
    sub=big[big.era==era]; X=sub[feats].values; y=sub['target'].values
    proba=cross_val_predict(mk(model,era),X,y,cv=cv,method='predict_proba',n_jobs=1)[:,1]
    np.savez(fn,y=y,proba=proba)
    print(f"done {era} {model} n={len(y)} [{time.time()-t0:.0f}s]",flush=True)
    if time.time()-t0>20: print("...bail",flush=True); raise SystemExit
print("ALL_DONE",flush=True)
