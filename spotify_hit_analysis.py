"""
Hit or Flop? Predicting Billboard chart success from Spotify audio features (1960-2019)
=======================================================================================
Full reproducible pipeline:
  1. Load the six per-decade CSVs; label decade + three industry eras; clean invalid rows.
  2. EXPLORATORY DATA ANALYSIS  -> descriptive statistics + EDA figures.
  3. HYPERPARAMETER TUNING (grid search, per era) for Logistic Regression, Random Forest, KNN.
  4. Evaluate the tuned models; report accuracy / F1 / ROC-AUC.
  5. Feature importance via SHAP (tuned Random Forest) + direction via Logistic-Regression coefficients.
  6. Save result tables (results/) and publication-quality figures (figures/).

Usage:
  pip install -r requirements.txt
  python spotify_hit_analysis.py
The six dataset-of-*.csv files must be in spotify_data/.
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
import shap

# ----------------------------------------------------------------------------- config
DATA_DIR, OUT, FIG = "spotify_data", "results", "figures"
RS = 42
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
DECADES = ["60s","70s","80s","90s","00s","10s"]
ERA_MAP = {"60s":"Analog (1960-89)","70s":"Analog (1960-89)","80s":"Analog (1960-89)",
           "90s":"CD/Digital (1990-2009)","00s":"CD/Digital (1990-2009)","10s":"Streaming (2010s)"}
ERAS = ["Analog (1960-89)","CD/Digital (1990-2009)","Streaming (2010s)"]
FEATS = ["danceability","energy","key","loudness","mode","speechiness","acousticness",
         "instrumentalness","liveness","valence","tempo","duration_ms","time_signature","chorus_hit","sections"]

def load_data():
    files = {f.split("-")[-1].replace(".csv",""): f for f in glob.glob(os.path.join(DATA_DIR,"dataset-of-*.csv"))}
    df = pd.concat([pd.read_csv(files[d]).assign(decade=d, era=ERA_MAP[d]) for d in DECADES], ignore_index=True)
    n0 = len(df)
    df = df[(df.tempo > 0) & (df.time_signature > 0)].reset_index(drop=True)   # drop sensor zeros
    print(f"Loaded {n0} tracks; {len(df)} after cleaning.")
    return df

# ----------------------------------------------------------------- 1. exploratory data analysis
def run_eda(df):
    df[FEATS].describe().T[["mean","std","min","25%","50%","75%","max"]].round(3).to_csv(f"{OUT}/eda_descriptive_stats.csv")
    byc = df.groupby("target")[FEATS].mean().T; byc.columns = ["flop_mean","hit_mean"]
    byc["hit_minus_flop"] = byc.hit_mean - byc.flop_mean
    byc.round(3).to_csv(f"{OUT}/eda_by_class_means.csv")
    sns.set_style("whitegrid")
    # class balance
    cnt = df.groupby(["decade","target"]).size().unstack().reindex(DECADES); cnt.columns = ["Flop","Hit"]
    ax = cnt.plot(kind="bar", figsize=(8,4.2), color=["#B5838D","#2D6A4F"], edgecolor="white")
    ax.set_xlabel("Decade"); ax.set_ylabel("Tracks"); ax.set_title("Class balance by decade", fontweight="bold", loc="left")
    plt.xticks(rotation=0); plt.savefig(f"{FIG}/eda1_class_balance.png", dpi=300, bbox_inches="tight"); plt.close()
    # distributions hit vs flop
    df = df.assign(Outcome=df.target.map({0:"Flop",1:"Hit"}))
    show = ["instrumentalness","loudness","danceability","acousticness","energy","valence","speechiness","duration_ms"]
    fig, axes = plt.subplots(2,4, figsize=(15,7)); axes = axes.ravel()
    for i,f in enumerate(show):
        d = df[df.duration_ms <= 600000] if f == "duration_ms" else df
        sns.kdeplot(data=d, x=f, hue="Outcome", hue_order=["Hit","Flop"], common_norm=False, fill=True, alpha=.4,
                    palette={"Hit":"#2D6A4F","Flop":"#B5838D"}, ax=axes[i], legend=(i==0))
        axes[i].set_title(f, fontweight="bold"); axes[i].set_ylabel("")
    fig.suptitle("Feature distributions: hits vs flops", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout(); fig.savefig(f"{FIG}/eda2_feature_distributions.png", dpi=300, bbox_inches="tight"); plt.close()
    # correlation
    fig, ax = plt.subplots(figsize=(9,7.5))
    sns.heatmap(df[FEATS].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size":7}, square=True, ax=ax)
    ax.set_title("Feature correlation matrix", fontweight="bold", loc="left")
    fig.savefig(f"{FIG}/eda3_correlation_heatmap.png", dpi=300, bbox_inches="tight"); plt.close()
    # standardized means by era
    z = df.copy(); z[FEATS] = (z[FEATS]-z[FEATS].mean())/z[FEATS].std()
    em = z.groupby("era")[FEATS].mean().T[ERAS]
    fig, ax = plt.subplots(figsize=(7,7.5))
    sns.heatmap(em, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size":8}, ax=ax)
    ax.set_title("Feature averages drift across eras", fontweight="bold", loc="left")
    fig.savefig(f"{FIG}/eda4_feature_means_by_era.png", dpi=300, bbox_inches="tight"); plt.close()
    print("EDA done.")

# ----------------------------------------------------------------- 2. hyperparameter tuning (grid search)
def tune_models(df):
    grids = {
        "LogReg": (Pipeline([("sc",StandardScaler()),("m",LogisticRegression(max_iter=4000,solver="liblinear"))]),
                   {"m__C":[0.01,0.1,1,10,100],"m__penalty":["l1","l2"]}),
        "RandomForest": (RandomForestClassifier(random_state=RS,n_jobs=-1),
                   {"n_estimators":[150,300],"max_depth":[12,20]}),
        "KNN": (Pipeline([("sc",StandardScaler()),("m",KNeighborsClassifier(n_jobs=-1))]),
                   {"m__n_neighbors":[15,45,75],"m__weights":["uniform","distance"]}),
    }
    cv3 = StratifiedKFold(3, shuffle=True, random_state=RS)
    rows, best = [], {}
    for era in ERAS:
        sub = df[df.era == era]; X = sub[FEATS].values; y = sub.target.values
        for name,(est,grid) in grids.items():
            gs = GridSearchCV(est, grid, scoring=["accuracy","f1","roc_auc"], refit="roc_auc", cv=cv3, n_jobs=1)
            gs.fit(X,y); r = gs.cv_results_; bi = gs.best_index_
            rows.append(dict(era=era, model=name, n=len(sub),
                             accuracy=r["mean_test_accuracy"][bi], f1=r["mean_test_f1"][bi], roc_auc=r["mean_test_roc_auc"][bi],
                             best_params=json.dumps(gs.best_params_)))
            best[f"{era}|{name}"] = gs.best_params_
            print(f"  tuned {era:24s} {name:13s} AUC={r['mean_test_roc_auc'][bi]:.3f}  {gs.best_params_}")
    pd.DataFrame(rows).to_csv(f"{OUT}/grid_final.csv", index=False)
    json.dump(best, open(f"{OUT}/grid_final_params.json","w"), indent=1)
    return best

# ----------------------------------------------------------------- 3. tuned interpretation (SHAP + direction)
def interpret(df, best):
    def lr_for(era):
        p = best[f"{era}|LogReg"]; return LogisticRegression(C=p["m__C"], penalty=p["m__penalty"], solver="liblinear", max_iter=4000)
    rf_shap = lambda: RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RS, n_jobs=-1)  # depth-capped for TreeSHAP
    rf_full = lambda: RandomForestClassifier(n_estimators=300, max_depth=20, random_state=RS, n_jobs=-1)  # tuned model (AUC)
    cv3 = StratifiedKFold(3, shuffle=True, random_state=RS)
    shp, imp, coef, dec, bee = [], [], [], [], {}
    for level, groups in [("era",ERAS),("decade",DECADES)]:
        for g in groups:
            sub = df[df[level]==g]; X = sub[FEATS].values; y = sub.target.values
            rf = rf_shap().fit(X,y)
            Xs = X[np.random.RandomState(RS).choice(len(sub), min(250,len(sub)), replace=False)]
            sv = np.asarray(shap.TreeExplainer(rf).shap_values(Xs, check_additivity=False)); sv = sv[:,:,1] if sv.ndim==3 else sv
            for i,f in enumerate(FEATS):
                imp.append(dict(level=level, group=g, feature=f, importance=rf.feature_importances_[i]))
                shp.append(dict(level=level, group=g, feature=f, mean_abs_shap=float(np.abs(sv[:,i]).mean()), mean_signed_shap=float(sv[:,i].mean())))
            if level == "era":
                sc = StandardScaler().fit(X); lr = lr_for(g).fit(sc.transform(X), y)
                for i,f in enumerate(FEATS): coef.append(dict(level="era", group=g, feature=f, coef=float(lr.coef_[0][i])))
                bee[g] = (sv, Xs)
            else:
                a = cross_val_score(rf_full(), X, y, cv=cv3, scoring="roc_auc", n_jobs=1)
                dec.append(dict(decade=g, roc_auc_mean=float(a.mean()), roc_auc_std=float(a.std())))
    pd.DataFrame(imp).to_csv(f"{OUT}/rf_importance_tuned.csv", index=False)
    pd.DataFrame(shp).to_csv(f"{OUT}/shap_meanabs_tuned.csv", index=False)
    pd.DataFrame(coef).to_csv(f"{OUT}/lr_coef_tuned.csv", index=False)
    pd.DataFrame(dec).to_csv(f"{OUT}/decade_rf_auc_tuned.csv", index=False)
    print("Interpretation done.")
    return bee

# ----------------------------------------------------------------- 4. figures (see repo for full plotting)
def make_figures(best, bee):
    """Performance, SHAP heatmap, decade trend, LR-direction, and per-era beeswarms.
    Plotting code is intentionally compact; see figures/ for the rendered outputs."""
    g = pd.read_csv(f"{OUT}/grid_final.csv"); sh = pd.read_csv(f"{OUT}/shap_meanabs_tuned.csv")
    dec = pd.read_csv(f"{OUT}/decade_rf_auc_tuned.csv"); lr = pd.read_csv(f"{OUT}/lr_coef_tuned.csv")
    es = ["Analog\n1960-89","CD/Digital\n1990-2009","Streaming\n2010s"]; modc = {"LogReg":"#8C9EB2","RandomForest":"#2D6A4F","KNN":"#B5838D"}
    # performance
    fig, (a1,a2) = plt.subplots(1,2, figsize=(12,4.6), gridspec_kw=dict(width_ratios=[1.25,1])); x=np.arange(3); w=.26
    for i,m in enumerate(["LogReg","KNN","RandomForest"]):
        d = g[g.model==m].set_index("era").loc[ERAS]
        a1.bar(x+(i-1)*w, d.roc_auc, w, color=modc[m], label={"LogReg":"Logistic Reg.","KNN":"KNN","RandomForest":"Random Forest"}[m])
    a1.set_xticks(x); a1.set_xticklabels(es); a1.set_ylim(.7,.95); a1.set_ylabel("ROC-AUC (tuned)"); a1.legend(frameon=False); a1.set_title("(a) Tuned performance by era", loc="left", fontweight="bold")
    a2.plot(DECADES, dec.set_index("decade").loc[DECADES].roc_auc_mean, "-o", color="#2D6A4F", lw=2.4); a2.set_ylim(.8,.95); a2.set_title("(b) Predictability over time (RF)", loc="left", fontweight="bold"); a2.set_xlabel("Decade")
    fig.savefig(f"{FIG}/fig1_model_performance.png", dpi=300, bbox_inches="tight"); plt.close()
    # SHAP heatmap
    piv = sh[sh.level=="era"].pivot(index="feature", columns="group", values="mean_abs_shap")[ERAS]
    piv = piv.loc[piv.mean(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.4,7)); im = ax.imshow(piv.values, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(3)); ax.set_xticklabels(es); ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    fig.colorbar(im, fraction=.046, pad=.04); ax.set_title("Feature importance by era (SHAP)", fontweight="bold", loc="left")
    fig.savefig(f"{FIG}/fig2_shap_heatmap_era.png", dpi=300, bbox_inches="tight"); plt.close()
    # decade trend
    dp = sh[sh.level=="decade"].pivot(index="feature", columns="group", values="mean_abs_shap")[DECADES]
    fig, ax = plt.subplots(figsize=(9.2,5.4))
    for f in ["instrumentalness","danceability","loudness","acousticness","speechiness","duration_ms"]:
        ax.plot(DECADES, dp.loc[f], "-o", lw=2.2, label=f)
    ax.set_xlabel("Decade"); ax.set_ylabel("mean |SHAP|"); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5,-.12)); ax.set_title("Feature-importance trend across decades", fontweight="bold", loc="left")
    fig.savefig(f"{FIG}/fig3_shap_trend_decade.png", dpi=300, bbox_inches="tight"); plt.close()
    # LR direction
    lp = lr[lr.level=="era"].pivot(index="feature", columns="group", values="coef")[ERAS]; lp = lp.loc[lp.abs().mean(1).sort_values(ascending=False).index]; mx = np.abs(lp.values).max()
    fig, ax = plt.subplots(figsize=(7.2,7)); im = ax.imshow(lp.values, aspect="auto", cmap="RdBu_r", vmin=-mx, vmax=mx)
    ax.set_xticks(range(3)); ax.set_xticklabels(es); ax.set_yticks(range(len(lp))); ax.set_yticklabels(lp.index)
    fig.colorbar(im, fraction=.046, pad=.04); ax.set_title("Direction of effect by era (LR coefficients)", fontweight="bold", loc="left")
    fig.savefig(f"{FIG}/fig4_lr_direction_era.png", dpi=300, bbox_inches="tight"); plt.close()
    # beeswarms
    for era,(sv,Xs) in bee.items():
        plt.figure(); shap.summary_plot(sv, Xs, feature_names=FEATS, show=False, max_display=12, plot_size=(7.5,5.2))
        plt.title(f"SHAP summary — {era}", fontweight="bold", loc="left"); plt.tight_layout()
        plt.savefig(f"{FIG}/fig5_beeswarm_{era.split(' ')[0].replace('/','_')}.png", dpi=300, bbox_inches="tight"); plt.close()
    print("Figures done.")

if __name__ == "__main__":
    df = load_data()
    print("\n[1/4] Exploratory data analysis ..."); run_eda(df)
    print("\n[2/4] Hyperparameter tuning (grid search) ..."); best = tune_models(df)
    print("\n[3/4] Tuned interpretation (SHAP) ..."); bee = interpret(df, best)
    print("\n[4/4] Figures ..."); make_figures(best, bee)
    print("\nDone. Tables in results/ , figures in figures/")
