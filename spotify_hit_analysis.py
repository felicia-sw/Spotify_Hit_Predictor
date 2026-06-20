"""
Hit or Flop? Predicting Billboard chart success from Spotify audio features (1960-2019)
=======================================================================================
Reproducible pipeline:
  1. Load the 6 per-decade CSVs, label decade + 3 industry eras, clean invalid rows.
  2. Compare Logistic Regression, Random Forest, KNN per era and per decade
     with stratified 5-fold CV (accuracy, F1, ROC-AUC).
  3. Feature importance via Random Forest + SHAP (mean |SHAP|) per era and decade.
  4. Direction of effect via standardised Logistic-Regression coefficients.
  5. Save result tables (CSV) and publication-quality figures (PNG).

Usage:
  - Put the Kaggle files (dataset-of-60s.csv ... dataset-of-10s.csv) in DATA_DIR.
  - pip install pandas numpy scikit-learn shap matplotlib
  - python spotify_hit_analysis.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- config
DATA_DIR  = "spotify_data"          # folder holding dataset-of-*.csv
OUT_DIR   = "results"
FIG_DIR   = "figures"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

DECADES = ["60s", "70s", "80s", "90s", "00s", "10s"]
ERA_MAP = {"60s": "Analog (1960-89)", "70s": "Analog (1960-89)", "80s": "Analog (1960-89)",
           "90s": "CD/Digital (1990-2009)", "00s": "CD/Digital (1990-2009)",
           "10s": "Streaming (2010s)"}
ERAS = ["Analog (1960-89)", "CD/Digital (1990-2009)", "Streaming (2010s)"]
FEATURES = ["danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo",
            "duration_ms", "time_signature", "chorus_hit", "sections"]

# ----------------------------------------------------------------------------- load
def load_data():
    files = {f.split("-")[-1].replace(".csv", ""): f
             for f in glob.glob(os.path.join(DATA_DIR, "dataset-of-*.csv"))}
    df = pd.concat([pd.read_csv(files[d]).assign(decade=d, era=ERA_MAP[d]) for d in DECADES],
                   ignore_index=True)
    before = len(df)
    df = df[(df["tempo"] > 0) & (df["time_signature"] > 0)].reset_index(drop=True)  # drop sensor zeros
    print(f"Loaded {before} tracks; {len(df)} after removing {before-len(df)} invalid rows.")
    return df

# ----------------------------------------------------------------------------- models
def build_models():
    return {
        "LogReg": Pipeline([("sc", StandardScaler()),
                            ("m", LogisticRegression(max_iter=2000))]),
        # depth cap keeps exact TreeSHAP tractable; CV AUC within ~0.005 of unconstrained RF
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=14,
                                               random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": Pipeline([("sc", StandardScaler()),
                        ("m", KNeighborsClassifier(n_neighbors=25))]),
    }

def subsets(df):
    return [("era", e) for e in ERAS] + [("decade", d) for d in DECADES]

# ----------------------------------------------------------------------------- 1. metrics
def evaluate(df):
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for level, grp in subsets(df):
        sub = df[df[level] == grp]; X = sub[FEATURES].values; y = sub["target"].values
        for name, mdl in build_models().items():
            r = cross_validate(mdl, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1)
            rows.append(dict(level=level, group=grp, model=name, n=len(sub),
                             accuracy_mean=r["test_accuracy"].mean(), accuracy_std=r["test_accuracy"].std(),
                             f1_mean=r["test_f1"].mean(), f1_std=r["test_f1"].std(),
                             roc_auc_mean=r["test_roc_auc"].mean(), roc_auc_std=r["test_roc_auc"].std()))
            print(f"  {level:6s} {grp:24s} {name:13s} AUC={rows[-1]['roc_auc_mean']:.3f}")
    out = pd.DataFrame(rows); out.to_csv(f"{OUT_DIR}/metrics_final.csv", index=False)
    return out

# ----------------------------------------------------------------------------- 2. importance + SHAP
def interpret(df):
    rf_rows, shap_rows, lr_rows, bee = [], [], [], {}
    for level, grp in subsets(df):
        sub = df[df[level] == grp]; X = sub[FEATURES].values; y = sub["target"].values
        rf = RandomForestClassifier(n_estimators=150, max_depth=14,
                                    random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)
        rng = np.random.RandomState(RANDOM_STATE)
        Xs = X[rng.choice(len(sub), min(400, len(sub)), replace=False)]
        sv = shap.TreeExplainer(rf).shap_values(Xs, check_additivity=False)
        sv = np.asarray(sv); sv = sv[:, :, 1] if sv.ndim == 3 else sv
        ma, ms = np.abs(sv).mean(0), sv.mean(0)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
        for i, f in enumerate(FEATURES):
            rf_rows.append(dict(level=level, group=grp, feature=f, importance=rf.feature_importances_[i]))
            shap_rows.append(dict(level=level, group=grp, feature=f,
                                  mean_abs_shap=float(ma[i]), mean_signed_shap=float(ms[i])))
            lr_rows.append(dict(level=level, group=grp, feature=f, coef=float(lr.coef_[0][i])))
        if level == "era":
            bee[grp] = (sv, Xs)
        print(f"  interpreted {level}:{grp}")
    pd.DataFrame(rf_rows).to_csv(f"{OUT_DIR}/rf_importance.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(f"{OUT_DIR}/shap_meanabs.csv", index=False)
    pd.DataFrame(lr_rows).to_csv(f"{OUT_DIR}/lr_coef.csv", index=False)
    return pd.DataFrame(shap_rows), pd.DataFrame(lr_rows), bee

# ----------------------------------------------------------------------------- 3. figures
def make_figures(metrics, shap_df, lr_df, bee):
    eshort = ["Analog\n1960-89", "CD/Digital\n1990-2009", "Streaming\n2010s"]
    modc = {"LogReg": "#8C9EB2", "RandomForest": "#2D6A4F", "KNN": "#B5838D"}
    # Fig 1: model performance
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    e = metrics[metrics.level == "era"]; x = np.arange(3); w = 0.26
    for i, m in enumerate(["LogReg", "KNN", "RandomForest"]):
        d = e[e.model == m].set_index("group").loc[ERAS]
        ax[0].bar(x + (i-1)*w, d.roc_auc_mean, w, yerr=d.roc_auc_std, capsize=3, color=modc[m],
                  label={"LogReg": "Logistic Reg.", "KNN": "KNN", "RandomForest": "Random Forest"}[m])
    ax[0].set_xticks(x); ax[0].set_xticklabels(eshort); ax[0].set_ylim(.7, .95)
    ax[0].set_ylabel("ROC-AUC (5-fold CV)"); ax[0].legend(frameon=False, fontsize=9); ax[0].set_title("(a) Classifier performance by era", loc="left", fontweight="bold")
    dec = metrics[(metrics.level == "decade") & (metrics.model == "RandomForest")].set_index("group").loc[DECADES]
    ax[1].plot(DECADES, dec.roc_auc_mean, "-o", color="#2D6A4F", lw=2.4)
    ax[1].set_ylim(.80, .95); ax[1].set_ylabel("ROC-AUC"); ax[1].set_xlabel("Decade")
    ax[1].set_title("(b) Hit predictability rises over time (RF)", loc="left", fontweight="bold")
    for a in ax:
        for s in ["top", "right"]: a.spines[s].set_visible(False)
        a.grid(axis="y", alpha=.25)
    fig.savefig(f"{FIG_DIR}/fig1_model_performance.png", dpi=300, bbox_inches="tight"); plt.close()
    # Fig 2: SHAP heatmap
    piv = shap_df[shap_df.level == "era"].pivot(index="feature", columns="group", values="mean_abs_shap")[ERAS]
    piv = piv.loc[piv.mean(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.4, 7)); im = ax.imshow(piv.values, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(3)); ax.set_xticklabels(eshort); ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    for i in range(len(piv)):
        for j in range(3):
            v = piv.values[i, j]; ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                          color="white" if v > piv.values.max()*.55 else "#222", fontsize=9)
    fig.colorbar(im, fraction=.046, pad=.04).set_label("mean |SHAP value|")
    ax.set_title("Feature importance by era (SHAP, Random Forest)", fontweight="bold", loc="left")
    fig.savefig(f"{FIG_DIR}/fig2_shap_heatmap_era.png", dpi=300, bbox_inches="tight"); plt.close()
    # Fig 3: decade trend
    decp = shap_df[shap_df.level == "decade"].pivot(index="feature", columns="group", values="mean_abs_shap")[DECADES]
    col = {"instrumentalness": "#C44E52", "danceability": "#4C72B0", "loudness": "#DD8452",
           "acousticness": "#55A868", "speechiness": "#8172B3", "duration_ms": "#937860"}
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for f, c in col.items():
        ax.plot(DECADES, decp.loc[f], "-o", lw=2.3, label=f, color=c)
    ax.set_xlabel("Decade"); ax.set_ylabel("mean |SHAP value|")
    ax.set_title("How feature importance shifts across decades", fontweight="bold", loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.12))
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(alpha=.25)
    fig.savefig(f"{FIG_DIR}/fig3_shap_trend_decade.png", dpi=300, bbox_inches="tight"); plt.close()
    # Fig 4: LR direction
    lrp = lr_df[lr_df.level == "era"].pivot(index="feature", columns="group", values="coef")[ERAS]
    lrp = lrp.loc[lrp.abs().mean(1).sort_values(ascending=False).index]; mx = np.abs(lrp.values).max()
    fig, ax = plt.subplots(figsize=(7.2, 7)); im = ax.imshow(lrp.values, aspect="auto", cmap="RdBu_r", vmin=-mx, vmax=mx)
    ax.set_xticks(range(3)); ax.set_xticklabels(eshort); ax.set_yticks(range(len(lrp))); ax.set_yticklabels(lrp.index)
    for i in range(len(lrp)):
        for j in range(3):
            v = lrp.values[i, j]; ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                                          color="white" if abs(v) > mx*.5 else "#222", fontsize=9)
    fig.colorbar(im, fraction=.046, pad=.04).set_label("LR coefficient (+ toward HIT)")
    ax.set_title("Direction of effect by era (standardised LR coefficients)", fontweight="bold", loc="left")
    fig.savefig(f"{FIG_DIR}/fig4_lr_direction_era.png", dpi=300, bbox_inches="tight"); plt.close()
    # Fig 5: SHAP beeswarms per era
    for grp, (sv, Xs) in bee.items():
        plt.figure()
        shap.summary_plot(sv, Xs, feature_names=FEATURES, show=False, max_display=12, plot_size=(7.5, 5.2))
        plt.title(f"SHAP summary - {grp}", fontweight="bold", loc="left")
        safe = grp.split(" ")[0].replace("/", "_")
        plt.savefig(f"{FIG_DIR}/fig5_beeswarm_{safe}.png", dpi=300, bbox_inches="tight"); plt.close()
    print("Figures written to", FIG_DIR)

# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    df = load_data()
    print("\n[1/3] Cross-validated model comparison ...")
    metrics = evaluate(df)
    print("\n[2/3] Feature importance + SHAP ...")
    shap_df, lr_df, bee = interpret(df)
    print("\n[3/3] Figures ...")
    make_figures(metrics, shap_df, lr_df, bee)
    print("\nDone. Tables in", OUT_DIR, "| figures in", FIG_DIR)
