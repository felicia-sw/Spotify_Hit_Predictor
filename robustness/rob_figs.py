import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ERAS=['Analog (1960-89)','CD/Digital (1990-2009)','Streaming (2010s)']; esh=['Analog\n1960-89','CD/Digital\n1990-2009','Streaming\n2010s']
COL={'instrumentalness':'#C44E52','danceability':'#4C72B0','loudness':'#DD8452','acousticness':'#55A868','speechiness':'#8172B3','duration_ms':'#937860'}

# FIG A: importance with 95% CI across eras
s=pd.read_csv("results/shap_seed_summary.csv")
s['ci95']=1.96*s['std']/np.sqrt(s['count'])
fig,ax=plt.subplots(figsize=(9,5.4)); x=np.arange(3)
for f in COL:
    m=[s[(s.era==e)&(s.feature==f)]['mean'].values[0] for e in ERAS]
    c=[s[(s.era==e)&(s.feature==f)]['ci95'].values[0] for e in ERAS]
    ax.errorbar(x,m,yerr=c,marker='o',lw=2.2,capsize=4,color=COL[f],label=f)
ax.set_xticks(x); ax.set_xticklabels(esh); ax.set_ylabel("mean |SHAP value|  (±95% CI, 5 seeds)")
ax.set_title("Feature importance by era with 95% confidence intervals",fontweight='bold',loc='left')
ax.legend(frameon=False,ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(0.5,-0.12)); ax.grid(alpha=.25)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.savefig("figures/fig8_importance_ci.png",dpi=300,bbox_inches='tight'); plt.close()

# FIG B: forest plot dAUC
mc=pd.read_csv("results/model_compare_bootstrap.csv")
lab={'Analog (1960-89)':'Analog','CD/Digital (1990-2009)':'CD/Digital','Streaming (2010s)':'Streaming'}
mc['row']=mc['era'].map(lab)+": "+mc['comparison']
mc=mc.iloc[::-1].reset_index(drop=True)
fig,ax=plt.subplots(figsize=(8,4.6)); yy=np.arange(len(mc))
ax.errorbar(mc['dAUC'],yy,xerr=[mc['dAUC']-mc['ci_lo'],mc['ci_hi']-mc['dAUC']],fmt='o',color='#2D6A4F',capsize=4,lw=1.8)
ax.axvline(0,ls='--',color='#c0392b',lw=1); ax.set_yticks(yy); ax.set_yticklabels(mc['row'],fontsize=9)
ax.set_xlabel("ΔAUC = AUC(Random Forest) − AUC(comparator),  95% CI"); ax.set_title("Random Forest vs other classifiers (paired bootstrap)",fontweight='bold',loc='left',fontsize=12)
ax.grid(axis='x',alpha=.25)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.savefig("figures/fig9_dauc_forest.png",dpi=300,bbox_inches='tight'); plt.close()

# FIG C: permutation vs SHAP scatter (3 panels)
perm=pd.read_csv("results/perm_importance.csv"); sh=pd.read_csv("results/shap_meanabs_tuned.csv"); sh=sh[sh.level=='era']
from scipy.stats import spearmanr
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
for j,e in enumerate(ERAS):
    p=perm[perm.era==e].set_index('feature')['perm_imp_mean']; v=sh[sh.group==e].set_index('feature')['mean_abs_shap']
    common=p.index.intersection(v.index); rho,_=spearmanr(p[common],v[common])
    ax=axes[j]; ax.scatter(p[common],v[common],color='#2D6A4F',s=28)
    for f in ['instrumentalness','loudness','danceability','acousticness','speechiness']:
        if f in common: ax.annotate(f,(p[f],v[f]),fontsize=7,xytext=(4,2),textcoords='offset points')
    ax.set_title(f"{esh[j].replace(chr(10),' ')}  (ρ={rho:.2f})",fontweight='bold',fontsize=10,loc='left')
    ax.set_xlabel("Permutation importance (AUC drop)"); ax.set_ylabel("mean |SHAP|" if j==0 else "")
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.suptitle("Permutation importance vs SHAP agreement by era",fontweight='bold',fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig("figures/fig10_perm_vs_shap.png",dpi=300,bbox_inches='tight'); plt.close()

# FIG D: robustness of shift across schemes (3 key features)
sens=pd.read_csv("results/sensitivity.csv"); sens=sens[sens.metric=='mean_abs_shap']
shm=sh.pivot(index='feature',columns='group',values='mean_abs_shap')
def val(scheme,grp,f): 
    r=sens[(sens.scheme==scheme)&(sens.group==grp)&(sens.feature==f)]['value']; return r.values[0] if len(r) else np.nan
schemes=[('Main',('Analog (1960-89)','Streaming (2010s)'),'main'),('2-era',('Pre-2000','2000s+'),'alt2_2era'),
         ('3-pairs',('Early','Late'),'alt3_pairs'),('Dedup',('Analog','Streaming'),'dedup_main')]
feats3=['instrumentalness','loudness','speechiness']
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
for k,f in enumerate(feats3):
    ax=axes[k]; xs=np.arange(len(schemes)); w=0.36
    early=[]; late=[]
    for name,(g0,g1),sc in schemes:
        if sc=='main': early.append(shm.loc[f,g0]); late.append(shm.loc[f,g1])
        else: early.append(val(sc,g0,f)); late.append(val(sc,g1,f))
    ax.bar(xs-w/2,early,w,label='earliest era',color='#9DB4C0'); ax.bar(xs+w/2,late,w,label='latest era',color='#2D6A4F')
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in schemes],fontsize=9); ax.set_title(f,fontweight='bold',fontsize=11)
    if k==0: ax.legend(frameon=False,fontsize=9); ax.set_ylabel("mean |SHAP|")
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
fig.suptitle("Robustness of the era shift across grouping schemes (earliest vs latest era)",fontweight='bold',fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig("figures/fig11_robustness_schemes.png",dpi=300,bbox_inches='tight'); plt.close()
print("saved fig8_importance_ci, fig9_dauc_forest, fig10_perm_vs_shap, fig11_robustness_schemes")
