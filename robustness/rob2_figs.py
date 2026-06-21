import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
eras=['Analog (1960-89)','CD/Digital (1990-2009)','Streaming (2010s)']
esh=['Analog 1960-89','CD/Digital 1990-2009','Streaming 2010s']
def safe(e): return e.split(' ')[0].replace('/','_')
def load(e,m):
    z=np.load(f"results/oof/oof_{safe(e)}_{m}.npz"); return z['y'],z['proba']
MC={'RandomForest':'#2D6A4F','LogReg':'#8C9EB2','KNN':'#B5838D'}; LAB={'RandomForest':'Random Forest','LogReg':'Logistic Reg.','KNN':'KNN'}

# ROC + PR (2x3)
fig,axes=plt.subplots(2,3,figsize=(13.5,8))
for j,e in enumerate(eras):
    ax=axes[0,j]
    for m in ['LogReg','KNN','RandomForest']:
        y,p=load(e,m); fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,color=MC[m],lw=2,label=f"{LAB[m]} ({auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],'--',color='#bbb',lw=1); ax.set_title(f"ROC — {esh[j]}",fontweight='bold',fontsize=11,loc='left')
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate" if j==0 else ""); ax.legend(fontsize=8,loc='lower right',frameon=False)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax2=axes[1,j]
    for m in ['LogReg','KNN','RandomForest']:
        y,p=load(e,m); pr,rc,_=precision_recall_curve(y,p); ax2.plot(rc,pr,color=MC[m],lw=2,label=f"{LAB[m]} ({auc(rc,pr):.3f})")
    ax2.set_title(f"Precision–Recall — {esh[j]}",fontweight='bold',fontsize=11,loc='left')
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision" if j==0 else ""); ax2.set_ylim(0.4,1.02); ax2.legend(fontsize=8,loc='lower left',frameon=False)
    for s in ['top','right']: ax2.spines[s].set_visible(False)
fig.suptitle("ROC and Precision–Recall curves by era (5-fold out-of-fold predictions)",fontweight='bold',fontsize=13,y=1.01)
fig.tight_layout(); fig.savefig("figures/fig6_roc_pr_by_era.png",dpi=300,bbox_inches='tight'); plt.close()

# Confusion matrices (RF, normalized) 1x3
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for j,e in enumerate(eras):
    y,p=load(e,'RandomForest'); pred=(p>=0.5).astype(int)
    cm=confusion_matrix(y,pred); cmn=cm/cm.sum(1,keepdims=True)
    ax=axes[j]; im=ax.imshow(cmn,cmap='Greens',vmin=0,vmax=1)
    for a in range(2):
        for b in range(2):
            ax.text(b,a,f"{cmn[a,b]:.2f}\n(n={cm[a,b]})",ha='center',va='center',color='white' if cmn[a,b]>0.5 else '#222',fontsize=10)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Pred Flop','Pred Hit']); ax.set_yticks([0,1]); ax.set_yticklabels(['True Flop','True Hit'])
    ax.set_title(f"{esh[j]}",fontweight='bold',fontsize=11)
fig.suptitle("Random Forest confusion matrices by era (row-normalised, threshold 0.5)",fontweight='bold',fontsize=12.5,y=1.04)
fig.tight_layout(); fig.savefig("figures/fig7_confusion_rf.png",dpi=300,bbox_inches='tight'); plt.close()
print("saved fig6_roc_pr_by_era.png, fig7_confusion_rf.png")
