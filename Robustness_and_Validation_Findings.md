# Robustness & Validation Findings

Additional rigor analyses to harden the paper for review (the "modelling is the weak point" concern).
All results are computed from the same 41,103-track dataset and tuned models. Draft text below is written
to paste into the manuscript's *Results and Discussion*; figures are in `figures/`.

---

## 1. The era shift is statistically robust (multi-seed SHAP with 95% CIs)

Per-era SHAP importance was recomputed across **5 independent random seeds** (each re-fits the forest and
re-samples the SHAP background), using the same forest configuration as the main analysis (300 trees,
depth 12; the main table's seed is among the five). Confidence intervals are tight relative to the
cross-era differences, and every key shift is consistent in direction across all seeds.

**Mean |SHAP| ± 95% CI by era (5 seeds):**

| Feature | Analog (1960–89) | CD/Digital (1990–2009) | Streaming (2010s) |
|---|---|---|---|
| instrumentalness | 0.091 ± 0.003 | 0.122 ± 0.005 | 0.170 ± 0.014 |
| loudness | 0.029 ± 0.002 | 0.040 ± 0.001 | 0.062 ± 0.002 |
| speechiness | 0.052 ± 0.001 | 0.024 ± 0.001 | 0.008 ± 0.001 |
| danceability | 0.046 ± 0.001 | 0.092 ± 0.002 | 0.057 ± 0.002 |
| acousticness | 0.053 ± 0.002 | 0.064 ± 0.001 | 0.044 ± 0.003 |

*(Table 4 in the manuscript reports a single representative run; the means above are over five seeds and agree with it.)*

**Analog → Streaming shift (paired across seeds):** instrumentalness **+0.079**, loudness **+0.033**,
speechiness **−0.044**, acousticness **−0.009**, danceability **+0.010** — for every feature the per-seed
range excludes zero, i.e. the shifts are statistically reliable, not single-run artifacts. *(Figure: `fig8_importance_ci.png`.)*

> **Draft text.** *To confirm that the cross-era importance shifts are not artifacts of a single fit, we
> recomputed per-era SHAP importance over five independent random seeds and report 95% confidence intervals.
> The intervals are narrow relative to the between-era differences (Figure 8). The increase in instrumentalness
> importance from the Analog to the Streaming era (+0.079) and the rise of loudness (+0.033), together with the
> decline of speechiness (−0.044) and acousticness (−0.009), are consistent across all seeds, indicating that
> the observed concept drift is statistically reliable.*

---

## 2. Random Forest's advantage is statistically significant (paired bootstrap on AUC)

Using 5-fold out-of-fold predictions and a 2,000-sample paired bootstrap, RF beats both comparators in
**every era** (all 95% CIs exclude zero).

| Era | RF − Logistic Reg. (ΔAUC, 95% CI) | RF − KNN (ΔAUC, 95% CI) |
|---|---|---|
| Analog | +0.062 [+0.059, +0.066] | +0.037 [+0.033, +0.040] |
| CD/Digital | +0.043 [+0.039, +0.048] | +0.028 [+0.024, +0.032] |
| Streaming | +0.048 [+0.041, +0.055] | +0.048 [+0.041, +0.056] |

Out-of-fold AUCs (RF): 0.849 / 0.917 / 0.913 — consistent with the cross-validated grid results.
*(Figures: `fig6_roc_pr_by_era.png` — ROC & PR curves; `fig7_confusion_rf.png` — confusion matrices;
`fig9_dauc_forest.png` — ΔAUC forest plot.)*

> **Draft text.** *Differences between classifiers were tested with a paired bootstrap (2,000 resamples) on
> the pooled out-of-fold predictions. The Random Forest significantly outperformed both Logistic Regression
> (ΔAUC +0.043 to +0.062) and KNN (ΔAUC +0.028 to +0.048) in every era, with all 95% confidence intervals
> excluding zero (Figure 9). ROC and precision–recall curves (Figure 6) and confusion matrices (Figure 7)
> corroborate the ranking.*

---

## 3. Findings survive alternative choices (sensitivity analyses)

**(a) Alternative era boundaries.** The shift and the rise in predictability persist under a coarser 2-era
split and a different 3-way split:

- *2 eras* (Pre-2000 → 2000s+): AUC 0.853 → 0.916; instrumentalness 0.099 → 0.155; loudness 0.031 → 0.062; speechiness 0.053 → 0.013.
- *3 decade-pairs* (Early/Mid/Late): AUC 0.844 → 0.890 → 0.916; instrumentalness 0.089 → 0.109 → 0.155; loudness 0.032 → 0.033 → 0.062; speechiness 0.050 → 0.032 → 0.013.

**(b) De-duplication.** Removing the ~1.3% of tracks that appear in more than one decade file barely changes
anything: era AUCs 0.845 / 0.916 / 0.914 (vs 0.848 / 0.917 / 0.912), and the importance trajectories are
essentially unchanged.

**(c) Permutation importance vs SHAP.** Model-agnostic permutation importance agrees almost perfectly with
SHAP (Spearman ρ = 0.97 Analog, 0.96 CD/Digital, 0.94 Streaming; near-identical top-5 features), so the
ranking is not an artifact of the SHAP method. *(Figures: `fig10_perm_vs_shap.png`, `fig11_robustness_schemes.png`.)*

> **Draft text.** *Three sensitivity analyses support the conclusions. First, re-grouping the decades into a
> 2-era or an alternative 3-era scheme preserves both the rise in predictability and the instrumentalness/
> loudness increase with the speechiness/acousticness decline (Figure 11). Second, de-duplicating tracks that
> recur across decade files leaves the per-era AUCs (0.845/0.916/0.914) and importances essentially unchanged.
> Third, model-agnostic permutation importance is in strong agreement with SHAP (Spearman ρ = 0.94–0.97;
> Figure 10), confirming the importance ranking is not specific to one attribution method.*

---

## Files
- `figures/fig6_roc_pr_by_era.png`, `fig7_confusion_rf.png`, `fig8_importance_ci.png`,
  `fig9_dauc_forest.png`, `fig10_perm_vs_shap.png`, `fig11_robustness_schemes.png`
- `results/shap_seed_summary.csv`, `model_compare_bootstrap.csv`, `sensitivity.csv`, `perm_importance.csv`
- `robustness/` — the scripts that produced these (multi-seed SHAP, OOF predictions, sensitivity, permutation importance)

## Suggested placement in the paper
Add a short subsection **"4.5 Robustness and validation"** summarising §1–§3, and cite the new figures.
This directly answers the modelling-rigor concern: the shift is significant (CIs), the model choice is
justified statistically (bootstrap ΔAUC), and the result is stable across era definitions, de-duplication,
and importance method.
