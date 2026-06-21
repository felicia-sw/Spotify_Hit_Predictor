# Hit or Flop? Predicting Billboard Chart Success from Spotify Audio Features (1960–2019)

A classification study that goes beyond a model bake-off: it models each musical **era separately**
and compares **feature importance (SHAP)** across eras to show *how the drivers of a hit have changed*.

## Research questions
1. Which classifier — Logistic Regression, Random Forest, or KNN — best predicts whether a track
   reached the Billboard Hot 100 from its Spotify audio features?
2. Which features matter most, and **does their importance shift across eras** (1960–2019)?

## Headline findings
- **Random Forest wins in every era** (tuned ROC-AUC 0.85 / 0.92 / 0.91).
- **Hits became more predictable over time** — per-decade RF AUC rises from ~0.85 (1960s) to 0.92 (2000s).
- **`instrumentalness` is the #1 predictor in every era and keeps rising** — modern hits are vocal-forward.
- **`loudness` surged** from rank #5 (Analog) to #2 (Streaming); **`speechiness` and `acousticness` faded**;
  **`danceability` peaked** in the CD/Digital era.

## Method (brief)
- **EDA** first: descriptive statistics, class balance, hit-vs-flop distributions, correlations, era drift.
- 41,103 tracks, 15 numeric audio/structural features, z-score scaling for LR/KNN.
- Six decades grouped into three industry eras: Analog (1960–89), CD/Digital (1990–2009), Streaming (2010s);
  per-decade models also fitted for the trend.
- **Hyperparameters chosen by grid search (cross-validation)** per era — not hand-picked — for all three models.
- Importance via Random Forest + SHAP; direction via standardised Logistic-Regression coefficients.
- The SHAP forest is depth-capped (`max_depth=12`) for tractable exact TreeSHAP; the tuned forest (depth 20)
  is the one reported for performance.

## Files
| File | What it is |
|------|------------|
| `Hit_or_Flop_Analysis.docx` | Full write-up — intro, **EDA**, methods (incl. **model justification + grid search**), results, discussion, limitations, references. |
| `spotify_hit_dashboard.html` | Self-contained interactive dashboard (open in any browser). |
| `spotify_hit_analysis.py` | One-file reproducible pipeline: EDA → **grid search** → tuned models → SHAP → figures. |
| `notebooks/01_eda.ipynb` | Exploratory data analysis, rendered inline. |
| `notebooks/02_modeling_and_shap.ipynb` | Grid search, model comparison, and SHAP, rendered inline. |
| `figures/` | EDA figures (`eda*`) and result figures (`fig*`). |
| `results/` | `grid_final.csv` (tuned hyperparameters + scores), SHAP/importance/coefficient tables, EDA stats, and robustness tables (multi-seed SHAP, bootstrap model comparison, sensitivity, permutation importance). |
| `robustness/` | Scripts for the rigor analyses: multi-seed SHAP confidence intervals, out-of-fold model comparison, sensitivity checks, and permutation importance. |
| `Robustness_and_Validation_Findings.md` | Paste-ready robustness write-up (draft text + tables) for the manuscript's §4.5. |

## Reproduce
1. The dataset (six `dataset-of-*.csv`) is already in `spotify_data/`.
2. `pip install -r requirements.txt`
3. `python spotify_hit_analysis.py`  → regenerates `results/` and `figures/`.

For an inline view without rerunning, open the notebooks in `notebooks/`.

## Data source
The Spotify Hit Predictor Dataset (1960–2019), Kaggle (theoverman):
https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset
