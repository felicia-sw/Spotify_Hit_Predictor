# Hit or Flop? Predicting Billboard Chart Success from Spotify Audio Features (1960–2019)

A classification study that goes beyond a model bake-off: it models each musical **era separately**
and compares **feature importance (SHAP)** across eras to show *how the drivers of a hit have changed*.

## Research questions
1. Which classifier — Logistic Regression, Random Forest, or KNN — best predicts whether a track
   reached the Billboard Hot 100 from its Spotify audio features?
2. Which features matter most, and **does their importance shift across eras** (1960–2019)?

## Headline findings
- **Random Forest wins in every era** (ROC-AUC 0.844 / 0.916 / 0.911).
- **Hits became more predictable over time** — per-decade RF AUC rises from ~0.85 (1960s) to 0.92 (2000s).
- **`instrumentalness` is the #1 predictor in every era and keeps rising** — modern hits are vocal-forward.
- **`loudness` surged** from rank #5 (Analog) to #2 (Streaming); **`speechiness` and `acousticness` faded**;
  **`danceability` peaked** in the CD/Digital era.

## Method (brief)
- 41,103 tracks, 15 numeric audio/structural features, z-score scaling for LR/KNN.
- Stratified 5-fold CV; accuracy, F1, ROC-AUC.
- Six decades grouped into three industry eras: Analog (1960–89), CD/Digital (1990–2009), Streaming (2010s);
  per-decade models also fitted for the trend.
- Importance via Random Forest + SHAP (TreeExplainer); direction via standardised LR coefficients.
- The interpretation forest is depth-capped (`max_depth=14`) so exact TreeSHAP is tractable; its CV AUC is
  within 0.005 of the unconstrained forest.

## Files
| File | What it is |
|------|------------|
| `Hit_or_Flop_Analysis.docx` | Full write-up — intro, data, methods, results, discussion, limitations, references (with figures + tables). |
| `spotify_hit_dashboard.html` | Self-contained interactive dashboard (open in any browser). |
| `spotify_hit_analysis.py` | One-file reproducible pipeline (metrics + SHAP + figures). |
| `figures/` | The 7 publication-quality PNGs. |
| `results/` | Result tables: `metrics_final.csv`, `shap_meanabs.csv`, `rf_importance.csv`, `lr_coef.csv`. |

## Reproduce
1. Download the dataset from Kaggle (theoverman, *The Spotify Hit Predictor Dataset 1960–2019*) and place the
   six `dataset-of-*.csv` files in a folder named `spotify_data/`.
2. `pip install pandas numpy scikit-learn shap matplotlib`
3. `python spotify_hit_analysis.py`

## Data source
The Spotify Hit Predictor Dataset (1960–2019), Kaggle (theoverman):
https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset
