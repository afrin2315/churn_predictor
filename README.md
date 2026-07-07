# Customer Churn Predictor

**🔗 Live demo:** https://churn-predictor-afrin.streamlit.app

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-predictor-afrin.streamlit.app)

An end-to-end machine-learning project that predicts customer churn for the IBM Telco Customer Churn dataset. The project includes data download, exploratory analysis, a baseline logistic regression model, a tuned XGBoost model, SHAP-based explainability, and a Streamlit app for live scoring.

## Project structure

```text
churn_predictor/
  data/
    download_data.py
    README.md
  notebooks/
    eda.ipynb
  reports/
  models/
  src/
    __init__.py
    preprocess.py
    train.py
    explain.py
  app.py
  requirements.txt
  .gitignore
```

## Dataset

Primary source:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

If the raw URL is unavailable, download the dataset manually from Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV at `data/Telco-Customer-Churn.csv`.

## What’s included

- Data download helper and instructions
- EDA notebook with churn rate, churn by contract, tenure, monthly charges, internet service, and correlation heatmap
- Clean preprocessing with numeric coercion, imputation, one-hot encoding, and stratified split
- Baseline logistic regression and tuned XGBoost training with class imbalance handling
- Saved metrics, confusion matrix, and precision-recall curve
- SHAP explanations with a global summary plot and a per-customer waterfall plot
- Streamlit app for single-customer scoring and batch CSV scoring

## Results

Metrics below are the **actual** numbers produced by `python src/train.py` on a
stratified 80/20 split (5,634 train / 1,409 test rows, 45 features after one-hot
encoding). The dataset is imbalanced (~26.5% churn), so the churn-class
precision/recall and ROC-AUC / PR-AUC matter more than raw accuracy. Class
imbalance is handled with `scale_pos_weight ≈ 2.77` (XGBoost) and
`class_weight="balanced"` (logistic regression), which deliberately trades some
precision for high recall on churners — usually the right call for retention.

| Model                          | ROC-AUC | Precision (churn) | Recall (churn) | F1 (churn) | PR-AUC |
| ------------------------------ | :-----: | :---------------: | :------------: | :--------: | :----: |
| Logistic Regression (baseline) |  0.841  |       0.504       |     0.783      |   0.614    |   —    |
| **XGBoost (tuned)**            | **0.847** |     **0.520**     |   **0.794**    | **0.629**  | 0.663  |

- **XGBoost 5-fold CV ROC-AUC (train):** 0.849
- **XGBoost test confusion matrix** `[[TN, FP], [FN, TP]] = [[761, 274], [77, 297]]`
  — of 374 real churners in the test set, the model catches **297 (79%)**.
- **Best XGBoost params:** `colsample_bytree=1.0, learning_rate=0.05, max_depth=3,
  min_child_weight=5, n_estimators=150, subsample=0.8`

Full metrics are saved to [reports/metrics.json](reports/metrics.json) and
[reports/metrics.txt](reports/metrics.txt). Charts (confusion matrix, PR curve,
SHAP summary, per-customer waterfall, and all EDA figures) are in [reports/](reports/).
Re-running `src/train.py` reproduces these numbers exactly (fixed `random_state=42`).

## How to run locally

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:

   ```bash
   python data/download_data.py
   ```

4. Train the models and generate reports:

   ```bash
   python src/train.py
   python src/explain.py
   ```

5. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Generated charts & screenshots

Running the pipeline produces these figures in `reports/` (already generated):

- Model: [confusion_matrix_xgb.png](reports/confusion_matrix_xgb.png),
  [pr_curve_xgb.png](reports/pr_curve_xgb.png)
- Explainability: [shap_summary.png](reports/shap_summary.png),
  [shap_waterfall_customer.png](reports/shap_waterfall_customer.png)
- EDA: [eda_churn_rate.png](reports/eda_churn_rate.png),
  [eda_churn_by_contract.png](reports/eda_churn_by_contract.png),
  [eda_churn_by_tenure.png](reports/eda_churn_by_tenure.png),
  [eda_churn_by_monthlycharges.png](reports/eda_churn_by_monthlycharges.png),
  [eda_churn_by_internetservice.png](reports/eda_churn_by_internetservice.png),
  [eda_correlation_heatmap.png](reports/eda_correlation_heatmap.png)

After launching the app, add UI screenshots here (create `reports/screenshots/`):

- App home / single-customer form: `reports/screenshots/app-home.png`
- Batch scoring: `reports/screenshots/batch-scoring.png`

## How to push to GitHub

```bash
cd churn_predictor
git init
git add .
git commit -m "Customer churn predictor: EDA, models, SHAP, Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

The trained model bundle in `models/` and the dataset CSV in `data/` are both
**committed** so the deployed app is fully self-contained — it loads the model and
its SHAP background sample instantly with no runtime download.

## How to deploy on Streamlit Community Cloud (get a public live link)

1. Push the repo to GitHub (above). Confirm `app.py`, `requirements.txt`, and
   `models/churn_xgb_bundle.joblib` are present in the repo on GitHub.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Select your repository and branch (`main`), and set **Main file path** to `app.py`.
5. Open **Advanced settings** and set **Python version to 3.11** — the pinned
   dependency versions (e.g. `numpy==1.26.4`) ship prebuilt wheels for 3.11, which
   avoids slow/failing source builds on newer Python. (`packages.txt` installs
   `libgomp1`, the OpenMP runtime XGBoost needs on Streamlit Cloud's Linux image.)
6. Click **Deploy**. Streamlit installs `requirements.txt`, builds the app, and gives
   you a public URL like `https://<your-app-name>.streamlit.app`.
7. Paste that URL back into the top of this README so recruiters can click straight
   through to the live demo.

## Notes

- The app expects the saved model bundle at `models/churn_xgb_bundle.joblib`
  (produced by `src/train.py`).
- The data loader tries the IBM raw CSV first and falls back to a manual Kaggle
  download instruction if the URL is unavailable.
- All results are reproducible: the train/test split and both models use
  `random_state=42`.
