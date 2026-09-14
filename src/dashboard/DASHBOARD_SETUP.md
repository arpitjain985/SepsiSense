# Running the SepsiSense Dashboard

## 1. Install dependencies (on your own machine)
```bash
pip install streamlit pandas numpy plotly
```

## 2. Get the required data files from Colab
After running Section 15 (SHAP Explainability) in the Colab notebook, two
files get saved to your Google Drive under `sepsisense_training_data/../`:
- `dashboard_export.csv`
- `global_feature_importance.csv`

Also grab `final_three_way_comparison.csv` (from Section 10/12) if you want
the model comparison table to show up in the "Model Insights" tab.

Download all three from Google Drive (right-click -> Download).

## 3. Place the files correctly
```
sepsisense/
└── src/
    └── dashboard/
        ├── dashboard.py
        └── data/                          <- create this folder
            ├── dashboard_export.csv
            ├── global_feature_importance.csv
            └── final_three_way_comparison.csv
```

## 4. Run it
```bash
cd src/dashboard
streamlit run dashboard.py
```
This opens automatically in your browser at `http://localhost:8501`.

## What you'll see
- **Ward View**: every test patient sorted by current risk score, with an
  adjustable alert threshold slider — mirrors how a nurse would triage
  attention across multiple patients.
- **Patient Detail**: pick any patient, see their risk trajectory over their
  ICU stay (with the actual sepsis onset marked, if applicable), plus a
  "Why this alert?" panel showing the top 5 features driving their current
  score via SHAP, with direction (increases/decreases risk).
- **Model Insights**: global feature importance across all patients, and
  the SOFA vs. GBT vs. LSTM comparison table.

## For your report/demo
This dashboard replays real, held-out PhysioNet Challenge 2019 test patients
retrospectively — the caption at the top of the app states this explicitly.
This is standard practice for demoing clinical ML systems without live
hospital monitor integration, and preempts the "is this actually connected
to real monitors" question before anyone asks it.
