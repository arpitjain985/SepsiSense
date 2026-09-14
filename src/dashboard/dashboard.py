"""
dashboard.py
-------------
SepsiSense clinician-facing dashboard. Retrospectively replays real ICU
test-set patients through the trained model, showing:
  1. A ward-level view: all patients sorted by current risk score
  2. A patient detail view: risk trend over time + SHAP explainability panel
     showing which vitals/labs are driving the current risk score

This is framed explicitly (see the caption in the UI) as a retrospective
simulation on held-out test patients -- standard practice for demoing
clinical ML systems without live hospital monitor integration.

SETUP (run these on YOUR machine, not in this sandbox):
  pip install streamlit pandas numpy plotly

REQUIRED DATA FILES (place in the same folder as this script, under ./data/):
  - dashboard_export.csv        <- exported from the Colab SHAP cell (Section 15)
  - global_feature_importance.csv  <- also from Colab Section 15

RUN:
  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="SepsiSense — ICU Risk Dashboard", layout="wide")

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    export_path = DATA_DIR / "dashboard_export.csv"
    importance_path = DATA_DIR / "global_feature_importance.csv"

    if not export_path.exists():
        return None, None

    df = pd.read_csv(export_path)
    importance = None
    if importance_path.exists():
        importance = pd.read_csv(importance_path, index_col=0).squeeze("columns")
    return df, importance


def get_shap_cols(df):
    return [c for c in df.columns if c.startswith("shap_")]


def get_val_cols(df):
    return [c for c in df.columns if c.startswith("val_")]


def get_latest_risk_per_patient(df, risk_col="gbt_prob"):
    """Ward view: latest known risk score per patient."""
    latest = df.sort_values("hour").groupby("patient_id").tail(1).copy()
    latest = latest.sort_values(risk_col, ascending=False)
    return latest


def explain_row(df, row, top_n=5):
    """Pull the top contributing SHAP features for a specific row."""
    shap_cols = get_shap_cols(df)
    val_cols = get_val_cols(df)
    shap_vals = row[shap_cols]
    shap_vals.index = [c.replace("shap_", "") for c in shap_cols]
    top = shap_vals.reindex(shap_vals.abs().sort_values(ascending=False).index).head(top_n)

    explanations = []
    for feat, sv in top.items():
        val_col = f"val_{feat}"
        raw_val = row[val_col] if val_col in row.index else None
        explanations.append({
            "feature": feat,
            "shap_value": sv,
            "patient_value": raw_val,
            "direction": "↑ increases risk" if sv > 0 else "↓ decreases risk",
        })
    return explanations


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏥 SepsiSense — ICU Early Sepsis Risk Dashboard")
st.caption(
    "Retrospective simulation on real, held-out PhysioNet Challenge 2019 test "
    "patients. This demonstrates the intended clinical workflow: in a live "
    "deployment, this view would update continuously from bedside monitors. "
    "**Decision-support only — not a diagnostic device.**"
)

df, importance = load_data()

if df is None:
    st.error(
        "No data found. Run the SHAP explainability cell (Section 15) in the "
        "Colab notebook, download `dashboard_export.csv` and "
        "`global_feature_importance.csv` from your Google Drive, and place "
        "them in a `data/` folder next to this script."
    )
    st.stop()

risk_col = "lstm_prob" if "lstm_prob" in df.columns and df["lstm_prob"].notna().any() else "gbt_prob"
st.sidebar.markdown(f"**Primary risk model:** `{risk_col}`")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏨 Ward View", "🔍 Patient Detail", "📊 Model Insights", "🔴 Live Monitor (Simulated)"]
)

# --- TAB 1: Ward view ---
with tab1:
    st.subheader("Patients sorted by current risk score")
    latest = get_latest_risk_per_patient(df, risk_col)

    threshold = st.slider("Alert threshold", 0.0, 1.0, 0.05, 0.01,
                           help="Patients above this risk score are highlighted")

    display_df = latest[["patient_id", "hour", risk_col, "label"]].copy()
    display_df.columns = ["Patient ID", "ICU Hour", "Risk Score", "Actual Sepsis (ground truth)"]
    display_df["Alert"] = display_df["Risk Score"] >= threshold

    def highlight_alert(row):
        color = "background-color: #ffcccc" if row["Alert"] else ""
        return [color] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_alert, axis=1).format({"Risk Score": "{:.3f}"}),
        use_container_width=True,
        height=500,
    )
    n_alerts = display_df["Alert"].sum()
    st.metric("Patients currently flagged", f"{n_alerts} / {len(display_df)}")

# --- TAB 2: Patient detail ---
with tab2:
    patient_ids = sorted(df["patient_id"].unique())
    selected_pid = st.selectbox("Select a patient", patient_ids)

    patient_df = df[df["patient_id"] == selected_pid].sort_values("hour")
    is_sepsis_case = (patient_df["label"] == 1).any()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Risk trajectory — {selected_pid}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=patient_df["hour"], y=patient_df[risk_col],
            mode="lines+markers", name="Risk score", line=dict(color="#d62728", width=2)
        ))
        if is_sepsis_case:
            onset_hour = patient_df[patient_df["label"] == 1]["hour"].min()
            fig.add_vline(x=onset_hour, line_dash="dash", line_color="black",
                          annotation_text="Clinical sepsis onset")
        fig.update_layout(xaxis_title="ICU hour", yaxis_title="Risk score",
                          yaxis_range=[0, 1], height=400)
        st.plotly_chart(fig, use_container_width=True)

        if is_sepsis_case:
            st.warning(f"⚠️ This patient developed sepsis at ICU hour {onset_hour}")
        else:
            st.success("✅ This patient did not develop sepsis in this ICU stay")

    with col2:
        st.subheader("Why this alert?")
        latest_row = patient_df.iloc[-1]
        shap_cols = get_shap_cols(df)
        if shap_cols:
            explanations = explain_row(df, latest_row, top_n=5)
            for item in explanations:
                icon = "🔴" if item["shap_value"] > 0 else "🟢"
                val_str = f"{item['patient_value']:.1f}" if pd.notna(item["patient_value"]) else "N/A"
                st.markdown(
                    f"{icon} **{item['feature']}** = {val_str}  \n"
                    f"_{item['direction']}_ (SHAP: {item['shap_value']:+.3f})"
                )
        else:
            st.info("No SHAP explanation data available for this row "
                     "(only sampled rows from Colab have explanations).")

# --- TAB 3: Model insights ---
with tab3:
    st.subheader("Global feature importance")
    st.caption("Which vitals/labs matter most across all predictions (mean |SHAP value|)")
    if importance is not None:
        top_features = importance.head(15)
        fig2 = go.Figure(go.Bar(
            x=top_features.values, y=top_features.index, orientation="h",
            marker_color="#1f4e5f"
        ))
        fig2.update_layout(height=500, xaxis_title="Mean |SHAP value|",
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Global feature importance not available — export it from "
                "the Colab SHAP cell.")

    st.subheader("Model comparison")
    st.caption("From the full-dataset evaluation notebook")
    comparison_path = DATA_DIR / "final_three_way_comparison.csv"
    if comparison_path.exists():
        comp_df = pd.read_csv(comparison_path)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Place `final_three_way_comparison.csv` in the data/ folder to show this table.")

# --- TAB 4: Live Monitor (Simulated) ---
with tab4:
    import time

    st.subheader("Simulated live ICU monitor")
    st.caption(
        "Replays a real patient's ICU stay hour-by-hour with a short delay, "
        "as if bedside monitor readings were arriving in real time. This is "
        "how the same model + dashboard would behave in a live hospital "
        "deployment — only the data source changes (live monitor feed "
        "instead of a saved file), the model and UI logic stay identical."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        live_pid = st.selectbox("Select a patient to monitor", patient_ids, key="live_pid")
    with col_b:
        speed = st.slider("Playback speed (seconds per ICU hour)", 0.1, 2.0, 0.4, 0.1)

    start_clicked = st.button("▶ Start Live Simulation", type="primary")

    live_patient_df = df[df["patient_id"] == live_pid].sort_values("hour").reset_index(drop=True)
    val_cols_all = get_val_cols(df)
    core_vital_map = {
        "HR": "val_HR_last", "O2Sat": "val_O2Sat_last", "Temp": "val_Temp_last",
        "SBP": "val_SBP_last", "MAP": "val_MAP_last", "Resp": "val_Resp_last",
    }
    available_vitals = {k: v for k, v in core_vital_map.items() if v in df.columns}

    if start_clicked:
        chart_placeholder = st.empty()
        vitals_placeholder = st.empty()
        alert_placeholder = st.empty()
        explain_placeholder = st.empty()

        hours_seen, risk_seen = [], []

        for i in range(len(live_patient_df)):
            row = live_patient_df.iloc[i]
            hours_seen.append(row["hour"])
            risk_seen.append(row[risk_col])

            # --- live risk trajectory chart, built up point by point ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours_seen, y=risk_seen, mode="lines+markers",
                line=dict(color="#d62728", width=2), name="Risk score"
            ))
            fig.update_layout(
                title=f"Live risk trajectory — {live_pid} (ICU hour {int(row['hour'])})",
                xaxis_title="ICU hour", yaxis_title="Risk score",
                yaxis_range=[0, 1], height=350,
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"live_chart_{i}")

            # --- simulated bedside vitals readout ---
            with vitals_placeholder.container():
                cols = st.columns(len(available_vitals)) if available_vitals else []
                for c, (label, colname) in zip(cols, available_vitals.items()):
                    val = row.get(colname, None)
                    c.metric(label, f"{val:.1f}" if pd.notna(val) else "—")

            # --- alert banner ---
            current_risk = row[risk_col]
            if current_risk >= threshold:
                alert_placeholder.error(
                    f"🚨 ALERT — Risk score {current_risk:.3f} exceeds threshold "
                    f"({threshold:.2f}) at ICU hour {int(row['hour'])}"
                )
            else:
                alert_placeholder.success(
                    f"Risk score {current_risk:.3f} — below alert threshold"
                )

            # --- live explanation, if SHAP columns available ---
            shap_cols_live = get_shap_cols(df)
            if shap_cols_live:
                with explain_placeholder.container():
                    st.markdown("**Why this reading?**")
                    explanations = explain_row(df, row, top_n=3)
                    ex_cols = st.columns(len(explanations))
                    for c, item in zip(ex_cols, explanations):
                        icon = "🔴" if item["shap_value"] > 0 else "🟢"
                        val_str = f"{item['patient_value']:.1f}" if pd.notna(item["patient_value"]) else "N/A"
                        c.markdown(f"{icon} **{item['feature']}**={val_str}  \n_{item['direction']}_")

            if row["label"] == 1:
                st.warning(f"⚠️ Patient met clinical sepsis criteria at ICU hour {int(row['hour'])}")
                break

            time.sleep(speed)

        st.info("Simulation complete — this replayed real historical data hour-by-hour. "
                "In a live deployment, this loop would instead wait for the next real "
                "monitor reading rather than a fixed delay.")
