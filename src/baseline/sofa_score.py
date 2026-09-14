"""
sofa_score.py
--------------
Rule-based clinical baseline: qSOFA + a Modified SOFA score, computed per
hour from real patient data. This is what your model needs to beat/match
with an earlier lead-time to make the "why AI adds value" argument concrete.

IMPORTANT DOCUMENTED LIMITATION (say this explicitly in your report -- it's
a sign of rigor, not a weakness):
  Full clinical SOFA requires:
    - GCS (Glasgow Coma Scale) for the neurological component -- NOT present
      in this dataset.
    - PaO2/FiO2 ratio for the respiratory component -- this dataset has
      FiO2 but not PaO2 (only PaCO2), so respiratory SOFA can't be computed
      either.
    - Vasopressor dosing for the full cardiovascular component -- not present;
      we use a MAP-only simplification instead.
  We therefore compute a "Modified SOFA" using the 4 components that ARE
  computable from available columns: Coagulation (Platelets), Liver
  (Bilirubin_total), Cardiovascular (MAP), Renal (Creatinine). qSOFA is
  computed similarly, omitting the "altered mental status" criterion (no
  GCS), so our qSOFA is scored out of 2 instead of the standard 3.
  This is a standard, defensible adaptation when GCS/PaO2 aren't available
  in a dataset -- multiple published papers using this exact PhysioNet
  Challenge dataset make the same adjustment.

Run:
  python src/baseline/sofa_score.py
"""

import glob
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "real_subset" / "subset_data"
SPLIT_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "split_ids.txt"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

QSOFA_ALERT_THRESHOLD = 1   # out of max 2 (RR + SBP components only, no GCS)
SOFA_ALERT_THRESHOLD = 2    # out of max 16 (4 components x 0-4 each)


def load_test_patient_ids():
    text = SPLIT_FILE.read_text()
    test_block = text.split("TEST:\n")[1]
    ids = [l.strip() for l in test_block.strip().split("\n") if l.strip()]
    return ids


def qsofa_component(row):
    score = 0
    if pd.notna(row["Resp"]) and row["Resp"] >= 22:
        score += 1
    if pd.notna(row["SBP"]) and row["SBP"] <= 100:
        score += 1
    return score


def modified_sofa_component(row):
    score = 0
    # Coagulation - platelets (x10^3/uL)
    p = row["Platelets"]
    if pd.notna(p):
        if p < 20: score += 4
        elif p < 50: score += 3
        elif p < 100: score += 2
        elif p < 150: score += 1
    # Liver - bilirubin total (mg/dL)
    b = row["Bilirubin_total"]
    if pd.notna(b):
        if b >= 12: score += 4
        elif b >= 6: score += 3
        elif b >= 2: score += 2
        elif b >= 1.2: score += 1
    # Cardiovascular - MAP only (simplified, no vasopressor data)
    m = row["MAP"]
    if pd.notna(m) and m < 70:
        score += 1
    # Renal - creatinine (mg/dL)
    c = row["Creatinine"]
    if pd.notna(c):
        if c >= 5: score += 4
        elif c >= 3.5: score += 3
        elif c >= 2: score += 2
        elif c >= 1.2: score += 1
    return score


def score_patient(fp):
    df = pd.read_csv(fp, sep="|")
    # forward-fill so a lab drawn a few hours ago still counts "now",
    # same clinical-reasoning approach used in preprocess.py
    cols_to_ffill = ["Resp", "SBP", "Platelets", "Bilirubin_total", "MAP", "Creatinine"]
    df[cols_to_ffill] = df[cols_to_ffill].ffill()

    df["qsofa"] = df.apply(qsofa_component, axis=1)
    df["mod_sofa"] = df.apply(modified_sofa_component, axis=1)
    df["qsofa_alert"] = df["qsofa"] >= QSOFA_ALERT_THRESHOLD
    df["sofa_alert"] = df["mod_sofa"] >= SOFA_ALERT_THRESHOLD
    df["combined_alert"] = df["qsofa_alert"] | df["sofa_alert"]

    onset_idx = df.index[df["SepsisLabel"] == 1]
    onset_hour = int(onset_idx[0]) if len(onset_idx) > 0 else None

    if onset_hour is not None:
        df = df.loc[:onset_hour]  # match preprocess.py truncation

    return df, onset_hour


def evaluate():
    test_ids = load_test_patient_ids()
    print(f"Evaluating SOFA/qSOFA baseline on {len(test_ids)} held-out test patients")

    lead_times = []       # for sepsis-positive patients: onset_hour - first_alert_hour
    missed = 0             # sepsis patients never alerted before/at onset
    n_sepsis = 0
    tp_hours, fp_hours, tn_hours, fn_hours = 0, 0, 0, 0

    for pid in test_ids:
        fp = RAW_DIR / f"{pid}.psv"
        if not fp.exists():
            continue
        df, onset_hour = score_patient(fp)

        is_sepsis = onset_hour is not None
        alerts = df.index[df["combined_alert"]].tolist()

        if is_sepsis:
            n_sepsis += 1
            pre_onset_alerts = [a for a in alerts if a <= onset_hour]
            if pre_onset_alerts:
                first_alert = pre_onset_alerts[0]
                lead_times.append(onset_hour - first_alert)
            else:
                missed += 1

        # hour-level confusion matrix
        for h in range(len(df)):
            true_label = df["SepsisLabel"].iloc[h]
            pred = df["combined_alert"].iloc[h]
            if true_label == 1 and pred:
                tp_hours += 1
            elif true_label == 1 and not pred:
                fn_hours += 1
            elif true_label == 0 and pred:
                fp_hours += 1
            else:
                tn_hours += 1

    precision = tp_hours / (tp_hours + fp_hours) if (tp_hours + fp_hours) > 0 else 0
    recall = tp_hours / (tp_hours + fn_hours) if (tp_hours + fn_hours) > 0 else 0
    specificity = tn_hours / (tn_hours + fp_hours) if (tn_hours + fp_hours) > 0 else 0

    print("\n=== SOFA/qSOFA Baseline Results (real test-set patients) ===")
    print(f"Sepsis-positive test patients: {n_sepsis}")
    print(f"Detected before/at onset: {n_sepsis - missed} "
          f"({100*(n_sepsis-missed)/n_sepsis:.1f}%)" if n_sepsis else "")
    print(f"Missed entirely: {missed}")
    if lead_times:
        print(f"Average lead-time before clinical onset: "
              f"{np.mean(lead_times):.2f} hours (median: {np.median(lead_times):.1f})")
    print(f"\nHour-level: precision={precision:.3f}, recall={recall:.3f}, "
          f"specificity={specificity:.3f}")
    print(f"TP={tp_hours}, FP={fp_hours}, TN={tn_hours}, FN={fn_hours}")

    results = {
        "n_sepsis_patients": n_sepsis,
        "detected_before_onset": n_sepsis - missed,
        "missed": missed,
        "avg_lead_time_hours": float(np.mean(lead_times)) if lead_times else None,
        "median_lead_time_hours": float(np.median(lead_times)) if lead_times else None,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp_hours": tp_hours, "fp_hours": fp_hours,
        "tn_hours": tn_hours, "fn_hours": fn_hours,
    }
    pd.DataFrame([results]).to_csv(OUT_DIR / "sofa_baseline_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'sofa_baseline_results.csv'}")
    return results


if __name__ == "__main__":
    evaluate()
