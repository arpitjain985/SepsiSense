"""
preprocess.py
--------------
Runs on REAL PhysioNet Challenge 2019 sepsis data (.psv files, one per patient).

What this does:
  1. Loads every patient file.
  2. Truncates septic patients' records at their first SepsisLabel==1 hour
     (predicting AFTER a patient is already flagged septic is not clinically
     useful -- we only want pre-onset / at-onset hours).
  3. For every hour, builds a feature vector from a trailing window of the
     last WINDOW_HOURS hours: mean/min/max/last value + a missingness flag,
     for a clinically-relevant subset of vitals and labs.
  4. Imputes missing values: forward-fill within a patient first (carry last
     known reading forward, which is how a clinician would reason about it),
     then column-median fill for anything still missing (e.g. a value never
     recorded yet for that patient).
  5. Splits by PATIENT (stay file), not by row, into train/val/test --
     this prevents the single most common leakage bug in this kind of project.
  6. Saves processed_features.csv + train/val/test patient ID lists.

Run:
  python src/data/preprocess.py
"""

import glob
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "real_subset" / "subset_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_HOURS = 6          # trailing window used to build each row's features
RANDOM_SEED = 42

# Vitals: fairly complete, always include
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
# Labs: clinically important for sepsis (SIRS/organ-dysfunction signal) even
# though sparsely sampled -- we use "last known value" + "how many hours since
# last reading" rather than per-hour presence, which is the correct way to
# handle sparse labs (a lactate drawn 3 hours ago is still informative now).
LABS = ["WBC", "Creatinine", "Lactate", "Platelets", "BUN"]
STATIC = ["Age", "Gender"]

FEATURE_COLS = VITALS + LABS + STATIC


def load_patient(fp):
    df = pd.read_csv(fp, sep="|")
    df["patient_id"] = Path(fp).stem
    return df


def truncate_at_onset(df):
    """Keep only hours up to and including the first SepsisLabel==1 hour."""
    pos = df.index[df["SepsisLabel"] == 1]
    if len(pos) == 0:
        return df
    first_onset_idx = pos[0]
    return df.loc[:first_onset_idx].reset_index(drop=True)


def forward_fill_vitals(df):
    """Carry forward last known reading -- mirrors clinical reasoning
    ('last measured HR was X, 2 hours ago')."""
    df[VITALS + LABS] = df[VITALS + LABS].ffill()
    return df


def build_window_features(df):
    """
    For every hour t, compute features from the trailing WINDOW_HOURS window
    (t-WINDOW_HOURS+1 ... t): mean, min, max, last, and a trend (last - first)
    for vitals; last known value + hours-since-updated for labs.
    """
    rows = []
    n = len(df)
    for t in range(n):
        start = max(0, t - WINDOW_HOURS + 1)
        window = df.iloc[start:t + 1]

        feat = {"patient_id": df["patient_id"].iloc[0], "hour": t}

        for v in VITALS:
            series = window[v]
            feat[f"{v}_mean"] = series.mean()
            feat[f"{v}_min"] = series.min()
            feat[f"{v}_max"] = series.max()
            feat[f"{v}_last"] = series.iloc[-1]
            valid = series.dropna()
            feat[f"{v}_trend"] = (valid.iloc[-1] - valid.iloc[0]) if len(valid) >= 2 else 0.0
            feat[f"{v}_missing_frac"] = series.isna().mean()

        for l in LABS:
            series = df[l].iloc[:t + 1]  # full history so far, not just window
            last_valid = series.dropna()
            feat[f"{l}_last"] = last_valid.iloc[-1] if len(last_valid) > 0 else np.nan
            if len(last_valid) > 0:
                feat[f"{l}_hours_since"] = t - last_valid.index[-1]
            else:
                feat[f"{l}_hours_since"] = np.nan

        for s in STATIC:
            feat[s] = df[s].iloc[0]

        feat["label"] = df["SepsisLabel"].iloc[t]
        rows.append(feat)

    return pd.DataFrame(rows)


def impute_remaining(df, medians=None):
    """Column-median fill for anything still missing after forward-fill/window
    construction (e.g., a lab never drawn yet for that patient)."""
    feature_cols = [c for c in df.columns if c not in ("patient_id", "hour", "label")]
    if medians is None:
        medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)
    return df, medians


def patient_level_split(patient_ids, sepsis_status, seed=RANDOM_SEED,
                         train_frac=0.7, val_frac=0.15):
    """Stratified split by patient so no patient's hours appear in more than
    one split -- prevents leakage."""
    rng = np.random.default_rng(seed)
    ids = np.array(patient_ids)
    status = np.array(sepsis_status)

    train_ids, val_ids, test_ids = [], [], []
    for cls in [0, 1]:
        cls_ids = ids[status == cls]
        rng.shuffle(cls_ids)
        n = len(cls_ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_ids.extend(cls_ids[:n_train])
        val_ids.extend(cls_ids[n_train:n_train + n_val])
        test_ids.extend(cls_ids[n_train + n_val:])

    return set(train_ids), set(val_ids), set(test_ids)


def main():
    files = sorted(glob.glob(str(RAW_DIR / "*.psv")))
    print(f"Found {len(files)} real patient files")

    all_windows = []
    patient_ids, sepsis_status = [], []

    for fp in files:
        df = load_patient(fp)
        df = truncate_at_onset(df)
        df = forward_fill_vitals(df)
        wdf = build_window_features(df)
        all_windows.append(wdf)

        pid = Path(fp).stem
        patient_ids.append(pid)
        sepsis_status.append(int((df["SepsisLabel"] == 1).any()))

    full_df = pd.concat(all_windows, ignore_index=True)
    print(f"Total rows (patient-hours) after truncation: {len(full_df)}")
    print(f"Positive rows: {full_df['label'].sum()} "
          f"({100*full_df['label'].mean():.2f}% of rows)")

    train_ids, val_ids, test_ids = patient_level_split(patient_ids, sepsis_status)
    print(f"Patient split -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    train_df = full_df[full_df["patient_id"].isin(train_ids)].copy()
    val_df = full_df[full_df["patient_id"].isin(val_ids)].copy()
    test_df = full_df[full_df["patient_id"].isin(test_ids)].copy()

    train_df, medians = impute_remaining(train_df)
    val_df, _ = impute_remaining(val_df, medians=medians)
    test_df, _ = impute_remaining(test_df, medians=medians)

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    medians.to_csv(OUT_DIR / "impute_medians.csv")

    with open(OUT_DIR / "split_ids.txt", "w") as f:
        f.write("TRAIN:\n" + "\n".join(sorted(train_ids)) + "\n\n")
        f.write("VAL:\n" + "\n".join(sorted(val_ids)) + "\n\n")
        f.write("TEST:\n" + "\n".join(sorted(test_ids)) + "\n")

    print(f"\nSaved: {OUT_DIR / 'train.csv'} ({len(train_df)} rows)")
    print(f"Saved: {OUT_DIR / 'val.csv'} ({len(val_df)} rows)")
    print(f"Saved: {OUT_DIR / 'test.csv'} ({len(test_df)} rows)")
    print(f"Train positive rate: {100*train_df['label'].mean():.2f}%")
    print(f"Val positive rate:   {100*val_df['label'].mean():.2f}%")
    print(f"Test positive rate:  {100*test_df['label'].mean():.2f}%")


if __name__ == "__main__":
    main()
