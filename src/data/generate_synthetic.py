"""
generate_synthetic.py
----------------------
Generates a synthetic ICU vitals dataset that mirrors the MIMIC-III schema
(patient stays -> hourly vitals -> sepsis onset label) closely enough that
every downstream script (preprocessing, baseline, model, dashboard) can be
built and fully tested WITHOUT real patient data.

Why this exists:
  MIMIC-III requires individual PhysioNet credentialing (CITI training +
  approval, which can take days). This generator lets your team build and
  debug the ENTIRE pipeline in parallel, right now. Once credentialed access
  comes through, swap `extract_mimic.py` in for this file's output -- the
  schema is designed to match, so nothing downstream needs to change.

Output: data/raw/synthetic_icu_vitals.csv
Columns:
  stay_id, hour, heart_rate, resp_rate, sbp, dbp, temp_c, spo2,
  lactate, wbc, creatinine, sepsis_onset_hour (NaN if never), is_sepsis_case
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

N_PATIENTS = 600          # number of simulated ICU stays
SEPSIS_RATE = 0.25         # fraction of patients who develop sepsis
STAY_LENGTH_HOURS = (24, 96)  # min/max length of an ICU stay
DETERIORATION_WINDOW = 8  # hours of build-up before clinical sepsis onset

OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "synthetic_icu_vitals.csv"


def simulate_stable_patient(n_hours):
    """Vitals that fluctuate mildly around healthy baselines (correlated noise)."""
    hr = 75 + RNG.normal(0, 4, n_hours).cumsum() * 0.05
    hr = np.clip(hr, 55, 100)

    resp = 16 + RNG.normal(0, 1.2, n_hours).cumsum() * 0.03
    resp = np.clip(resp, 10, 22)

    sbp = 120 + RNG.normal(0, 5, n_hours).cumsum() * 0.04
    sbp = np.clip(sbp, 95, 140)

    dbp = sbp * 0.62 + RNG.normal(0, 3, n_hours)
    temp = 36.8 + RNG.normal(0, 0.15, n_hours)
    spo2 = np.clip(97 + RNG.normal(0, 0.8, n_hours), 92, 100)

    lactate = np.clip(1.0 + RNG.normal(0, 0.2, n_hours), 0.4, 2.0)
    wbc = np.clip(8 + RNG.normal(0, 1.0, n_hours), 4, 11)
    creat = np.clip(0.9 + RNG.normal(0, 0.1, n_hours), 0.5, 1.2)

    return hr, resp, sbp, dbp, temp, spo2, lactate, wbc, creat


def simulate_sepsis_patient(n_hours, onset_hour):
    """
    Vitals start stable, then show a realistic deterioration pattern
    building up to `onset_hour`: rising HR & RR, falling BP, rising temp
    or hypothermia, falling SpO2, rising lactate/WBC, rising creatinine.
    This mirrors the real physiological signature of sepsis (SIRS response
    -> compensated shock -> organ dysfunction).
    """
    hr, resp, sbp, dbp, temp, spo2, lactate, wbc, creat = simulate_stable_patient(n_hours)

    ramp_start = max(0, onset_hour - DETERIORATION_WINDOW)
    for h in range(n_hours):
        if h < ramp_start:
            continue
        # progress: 0 -> 1 as we approach and pass onset
        progress = np.clip((h - ramp_start) / DETERIORATION_WINDOW, 0, 1.3)
        hr[h] += 28 * progress + RNG.normal(0, 2)
        resp[h] += 10 * progress + RNG.normal(0, 1)
        sbp[h] -= 22 * progress + RNG.normal(0, 3)
        dbp[h] -= 12 * progress + RNG.normal(0, 2)
        temp[h] += (2.0 if RNG.random() > 0.3 else -1.5) * progress
        spo2[h] -= 6 * progress + RNG.normal(0, 0.5)
        lactate[h] += 2.8 * progress + RNG.normal(0, 0.2)
        wbc[h] += 6.5 * progress + RNG.normal(0, 0.5)
        creat[h] += 0.9 * progress + RNG.normal(0, 0.1)

    hr = np.clip(hr, 50, 180)
    resp = np.clip(resp, 8, 45)
    sbp = np.clip(sbp, 55, 145)
    dbp = np.clip(dbp, 30, 95)
    spo2 = np.clip(spo2, 78, 100)
    lactate = np.clip(lactate, 0.4, 9.0)
    wbc = np.clip(wbc, 3, 28)
    creat = np.clip(creat, 0.4, 4.5)

    return hr, resp, sbp, dbp, temp, spo2, lactate, wbc, creat


def inject_missingness(df, cols, rate=0.08):
    """MIMIC-III vitals are irregularly sampled -- simulate missing readings."""
    mask = RNG.random(size=(len(df), len(cols))) < rate
    for i, c in enumerate(cols):
        df.loc[mask[:, i], c] = np.nan
    return df


def main():
    rows = []
    vital_cols = ["heart_rate", "resp_rate", "sbp", "dbp", "temp_c", "spo2",
                  "lactate", "wbc", "creatinine"]

    for stay_id in range(1, N_PATIENTS + 1):
        n_hours = int(RNG.integers(*STAY_LENGTH_HOURS))
        is_sepsis = RNG.random() < SEPSIS_RATE

        if is_sepsis:
            # onset must leave at least DETERIORATION_WINDOW hours of runway
            onset_hour = int(RNG.integers(DETERIORATION_WINDOW + 4, n_hours))
            vitals = simulate_sepsis_patient(n_hours, onset_hour)
        else:
            onset_hour = np.nan
            vitals = simulate_stable_patient(n_hours)

        hr, resp, sbp, dbp, temp, spo2, lactate, wbc, creat = vitals

        stay_df = pd.DataFrame({
            "stay_id": stay_id,
            "hour": np.arange(n_hours),
            "heart_rate": hr, "resp_rate": resp, "sbp": sbp, "dbp": dbp,
            "temp_c": temp, "spo2": spo2, "lactate": lactate, "wbc": wbc,
            "creatinine": creat,
            "sepsis_onset_hour": onset_hour,
            "is_sepsis_case": is_sepsis,
        })
        rows.append(stay_df)

    full_df = pd.concat(rows, ignore_index=True)
    full_df = inject_missingness(full_df, vital_cols, rate=0.08)
    full_df = full_df.round(2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_PATH, index=False)

    n_sepsis = full_df.groupby("stay_id")["is_sepsis_case"].first().sum()
    print(f"Generated {N_PATIENTS} patient stays ({n_sepsis} sepsis cases, "
          f"{N_PATIENTS - n_sepsis} controls)")
    print(f"Total rows (patient-hours): {len(full_df)}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
