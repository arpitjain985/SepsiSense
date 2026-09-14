"""
train_gbt_model.py
--------------------
Trains a class-imbalance-aware Gradient Boosted Trees model on the REAL
processed PhysioNet Challenge 2019 data (train.csv / val.csv / test.csv from
preprocess.py).

Why Gradient Boosted Trees as the first real model (not "instead of" the LSTM
-- "in addition to" it, see train_lstm_colab.py for the sequence model to run
on Colab):
  - Our preprocess.py already converts each patient-hour into an engineered
    feature vector (windowed mean/min/max/trend for vitals, recency-aware
    last-known-value for labs). This is fundamentally tabular data, and GBTs
    are a strong, standard, well-published choice for exactly this kind of
    engineered clinical feature table (this is in fact what many top-ranking
    PhysioNet Challenge 2019 leaderboard entries used).
  - It runs entirely offline/CPU, so we get REAL trained-model metrics today
    instead of blocking on GPU/cloud access.
  - It gives you a second, different model family to compare against the
    LSTM later -- a stronger project than a single model alone.

Handles the 463:1 class imbalance found during preprocessing via
`class_weight="balanced"` (this reweights the loss so rare positive rows
aren't just ignored by the model).

Run:
  python src/models/train_gbt_model.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, confusion_matrix,
                              classification_report)
import joblib

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURE_COLS = None  # inferred at load time (everything except id/hour/label)


def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    feature_cols = [c for c in df.columns if c not in ("patient_id", "hour", "label")]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y, feature_cols, df


def find_best_threshold(y_true, y_prob):
    """Pick the probability threshold that maximizes F1, since accuracy is
    meaningless under 463:1 imbalance."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1s[:-1])  # last point has no matching threshold
    return thresholds[best_idx], f1s[best_idx]


def main():
    X_train, y_train, feature_cols, _ = load_split("train")
    X_val, y_val, _, _ = load_split("val")
    X_test, y_test, _, test_df = load_split("test")

    print(f"Train: {X_train.shape}, positives: {y_train.sum()} ({100*y_train.mean():.3f}%)")
    print(f"Val:   {X_val.shape}, positives: {y_val.sum()} ({100*y_val.mean():.3f}%)")
    print(f"Test:  {X_test.shape}, positives: {y_test.sum()} ({100*y_test.mean():.3f}%)")

    print("\nTraining Random Forest (class_weight='balanced')...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Threshold tuned on VALIDATION set only (never touch test set for this)
    val_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = find_best_threshold(y_val, val_prob)
    print(f"\nBest threshold (tuned on val set): {best_thresh:.4f} (val F1={best_f1:.3f})")

    # Final evaluation on held-out TEST set
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_thresh).astype(int)

    auroc = roc_auc_score(y_test, test_prob)
    auprc = average_precision_score(y_test, test_prob)
    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n=== Test Set Results (real held-out patients) ===")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}  (primary metric under severe imbalance)")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Confusion matrix -> TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Feature importance -- useful context for your explainability chapter
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\nTop 10 most important features:")
    print(importances.head(10))

    # Save everything
    joblib.dump(model, MODEL_DIR / "gbt_model.joblib")
    joblib.dump(best_thresh, MODEL_DIR / "gbt_threshold.joblib")

    test_df = test_df.copy()
    test_df["pred_prob"] = test_prob
    test_df["pred_label"] = test_pred
    test_df.to_csv(DATA_DIR / "test_predictions.csv", index=False)

    results = {
        "auroc": auroc, "auprc": auprc,
        "precision": precision, "recall": recall,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(best_thresh),
    }
    pd.DataFrame([results]).to_csv(DATA_DIR / "gbt_model_results.csv", index=False)
    importances.to_csv(DATA_DIR / "feature_importances.csv")

    print(f"\nSaved model to: {MODEL_DIR / 'gbt_model.joblib'}")
    print(f"Saved predictions to: {DATA_DIR / 'test_predictions.csv'}")
    print(f"Saved results to: {DATA_DIR / 'gbt_model_results.csv'}")


if __name__ == "__main__":
    main()
