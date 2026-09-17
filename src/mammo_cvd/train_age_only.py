"""
Age-only baseline: a plain (unweighted) LogisticRegression(age_at_baseline).
Deterministic (closed-form sklearn fit, no training-seed randomness) 

Output: outputs/mammo_cvd/test_predictions_age_only_{run_tag}.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(os.environ.get("MAMMOCVD_ROOT", "."))  # set to your repo checkout root
OUT_DIR = PROJECT_ROOT / "outputs" / "mammo_cvd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_csv", type=str, required=True)
    ap.add_argument("--splits_csv", type=str, required=True)
    ap.add_argument("--run_tag", type=str, required=True)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort_csv, dtype={"patient_id": str})
    splits = pd.read_csv(args.splits_csv, dtype={"patient_id": str})
    df = cohort.merge(splits, on="patient_id", how="inner").dropna(subset=["age_at_baseline"])

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    clf = LogisticRegression().fit(train[["age_at_baseline"]].values, train["label_5yr"].values)

    probs = clf.predict_proba(test[["age_at_baseline"]].values)[:, 1]
    labels = test["label_5yr"].values
    auroc = roc_auc_score(labels, probs)
    print(f"FINAL TEST (age-only, {args.run_tag}): auroc={auroc:.4f} (baseline prevalence={labels.mean():.4f})")

    pred_df = pd.DataFrame({"patient_id": test["patient_id"].values, "label": labels, "prob": probs})
    pred_path = OUT_DIR / f"test_predictions_age_only_{args.run_tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
