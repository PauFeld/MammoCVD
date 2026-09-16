"""
Age-only baseline: a plain (unweighted) LogisticRegression(age_at_baseline),
generalized to run on any cohort_csv/splits_csv -- 2026-09-03, needed
because the original age-only baseline (test_predictions_age_only.csv) was
never a proper standalone script; it was reconstructed this session
(verified to reproduce it almost exactly: e.g. empi=1000077043 -> 0.0295
reconstructed vs 0.02954 on-file) to score one patient outside the test
set. Now formalized so it can run on the 4 new expanded/symmetric-landmark
cohorts too, matching the user's point that age-only (and tabular-only)
need to be recomputed per-experiment, not just the image arms.

Deterministic (closed-form sklearn fit, no training-seed randomness) --
run once per cohort, not 10 seeds like the neural-net arms.

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

    cohort = pd.read_csv(args.cohort_csv, dtype={"empi": str})
    splits = pd.read_csv(args.splits_csv, dtype={"empi": str})
    df = cohort.merge(splits, on="empi", how="inner").dropna(subset=["age_at_baseline"])

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    clf = LogisticRegression().fit(train[["age_at_baseline"]].values, train["label_5yr"].values)

    probs = clf.predict_proba(test[["age_at_baseline"]].values)[:, 1]
    labels = test["label_5yr"].values
    auroc = roc_auc_score(labels, probs)
    print(f"FINAL TEST (age-only, {args.run_tag}): auroc={auroc:.4f} (baseline prevalence={labels.mean():.4f})")

    pred_df = pd.DataFrame({"empi": test["empi"].values, "label": labels, "prob": probs})
    pred_path = OUT_DIR / f"test_predictions_age_only_{args.run_tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
