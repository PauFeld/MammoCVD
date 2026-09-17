"""
Cross-arm comparison on the strict_cv held-out test set: AUROC + bootstrap CI + sensitivity/specificity at
the Youden-J optimal threshold for each of the four reported arms
(age-only, tabular, Mammo-CLIP, Mammo-FM), plus pairwise DeLong
significance tests.

Expects the four arms' `test_predictions_*.csv` files (patient_id,label,prob)
under $MAMMOCVD_ROOT/outputs/mammo_cvd/, produced by
train_age_only.py / train_finetune_tabular_only.py /
train_finetune_simplefusion_scankeyed.py against the strict_cv_test.csv
cohort.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(os.environ.get("MAMMOCVD_ROOT", "."))  # set to your repo checkout root
OUT_DIR = PROJECT_ROOT / "outputs" / "mammo_cvd"

CSV_SOURCES = {
    "age_only": "test_predictions_age_only.csv",
    "tabular": "test_predictions_tabular.csv",
    "mammoclip": "test_predictions_mammoclip_simplefusion_test.csv",
    "mammofm": "test_predictions_mammofm_simplefusion_test.csv",
}

PAIRS = [
    ("age_only", "tabular"),
    ("age_only", "mammoclip"),
    ("tabular", "mammoclip"),
    ("tabular", "mammofm"),
    ("mammoclip", "mammofm"),
]


def sens_spec_at_youden(labels, probs):
    fpr, tpr, thresh = roc_curve(labels, probs)
    j = tpr - fpr
    idx = np.argmax(j)
    return {"threshold": thresh[idx], "sensitivity": tpr[idx], "specificity": 1 - fpr[idx]}


def delong_auc_var(labels, probs):
    """DeLong (1988) variance of the AUC estimator, via the Mann-Whitney U /
    placement-values formulation (no external dependency needed)."""
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    n_pos, n_neg = len(pos), len(neg)

    def midrank(x):
        order = np.argsort(x)
        ranks = np.empty(len(x))
        sorted_x = x[order]
        i = 0
        while i < len(x):
            j = i
            while j < len(x) - 1 and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return ranks

    all_scores = np.concatenate([pos, neg])
    all_ranks = midrank(all_scores)
    pos_ranks = all_ranks[:n_pos]
    v01 = (pos_ranks - midrank(pos)) / n_neg
    neg_ranks = all_ranks[n_pos:]
    v10 = 1 - (neg_ranks - midrank(neg)) / n_pos

    auc = v01.mean()
    s01 = np.var(v01, ddof=1) / n_pos
    s10 = np.var(v10, ddof=1) / n_neg
    return auc, s01 + s10


def bootstrap_auroc_ci(labels, probs, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI, resampling patients (not events) with
    replacement -- standard nonparametric AUROC CI, doesn't assume the
    DeLong normal approximation."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_aucs = np.empty(n_boot)
    idx_all = np.arange(n)
    i = 0
    while i < n_boot:
        idx = rng.choice(idx_all, size=n, replace=True)
        y, p = labels[idx], probs[idx]
        if y.min() == y.max():  # degenerate resample, no positives or no negatives
            continue
        boot_aucs[i] = roc_auc_score(y, p)
        i += 1
    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, boot_aucs.std()


def delong_test(labels, probs_a, probs_b):
    """Two-sided p-value for AUC(a) == AUC(b) on the SAME patients
    (paired, correlated test sets) -- ignores the covariance term between
    the two correlated AUCs (conservative: true variance of the
    difference is <= this, so this p-value is an upper bound / slightly
    conservative, not anti-conservative)."""
    auc_a, var_a = delong_auc_var(labels, probs_a)
    auc_b, var_b = delong_auc_var(labels, probs_b)
    se_diff = np.sqrt(var_a + var_b)
    z = (auc_a - auc_b) / se_diff
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, z, p


def main():
    frames = {}
    for arm, fname in CSV_SOURCES.items():
        path = OUT_DIR / fname
        if not path.exists():
            print(f"  skipping {arm}: {fname} not found yet")
            continue
        d = pd.read_csv(path, dtype={"patient_id": str}).rename(columns={"prob": arm})
        frames[arm] = d[["patient_id", "label", arm]] if not frames else d[["patient_id", arm]]

    if "tabular" not in frames:
        raise SystemExit("need at least test_predictions_tabular.csv (carries the label column) to proceed")

    df = frames.pop("tabular")
    df = df.rename(columns={list(df.columns)[-1]: "tabular"}) if "tabular" not in df.columns else df
    for arm, d in frames.items():
        df = df.merge(d, on="patient_id", how="inner")
    print(f"matched patients across all arms: {len(df)} (positives={df['label'].sum()})")

    labels = df["label"].values
    arms = {a: a for a in CSV_SOURCES if a in df.columns}

    print("\n=== AUROC + sensitivity/specificity (Youden-J optimal threshold) ===")
    print("(bootstrap CIs: 2000 resamples, percentile method, resampling patients not events)")
    rows = []
    for name, col in arms.items():
        auroc = roc_auc_score(labels, df[col])
        ci_lo, ci_hi, ci_std = bootstrap_auroc_ci(labels, df[col].values)
        ss = sens_spec_at_youden(labels, df[col].values)
        rows.append({"arm": name, "auroc": auroc, "auroc_ci_lo": ci_lo, "auroc_ci_hi": ci_hi,
                     "auroc_boot_std": ci_std, **ss})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "compare_arms_summary.csv", index=False)

    print("\n=== DeLong pairwise significance tests (paired, same patients) ===")
    for name_a, name_b in PAIRS:
        if name_a not in df.columns or name_b not in df.columns:
            continue
        auc_a, auc_b, z, p = delong_test(labels, df[name_a].values, df[name_b].values)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name_a} ({auc_a:.4f}) vs {name_b} ({auc_b:.4f}): z={z:.3f} p={p:.4f} {sig}")

    df.to_csv(OUT_DIR / "compare_arms_predictions.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'compare_arms_summary.csv'} and compare_arms_predictions.csv")


if __name__ == "__main__":
    main()
