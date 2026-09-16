"""
Tabular-only baseline for the X-Cardia-style fixed-5yr-MACE binary task
(finetune_cohort.csv / finetune_splits.csv). No image, no GPU needed --
runs while DINO pretraining occupies the GPU. This is the number the
DINO-pretrained and no-pretraining image models both need to beat (or at
least understand their relationship to) once fine-tuning runs.

Architecture: TabularEncoder (missingness-aware, from mirai_survival.py)
+ small MLP head -> single logit. Uses the expanded v2 tabular feature set
(build_tabular_features_v2.py): labs, diabetes/hypertension flags,
medication flags, smoking/alcohol, family history of CVD, age.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.mammo_cvd.mirai_survival import TabularEncoder

PROJECT_ROOT = Path(os.environ.get("MAMMOCVD_ROOT", "."))  # set to your repo checkout root
OUT_DIR = PROJECT_ROOT / "outputs" / "mammo_cvd"
COHORT_CSV = OUT_DIR / "finetune_cohort.csv"
SPLITS_CSV = OUT_DIR / "finetune_splits.csv"
TABULAR_CSV = OUT_DIR / "tabular_features_v2.csv"

# Split per user's three-question framing (2026-08-19): "cardiovascular workup"
# features require the patient to have had CVD-relevant testing/diagnosis;
# "demographic" features are routinely available with no workup needed.
#
# statin_med and antihypertensive_med deliberately EXCLUDED from the
# cardiovascular set (2026-08-19, per user): a medication flag reflects
# "a clinician already diagnosed elevated risk and chose to treat it," not
# a raw physiological measurement -- including it makes the "cardiovascular
# workup" baseline partly a "was this patient already flagged by a doctor"
# detector, which is close to circular for testing whether the model finds
# genuine risk signal. Diagnosis flags (diabetes, hypertension) are kept --
# also clinical-judgment-derived, but they represent actual risk-factor
# conditions rather than a treatment decision already made in response to
# assessed risk, a meaningfully different level of circularity.
CARDIOVASCULAR_CONTINUOUS = ["total_cholesterol", "ldl", "hdl", "triglycerides", "creatinine", "hba1c"]
CARDIOVASCULAR_BINARY = ["diabetes", "hypertension_dx"]
# BMI moved to demographic 2026-08-19 (per user): it's a routine vital-sign
# measurement (height/weight), not something requiring a dedicated
# cardiovascular workup order the way a lipid panel does -- currently 0%
# coverage in our data regardless (same "lab code postdates our cohort's
# era" issue as eGFR), but the tier assignment matters if that's ever fixed.
DEMOGRAPHIC_CONTINUOUS = ["age_at_baseline", "bmi"]
DEMOGRAPHIC_BINARY = ["smoker", "alcohol_use", "family_hx_cvd"]
MEDICATION_BINARY = ["antihypertensive_med", "statin_med"]  # excluded from all tiers below by default

FEATURE_SETS = {
    "all": (CARDIOVASCULAR_CONTINUOUS + DEMOGRAPHIC_CONTINUOUS, CARDIOVASCULAR_BINARY + DEMOGRAPHIC_BINARY),
    "cardiovascular": (CARDIOVASCULAR_CONTINUOUS, CARDIOVASCULAR_BINARY),
    "demographic": (DEMOGRAPHIC_CONTINUOUS, DEMOGRAPHIC_BINARY),
    "all_with_meds": (CARDIOVASCULAR_CONTINUOUS + DEMOGRAPHIC_CONTINUOUS,
                       CARDIOVASCULAR_BINARY + DEMOGRAPHIC_BINARY + MEDICATION_BINARY),
    # Isolates statin_med specifically (2026-08-22, per user) -- separate from
    # all_with_meds (which also adds antihypertensive_med) so the effect of
    # this one feature can be read cleanly before deciding whether to add
    # antihypertensive_med too. 100% coverage in tabular_features_v2.csv
    # (20.1% prevalence), no missingness handling needed.
    "all_with_statin": (CARDIOVASCULAR_CONTINUOUS + DEMOGRAPHIC_CONTINUOUS,
                         CARDIOVASCULAR_BINARY + DEMOGRAPHIC_BINARY + ["statin_med"]),
}


class TabularOnlyDataset(Dataset):
    def __init__(self, split: str, continuous_features: list, binary_features: list,
                 norm_stats: dict | None = None, knockout_prob: float = 0.0,
                 cohort_csv=None, splits_csv=None, tabular_csv=None):
        self.continuous_features = continuous_features
        self.binary_features = binary_features
        self.all_features = continuous_features + binary_features
        # Knockout (arxiv.org/abs/2405.20448), see pretrain_mmcl.py's
        # MMCLDataset.knockout_prob docstring for the full rationale --
        # only ever applied when split=="train" (set below), never val/test.
        self.knockout_prob = knockout_prob if split == "train" else 0.0

        cohort_csv = cohort_csv or COHORT_CSV
        splits_csv = splits_csv or SPLITS_CSV
        tabular_csv = tabular_csv or TABULAR_CSV
        cohort = pd.read_csv(cohort_csv, dtype={"empi": str})
        tab = pd.read_csv(tabular_csv, dtype={"empi": str})
        splits = pd.read_csv(splits_csv, dtype={"empi": str})

        cohort_cols = ["empi", "label_5yr"]
        # landmark_cohort.csv carries its own (re-anchored) age_at_baseline
        # for the 267 relabeled patients -- tabular_features_v2.csv's age
        # is stale for exactly those rows (built against the ORIGINAL
        # baseline dates), so prefer the cohort file's age when present.
        has_age_override = "age_at_baseline" in cohort.columns
        if has_age_override:
            cohort_cols.append("age_at_baseline")
        # 2026-09-03: multi-instance-aware tabular files (dual-pair cohort)
        # have one row per (empi, study_date) instance, not one per empi --
        # merging on empi alone would cross-multiply a patient's multiple
        # cohort rows against their multiple tabular rows. Join on both
        # keys whenever the tabular file carries study_date (built by
        # build_tabular_features_v2_multiinstance.py); falls back to the
        # original empi-only join for the single-instance tabular files.
        join_keys = ["empi"]
        if "study_date" in tab.columns and "study_date" in cohort.columns:
            cohort_cols.append("study_date")
            join_keys.append("study_date")
        df = cohort[cohort_cols].merge(tab, on=join_keys, how="left", suffixes=("_cohort", ""))
        if has_age_override:
            df["age_at_baseline"] = df["age_at_baseline_cohort"].combine_first(df["age_at_baseline"])
            df = df.drop(columns=["age_at_baseline_cohort"])
        df = df.merge(splits, on="empi", how="inner")
        self.df = df[df["split"] == split].reset_index(drop=True)

        if norm_stats is None:
            self.norm_stats = {}
            for col in self.continuous_features:
                vals = self.df[col].dropna()
                self.norm_stats[col] = (vals.mean(), vals.std() if vals.std() > 0 else 1.0)
        else:
            self.norm_stats = norm_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        values, mask = [], []
        for col in self.all_features:
            raw = row.get(col)
            if col in self.binary_features:
                if pd.notna(raw):
                    values.append(float(raw)); mask.append(1.0)
                else:
                    values.append(0.0); mask.append(0.0)
            else:
                if pd.notna(raw):
                    mean, std = self.norm_stats[col]
                    values.append(float((raw - mean) / std)); mask.append(1.0)
                else:
                    values.append(0.0); mask.append(0.0)
        if self.knockout_prob > 0:
            for i in range(len(mask)):
                if mask[i] == 1.0 and np.random.rand() < self.knockout_prob:
                    values[i], mask[i] = 0.0, 0.0
        return (torch.tensor(values, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(float(row["label_5yr"])),
                row["empi"])


class TabularOnlyClassifier(nn.Module):
    def __init__(self, n_features: int, d_feature: int = 32, dropout: float = 0.3):
        super().__init__()
        self.encoder = TabularEncoder(n_features, d_feature=d_feature)
        self.head = nn.Sequential(
            nn.Linear(d_feature, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, values, mask):
        emb = self.encoder(values, mask)
        return self.head(emb)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for values, mask, label, _ in loader:
            values, mask = values.to(device), mask.to(device)
            logits = model(values, mask).squeeze(-1)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(label.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return roc_auc_score(labels, probs), probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--feature_set", type=str, default="all", choices=list(FEATURE_SETS))
    ap.add_argument("--tabular_knockout_prob", type=float, default=0.0,
                     help="Knockout (arxiv.org/abs/2405.20448), see TabularOnlyDataset's docstring "
                          "-- 0 = off, matches prior behavior exactly")
    ap.add_argument("--run_tag", type=str, default=None,
                     help="distinguishes output filenames from --feature_set alone, e.g. "
                          "'all_knockout' -- defaults to --feature_set if unset")
    ap.add_argument("--seed", type=int, default=0,
                     help="for multi-seed variance estimates -- seed 0 keeps the original, "
                          "un-suffixed checkpoint/prediction filenames exactly as before; "
                          "seed!=0 appends _seedN so repeat runs don't collide")
    ap.add_argument("--cohort_csv", type=str, default=None, help="override COHORT_CSV, e.g. landmark_cohort.csv")
    ap.add_argument("--splits_csv", type=str, default=None, help="override SPLITS_CSV, e.g. landmark_cohort_splits.csv")
    ap.add_argument("--tabular_csv", type=str, default=None, help="override TABULAR_CSV, e.g. a per-experiment baseline-restricted tabular_features_v2_{tag}.csv")
    args = ap.parse_args()

    # 2026-08-23: no script in this project set a fixed seed before this --
    # every training run got a different random init/shuffle order, which
    # is how an old, un-`--run_tag`-ed checkpoint (best_finetune_tabular_all.pt)
    # ended up mismatching its own logged result (two different invocations
    # silently overwrote the same unversioned path with two different valid
    # but non-identical models). Model init/dropout are covered by the two
    # seed calls below; DataLoader shuffle order is covered by the seeded
    # `generator=` passed to it explicitly (a bare global seed does not
    # fully control shuffle order).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    continuous, binary = FEATURE_SETS[args.feature_set]
    all_features = continuous + binary
    print(f"device: {device}  feature_set: {args.feature_set} ({all_features})  "
          f"knockout_prob: {args.tabular_knockout_prob}  seed: {args.seed}")

    train_ds = TabularOnlyDataset("train", continuous, binary, knockout_prob=args.tabular_knockout_prob,
                                   cohort_csv=args.cohort_csv, splits_csv=args.splits_csv, tabular_csv=args.tabular_csv)
    val_ds = TabularOnlyDataset("val", continuous, binary, norm_stats=train_ds.norm_stats,
                                 cohort_csv=args.cohort_csv, splits_csv=args.splits_csv, tabular_csv=args.tabular_csv)
    test_ds = TabularOnlyDataset("test", continuous, binary, norm_stats=train_ds.norm_stats,
                                  cohort_csv=args.cohort_csv, splits_csv=args.splits_csv, tabular_csv=args.tabular_csv)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = TabularOnlyClassifier(len(all_features)).to(device)
    n_pos = train_ds.df["label_5yr"].sum()
    n_neg = len(train_ds.df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"train positives={n_pos} negatives={n_neg} pos_weight={pos_weight.item():.2f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auroc = -1.0
    tag = args.run_tag or args.feature_set
    if args.seed != 0:
        tag = f"{tag}_seed{args.seed}"
    ckpt_path = OUT_DIR / f"best_finetune_tabular_{tag}.pt"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for values, mask, label, _ in train_loader:
            values, mask, label = values.to(device), mask.to(device), label.to(device)
            logits = model(values, mask).squeeze(-1)
            loss = criterion(logits, label)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        val_auroc, _, _ = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={np.mean(losses):.4f} val_auroc={val_auroc:.4f}")
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))
    test_auroc, probs, labels = evaluate(model, test_loader, device)
    print(f"\nFINAL TEST: auroc={test_auroc:.4f} (baseline prevalence={labels.mean():.4f})")

    empis = test_ds.df["empi"].values
    pred_df = pd.DataFrame({"empi": empis, "label": labels, "prob": probs})
    pred_path = OUT_DIR / f"test_predictions_tabular_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
