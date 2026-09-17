"""
Tabular-only baseline for the fixed-5yr-MACE binary task. No image
needed.

Architecture: TabularEncoder (missingness-aware) + small MLP head -> single
logit. Expects a tabular_csv of per-patient risk factors: labs,
diabetes/hypertension flags, medication flags, smoking/alcohol, family
history of CVD, age.
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

PROJECT_ROOT = Path(os.environ.get("MAMMOCVD_ROOT", "."))  # set to your repo checkout root
OUT_DIR = PROJECT_ROOT / "outputs" / "mammo_cvd"
COHORT_CSV = OUT_DIR / "finetune_cohort.csv"
SPLITS_CSV = OUT_DIR / "finetune_splits.csv"
TABULAR_CSV = OUT_DIR / "tabular_features_v2.csv"


CARDIOVASCULAR_CONTINUOUS = ["total_cholesterol", "ldl", "hdl", "triglycerides", "creatinine", "hba1c"]
CARDIOVASCULAR_BINARY = ["diabetes", "hypertension_dx"]
DEMOGRAPHIC_CONTINUOUS = ["age_at_baseline", "bmi"]
DEMOGRAPHIC_BINARY = ["smoker", "alcohol_use", "family_hx_cvd"]

FEATURE_SETS = {
    "all": (CARDIOVASCULAR_CONTINUOUS + DEMOGRAPHIC_CONTINUOUS, CARDIOVASCULAR_BINARY + DEMOGRAPHIC_BINARY),
    "cardiovascular": (CARDIOVASCULAR_CONTINUOUS, CARDIOVASCULAR_BINARY),
    "demographic": (DEMOGRAPHIC_CONTINUOUS, DEMOGRAPHIC_BINARY),
}


class TabularOnlyDataset(Dataset):
    def __init__(self, split: str, continuous_features: list, binary_features: list,
                 norm_stats: dict | None = None, knockout_prob: float = 0.0,
                 cohort_csv=None, splits_csv=None, tabular_csv=None):
        self.continuous_features = continuous_features
        self.binary_features = binary_features
        self.all_features = continuous_features + binary_features
        self.knockout_prob = knockout_prob if split == "train" else 0.0

        cohort_csv = cohort_csv or COHORT_CSV
        splits_csv = splits_csv or SPLITS_CSV
        tabular_csv = tabular_csv or TABULAR_CSV
        cohort = pd.read_csv(cohort_csv, dtype={"patient_id": str})
        tab = pd.read_csv(tabular_csv, dtype={"patient_id": str})
        splits = pd.read_csv(splits_csv, dtype={"patient_id": str})

        cohort_cols = ["patient_id", "label_5yr"]
        has_age_override = "age_at_baseline" in cohort.columns
        if has_age_override:
            cohort_cols.append("age_at_baseline")

        join_keys = ["patient_id"]
        if "study_date" in tab.columns and "study_date" in cohort.columns:
            cohort_cols.append("study_date")
            join_keys.append("study_date")
        df = cohort[cohort_cols].merge(tab, on=join_keys, how="left", suffixes=("_cohort", ""))
        if has_age_override:
            df["age_at_baseline"] = df["age_at_baseline_cohort"].combine_first(df["age_at_baseline"])
            df = df.drop(columns=["age_at_baseline_cohort"])
        df = df.merge(splits, on="patient_id", how="inner")
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
                row["patient_id"])


class TabularEncoder(nn.Module):
    """Per-feature: linear(value) if present, learned "missing" vector if
    not. Sums the per-feature embeddings into one tabular representation."""

    def __init__(self, n_features: int, d_feature: int = 32):
        super().__init__()
        self.n_features = n_features
        self.d_feature = d_feature
        self.value_proj = nn.ModuleList([nn.Linear(1, d_feature) for _ in range(n_features)])
        self.missing_embed = nn.Parameter(torch.randn(n_features, d_feature) * 0.02)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """values: (B, F) z-scored, NaN-safe (garbage where mask=0).
        mask: (B, F) 1=present, 0=missing. Returns (B, d_feature)."""
        b, f = values.shape
        out = torch.zeros(b, self.d_feature, device=values.device)
        for i in range(f):
            present = mask[:, i : i + 1]  # (B,1)
            v = values[:, i : i + 1]
            projected = self.value_proj[i](v)  # (B, d_feature)
            missing = self.missing_embed[i].unsqueeze(0).expand(b, -1)
            out = out + present * projected + (1 - present) * missing
        return out


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
                     help="for multi-seed variance estimates -- seed!=0 appends _seedN to "
                          "checkpoint/prediction filenames so repeat runs don't collide")
    ap.add_argument("--cohort_csv", type=str, default=None, help="override COHORT_CSV")
    ap.add_argument("--splits_csv", type=str, default=None, help="override SPLITS_CSV")
    ap.add_argument("--tabular_csv", type=str, default=None, help="override TABULAR_CSV")
    args = ap.parse_args()

    # Seeds model init/dropout; DataLoader shuffle order is separately
    # covered by the seeded `generator=` passed to it below (a bare global
    # seed does not fully control shuffle order).
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
