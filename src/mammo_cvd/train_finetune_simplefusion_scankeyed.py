"""
Generalized simplefusion trainer for the 2026-09-03 expanded/symmetric-
landmark cohort experiments (3, 4, 5a, 5b) -- same architecture as
train_finetune_mammoclip_simplefusion.py / train_finetune_mammofm_simplefusion.py
(frozen backbone + mean-pool + small MLP head), but:

1. --backbone selects mammoclip/mammofm at runtime instead of being a
   separate hardcoded script.
2. The embedding dataset is keyed by SCAN IDENTITY (empi, study_date,
   view), reading from *_embeddings_scankeyed/ (built by
   extract_embeddings_scankeyed.py), not the old empi-only cache --
   required because these cohorts can select a different physical
   baseline scan for the same patient across variants (confirmed 11,879/
   18,017 patients differ between the plain-expanded and symmetric-
   landmark cohorts), so an empi-only key would silently mix scans
   across experiments.

MLO-only (L_MLO/R_MLO), matching this experiment set's scope.
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

from src.mammo_cvd.mirai_encoder import STANDARD_VIEWS

PROJECT_ROOT = Path(os.environ.get("MAMMOCVD_ROOT", "."))  # set to your repo checkout root
OUT_DIR = PROJECT_ROOT / "outputs" / "mammo_cvd"

MLO_VIEWS = ["L_MLO", "R_MLO"]
VIEW_TO_IDX = {v: i for i, v in enumerate(STANDARD_VIEWS)}


class ScanKeyedEmbedDataset(Dataset):
    """2026-09-04 perf fix: __getitem__ used to re-open both view .npy
    files from disk on every sample access, every epoch -- for a frozen-
    embedding probe head this tiny (a few thousand params) that made the
    dataset almost entirely I/O-bound (observed ~51s/epoch, worse under
    concurrent extraction jobs competing for the same scratch filesystem),
    when the actual compute per epoch is milliseconds. A cohort's full set
    of embeddings easily fits in RAM (~2048 floats x 2 views x ~22k
    patients =~ 360MB), so load everything once at init and serve from an
    in-memory dict instead."""
    def __init__(self, split: str, cohort_csv: str, splits_csv: str, embed_dir: Path, embed_dim: int):
        cohort = pd.read_csv(cohort_csv, dtype={"empi": str})
        cohort["study_date"] = pd.to_datetime(cohort["study_date"]).dt.strftime("%Y%m%d")
        splits = pd.read_csv(splits_csv, dtype={"empi": str})
        df = cohort.merge(splits, on="empi", how="inner")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.embed_dim = embed_dim

        self._cache: dict[str, np.ndarray] = {}
        for row in self.df.itertuples(index=False):
            for v in MLO_VIEWS:
                key = f"{row.empi}_{row.study_date}_{v}"
                p = embed_dir / f"{key}.npy"
                if p.exists():
                    self._cache[key] = np.load(p).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        embeds, mask = [], []
        for v in MLO_VIEWS:
            key = f"{row['empi']}_{row['study_date']}_{v}"
            cached = self._cache.get(key)
            if cached is not None:
                embeds.append(cached)
                mask.append(1.0)
            else:
                embeds.append(np.zeros(self.embed_dim, dtype=np.float32))
                mask.append(0.0)
        embeds = torch.from_numpy(np.stack(embeds).astype(np.float32))  # (V, D)
        mask = torch.tensor(mask, dtype=torch.float32)
        label = torch.tensor(float(row["label_5yr"]))
        return embeds, mask, label


class SimpleFusionHead(nn.Module):
    """Plain mean-pool across available views + a small MLP head -- no
    view-conditioning, no cross-view attention. Matches the level of
    multi-view fusion sophistication actually used in the Mammo-CLIP/
    Mammo-FM papers' own downstream evaluations."""

    def __init__(self, d_model: int, n_outputs: int = 1, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_outputs),
        )

    def forward(self, embeds, mask):
        # mean-pool over valid views only
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        exam = (embeds * mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.mlp(exam)
        return logits, exam


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for embeds, mask, label in loader:
            embeds, mask = embeds.to(device), mask.to(device)
            logits, _ = model(embeds, mask)
            all_probs.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            all_labels.append(label.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return roc_auc_score(labels, probs), probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                     help="for multi-seed variance estimates -- seed 0 keeps the original, "
                          "un-suffixed checkpoint/prediction filenames exactly as before; "
                          "seed!=0 appends _seedN so repeat runs don't collide")
    ap.add_argument("--backbone", choices=["mammoclip", "mammofm"], required=True)
    ap.add_argument("--cohort_csv", type=str, required=True)
    ap.add_argument("--splits_csv", type=str, required=True)
    ap.add_argument("--run_tag", type=str, required=True,
                     help="distinguishes checkpoint/predictions filenames across the different "
                          "cohort experiments (3/4/5a/5b) -- required, not optional, since there "
                          "is no single 'default' cohort here")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, backbone: {args.backbone}")

    if args.backbone == "mammoclip":
        from src.mammo_cvd.mammo_clip_features import OUT_DIM as EMBED_DIM
        embed_dir = OUT_DIR / "mammo_clip_embeddings_scankeyed"
    else:
        from src.mammo_cvd.mammo_fm_features import OUT_DIM as EMBED_DIM
        embed_dir = OUT_DIR / "mammo_fm_embeddings_scankeyed"

    train_ds = ScanKeyedEmbedDataset("train", args.cohort_csv, args.splits_csv, embed_dir, EMBED_DIM)
    val_ds = ScanKeyedEmbedDataset("val", args.cohort_csv, args.splits_csv, embed_dir, EMBED_DIM)
    test_ds = ScanKeyedEmbedDataset("test", args.cohort_csv, args.splits_csv, embed_dir, EMBED_DIM)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SimpleFusionHead(d_model=EMBED_DIM).to(device)

    n_pos = train_ds.df["label_5yr"].sum()
    n_neg = len(train_ds.df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"train positives={n_pos} negatives={n_neg} pos_weight={pos_weight.item():.2f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auroc = -1.0
    epochs_since_improvement = 0
    seed_suffix = f"_seed{args.seed}" if args.seed != 0 else ""
    ckpt_path = OUT_DIR / f"best_finetune_{args.backbone}_simplefusion_{args.run_tag}{seed_suffix}.pt"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for embeds, mask, label in train_loader:
            embeds, mask, label = embeds.to(device), mask.to(device), label.to(device)
            logits, _ = model(embeds, mask)
            loss = criterion(logits.squeeze(-1), label)
            opt.zero_grad(); loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(loss.item())

        val_auroc, _, _ = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={np.mean(losses):.4f} val_auroc={val_auroc:.4f}")
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_since_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f"early stopping: no val_auroc improvement for {args.patience} epochs "
                      f"(best={best_val_auroc:.4f})")
                break

    model.load_state_dict(torch.load(ckpt_path))
    test_auroc, probs, labels = evaluate(model, test_loader, device)
    print(f"\nFINAL TEST ({args.backbone} simple mean-pool fusion, {args.run_tag}): auroc={test_auroc:.4f} "
          f"(baseline prevalence={labels.mean():.4f})")

    empis = test_ds.df["empi"].values
    pred_df = pd.DataFrame({"empi": empis, "label": labels, "prob": probs})
    pred_path = OUT_DIR / f"test_predictions_{args.backbone}_simplefusion_{args.run_tag}_test{seed_suffix}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
