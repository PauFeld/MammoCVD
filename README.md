# mammocvd

<p align="center">
  <img src="docs/fig1_method.png" alt="Method: frozen mammography foundation model encodes L-MLO/R-MLO views, embeddings are pooled into an exam representation, and an MLP head predicts 5-year MACE risk; Grad-CAM shown per view." width="720">
</p>

Code accompanying **"Mammography Foundation Models for Opportunistic Prediction of
Major Adverse Cardiovascular Events."**
Paula Feldman, Nusrat Binta Nizam, Sunwoo Kwak, Batuhan Karaman, Katerina Dodelzon,
Mert Sabuncu — Weill Cornell Medicine / Cornell Tech. **Preprint:** arXiv link TBD.

## Summary

We reuse **frozen mammography foundation models** (Mammo-CLIP, Mammo-FM) —
pretrained only for breast-cancer tasks, no cardiovascular supervision — to predict
5-year MACE directly from raw screening mammograms. No calcification segmentation
or BAC annotation is needed anywhere in the pipeline: the exam-level embedding
alone gets AUROC ≈ 0.82 (vs. 0.765 age-only, 0.859 full tabular), and Grad-CAM
shows the models attend to vessel-like structures without ever being told where
they are.

## Results (held-out test set, n = 5,287; 118 events)

| Model       | AUROC | 95% CI          |
|-------------|-------|-----------------|
| Age-only    | 0.765 | [0.722, 0.805]  |
| Tabular     | 0.859 | [0.825, 0.889]  |
| Mammo-CLIP  | 0.823 | [0.785, 0.856]  |
| Mammo-FM    | 0.822 | [0.784, 0.858]  |

At the top 20% of predicted risk, Mammo-CLIP and Mammo-FM each captured 68.6% of
future MACE events at 7.7% precision (~3.4-fold enrichment over the base rate).

## Repository contents

This repo ships the **modeling code** — embedding extraction, the four trained
arms, and cross-arm statistical comparison — starting from an already-built cohort
CSV. It does **not** include cohort construction / patient labeling, since that
code runs directly against identifiable institutional EHR and DICOM data.

**No patient data, split files, or model checkpoints are included or will be
published.** See `dummy_data/` for a small synthetic dataset (public-domain
mammogram images + invented labels) that smoke-tests every script here end-to-end.

```
src/mammo_cvd/
  dataset.py                                DICOM loading, breast-region auto-crop
  mirai_encoder.py / mirai_survival.py      view-conditioning aggregator components
  mammo_clip_features.py / mammo_fm_features.py   frozen EfficientNet-B5 encoders
  extract_embeddings_scankeyed.py           precompute per-view embeddings
  train_finetune_simplefusion_scankeyed.py  mean-pool + MLP head trainer (image arms)
  train_finetune_tabular_only.py / train_age_only.py   baselines
  compare_arms.py                           bootstrap CIs + DeLong tests across arms

dummy_data/
  make_dummy_dataset.py   regenerates the synthetic smoke-test dataset
  dummy_finetune_cohort.csv / dummy_tabular_features.csv / dummy_splits.csv
  images/*.dcm             synthetic DICOMs from public-domain mammograms
```

Expected cohort CSV schema (see `dummy_data/dummy_finetune_cohort.csv`): one row
per instance — `empi, study_date, label_5yr, path_L_MLO, path_R_MLO,
age_at_baseline` — plus a `splits.csv` (`empi, split`) and, for the tabular arm, a
`tabular_features.csv` of per-patient risk factors.

## Setup

```bash
pip install torch pandas numpy scikit-learn matplotlib pillow pydicom requests
```

Try it now with no real data: [`dummy_data/README.md`](dummy_data/README.md) runs
the age-only and tabular baselines end-to-end in a couple of commands.

All paths come from environment variables (nothing institutional is hardcoded):

| Variable                        | Purpose                                        |
|----------------------------------|------------------------------------------------|
| `MAMMOCVD_ROOT`                  | repo checkout root (default: current directory)|
| `MAMMOCVD_FINETUNE_COHORT_CSV`   | path to your cohort CSV (schema above)         |
| `MAMMO_CLIP_REPO` / `MAMMO_CLIP_CKPT` | local [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP) clone + checkpoint (`shawn24/Mammo-CLIP` on HF) |
| `MAMMO_FM_CKPT`                  | Mammo-FM checkpoint (`batmanlab/Mammo-FM` on HF) |

Pipeline, given a cohort CSV: `extract_embeddings_scankeyed.py` →
`train_finetune_simplefusion_scankeyed.py` (image arms) /
`train_finetune_tabular_only.py` / `train_age_only.py` (baselines) →
`compare_arms.py` (bootstrap CIs + DeLong tests across arms).

## Citation

```
@article{feldman2026mammocvd,
  title   = {Mammography Foundation Models for Opportunistic Prediction of Major Adverse Cardiovascular Events},
  author  = {Feldman, Paula and Nizam, Nusrat Binta and Kwak, Sunwoo and Karaman, Batuhan and Dodelzon, Katerina and Sabuncu, Mert},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv:TBD}
}
```
