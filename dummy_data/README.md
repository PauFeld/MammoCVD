# Dummy smoke-test dataset

**This is not real patient data.** It exists only so someone cloning this repo
can run the cohort → embedding → training → evaluation pipeline end-to-end
without access to the real, non-public institutional cohort described in the
paper. Any numbers this dataset produces are meaningless (100 patients, 20
underlying images reused across patients, invented labels) and must never be
reported as results.

## What's in here

- `images/*.dcm` — 40 single-frame synthetic DICOM files (20 base images ×
  L/R), generated from 20 real, public-domain mammograms.
- `dummy_finetune_cohort.csv` — 100 synthetic patients: `empi`, `study_date`,
  `label_5yr` (invented, sampled at the paper's reported 2.22% prevalence),
  `path_L_MLO` / `path_R_MLO` (pointing at two of the DICOM files above), and
  `age_at_baseline`.
- `dummy_tabular_features.csv` — the same patients' invented tabular risk
  factors (lipid panel, BMI, diabetes/hypertension flags, smoking, alcohol,
  family history, medication flags), sampled from the mean/SD/prevalence
  reported in the paper's Table 2 where available (not from any real
  patient); `bmi` and the two medication flags aren't in Table 2, so those
  are just plausible invented placeholders needed to satisfy
  `train_finetune_tabular_only.py`'s feature sets.
- `dummy_splits.csv` — `empi`, `split` (train/val/test, ~61.5/15.4/23.1,
  matching the paper's ratio).
- `make_dummy_dataset.py` — regenerates all of the above from scratch.

These three CSVs match the exact schema `src/mammo_cvd/train_age_only.py`
and `src/mammo_cvd/train_finetune_tabular_only.py` expect (verified by
actually running both against this dummy data — see below).

## Image source and license

The underlying mammogram images come from the **DMID (Digital Mammography
Dataset for Breast Cancer Diagnosis Research)**, Oza, Parita; Oza, Rajiv; Oza,
Urvi; Sharma, Paawan; Patel, Samir; Kumar, Pankaj; et al. (2023), figshare,
https://doi.org/10.6084/m9.figshare.24522883.v2, mirrored on Hugging Face at
`MyTwinLab/DMID_Breast_Cancer_Mammography_Dataset`. Licensed **CC BY 4.0**.
These images are unrelated to this project's cohort or institution — they are
used only to exercise the DICOM-loading and preprocessing code paths with
real mammogram-shaped pixel data.

## Regenerating

```bash
pip install pandas numpy pydicom pillow requests
python dummy_data/make_dummy_dataset.py
```

## Using it to smoke-test the pipeline

The tabular and age-only baselines need no foundation-model weights or image
loading at all, and are the fastest way to confirm the pipeline runs
end-to-end. From the repo root:

```bash
export MAMMOCVD_ROOT=$(pwd)          # or any writable directory for outputs
mkdir -p "$MAMMOCVD_ROOT/outputs/mammo_cvd"

python -m src.mammo_cvd.train_age_only \
  --cohort_csv dummy_data/dummy_finetune_cohort.csv \
  --splits_csv dummy_data/dummy_splits.csv \
  --run_tag dummy_smoketest

python -m src.mammo_cvd.train_finetune_tabular_only \
  --cohort_csv dummy_data/dummy_finetune_cohort.csv \
  --splits_csv dummy_data/dummy_splits.csv \
  --tabular_csv dummy_data/dummy_tabular_features.csv \
  --epochs 2 --run_tag dummy_smoketest
```

Both were verified to run against this exact dummy data. With only 100
patients (3 positive) the reported AUROC is often `nan` (a split can easily
land zero positives) — that's expected for a toy dataset, not a bug.

To exercise the image arms (Mammo-CLIP / Mammo-FM), you additionally need
their real checkpoints (see the top-level README for links) and to run
`src/mammo_cvd/extract_embeddings_scankeyed.py` against
`dummy_finetune_cohort.csv` first to populate the `.npy` embedding cache that
`train_finetune_simplefusion_scankeyed.py` reads.
