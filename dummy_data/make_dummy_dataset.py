"""
Builds a small, fully-synthetic smoke-test dataset for the mammocvd pipeline.

This is NOT real patient data. It combines:
  - A handful of real, public-domain mammogram images from the DMID dataset
    (Oza et al., 2023, CC-BY-4.0, https://doi.org/10.6084/m9.figshare.24522883.v2,
    mirrored on Hugging Face at MyTwinLab/DMID_Breast_Cancer_Mammography_Dataset),
    re-saved as minimal single-frame DICOM files so they load through the
    project's real `dataset.load_mammo_view()` path unmodified.
  - Entirely invented patient identifiers, tabular risk factors, and MACE
    labels, sampled from distributions matching the *summary statistics*
    reported in Table 2 of the preprint (mean/SD/prevalence only -- no real
    patient's values are used or reconstructable from this).

Purpose: let someone clone this repo and run the cohort -> embedding ->
training -> evaluation pipeline end-to-end without access to the real,
non-public institutional cohort. Numbers produced from this dataset are
meaningless and must never be reported as if they were the paper's results.

Usage:
    python dummy_data/make_dummy_dataset.py
Output:
    dummy_data/images/*.dcm            (synthetic single-frame DICOMs)
    dummy_data/dummy_cohort.csv         (100 synthetic patients)
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

SEED = 0
N_BASE_IMAGES = 20          # real public mammograms to download and convert
N_PATIENTS = 100            # synthetic patients in the dummy cohort
MAX_DIM = 512               # longest side after downsizing, keeps file sizes small
MACE_PREVALENCE = 0.0222    # matches the preprint's reported 5-year MACE prevalence

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images"

HF_BASE = (
    "https://huggingface.co/datasets/MyTwinLab/DMID_Breast_Cancer_Mammography_Dataset"
    "/resolve/main/TIFF%20Images/TIFF%20Images/IMG{idx:03d}.tif"
)

# Continuous/binary feature stats (mean, sd, fraction available), matched to
# Table 2 of the preprint where reported. `bmi` and the medication flags
# aren't in the preprint's table but are needed by
# src/mammo_cvd/train_finetune_tabular_only.py's feature sets, so their
# distributions are just plausible invented placeholders, not sourced.
TABULAR_STATS = {
    "age_at_baseline":   dict(mean=55.7, sd=11.9, avail=1.000, lo=22, hi=95),
    "total_cholesterol": dict(mean=196.5, sd=34.1, avail=0.657, lo=80, hi=400),
    "ldl":               dict(mean=112.5, sd=28.6, avail=0.619, lo=30, hi=300),
    "hdl":               dict(mean=64.5, sd=17.2, avail=0.625, lo=20, hi=150),
    "triglycerides":     dict(mean=97.2, sd=56.8, avail=0.621, lo=30, hi=600),
    "creatinine":        dict(mean=0.8, sd=0.4, avail=0.758, lo=0.3, hi=3.0),
    "hba1c":             dict(mean=5.8, sd=0.8, avail=0.355, lo=4.5, hi=12.0),
    "bmi":               dict(mean=26.6, sd=6.2, avail=0.300, lo=15, hi=55),  # invented, not in Table 2
}
BINARY_STATS = {
    "diabetes":              dict(p=0.118, avail=1.000),
    "hypertension_dx":       dict(p=0.226, avail=1.000),
    "smoker":                dict(p=0.070, avail=0.850),
    "alcohol_use":           dict(p=0.476, avail=0.761),
    "family_hx_cvd":         dict(p=0.589, avail=0.589),
    "antihypertensive_med":  dict(p=0.200, avail=1.000),  # invented, not in Table 2
    "statin_med":            dict(p=0.201, avail=1.000),  # invented, not in Table 2
}
STUDY_DATE_START = pd.Timestamp("2014-01-01")
STUDY_DATE_END = pd.Timestamp("2018-12-31")


def download_base_images(rng: random.Random) -> list[np.ndarray]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    indices = rng.sample(range(1, 514), N_BASE_IMAGES)
    arrays = []
    for i, idx in enumerate(indices):
        url = HF_BASE.format(idx=idx)
        print(f"[{i+1}/{N_BASE_IMAGES}] downloading {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("L")
        w, h = img.size
        scale = MAX_DIM / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        arrays.append(np.array(img, dtype=np.uint16) * 257)  # 8-bit -> 16-bit range
    return arrays


def save_as_dicom(arr: np.ndarray, path: Path, laterality: str) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"  # Digital Mammography X-Ray Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = "DUMMY"
    ds.PatientName = "Dummy^Patient"
    ds.Modality = "MG"
    ds.ImageLaterality = laterality
    ds.ViewPosition = "MLO"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    try:
        ds.save_as(str(path), enforce_file_format=True)  # pydicom >= 3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)  # pydicom < 3


def build_base_pairs(rng: random.Random) -> list[tuple[Path, Path]]:
    arrays = download_base_images(rng)
    pairs = []
    for i, arr in enumerate(arrays):
        l_path = IMG_DIR / f"base{i:03d}_L.dcm"
        r_path = IMG_DIR / f"base{i:03d}_R.dcm"
        save_as_dicom(arr, l_path, "L")
        save_as_dicom(np.fliplr(arr).copy(), r_path, "R")  # mirrored, synthetic "other breast"
        pairs.append((l_path, r_path))
    return pairs


def sample_tabular_row(rng: np.random.Generator) -> dict:
    row = {}
    for name, s in TABULAR_STATS.items():
        val = np.clip(rng.normal(s["mean"], s["sd"]), s["lo"], s["hi"])
        row[name] = round(float(val), 2) if rng.random() < s["avail"] else np.nan
    for name, s in BINARY_STATS.items():
        row[name] = int(rng.random() < s["p"]) if rng.random() < s["avail"] else np.nan
    return row


def sample_study_date(rng: np.random.Generator) -> str:
    span_days = (STUDY_DATE_END - STUDY_DATE_START).days
    return (STUDY_DATE_START + pd.Timedelta(days=int(rng.integers(0, span_days)))).strftime("%Y-%m-%d")


def assign_splits(n: int, rng: np.random.Generator) -> list[str]:
    # matches the paper's approximate 61.5/15.4/23.1 train/val/test ratio
    choices = rng.choice(["train", "val", "test"], size=n, p=[0.615, 0.154, 0.231])
    return list(choices)


def main():
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    base_pairs = build_base_pairs(rng)

    records = []
    for i in range(N_PATIENTS):
        l_path, r_path = base_pairs[i % len(base_pairs)]
        rec = {
            "empi": f"DUMMY{i:04d}",
            "study_date": sample_study_date(np_rng),
            "path_L_MLO": str(l_path.relative_to(HERE.parent)),
            "path_R_MLO": str(r_path.relative_to(HERE.parent)),
            "label_5yr": int(np_rng.random() < MACE_PREVALENCE),
        }
        rec.update(sample_tabular_row(np_rng))
        records.append(rec)

    df = pd.DataFrame(records)
    df["split"] = assign_splits(len(df), np_rng)

    cohort_cols = ["empi", "study_date", "label_5yr", "path_L_MLO", "path_R_MLO", "age_at_baseline"]
    tabular_cols = ["empi", "study_date"] + list(TABULAR_STATS) + list(BINARY_STATS)
    splits_cols = ["empi", "split"]

    df[cohort_cols].to_csv(HERE / "dummy_finetune_cohort.csv", index=False)
    df[tabular_cols].to_csv(HERE / "dummy_tabular_features.csv", index=False)
    df[splits_cols].to_csv(HERE / "dummy_splits.csv", index=False)

    print(f"wrote dummy_finetune_cohort.csv / dummy_tabular_features.csv / dummy_splits.csv "
          f"({len(df)} synthetic patients, {df['label_5yr'].sum()} positive, "
          f"{len(base_pairs)} unique base images)")


if __name__ == "__main__":
    main()
