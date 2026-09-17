"""DICOM loading / preprocessing for mammogram views, plus a placeholder
Dataset (see ExamDataset below) -- fill in your own cohort_csv schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

STANDARD_VIEWS = ["L_CC", "L_MLO", "R_CC", "R_MLO"]

IMG_SIZE = 512


def _count_up_continuing_ones(b_arr: np.ndarray) -> np.ndarray:
    """For a boolean array, returns for each position the length of the
    longest run of True values it belongs to (or -1 if False). Used to find
    the widest contiguous non-background stripe -- the breast tissue,
    since scanner black-background is by far the most common single value
    and thus the longest constant run, while burned-in text/markers are
    small isolated blobs that don't form a wide contiguous run."""
    n = len(b_arr)
    left = np.arange(n)
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)
    rev = b_arr[::-1]
    right = np.arange(n)
    right[rev > 0] = 0
    right = np.maximum.accumulate(right)
    right = n - 1 - right[::-1]
    return right - left - 1


def _breast_region_indices(arr_255: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """arr_255 must already be normalized to 0-255 scale (bg_thresh=40 is
    calibrated to that range). Returns (row_idx, col_idx) into arr_255
    covering the widest contiguous non-background stripe in each axis --
    the breast tissue silhouette."""
    img = np.where(arr_255 <= 40.0, 0, arr_255)
    height, _ = img.shape
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    col_is_tissue = img[y_b:y_a].std(axis=0) != 0
    run_len = _count_up_continuing_ones(col_is_tissue)
    col_idx = np.where(run_len == run_len.max())[0]

    img_cols = arr_255[:, col_idx]
    _, width = img_cols.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    masked_cols = np.where(img_cols <= 40.0, 0, img_cols)
    row_is_tissue = masked_cols[:, x_b:x_a].std(axis=1) != 0
    run_len = _count_up_continuing_ones(row_is_tissue)
    row_idx = np.where(run_len == run_len.max())[0]
    return row_idx, col_idx


def crop_breast_region(arr: np.ndarray) -> np.ndarray:
    """Auto-crops to the breast tissue silhouette, discarding black
    background -- and with it, any burned-in laterality/view marker text
    (e.g. "L MLO"), which sits in the background corners, not on the
    breast itself. Ports Mammo-CLIP's own preprocessing algorithm
    (external/Mammo-CLIP/src/preprocessing/preprocess_image_to_png_kaggle.py,
    np_ExtractBreast) rather than inventing a different heuristic, so our
    reproduction of their pipeline actually matches it -- their bg_thresh=40
    is calibrated to a 0-255 scale, so `arr` is normalized to that range
    internally to compute the crop indices, then those indices are applied
    to the ORIGINAL (real-intensity) array so downstream percentile-clip
    normalization still sees true DICOM intensities, not a lossy 0-255
    intermediate.

    Without this, a model can fixate on that corner marker text as a
    shortcut instead of learning from breast tissue."""
    arr_min, arr_max = arr.min(), arr.max()
    arr_255 = (arr - arr_min) / max(arr_max - arr_min, 1e-6) * 255.0
    row_idx, col_idx = _breast_region_indices(arr_255)
    cropped = arr[row_idx][:, col_idx]
    # degenerate fallback (near-uniform image, e.g. a corrupted/blank DICOM)
    if cropped.size == 0:
        return arr
    return cropped


def _letterbox_pad(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Pads arr (H,W) with zeros, centered, so its aspect ratio matches
    target_w/target_h -- WITHOUT stretching/distorting content. Needed
    because crop_breast_region's output aspect ratio varies per patient
    (breast shape/positioning differs), unlike the uncropped DICOM's
    fairly consistent ratio -- a plain resize straight to a fixed target
    size after cropping would apply a different, unpredictable stretch
    factor to every patient."""
    h, w = arr.shape
    target_ratio = target_w / target_h
    cur_ratio = w / h
    if cur_ratio < target_ratio:
        # too narrow for the target -- pad width
        new_w = int(round(h * target_ratio))
        pad_total = max(new_w - w, 0)
        pad_l, pad_r = pad_total // 2, pad_total - pad_total // 2
        return np.pad(arr, ((0, 0), (pad_l, pad_r)), mode="constant")
    elif cur_ratio > target_ratio:
        # too wide for the target -- pad height
        new_h = int(round(w / target_ratio))
        pad_total = max(new_h - h, 0)
        pad_t, pad_b = pad_total // 2, pad_total - pad_total // 2
        return np.pad(arr, ((pad_t, pad_b), (0, 0)), mode="constant")
    return arr


def load_mammo_view(path: str, size: int | tuple[int, int] = IMG_SIZE,
                     laterality: str | None = None, crop_breast: bool = True,
                     letterbox: bool = True) -> np.ndarray:
    """laterality: "L" or "R" (or None to skip left-alignment). When given,
    R-laterality images are flipped horizontally so breast tissue is always
    positioned the same way across L and R views ("left-align for consistent
    positioning", matching Mirai's preprocessing) -- otherwise the same
    anatomical structures appear mirrored depending on which breast was
    imaged, which is a spurious cue the encoder would otherwise have to
    learn to ignore.

    crop_breast=True (default): auto-crops to the breast silhouette BEFORE
    resizing, discarding black background and any burned-in laterality/view
    marker text with it -- see crop_breast_region. The crop's aspect ratio
    varies per patient (unlike the raw DICOM's fairly consistent ratio), so
    a zero-padded letterbox step restores a fixed aspect ratio before the
    final resize -- without it, every patient gets a different,
    unpredictable stretch distortion.

    letterbox=True (default): pads to the target aspect ratio before
    resizing (see _letterbox_pad) -- correct for every model WE train
    end-to-end, since we control that distribution ourselves. Set False
    only for load_mammo_clip_view's use: Mammo-CLIP's own published
    preprocessing crops then stretches directly to their fixed size with
    no letterbox step (checked their code) -- their frozen pretrained
    backbone's weights were calibrated to that stretched distribution, so
    "fixing" the distortion for that specific arm would itself be a
    train/inference preprocessing mismatch, not an improvement."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    if crop_breast:
        arr = crop_breast_region(arr)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)

    target = (size, size) if isinstance(size, int) else size
    if crop_breast and letterbox:
        arr = _letterbox_pad(arr, target_w=target[0], target_h=target[1])

    from PIL import Image

    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize(target, Image.BILINEAR)
    out = (np.asarray(img).astype(np.float32)) / 255.0
    if laterality == "R":
        out = np.ascontiguousarray(out[:, ::-1])
    return out


# MLO is preferred over CC for CVD/BAC work -- the BAC paper (Dapamede et al.,
# EHJ 2026) uses MLO specifically because that plane is more perpendicular to
# breast arteries, so calcification is more visible there than on CC.
MLO_VIEWS = ["L_MLO", "R_MLO"]


VIEW_TO_IDX = {v: i for i, v in enumerate(STANDARD_VIEWS)}


class ExamDataset(Dataset):
    """PLACEHOLDER -- plug in your own cohort loading here. Expected to
    return, per exam: (views (V,1,H,W) via load_mammo_view() above, mask
    (V,) marking which views are present, view_idx (V,) into
    VIEW_TO_IDX/STANDARD_VIEWS, label (scalar 5yr MACE outcome), patient id).

    `cohort_csv` should have one row per patient/instance with columns
    `path_L_MLO` / `path_R_MLO` (and `path_L_CC` / `path_R_CC` if using all
    four standard views) pointing at real DICOM files, plus a label column.
    `splits_csv` should have `patient_id, split` (train/val/test)."""

    def __init__(self, split: str, cohort_csv: str, size: int = IMG_SIZE,
                 splits_csv: str | None = None, view_mode: str = "single_mlo"):
        assert view_mode in ("all", "single_mlo")
        self.view_mode = view_mode
        self.views_to_use = STANDARD_VIEWS if view_mode == "all" else MLO_VIEWS
        self.size = size

        df = pd.read_csv(cohort_csv, dtype={"patient_id": str})
        if splits_csv is not None:
            splits = pd.read_csv(splits_csv, dtype={"patient_id": str})
            df = df.merge(splits[["patient_id", "split"]], on="patient_id", how="inner")
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        imgs, mask, view_idx = [], [], []
        for v in self.views_to_use:
            p = row.get(f"path_{v}")
            if isinstance(p, str) and p:
                try:
                    imgs.append(load_mammo_view(p, self.size, laterality=v[0]))
                    mask.append(1.0)
                    view_idx.append(VIEW_TO_IDX[v])
                    continue
                except Exception:
                    pass
            imgs.append(np.zeros((self.size, self.size), dtype=np.float32))
            mask.append(0.0)
            view_idx.append(VIEW_TO_IDX[v])

        views = torch.from_numpy(np.stack(imgs)).unsqueeze(1)  # (V, 1, H, W)
        mask = torch.tensor(mask, dtype=torch.float32)  # (V,)
        view_idx = torch.tensor(view_idx, dtype=torch.long)
        label = torch.tensor(float(row["label_5yr"]))
        return views, mask, view_idx, label, row["patient_id"]
