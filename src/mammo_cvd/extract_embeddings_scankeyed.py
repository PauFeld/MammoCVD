"""
Generalized Mammo-CLIP/Mammo-FM embedding extraction, keyed by SCAN
IDENTITY ({empi}_{study_date}_{view}.npy) instead of the existing
{empi}_{view}.npy convention used by extract_mammo_clip_embeddings.py /
extract_mammo_fm_embeddings.py -- 2026-09-03, required for the new
expanded/symmetric-landmark cohort experiments.

Why this exists: experiments 3/4/5a/5b (build_unified_finetune_cohort.py)
can select DIFFERENT physical baseline scans for the SAME patient across
cohort variants (confirmed: 11,879 / 18,017 patients have a different
baseline study_date between the plain expanded cohort and its
symmetric-landmark counterpart). The old empi-only cache key assumes one
canonical scan per patient forever -- reusing it here would silently
overwrite one cohort's embedding with a different scan's features (or
silently reuse the wrong scan's embedding), with no error, for two-thirds
of the expanded cohort. Keying by (empi, study_date, view) makes every
scan-specific vector unique and content-addressed, so extracting for one
cohort variant can never corrupt another's, and running this again for an
overlapping cohort (same patient, same scan) correctly skips re-work.

2026-09-03 perf fix: original version loaded+forward-passed one image at
a time (batch size 1) -- at the observed rate this cohort's heaviest pass
alone was projecting to ~7-8 more hours. DICOM decode + preprocessing is
CPU-bound (pydicom read, percentile-clip, resize) while the actual
EfficientNet-B5 forward pass is fast on GPU, so the fix is to parallelize
loading across a worker pool (CPU-bound, overlaps I/O across workers) and
batch the GPU forward pass (16 workers load concurrently, N loaded views
get forward-passed together instead of one by one).

MLO-only (L_MLO/R_MLO), matching this experiment set's scope.

Output: outputs/mammo_cvd/{mammo_clip,mammo_fm}_embeddings_scankeyed/{empi}_{study_date}_{view}.npy

Run:
  python -m src.mammo_cvd.extract_embeddings_scankeyed --backbone mammoclip --cohort_csv <path>
  python -m src.mammo_cvd.extract_embeddings_scankeyed --backbone mammofm --cohort_csv <path>
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(os.environ.get("MAMMOCVD_ROOT", "."))  # set to your repo checkout root
OUT_DIR = PROJECT_ROOT / "outputs" / "mammo_cvd"

MLO_VIEWS = ["L_MLO", "R_MLO"]
BATCH_SIZE = 32
LOAD_TIMEOUT_S = 60
# 2026-09-04 per user: cache the correctly-preprocessed (crop+stretch,
# letterbox=False) grayscale PNG as a byproduct of extraction, keyed the
# same scan-specific way as the embeddings themselves -- distinct from
# build_png_cache_all_bathuan.py's cache, which uses the wrong (letterboxed)
# preprocessing for this arm. Mammo-CLIP and Mammo-FM share identical
# preprocessing (load_mammo_fm_view is an alias of load_mammo_clip_view),
# so one PNG per (empi, study_date, view) is valid input for both backbones
# -- whichever extraction pass runs first writes it, the other skips.
PNG_DIR = OUT_DIR / "stretched_png_cache_scankeyed"

_BACKBONE = None


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _load_one(args):
    """Runs in a worker process: decode + preprocess one view. Returns
    (out_key, view_array_or_None, error_str_or_None)."""
    out_key, p, laterality, backbone = args
    if backbone == "mammoclip":
        from src.mammo_cvd.mammo_clip_features import load_mammo_clip_view as load_view
    else:
        from src.mammo_cvd.mammo_fm_features import load_mammo_fm_view as load_view

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(LOAD_TIMEOUT_S)
    try:
        png_path = str(PNG_DIR / f"{out_key}.png")
        view = load_view(p, laterality=laterality, save_png_path=png_path)
        return out_key, view, None
    except _Timeout:
        return out_key, None, f"TIMEOUT (>{LOAD_TIMEOUT_S}s)"
    except Exception as e:
        return out_key, None, str(e)
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["mammoclip", "mammofm"], required=True)
    ap.add_argument("--cohort_csv", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=16)
    args = ap.parse_args()

    if args.backbone == "mammoclip":
        from src.mammo_cvd.mammo_clip_features import build_mammo_clip_encoder
        build_encoder = build_mammo_clip_encoder
        embed_dir = OUT_DIR / "mammo_clip_embeddings_scankeyed"
    else:
        from src.mammo_cvd.mammo_fm_features import build_mammo_fm_encoder
        build_encoder = build_mammo_fm_encoder
        embed_dir = OUT_DIR / "mammo_fm_embeddings_scankeyed"

    embed_dir.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, backbone: {args.backbone}")
    model = build_encoder(device)

    df = pd.read_csv(args.cohort_csv, dtype={"empi": str})
    df["study_date"] = pd.to_datetime(df["study_date"]).dt.strftime("%Y%m%d")
    print(f"cohort: {len(df)} patients")

    jobs = []
    for _, row in df.iterrows():
        for v in MLO_VIEWS:
            p = row.get(f"path_{v}")
            if not (isinstance(p, str) and p):
                continue
            out_key = f"{row['empi']}_{row['study_date']}_{v}"
            out_path = embed_dir / f"{out_key}.npy"
            if out_path.exists():
                continue
            jobs.append((out_key, p, v[0], args.backbone))

    print(f"jobs needing extraction (skip-if-exists already applied): {len(jobs)}", flush=True)

    n_written, n_failed = 0, 0
    with mp.Pool(args.num_workers) as pool:
        batch_keys, batch_views = [], []

        def flush_batch():
            nonlocal n_written
            if not batch_views:
                return
            with torch.no_grad():
                x = torch.from_numpy(np.stack(batch_views)).to(device)
                feats = model(x).cpu().numpy()
            for key, feat in zip(batch_keys, feats):
                np.save(embed_dir / f"{key}.npy", feat.astype(np.float32))
            n_written += len(batch_keys)
            batch_keys.clear()
            batch_views.clear()

        for i, (out_key, view, err) in enumerate(pool.imap_unordered(_load_one, jobs, chunksize=4)):
            if err is not None:
                n_failed += 1
                print(f"  FAILED {out_key}: {err}", flush=True)
                continue
            batch_keys.append(out_key)
            batch_views.append(view)
            if len(batch_views) >= BATCH_SIZE:
                flush_batch()
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(jobs)} jobs, written={n_written} failed={n_failed}", flush=True)

        flush_batch()

    print(f"\nDone. written={n_written} failed={n_failed}")
    print(f"Embed dir: {embed_dir}")


if __name__ == "__main__":
    main()
