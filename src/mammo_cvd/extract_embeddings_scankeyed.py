"""
Precomputes frozen Mammo-CLIP/Mammo-FM embeddings for every L_MLO/R_MLO
view in a cohort, cached to disk keyed by scan identity
({empi}_{study_date}_{view}.npy) rather than just {empi}_{view}.npy --
a patient can have more than one scan (e.g. across different cohort
labeling strategies), so the cache key includes study_date to avoid
silently mixing embeddings from the wrong scan.

Loading (DICOM decode + preprocessing) is parallelized across a CPU
worker pool while the GPU forward pass runs in batches, since decode is
the bottleneck relative to the encoder's forward pass.

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
# Cache the correctly-preprocessed (crop+stretch, letterbox=False)
# grayscale PNG as a byproduct of extraction, keyed the same scan-specific
# way as the embeddings themselves. Mammo-CLIP and Mammo-FM share identical
# preprocessing (load_mammo_fm_view is an alias of load_mammo_clip_view),
# so one PNG per (empi, study_date, view) is valid input for both backbones
# -- whichever extraction pass runs first writes it, the other skips.
PNG_DIR = OUT_DIR / "stretched_png_cache_scankeyed"


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
