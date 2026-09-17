"""
Frozen feature extraction using Mammo-FM's pretrained EfficientNet-B5
image encoder (batmanLab/Mammo-FM on HF, Batmanlab-trained variant --
Mammo-FM_BatmanlabTrained_CLIP.tar) -- Ghosh et al.'s newer, broader
successor to Mammo-CLIP (arXiv:2512.00198, "Breast-specific foundational
model for Integrated Mammographic Diagnosis, Prognosis, and Reporting"),
trained on UPMC + EMBED + Boston Medical Center + Mayo Clinic mammogram-
report pairs (multi-institution, larger and more diverse than Mammo-CLIP's
single-site UPMC pretraining).

Same core method as Mammo-CLIP though, not a different objective: same
backbone (tf_efficientnet_b5_ns-detect) and same CLIP-style image-report
contrastive alignment, just bigger/more diverse training data and a newer
text encoder (ModernBERT) -- the text side is irrelevant here since only
the frozen image encoder is ever used downstream.

Checkpoint structure and preprocessing constants (mean=0.3089279,
std=0.25053555408335154, image_size 1520x912, weight-key prefix
"image_encoder.") are byte-identical to mammo_clip_features.py's -- this
module reuses that module's EfficientNet loader, preprocessing function,
and embedding extractor unchanged, only pointing at a different checkpoint
file and building a fresh backbone instance.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from src.mammo_cvd.mammo_clip_features import (
    EfficientNet, OUT_DIM, load_mammo_clip_view, extract_embedding,  # noqa: F401 (re-exported)
)

CKPT_PATH = Path(os.environ.get(
    "MAMMO_FM_CKPT",
    # download from https://huggingface.co/batmanlab/Mammo-FM (Mammo-FM_BatmanlabTrained_CLIP.tar)
    "/path/to/Mammo-FM_BatmanlabTrained_CLIP.tar",
))


def build_mammo_fm_encoder(device) -> torch.nn.Module:
    model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1, in_channels=3)
    model.out_dim = OUT_DIM

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    full_state = ckpt["model"]
    prefix = "image_encoder."
    state = {k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not missing, f"missing keys: {missing}"

    model.eval()
    return model.to(device)


# alias matching mammo_clip_features.py's naming, for load_mammo_fm_view callers
load_mammo_fm_view = load_mammo_clip_view


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = build_mammo_fm_encoder(device)
    print(f"Mammo-FM encoder loaded, out_dim={model.out_dim}")

    import numpy as np
    import pandas as pd
    df = pd.read_csv(os.environ.get("MAMMOCVD_FINETUNE_COHORT_CSV", "outputs/mammo_cvd/finetune_cohort.csv"),
                      dtype={"patient_id": str}, nrows=3)
    for _, row in df.iterrows():
        p = row.get("path_L_MLO")
        if isinstance(p, str) and p:
            view = load_mammo_fm_view(p, laterality="L")
            embed = extract_embedding(model, view, device)
            print(f"patient_id={row['patient_id']} embed shape={embed.shape} "
                  f"norm={np.linalg.norm(embed):.3f} mean={embed.mean():.4f} std={embed.std():.4f}")
