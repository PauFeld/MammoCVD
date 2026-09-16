"""
Frozen feature extraction using Mammo-CLIP's pretrained EfficientNet-B5
image encoder (batmanlab, github.com/batmanlab/Mammo-CLIP, weights on HF
at shawn24/Mammo-CLIP) -- trained on UPMC mammogram-report pairs, ~5-8x
more patients than our own DINO pretraining pool.

Reuses their own vendored EfficientNet implementation and checkpoint's
embedded config directly (not a hand-reconstructed guess at their
preprocessing) -- confirmed by inspecting the actual downloaded checkpoint:
  config.base: mean=0.3089279, std=0.25053555408335154,
               image_size_h=1520, image_size_w=912 (portrait)
  config.model.image_encoder: name='tf_efficientnet_b5_ns-detect'
    -> EfficientNet.from_pretrained("efficientnet-b5", num_classes=1),
       in_channels=3 (RGB), out_dim=2048
  weight keys in checkpoint["model"] are prefixed "image_encoder."

Their own data pipeline loads pre-processed PNGs directly (PIL RGB) --
we don't have their exact PNG windowing recipe, so this instead reuses
our own DICOM decode (dataset.load_mammo_view: percentile-clip 0.5-99.5%,
normalize, uint8) as the source array, replicates to 3 channels, then
applies THEIR exact resize + min-max + z-score normalization. This is
the closest faithful adaptation achievable without their raw
preprocessing scripts -- worth flagging as an approximation, not a
guaranteed exact match to their own training-time pipeline.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

from src.mammo_cvd.dataset import load_mammo_view

MAMMO_CLIP_REPO = Path(os.environ.get(
    "MAMMO_CLIP_REPO",
    "/path/to/Mammo-CLIP/src/codebase",  # clone from https://github.com/batmanlab/Mammo-CLIP
))
CKPT_PATH = Path(os.environ.get(
    "MAMMO_CLIP_CKPT",
    "/path/to/b5-model-best-epoch-7.tar",  # download from https://huggingface.co/shawn24/Mammo-CLIP
))


def _load_efficientnet_class():
    """Load efficientnet_custom.py directly (bypassing breastclip/__init__.py,
    which eagerly imports their full training framework -- tensorboard,
    albumentations, transformers, etc -- none of which this standalone
    feature extractor needs). efficientnet_custom.py uses a relative import
    of its sibling efficient_net_custom_utils.py, which needs real package
    context to resolve -- registered here as a minimal fake package rooted
    at that directory rather than importing the real (heavy) breastclip
    package."""
    pkg_dir = MAMMO_CLIP_REPO / "breastclip" / "model" / "modules"
    pkg_name = "_mammo_clip_modules_standalone"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_dir)]
        sys.modules[pkg_name] = pkg

    mod_name = f"{pkg_name}.efficientnet_custom"
    spec = importlib.util.spec_from_file_location(mod_name, pkg_dir / "efficientnet_custom.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module.EfficientNet


EfficientNet = _load_efficientnet_class()

MEAN = 0.3089279
STD = 0.25053555408335154
IMG_SIZE_H = 1520
IMG_SIZE_W = 912
OUT_DIM = 2048


def build_mammo_clip_encoder(device) -> torch.nn.Module:
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


def load_mammo_clip_view(path: str, laterality: str | None = None,
                          save_png_path: str | None = None) -> np.ndarray:
    """Returns (3, IMG_SIZE_H, IMG_SIZE_W) float32, normalized per Mammo-CLIP's
    own convention (global min-max to [0,1], then z-score with their mean/std).

    letterbox=False: their own published preprocessing crops the breast
    region then stretches directly to the fixed target size, no aspect-
    ratio-preserving pad -- matching that exactly (not "fixing" it) since
    their pretrained backbone's weights were calibrated to that stretched
    distribution (checked 2026-08-21, see load_mammo_view's docstring).

    save_png_path (2026-09-04, per user -- "save the pngs of the properly
    extracted stretched [images], otherwise you will generate gradcam on
    the wrong pngs"): if given, writes the pre-normalization grayscale
    (post-crop, post-stretch, pre-RGB-stack, pre-z-score) as an 8-bit PNG
    -- this is the actual visual input the frozen backbone sees, distinct
    from build_png_cache_all_bathuan.py's cache (which uses letterbox=True,
    the wrong preprocessing for this arm -- see 2026-09-04 finding). Skips
    the write if the file already exists (cheap, no re-encode)."""
    gray = load_mammo_view(path, size=(IMG_SIZE_W, IMG_SIZE_H), laterality=laterality,
                            letterbox=False)  # (H,W) in [0,1]
    if save_png_path is not None and not os.path.exists(save_png_path):
        from PIL import Image
        Image.fromarray((gray * 255).astype(np.uint8)).save(save_png_path)
    rgb = np.stack([gray, gray, gray], axis=0).astype(np.float32)  # (3,H,W)
    rgb -= rgb.min()
    rgb /= max(rgb.max(), 1e-6)
    rgb = (rgb - MEAN) / STD
    return rgb


@torch.no_grad()
def extract_embedding(model: torch.nn.Module, view: np.ndarray, device) -> np.ndarray:
    """view: (3,H,W) -> (2048,) embedding."""
    x = torch.from_numpy(view).unsqueeze(0).to(device)  # (1,3,H,W)
    feat = model(x)  # (1, 2048)
    return feat.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = build_mammo_clip_encoder(device)
    print(f"Mammo-CLIP encoder loaded, out_dim={model.out_dim}")

    import pandas as pd
    df = pd.read_csv(os.environ.get("MAMMOCVD_FINETUNE_COHORT_CSV", "outputs/mammo_cvd/finetune_cohort.csv"),
                      dtype={"empi": str}, nrows=3)
    for _, row in df.iterrows():
        p = row.get("path_L_MLO")
        if isinstance(p, str) and p:
            view = load_mammo_clip_view(p, laterality="L")
            embed = extract_embedding(model, view, device)
            print(f"empi={row['empi']} embed shape={embed.shape} "
                  f"norm={np.linalg.norm(embed):.3f} mean={embed.mean():.4f} std={embed.std():.4f}")
