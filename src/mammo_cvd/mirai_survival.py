"""
Delta-t-conditioned, multimodal survival model.

Two extensions over `mirai_encoder.MiraiExamEncoder`:

1. **Time as input, not fixed output buckets**: instead of Mirai/LoGo-MR's
   (n_years+1)-dim hazard vector predicted all at once, this model takes a
   scalar time horizon `delta_t` (years) as an explicit input alongside
   the image, and predicts a single P(event occurs by delta_t) for that
   specific horizon -- closer to the Cox-Time/neural-Cox family (Kvamme et
   al.) than to Mirai's discrete-year-bucket approach. `delta_t` is
   embedded with a sinusoidal encoding (same construction as
   odelia_3d/logo_mr.py's `sinusoidal_position_encoding`, repurposed here
   for a continuous scalar rather than a discrete slice/visit index) then
   concatenated with the pooled exam embedding.

2. **Multimodal tabular fusion**: BMI, HDL, LDL, total cholesterol,
   triglycerides, smoking, and alcohol-use, fused with the image
   embedding before the risk head. Coverage for these fields is partial
   (see build_survival_cohort.py), so each feature gets a learned
   "missing" embedding rather than mean-imputation, matching this
   project's own stated design pattern for multimodal risk models (see
   README's "Multimodal extension" section for the MRI risk-prediction
   work) -- the model must still work on imaging alone for patients
   missing all tabular data.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.mammo_cvd.mirai_encoder import ResNetViewEncoder, ViewAttentionPooling

TABULAR_FEATURES = ["bmi", "hdl", "ldl", "total_cholesterol", "triglycerides", "smoker", "alcohol_use", "age_at_baseline"]


def sinusoidal_time_encoding(t: torch.Tensor, d_model: int, max_period: float = 20.0) -> torch.Tensor:
    """t: (B,) continuous years -> (B, d_model)."""
    device = t.device
    half = d_model // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=device).float() / half)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
    if enc.shape[-1] < d_model:
        enc = torch.cat([enc, torch.zeros(t.shape[0], d_model - enc.shape[-1], device=device)], dim=-1)
    return enc


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


class DeltaTSurvivalModel(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 time_embed_dim: int = 32, tabular_embed_dim: int = 32,
                 dropout: float = 0.3, n_tabular_features: int = len(TABULAR_FEATURES)):
        super().__init__()
        self.view_encoder = ResNetViewEncoder(backbone=backbone, pretrained=pretrained)
        self.pool = ViewAttentionPooling(self.view_encoder.embed_dim)
        self.time_embed_dim = time_embed_dim
        self.tabular_encoder = TabularEncoder(n_tabular_features, d_feature=tabular_embed_dim)

        fusion_dim = self.view_encoder.embed_dim + time_embed_dim + tabular_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, views: torch.Tensor, view_mask: torch.Tensor,
               tabular_values: torch.Tensor, tabular_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The expensive, delta_t-independent half: image + tabular ->
        (exam_embed (B,D_img), tab_embed (B,D_tab), attn_weights (B,V)).
        Call once per batch; reuse across many sampled delta_t via
        `predict_at_time` to avoid re-running the CNN per time sample."""
        b, v, c, h, w = views.shape
        feats = self.view_encoder(views.reshape(b * v, c, h, w)).reshape(b, v, -1)
        feats = feats * view_mask.unsqueeze(-1)
        exam, attn_weights = self.pool(feats, view_mask)
        tab_embed = self.tabular_encoder(tabular_values, tabular_mask)
        return exam, tab_embed, attn_weights

    def predict_at_time(self, exam_embed: torch.Tensor, tab_embed: torch.Tensor,
                         delta_t: torch.Tensor) -> torch.Tensor:
        """exam_embed: (B,D_img), tab_embed: (B,D_tab), delta_t: (B,) -> logits (B,1).
        Cheap: no CNN forward pass, just the time embedding + fusion MLP."""
        t_embed = sinusoidal_time_encoding(delta_t, self.time_embed_dim)
        fused = torch.cat([exam_embed, t_embed, tab_embed], dim=-1)
        return self.head(fused)

    def forward(self, views: torch.Tensor, view_mask: torch.Tensor, delta_t: torch.Tensor,
                tabular_values: torch.Tensor, tabular_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience single-delta_t path (used at inference for a fixed
        horizon). views: (B,V,1,H,W), view_mask: (B,V), delta_t: (B,),
        tabular_values/mask: (B,F). Returns (logits (B,1), attn (B,V))."""
        exam, tab_embed, attn_weights = self.encode(views, view_mask, tabular_values, tabular_mask)
        logits = self.predict_at_time(exam, tab_embed, delta_t)
        return logits, attn_weights


class TabularOnlySurvivalModel(nn.Module):
    """Ablation: same delta_t-conditioned formulation, tabular features
    only, no image at all -- no CNN, no view encoder. Used to check how
    much of DeltaTSurvivalModel's performance is coming from age/labs
    alone vs. the mammogram itself. Deliberately mirrors
    DeltaTSurvivalModel's encode()/predict_at_time() split so
    train_survival.py's training loop needs no changes to run this."""

    def __init__(self, time_embed_dim: int = 32, tabular_embed_dim: int = 32,
                 dropout: float = 0.3, n_tabular_features: int = len(TABULAR_FEATURES)):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.tabular_encoder = TabularEncoder(n_tabular_features, d_feature=tabular_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(time_embed_dim + tabular_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, views: torch.Tensor, view_mask: torch.Tensor,
               tabular_values: torch.Tensor, tabular_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Signature matches DeltaTSurvivalModel.encode for drop-in reuse
        in train_survival.py -- `views`/`view_mask` are accepted but
        ignored (no CNN pass), so a dummy zero image tensor is fine and
        cheap here; the real speed win is in the dataset skipping the
        DICOM decode entirely (see survival_dataset.py's skip_images)."""
        b = tabular_values.shape[0]
        tab_embed = self.tabular_encoder(tabular_values, tabular_mask)
        dummy_attn = torch.zeros(b, view_mask.shape[1], device=tabular_values.device)
        dummy_exam = torch.zeros(b, 0, device=tabular_values.device)
        return dummy_exam, tab_embed, dummy_attn

    def predict_at_time(self, exam_embed: torch.Tensor, tab_embed: torch.Tensor,
                         delta_t: torch.Tensor) -> torch.Tensor:
        t_embed = sinusoidal_time_encoding(delta_t, self.time_embed_dim)
        fused = torch.cat([t_embed, tab_embed], dim=-1)
        return self.head(fused)

    def forward(self, views, view_mask, delta_t, tabular_values, tabular_mask):
        _, tab_embed, attn = self.encode(views, view_mask, tabular_values, tabular_mask)
        logits = self.predict_at_time(_, tab_embed, delta_t)
        return logits, attn


def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Same construction as odelia_3d/logo_mr.py's masked_bce_loss, reused
    here per-sample (each row is one (patient, sampled delta_t) pair)
    instead of per-year-bucket -- only (patient, delta_t) pairs where the
    label is actually determinable from that patient's censored follow-up
    contribute to the loss."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    masked = bce * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom
