"""
Mirai-style image encoder for single-exam mammogram-based CVD risk.

Mirai (Yala et al., "Toward robust mammography-based models for breast
cancer risk", Nature Medicine 2021) encodes each standard mammogram view
(L-CC, L-MLO, R-CC, R-MLO) independently with a 2D CNN, pools across views
to get one exam-level representation, then (in the original paper) feeds a
sequence of these exam-level vectors -- one per prior visit -- into a
transformer with positional encoding based on time-since-first-exam, before
a per-year hazard head.

This module implements only the per-exam half: PER-VIEW ENCODER + POOLING
-> EXAM EMBEDDING. That's deliberate -- per user direction, we're checking
for single-image signal first, not building the longitudinal transformer
yet. The exam embedding here is exactly the unit that a later
visits-over-time transformer would consume as its per-timestep input, so
this stays a strict subset of the full Mirai architecture rather than a
different design.

Risk head: for this first pass, a single logit for prevalent-CVD binary
classification (see `RiskHead` below with `n_outputs=1`). To extend towards
Mirai's actual risk formulation later (discrete per-year hazard + "remained
healthy" bucket, trained with masked BCE for right-censoring -- see
odelia_3d/logo_mr.py's `masked_bce_loss`, which is architecture-agnostic
and reusable here unchanged), just increase `n_outputs` to n_years+1 and
swap the loss/labels for the incident/prospective cohort instead of the
prevalent one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

STANDARD_VIEWS = ["L_CC", "L_MLO", "R_CC", "R_MLO"]


class ResNetViewEncoder(nn.Module):
    """Per-view 2D CNN encoder. x: (B, 1, H, W) single-channel mammogram ->
    (B, embed_dim). Grayscale is replicated to 3 channels for the
    ImageNet-pretrained backbone, matching LoGo-MR's convention
    (odelia_3d/logo_mr.py:ResNet18SliceEncoder) elsewhere in this project."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet34(weights=weights)
        else:
            raise ValueError(f"unsupported backbone {backbone}")
        self.embed_dim = net.fc.in_features
        net.fc = nn.Identity()
        # Mirai (Yala et al. 2021) pools the final conv feature map with
        # global MAX pooling, not the ResNet-default global average pooling
        # -- swap it so a view's embedding is driven by its most salient
        # (highest-activation) spatial region rather than an average over
        # mostly-background tissue.
        net.avgpool = nn.AdaptiveMaxPool2d(1)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        return self.net(x)


class ViewConditioning(nn.Module):
    """FiLM-style view/laterality conditioning (Yala et al. 2021 Mirai):
    each of the 4 (view, laterality) combinations -- L_CC, L_MLO, R_CC,
    R_MLO -- gets a learned embedding e; e is projected to a per-channel
    scale and shift that's applied to that view's image embedding before
    the self-attention block, so the aggregator knows which physical view
    each embedding came from. W_scale/W_shift are shared across all 4
    combinations (only the embedding e differs)."""

    def __init__(self, d_model: int, n_combos: int = 4, d_embed: int = 128):
        super().__init__()
        self.view_embed = nn.Embedding(n_combos, d_embed)
        self.w_scale = nn.Linear(d_embed, d_model)
        self.w_shift = nn.Linear(d_embed, d_model)

    def forward(self, x: torch.Tensor, view_idx: torch.Tensor) -> torch.Tensor:
        """x: (B, V, D) per-view embeddings, view_idx: (B, V) long in
        [0, n_combos). Returns conditioned (B, V, D)."""
        e = self.view_embed(view_idx)  # (B, V, d_embed)
        return self.w_scale(e) * x + self.w_shift(e)


class ViewSelfAttention(nn.Module):
    """Self-attention block over the set of (conditioned) view embeddings
    of one exam, letting each view's representation be refined in context
    of the others (e.g. CC and MLO of the same breast, or the contralateral
    breast) before pooling -- as in Yala et al. 2021's image aggregator."""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, V, D), mask: (B, V) 1=valid/0=missing. Padded positions
        are excluded from attention via key_padding_mask, then re-zeroed so
        they can't leak stray bias terms into the pooling step."""
        key_padding_mask = mask == 0  # True = ignore
        # a fully-masked row would make softmax produce NaNs; guard against
        # it even though every exam here has >=1 valid view.
        safe_mask = key_padding_mask & (~key_padding_mask.all(dim=1, keepdim=True))
        out = self.layer(x, src_key_padding_mask=safe_mask)
        return out * mask.unsqueeze(-1)


class ViewAttentionPooling(nn.Module):
    """Additive attention pooling across the available views of one exam
    (Santos et al. 2016 attentive pooling, as used in Yala et al. 2021's
    image aggregator). Handles a variable number of views per patient (some
    exams are missing a standard view) via the `mask` argument."""

    def __init__(self, d_model: int, d_attn: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d_model, d_attn), nn.Tanh(), nn.Linear(d_attn, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, V, D) per-view embeddings, mask: (B, V) 1=valid/0=missing view.
        Returns (exam_repr (B, D), attn_weights (B, V))."""
        scores = self.attn(x).squeeze(-1)  # (B, V)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        exam = torch.einsum("bv,bvd->bd", weights, x)
        return exam, weights


class RiskHead(nn.Module):
    """n_outputs=1 -> single binary-risk logit (this first pass, prevalent
    CVD label). n_outputs=n_years+1 -> Mirai-style per-year hazard logits
    for a later incident/prospective survival formulation."""

    def __init__(self, d_model: int, n_outputs: int = 1, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MammoClipViewEncoderWrapper(nn.Module):
    """Mirrors pretrain_mmcl.MammoClipViewEncoder's exact submodule naming
    (self.net wrapping the raw EfficientNet-B5) so MMCL checkpoints
    pretrained with --image_backbone mammo_clip load here without a key
    mismatch -- not imported directly from pretrain_mmcl.py to avoid a
    circular import (that module imports FROM this one)."""

    def __init__(self, device):
        super().__init__()
        from src.mammo_cvd.mammo_clip_features import build_mammo_clip_encoder, OUT_DIM
        self.net = build_mammo_clip_encoder(device)
        self.embed_dim = OUT_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MiraiExamEncoder(nn.Module):
    """Full single-exam model: per-view encoder -> attention pooling over
    views -> risk head. This is the piece to reuse unchanged as the
    per-timestep encoder if/when a visits-over-time transformer is added
    later (per LoGo-MR's sinusoidal_position_encoding pattern in
    odelia_3d/logo_mr.py, applied across exam-level embeddings instead of
    slices)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 n_outputs: int = 1, dropout: float = 0.2, nhead: int = 8, device=None):
        super().__init__()
        if backbone == "mammo_clip":
            # 2026-08-23: added to evaluate MMCL checkpoints pretrained with
            # --image_backbone mammo_clip (e.g. mmcl_pretrain_mammoclip_init_exp2/)
            # -- previously every downstream eval script hardcoded resnet18,
            # so no mammo_clip-backbone MMCL checkpoint had ever been
            # evaluated downstream; failed loudly with clean shape-mismatch
            # errors (2048-dim EfficientNet-B5 fusion layers vs. this
            # class's 512-dim resnet18 default) rather than silently. Same
            # branching pattern as pretrain_mmcl.MMCLImageEncoder -- not
            # imported directly to avoid a circular import (that module
            # imports FROM this one).
            assert device is not None, "device required to build the Mammo-CLIP backbone"
            self.view_encoder = MammoClipViewEncoderWrapper(device)
        else:
            self.view_encoder = ResNetViewEncoder(backbone=backbone, pretrained=pretrained)
        d_model = self.view_encoder.embed_dim
        self.view_conditioning = ViewConditioning(d_model)
        self.self_attn = ViewSelfAttention(d_model, nhead=nhead, dropout=dropout)
        self.pool = ViewAttentionPooling(d_model)
        self.head = RiskHead(d_model, n_outputs=n_outputs, dropout=dropout)

    def forward(self, views: torch.Tensor, mask: torch.Tensor,
                view_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """views: (B, V, 1, H, W), mask: (B, V) 1=valid/0=missing, view_idx:
        (B, V) long in [0,4) identifying each view's (laterality, CC/MLO)
        combo (see STANDARD_VIEWS order: L_CC=0, L_MLO=1, R_CC=2, R_MLO=3).
        Returns (logits (B, n_outputs), exam_embedding (B, D), view_attn_weights (B, V))."""
        b, v, c, h, w = views.shape
        feats = self.view_encoder(views.reshape(b * v, c, h, w)).reshape(b, v, -1)
        # zero out missing-view embeddings so they can't leak through padding
        feats = feats * mask.unsqueeze(-1)
        feats = self.view_conditioning(feats, view_idx)
        feats = self.self_attn(feats, mask)
        exam, attn_weights = self.pool(feats, mask)
        logits = self.head(exam)
        return logits, exam, attn_weights
