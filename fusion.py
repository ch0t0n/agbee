"""
Feature-fusion strategies evaluated in this study.

This module does not define a proposed method. It implements the six visual
fusion mechanisms compared empirically in the paper (backbone-only /
descriptors-only reference points, body-masked whole image, part-crop late
fusion, attention-pooled part fusion, concatenation, and reference-protocol gated
residual fusion), so that all mechanisms share the same backbone, descriptor
head, and training loop and differ only in how they combine visual evidence.
Gated residual fusion and the classical descriptor formulation follow prior work (Choton et al., 2026)
(cited as prior art); they are evaluated here, not introduced here.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import timm


FusionMode = Literal[
    "backbone_only",
    "descriptors_only",
    "concat",
    "gated_residual",
    "attention_parts",
]


def _normalization_for(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the pretrained timm normalization attached to a model."""
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
    return mean, std


def _create_timm(backbone_name: str, image_size: int | None = None, **kwargs):
    """Build a timm model, passing ``img_size`` only to architectures that need it.

    Transformer backbones tie their positional embeddings to the resolution they
    were pretrained at: `deit_base_distilled_patch16_224` and
    `swin_base_patch4_window7_224` raise "Input height (320) doesn't match model
    (224)" at any other size unless constructed with ``img_size``, which makes
    timm interpolate the embeddings. Convolutional backbones accept any size and
    reject the argument, so it is only passed when the model supports it and the
    requested size actually differs from the pretrained default.
    """
    if image_size is not None:
        try:
            probe = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            data_cfg = timm.data.resolve_model_data_config(probe)
            native = int(data_cfg.get("input_size", (3, image_size, image_size))[-1])
            del probe
            if native != int(image_size):
                try:
                    return timm.create_model(
                        backbone_name, img_size=int(image_size), **kwargs
                    )
                except TypeError:
                    pass  # convolutional backbone: size is free, no img_size arg
        except Exception:
            pass
    return timm.create_model(backbone_name, **kwargs)


def feature_width(model: nn.Module, image_size: int | None = None) -> int:
    """Width of what ``model(x)`` actually returns, measured by one dry forward.

    `model.num_features` is NOT that width for every timm architecture, and
    trusting it builds fusion heads with the wrong `in_features`:

      * `mobilenetv3_small_050` reports `num_features=288` (the pre-conv-head
        channel count) but returns 1024 -- `forward_head(pre_logits=True)` runs
        the conv head. Building `GatedFusion(288, ...)` and then feeding it 1024
        is `mat1 and mat2 shapes cannot be multiplied (8x1312 and 576x288)`,
        which is how Stage C's `gated_residual` arm died.
      * `inception_next_tiny.sail_in1k` reports 768 but returns width **0** at
        `num_classes=0` -- its MLP head has nothing to return before the
        classifier. That one is worse than a crash: `TimmFeatureExtractor`
        exposes `num_features=768` while emitting an empty tensor.

    One forward pass at construction costs milliseconds and is the only source
    of truth that cannot drift from the architecture.
    """
    was_training = model.training
    model.eval()
    try:
        size = image_size
        if size is None:
            data_cfg = timm.data.resolve_model_data_config(model)
            size = int(data_cfg.get("input_size", (3, 224, 224))[-1])
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        with torch.no_grad():
            out = model(torch.zeros(1, 3, int(size), int(size), device=device))
    finally:
        model.train(was_training)
    width = int(out.shape[-1])
    if width <= 0:
        raise ValueError(
            f"backbone returns a zero-width feature vector at num_classes=0 "
            f"(declared num_features={getattr(model, 'num_features', '?')}). "
            "It cannot be used as a feature extractor for any fusion arm; use it "
            "only through TimmClassifier, which keeps its own classifier head."
        )
    return width


class TimmClassifier(nn.Module):
    """A timm classifier with model-specific input normalization."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        image_size: int | None = None,
    ):
        super().__init__()
        self.model = _create_timm(
            backbone_name,
            image_size=image_size,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        mean, std = _normalization_for(self.model)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


class TimmFeatureExtractor(nn.Module):
    """A normalized timm backbone returning pooled feature vectors."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        image_size: int | None = None,
    ):
        super().__init__()
        self.model = _create_timm(
            backbone_name,
            image_size=image_size,
            pretrained=pretrained,
            num_classes=0,
        )
        # Measured, not declared -- see feature_width().
        self.num_features = feature_width(self.model, image_size)
        mean, std = _normalization_for(self.model)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


class ShapeEncoder(nn.Module):
    """Lightweight conv encoder over aggregated foreground + edge map (Choton et al., 2026)."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(mask_tensor).flatten(1)
        return self.fc(feat)


class GatedFusion(nn.Module):
    """Gated residual fusion: Z_f = Z_b + g ⊙ (Ẑ_e − Z_b).

    Follows the reference formulation (Choton et al., 2026) exactly, including
    the LayerNorm between the gate's linear map and its sigmoid. That norm is
    not cosmetic: without it the pre-activations entering the sigmoid are
    unnormalised, the sigmoid saturates, and the gate collapses toward a
    near-constant that no longer selects between the visual and shape paths --
    which is a plausible cause of this arm scoring *below* the plain
    whole-image reference on Beemachine in the 2026-08 campaign, where the
    reference implementation scored above it.
    """

    def __init__(self, img_dim: int, shape_dim: int):
        super().__init__()
        self.shape_proj = nn.Linear(shape_dim, img_dim)
        self.gate = nn.Sequential(
            nn.Linear(img_dim * 2, img_dim),
            nn.LayerNorm(img_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_img: torch.Tensor, z_shape: torch.Tensor) -> torch.Tensor:
        z_shape_proj = self.shape_proj(z_shape)
        g = self.gate(torch.cat([z_img, z_shape_proj], dim=1))
        return z_img + g * (z_shape_proj - z_img)


class DescriptorMLP(nn.Module):
    """Descriptors-only classifier head."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z_p: torch.Tensor) -> torch.Tensor:
        return self.net(z_p)


class PartAttentionPool(nn.Module):
    """Learned attention over part feature vectors."""

    def __init__(self, dim: int, n_parts: int = 4):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.n_parts = n_parts

    def forward(self, part_feats: torch.Tensor) -> torch.Tensor:
        # part_feats: (B, K, D)
        w = torch.softmax(self.score(part_feats).squeeze(-1), dim=1)  # (B, K)
        return (part_feats * w.unsqueeze(-1)).sum(dim=1)


class PartAwareFusionClassifier(nn.Module):
    """
    Unified classifier for Stage B / B+ ablations.

    Modes
    -----
    backbone_only     : Z_b → FC
    descriptors_only  : Z_p → MLP
    concat            : [Z_b, Z_p] → FC
    gated_residual        : gated(Z_b, Z_e) then [Z_f, Z_p] → FC  (reference-protocol)
    attention_parts   : attention over part backbone crops then optional concat Z_p
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "convnext_nano",
        descriptor_dim: int = 937,
        fusion_mode: FusionMode = "gated_residual",
        # 512, matching the reference implementation (Choton et al., 2026),
        # which instantiates its fusion model with shape_embed_dim=512 on all
        # three datasets. This was 256, which halved the shape pathway's width
        # relative to the arm it is meant to reproduce. Overridable from
        # `config.descriptors.shape_embed_dim`.
        shape_embed_dim: int = 512,
        pretrained: bool = True,
        use_descriptors: bool = True,
        n_parts: int = 4,
        image_size: int | None = None,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.use_descriptors = use_descriptors and fusion_mode != "backbone_only"
        self.descriptor_dim = descriptor_dim

        # descriptors_only never calls the backbone -- its forward returns from
        # the descriptor MLP before any visual pathway runs. Building it anyway
        # put a full pretrained backbone's parameters into `n_params`, which is
        # exactly the number the accuracy-per-compute comparison (RQ5) reads:
        # the cheapest arm in the study was being reported as costing as much
        # as the arms it is meant to be the floor for. Same reasoning as the
        # shape/gate/attention submodules below.
        if fusion_mode == "descriptors_only":
            self.backbone = None
            self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
            self.register_buffer("std", torch.ones(1, 3, 1, 1))
            img_dim = 0
        else:
            self.backbone = _create_timm(
                backbone_name, image_size=image_size, pretrained=pretrained, num_classes=0
            )
            mean, std = _normalization_for(self.backbone)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            # Measured, not declared -- see feature_width().
            img_dim = feature_width(self.backbone, image_size)

        # Built only for the mode that actually uses them: an unconditional
        # build here would count every mode's parameters identically
        # (including dead, ungraded submodules), defeating the capacity-matched
        # control's whole purpose of isolating the fusion mechanism's true cost.
        self.shape_encoder = ShapeEncoder(shape_embed_dim) if fusion_mode == "gated_residual" else None
        self.gated = GatedFusion(img_dim, shape_embed_dim) if fusion_mode == "gated_residual" else None
        self.attn = PartAttentionPool(img_dim, n_parts=n_parts) if fusion_mode == "attention_parts" else None
        self.desc_mlp = DescriptorMLP(descriptor_dim, num_classes) if fusion_mode == "descriptors_only" else None

        if fusion_mode == "backbone_only":
            in_dim = img_dim
        elif fusion_mode == "descriptors_only":
            in_dim = descriptor_dim
        elif fusion_mode in ("concat", "gated_residual", "attention_parts"):
            in_dim = img_dim + (descriptor_dim if self.use_descriptors else 0)
        else:
            raise ValueError(fusion_mode)

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.3),
            nn.Linear(in_dim, num_classes),
        )

    @staticmethod
    def mask_to_shape_tensor(mask_logits_or_ids: torch.Tensor) -> torch.Tensor:
        """Build the 3-channel [F, E, F] shape-encoder input.

        Two input forms, deliberately handled differently:

        * **Logits, (B, C, H, W)** -- the predicted-mask regime. F is the sum of
          the foreground channels, `logits[:, 1:].sum(1)`, kept as a real-valued
          map exactly as in the reference implementation (Choton et al., 2026).
          The previous version took `argmax > 0` and produced a hard 0/1 map,
          which throws away everything the segmenter knows about *how confident*
          it is per pixel -- precisely the signal a shape encoder can use to
          discount an unreliable boundary. Restoring the soft map is what makes
          the predicted-mask arm structurally different from the GT arm, which
          is itself a result worth reporting under RQ3 rather than a detail to
          paper over.
        * **Ids, (B, H, W)** -- the ground-truth regime, where no confidence
          exists to preserve. Foreground is `ids > 0`, necessarily binary.

        The edge channel is the morphological residual of a 3x3 mean filter, and
        F is repeated in the third channel so the encoder's first Conv2d keeps
        its 3-channel signature under both regimes.
        """
        if mask_logits_or_ids.ndim == 4:
            foreground = mask_logits_or_ids[:, 1:, :, :].sum(dim=1, keepdim=True)
        else:
            foreground = (mask_logits_or_ids > 0).float().unsqueeze(1)
        edge = torch.abs(
            torch.nn.functional.avg_pool2d(foreground, 3, stride=1, padding=1)
            - foreground
        )
        return torch.cat([foreground, edge, foreground], dim=1)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        z_p: Optional[torch.Tensor] = None,
        mask_logits: Optional[torch.Tensor] = None,
        part_feats: Optional[torch.Tensor] = None,
        part_crops: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        part_feats  : precomputed (B, K, D) per-part backbone embeddings.
        part_crops  : raw (B, K, C, H, W) part crops; the shared backbone is
                      applied to every stream internally (mirrors the crop
                      stack used by the late-fusion arm, so the two mechanisms
                      differ only in aggregation: concat+FC vs. attention).
        """
        if self.fusion_mode == "descriptors_only":
            assert z_p is not None
            return self.desc_mlp(torch.nan_to_num(z_p, nan=0.0))

        if self.fusion_mode == "attention_parts":
            if part_crops is not None:
                b, k, c, h, w = part_crops.shape
                flat = ((part_crops.reshape(b * k, c, h, w) - self.mean) / self.std)
                part_feats = self.backbone(flat).view(b, k, -1)
            if part_feats is not None:
                z_vis = self.attn(part_feats)
            else:
                assert x is not None, "attention_parts needs part_crops, part_feats, or x"
                z_vis = self.backbone((x - self.mean) / self.std)
            if self.use_descriptors:
                assert z_p is not None
                combined = torch.cat([z_vis, torch.nan_to_num(z_p, nan=0.0)], dim=1)
            else:
                combined = z_vis
            return self.head(combined)

        assert x is not None, f"fusion_mode={self.fusion_mode!r} requires x"
        z_img = self.backbone((x - self.mean) / self.std)

        if self.fusion_mode == "backbone_only":
            return self.head(z_img)

        if self.fusion_mode == "gated_residual":
            assert mask_logits is not None
            shape_t = self.mask_to_shape_tensor(mask_logits)
            z_e = self.shape_encoder(shape_t)
            z_vis = self.gated(z_img, z_e)
        else:  # concat
            z_vis = z_img

        if self.use_descriptors:
            assert z_p is not None
            z_p = torch.nan_to_num(z_p, nan=0.0)
            combined = torch.cat([z_vis, z_p], dim=1)
        else:
            combined = z_vis
        return self.head(combined)
