#!/usr/bin/env python3
"""Stage B / B+ CLI: baseline | masked | partcrop | multitask | extract_desc | fusion | ablate."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config import (
    add_global_stage_args,
    cls_corpus,
    code_version,
    dataset_cli_args,
    cls_image_size_for,
    default_config_path,
    ensure_dir,
    foreground_part_ids,
    heavy_augmentation,
    label_smoothing,
    load_config,
    cls_patience,
    multitask_run_tag,
    multitask_seg,
    part_image_size,
    protocol,
    resolve_seg_ckpt,
    shape_embed_dim,
    source_digest,
    stage_run_dir,
)
from data import (
    build_part_dataset,
    fold_names,
    part_train_val_test,
    read_shard_csv,
    train_counts_from_indices,
)
from stage_c import large_cls_parts_splits
from descriptors import (
    confidence_weight_vector,
    descriptor_group_mask,
    extract_all_features,
    extract_learned_iqa_features_batch,
    fit_descriptor_standardizer,
    normalize_image_for_iqa,
    report_dead_descriptor_columns,
    set_iqa_device,
    standardize_descriptor_vector,
    validate_iqa_metrics,
)
from distributed_utils import (
    auto_batch_size,
    mib_per_cls_sample,
    mib_per_inf_sample,
    per_device_batch_size,
    resolve_num_gpus,
    run_commands_parallel,
    seed_everything,
    unwrap_model,
)
from fusion import PartAwareFusionClassifier, TimmClassifier, TimmFeatureExtractor
from image_cache import build_or_load as _cache_build_or_load
from metrics import (
    EarlyStopper,
    HeavyAug,
    evaluate_cls_loader,
    fit_cls_model,
    make_cls_criterion,
    classification_report_dict,
    long_tail_bin_metrics,
)
from segmenters import load_segmenter, make_seg_trainer


# The one backbone the controlled comparison grid is pinned to. Every arm that
# lands in `ablation_table` must use it, because that table groups by backbone:
# an arm trained on a different backbone lands in a cell with no reference arm
# and gets no paired-significance result. The 9-backbone cross-architecture
# check lives in `sweep_baselines` alone. Kept in one place so the CLI defaults,
# the sweeps, and run_all_experiments.sh cannot drift apart.
COMMON_BACKBONE = "convnext_nano.in12k"


def _pseudo_masks_dir(cfg) -> str:
    """The confidence-gated pseudo-mask directory every ``large_cls`` dataset's
    classification arms read (see ``stage_c.LargeClsPartsDataset``).

    0.7 matches the threshold `stage_c.py filter_sweep` selected on
    validation data and the pipeline has used ever since (README, Table
    "Behaviour Under Production-Like Masks"); this is not a second free
    parameter, it is the same one Stage C already fixed.
    """
    conf = float(cfg.get("pseudo_labeling", {}).get("selected_conf_threshold", 0.7))
    return str(Path(cfg["paths"]["output_root"]) / "stage_c" / f"pseudo_masks_conf{conf}")


def _mask_source_label(cfg, mask_source: str) -> str:
    """Validate --mask_source and return the label to persist.

    A ``large_cls`` corpus (CUB, Fish-Vista) has no ground truth: pixel
    annotations exist for a small part set only -- 1,888 of CUB's 11,788
    images, and a 6,132-image segmentation split for Fish-Vista that was never
    curated for classification. So those arms only ever consume the
    confidence-gated pseudo-masks already baked into the dataset returned by
    ``_part_cls_splits``. That is plumbed through internally as
    ``mask_source="gt"`` -- the direct-read code path in
    ``MaskedBeeDataset``/``PartCropDataset``/etc, with no live segmenter and
    no redundant prediction pass -- but reported and named as "pseudo" so no
    table or output directory reads as if real annotations were used.
    ``--mask_source pred`` is rejected outright: it would spin up a second,
    wasted segmentation pass over data that is already masked.

    Beemachine is now `large_cls` too (its part set moved to Stage-A-only use;
    see the `cls_corpus` comment in config.yaml), so this applies uniformly to
    all three datasets: none of them has a real gt/pred axis for
    classification, and RQ3's ground-truth-to-predicted question is not
    answered by this study -- a limitation to state rather than to hide.
    """
    if cls_corpus(cfg) != "large_cls":
        return mask_source
    if mask_source == "pred":
        raise SystemExit(
            f"{cfg['dataset']} classification arms already read "
            "confidence-gated pseudo-masks baked into the dataset "
            "(LargeClsPartsDataset); --mask_source pred would run a redundant "
            "live segmentation pass. Pass --mask_source gt instead -- this "
            "dataset's classification grid has no ground-truth option; 'gt' "
            "here selects the direct-read path to the same pseudo-masks Stage "
            "C uses, and is reported as mask_source='pseudo'."
        )
    return "pseudo"


# All internal call sites use the fish-prefixed names below; the bare names
# (_pseudo_masks_dir, _mask_source_label) are kept as aliases for any external
# callers that predate the multi-dataset generalisation.
_fish_pseudo_masks_dir = _pseudo_masks_dir
_fish_mask_source_label = _mask_source_label


def _part_cls_splits(cfg, image_size: int):
    """Shared Stage B setup: ref dataset + train/val/test sources + train counts.

    Which corpus each dataset uses is `config.cls_corpus`, not the on-disk
    layout:

    * ``large_cls`` (Beemachine, CUB, Fish-Vista -- all three datasets) -- the
      full classification corpus with confidence-gated pseudo-masks. CUB moved
      here because its part set spans only 67 of 200 species and 191 test
      images; Beemachine moved here because its part set has species with as
      few as 1 training photograph, neither of which is comparable to the
      corpus the deployed/published model actually serves. `use_train_aug`
      and the 6x train-aug fold below now apply only to each dataset's Stage A
      segmenter training, not to classification.
    * ``part_set`` -- the pixel-annotated part set, where every training image
      has a real mask. No configured dataset currently uses this for
      classification; the branch remains for a dataset whose part set is
      itself the intended classification corpus.
    """
    if cls_corpus(cfg) == "large_cls":
        ref, train_ds, val_ds, test_ds = large_cls_parts_splits(
            cfg, image_size, _pseudo_masks_dir(cfg)
        )
    else:
        entry = cfg.get("dataset_entry") or {}
        ref, train_ds, val_ds, test_ds = part_train_val_test(
            cfg, image_size, use_train_aug=bool(entry.get("use_train_aug", True))
        )
    if isinstance(train_ds, Subset):
        train_counts = train_counts_from_indices(ref.species_ids, list(train_ds.indices))
    else:
        train_counts = train_counts_from_indices(
            train_ds.species_ids, list(range(len(train_ds)))
        )
    return ref, train_ds, val_ds, test_ds, train_counts


def _as_index_dataset(ds):
    """Return (base_ds, indices) suitable for MaskedBeeDataset / PartCropDataset."""
    if isinstance(ds, Subset):
        return ds.dataset, list(ds.indices)
    return ds, list(range(len(ds)))


# ---------------------------------------------------------------------------
# Datasets / models (stage-specific; keep once here)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pred_mask_path(cache_dir: Path, image_name: str) -> Path:
    digest = hashlib.sha256(str(image_name).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.png"


def _prediction_cache_dir(cfg, seg_ckpt: str) -> Path:
    ckpt = Path(seg_ckpt)
    stat = ckpt.stat()
    identity = f"{ckpt.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    tag = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_b" / "pred_mask_cache" / tag)


@torch.no_grad()
def _ensure_pred_mask_cache(cfg, ds, indices, seg, device, cache_dir: Path) -> None:
    pending = [i for i in indices if not _pred_mask_path(cache_dir, ds.image_names[i]).exists()]
    if not pending:
        return

    class _Items(Dataset):
        def __len__(self):
            return len(pending)

        def __getitem__(self, j):
            idx = pending[j]
            return ds[idx][0], idx

    dev_idx = (device.index or 0) if hasattr(device, "index") else int(device)
    pseudo_bs = auto_batch_size(
        dev_idx, mib_per_inf_sample(int(cfg["image_size"])),
        cfg["pseudo_labeling"]["batch_size"],
    )
    loader = DataLoader(
        _Items(),
        batch_size=pseudo_bs,
        shuffle=False,
        **_loader_kwargs(cfg),
    )
    seg_size = int(cfg["image_size"])
    for imgs, batch_indices in tqdm(loader, desc="cache-pred-masks", leave=False):
        original_size = imgs.shape[-2:]
        inputs = F.interpolate(
            imgs.to(device),
            size=(seg_size, seg_size),
            mode="bilinear",
            align_corners=False,
        )
        pred = seg(inputs).argmax(1, keepdim=True).float()
        pred = F.interpolate(pred, size=original_size, mode="nearest").squeeze(1).byte().cpu()
        for mask, idx in zip(pred, batch_indices.tolist()):
            # Parallel sweeps (sweep_masked / sweep_partcrop / ablate) can share
            # the same --seg_ckpt and therefore the same cache_dir, so multiple
            # processes may race to fill the same missing entry. Write to a
            # process-unique temp file and atomically rename into place so a
            # concurrent reader never observes a partially-written PNG.
            dest = _pred_mask_path(cache_dir, ds.image_names[idx])
            tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            Image.fromarray(mask.numpy()).save(tmp, format="PNG")
            os.replace(tmp, dest)


def _pred_mask_ram_cache(cfg, ds, indices: list[int], pred_cache: Path):
    """Shared /dev/shm cache of predicted masks, keyed by position in ``indices``.

    `_ensure_pred_mask_cache` writes one PNG per image under `pred_cache`
    (whose name already embeds the segmenter checkpoint identity, so it
    doubles as a stable cache key). Without this, every dataset __getitem__
    re-opened that PNG from disk every epoch, every DataLoader worker --
    exactly the per-epoch decode cost `image_cache.py` exists to remove, and
    the same bug `stage_c.py`'s `build_mask_cache` fixed for pseudo-masks.
    Returns None (falls back to disk) when the RAM cache is disabled or
    cannot be built; ``cfg`` may be None to opt out explicitly.
    """
    if cfg is None:
        return None
    # The classification resolution, NOT part_image_size(). `_ensure_pred_mask_cache`
    # writes each PNG back at the *dataset's* size (`original_size`), and every
    # consumer multiplies the mask against an image from that same dataset. Sizing
    # the RAM cache by the segmentation resolution instead was invisible on
    # beemachine, where cls_image_size == image_size == 320, and broke both other
    # datasets: cub and fish_vista classify at 224 against a 320 segmentation
    # input, so `img * body` raised "The size of tensor a (224) must match the
    # size of tensor b (320)". Only the cache was wrong -- the disk fallback in
    # `_read_pred_mask` reads the PNG at its native (correct) size, which is why
    # the two paths have to agree here.
    size = int(getattr(ds, "image_size", 0) or part_image_size(cfg))
    paths = [str(_pred_mask_path(pred_cache, ds.image_names[i])) for i in indices]
    return _cache_build_or_load(f"predmask_{pred_cache.name}", paths, size, mode="mask", cfg=cfg)


def _read_pred_mask(cache, i: int, pred_cache: Path, ds, idx: int) -> torch.Tensor:
    """Predicted mask for position ``i`` (ds index ``idx``): RAM cache, else disk."""
    if cache is not None and i < len(cache):
        return torch.from_numpy(np.asarray(cache[i]).astype(np.int64))
    if pred_cache is None:
        raise RuntimeError("predicted-mask cache was not prepared")
    return torch.from_numpy(
        np.asarray(Image.open(_pred_mask_path(pred_cache, ds.image_names[idx])), dtype=np.int64)
    )


class MaskedBeeDataset(Dataset):
    def __init__(self, ds, indices, mask_source: str, pred_cache: Path | None = None, cfg=None):
        self.ds = ds
        self.indices = list(indices)
        self.mask_source = mask_source
        self.pred_cache = pred_cache
        self._pred_mask_cache = (
            _pred_mask_ram_cache(cfg, ds, self.indices, pred_cache)
            if mask_source == "pred" and pred_cache is not None
            else None
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, mask, y = self.ds[idx]
        if self.mask_source == "pred":
            pred = _read_pred_mask(self._pred_mask_cache, i, self.pred_cache, self.ds, idx)
            body = (pred > 0).float()
        else:
            body = (mask > 0).float()
        return img * body.unsqueeze(0), int(y)


class PartCropDataset(Dataset):
    def __init__(self, ds, indices, mask_source, part_ids, pred_cache=None, cfg=None):
        self.ds = ds
        self.indices = list(indices)
        self.mask_source = mask_source
        self.part_ids = list(part_ids)
        self.pred_cache = pred_cache
        self._pred_mask_cache = (
            _pred_mask_ram_cache(cfg, ds, self.indices, pred_cache)
            if mask_source == "pred" and pred_cache is not None
            else None
        )

    def __len__(self):
        return len(self.indices)

    def _mask(self, i, idx, mask):
        if self.mask_source == "gt":
            return mask
        return _read_pred_mask(self._pred_mask_cache, i, self.pred_cache, self.ds, idx)

    @staticmethod
    def _crop_part(img, binary_mask):
        coords = torch.nonzero(binary_mask, as_tuple=False)
        if coords.numel() == 0:
            return torch.zeros_like(img)
        y0, x0 = coords.min(dim=0).values.tolist()
        y1, x1 = coords.max(dim=0).values.tolist()
        crop = (img * binary_mask.float().unsqueeze(0))[
            :, y0 : y1 + 1, x0 : x1 + 1
        ].unsqueeze(0)
        return F.interpolate(
            crop, size=img.shape[-2:], mode="bilinear", align_corners=False
        )[0]

    def __getitem__(self, i):
        idx = self.indices[i]
        img, mask, y = self.ds[idx]
        mask = self._mask(i, idx, mask)
        crops = [self._crop_part(img, mask == pid) for pid in self.part_ids]
        crops.append(self._crop_part(img, mask > 0))
        return torch.stack(crops, dim=0), int(y)


class LateFusionModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        n_streams: int = 4,
        image_size: int | None = None,
    ):
        super().__init__()
        self.backbone = TimmFeatureExtractor(
            backbone_name, pretrained=True, image_size=image_size
        )
        self.n_streams = n_streams
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features * n_streams),
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features * n_streams, num_classes),
        )

    def forward(self, crops):
        b, k, c, h, w = crops.shape
        z = self.backbone(crops.view(b * k, c, h, w))
        return self.head(z.view(b, k * self.backbone.num_features))


class PartCropDescDataset(Dataset):
    """K foreground part crops + whole-body crop, joined to a descriptor row.

    Reuses PartCropDataset's crop logic so the attention-pooled fusion arm
    consumes exactly the same crop stack as the late-fusion (concat) arm —
    the two mechanisms are compared on identical inputs, differing only in
    how the K+1 streams are aggregated (concat+FC vs. learned attention).
    """

    def __init__(
        self, ds, indices, mask_source, part_ids, desc_df, feat_cols, group_mask,
        pred_cache=None, cfg=None, desc_stats=None,
    ):
        self.ds = ds
        self.indices = list(indices)
        self.mask_source = mask_source
        self.part_ids = list(part_ids)
        self.pred_cache = pred_cache
        self.feat_cols = feat_cols
        self.group_mask = group_mask
        self.desc_stats = desc_stats
        self.name_to_row = {r.image: r for r in desc_df.itertuples()}
        self.indices = _join_descriptor_indices(
            self.indices, ds.image_names, self.name_to_row, where='PartCropDescDataset'
        )
        self._pred_mask_cache = (
            _pred_mask_ram_cache(cfg, ds, self.indices, pred_cache)
            if mask_source == "pred" and pred_cache is not None
            else None
        )

    def __len__(self):
        return len(self.indices)

    def _mask(self, i, idx, mask):
        if self.mask_source == "gt":
            return mask
        return _read_pred_mask(self._pred_mask_cache, i, self.pred_cache, self.ds, idx)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, mask, y = self.ds[idx]
        mask = self._mask(i, idx, mask)
        crops = [PartCropDataset._crop_part(img, mask == pid) for pid in self.part_ids]
        crops.append(PartCropDataset._crop_part(img, mask > 0))
        row = self.name_to_row[self.ds.image_names[idx]]
        z = np.asarray([getattr(row, c) for c in self.feat_cols], dtype=np.float32)
        z = standardize_descriptor_vector(z, self.desc_stats, self.group_mask)
        return torch.stack(crops, dim=0), torch.from_numpy(z), int(y)


def _join_descriptor_indices(indices, image_names, name_to_row, *, where: str):
    """Keep only indices whose image has a descriptor row -- and refuse to
    proceed if that silently discards a meaningful share of the split.

    The unguarded filter is how a descriptor CSV that does not cover the
    training fold turns into a smaller, quieter experiment instead of an error.
    That is not hypothetical: turning on Beemachine's 6x train-aug for Stage B
    makes the training fold 34,470 augmented images, none of which appear in a
    descriptor CSV extracted over the 7,716-image unaugmented part set, so
    every fusion arm would have trained on the ~0 rows that happened to match
    and reported it as a normal run.
    """
    kept = [i for i in indices if image_names[i] in name_to_row]
    total = len(list(indices))
    if total and len(kept) < total:
        missing = total - len(kept)
        frac = missing / total
        msg = (
            f"{where}: {missing}/{total} images ({frac:.1%}) have no descriptor "
            f"row and would be dropped from training/evaluation."
        )
        if frac > 0.01:
            raise SystemExit(
                msg
                + "\n  The descriptor CSV does not cover this split. Re-run "
                "descriptor extraction against the same folds Stage B uses "
                "(`stage_b.py extract_desc` for a part_set dataset, "
                "`stage_c.py extract_desc` for a large_cls one), then retry."
            )
        print(f"WARNING: {msg}")
    return kept


class ImageDescDataset(Dataset):
    def __init__(
        self, part_ds, indices, desc_df, feat_cols, group_mask, pred_cache=None, cfg=None,
        desc_stats=None,
    ):
        self.ds = part_ds
        self.feat_cols = feat_cols
        self.group_mask = group_mask
        self.desc_stats = desc_stats
        self.pred_cache = pred_cache
        self.name_to_row = {r.image: r for r in desc_df.itertuples()}
        self.indices = _join_descriptor_indices(
            indices, part_ds.image_names, self.name_to_row, where='ImageDescDataset'
        )
        self._pred_mask_cache = (
            _pred_mask_ram_cache(cfg, part_ds, self.indices, pred_cache)
            if pred_cache is not None
            else None
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, mask, y = self.ds[idx]
        name = self.ds.image_names[idx]
        if self.pred_cache is not None:
            mask = _read_pred_mask(self._pred_mask_cache, i, self.pred_cache, self.ds, idx)
        row = self.name_to_row[name]
        z = np.asarray([getattr(row, c) for c in self.feat_cols], dtype=np.float32)
        z = standardize_descriptor_vector(z, self.desc_stats, self.group_mask)
        return img, mask, torch.from_numpy(z), int(y)


class MultiTaskBeeModel(pl.LightningModule):
    """Shared SMP encoder with seg + species heads (patterns from the reference multi-task model)."""

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        num_parts: int,
        num_classes: int,
        learning_rate: float = 1e-4,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
        t_max: int = 100,
        label_smoothing_value: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.t_max = t_max
        self.num_parts = num_parts

        self.seg = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=3, classes=num_parts
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.seg_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        # Smoothing comes from `classification.label_smoothing` like every other
        # arm; this module cannot call make_cls_criterion(cfg) because Lightning
        # constructs it without the config, so the value is passed in.
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing_value))

        feat_dim = getattr(self.seg.encoder, "out_channels", None)
        if isinstance(feat_dim, (list, tuple)):
            feat_dim = feat_dim[-1]
        if not isinstance(feat_dim, int):
            feat_dim = 2048
        # Set by cmd_multitask after construction, never a constructor arg:
        # save_hyperparameters() would otherwise try to serialize the augmenter
        # into the checkpoint, and a checkpoint that carries its training-time
        # augmentation is one that cannot be reloaded for evaluation.
        self.heavy_aug = None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )
        self._feat_dim = feat_dim

    def forward(self, image: torch.Tensor):
        x = (image - self.mean) / self.std
        seg_logits = self.seg(x)
        feats = self.seg.encoder(x)
        deep = feats[-1] if isinstance(feats, (list, tuple)) else feats
        cls_logits = self.cls_head(self.pool(deep))
        return seg_logits, cls_logits

    def _step(self, batch, stage: str):
        image, mask, y = batch
        if stage == "train" and self.heavy_aug is not None:
            # The full recipe works here, contrary to the usual claim that
            # Mixup and segmentation are incompatible. The incompatibility is
            # only with blending the *target*: DiceLoss takes hard class
            # indices, and a 0.6/0.4 blend of two part maps is not a part map.
            # Mixing the LOSS instead is exact and equivalent --
            #     lam * L(pred, m_a) + (1 - lam) * L(pred, m_b)
            # -- because the loss is linear in the target distribution. So this
            # arm runs the same Mixup/CutMix/RandomErasing as the other six
            # rather than being restricted to CutMix.
            plan = self.heavy_aug.plan_for(image)
            image = self.heavy_aug.transform_images(image, plan)
            seg_logits, cls_logits = self.forward(image)
            mask_a, mask_b = mask.long(), mask.long()[plan.perm]
            loss_s = (
                plan.lam * self.seg_loss(seg_logits, mask_a)
                + (1.0 - plan.lam) * self.seg_loss(seg_logits, mask_b)
            )
            soft = self.heavy_aug.soft_targets(y, plan)
            loss_c = self.heavy_aug.soft_ce(cls_logits, soft)
            loss = self.seg_weight * loss_s + self.cls_weight * loss_c
            acc = (cls_logits.argmax(1) == y).float().mean()
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_cls_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_seg_loss", loss_s, on_step=False, on_epoch=True)
            return loss
        seg_logits, cls_logits = self.forward(image)
        loss_s = self.seg_loss(seg_logits, mask.long())
        loss_c = self.cls_loss(cls_logits, y)
        loss = self.seg_weight * loss_s + self.cls_weight * loss_c
        acc = (cls_logits.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_cls_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_seg_loss", loss_s, on_step=False, on_epoch=True)
        if stage == "valid":
            pred = seg_logits.argmax(1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred, mask, mode="multiclass", num_classes=self.num_parts
            )
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").nanmean()
            self.log("valid_dataset_iou", iou, prog_bar=True, on_step=False, on_epoch=True)
            self.log("valid_miou", miou, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")

    def configure_optimizers(self):
        # AdamW + ReduceLROnPlateau(factor=0.5, patience=2) on validation loss,
        # matching every other Stage B arm's optimizer/schedule (fit_cls_model /
        # cmd_heavy_aug / cmd_fusion) instead of a CosineAnnealingLR schedule
        # unique to this arm.
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "valid_loss", "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Fusion helpers (mask_logits; fit_cls_model batch_fn cannot carry tuples safely)
# ---------------------------------------------------------------------------


def _fusion_mask_logits(model, imgs, masks, seg, num_parts: int, seg_size: int):
    raw = unwrap_model(model)
    if raw.fusion_mode != "gated_residual":
        return None
    if seg is not None:
        with torch.no_grad():
            inputs = F.interpolate(
                imgs, size=(seg_size, seg_size), mode="bilinear", align_corners=False
            )
            logits = seg(inputs)
            return F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
    b, h, w = masks.shape
    mask_logits = torch.zeros(b, num_parts, h, w, device=imgs.device)
    mask_logits.scatter_(1, masks.unsqueeze(1), 1.0)
    return mask_logits


def _run_fusion_epoch(
    model, loader, criterion, device, optimizer=None, seg=None, num_parts=4, seg_size=320,
    desc=None, heavy_aug=None,
):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    # See metrics.run_cls_epoch: model.train(False) does not disable autograd,
    # so the validation pass was recording a backward graph it never used.
    for imgs, masks, z_p, y in tqdm(loader, desc=desc, leave=False):
        imgs, masks, z_p, y = imgs.to(device), masks.to(device), z_p.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        soft = None
        if train and heavy_aug is not None:
            # One plan for all three inputs. The mask is mixed as a hard label
            # and the descriptors as a continuous vector, and both are done
            # BEFORE mask_logits is derived, so the shape channel describes the
            # same composite specimen the backbone sees.
            plan = heavy_aug.plan_for(imgs)
            imgs = heavy_aug.transform_images(imgs, plan)
            masks = heavy_aug.mix_hard(masks, plan)
            z_p = heavy_aug.mix_vector(z_p, plan)
            soft = heavy_aug.soft_targets(y, plan)
        with torch.set_grad_enabled(train):
            mask_logits = _fusion_mask_logits(model, imgs, masks, seg, num_parts, seg_size)
            logits = model(imgs, z_p=z_p, mask_logits=mask_logits)
            loss = criterion(logits, y) if soft is None else heavy_aug.soft_ce(logits, soft)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += imgs.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _eval_fusion(model, loader, device, train_counts, bins, seg=None, num_parts=4, seg_size=320):
    model.eval()
    ys, preds, probas = [], [], []
    for imgs, masks, z_p, y in loader:
        imgs, masks, z_p = imgs.to(device), masks.to(device), z_p.to(device)
        mask_logits = _fusion_mask_logits(model, imgs, masks, seg, num_parts, seg_size)
        logits = model(imgs, z_p=z_p, mask_logits=mask_logits)
        ys.append(y.numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        probas.append(torch.softmax(logits, 1).cpu().numpy())
    y_true, y_pred, y_proba = map(np.concatenate, (ys, preds, probas))
    report = classification_report_dict(
        y_true, y_pred, y_proba, labels=range(y_proba.shape[1])
    )
    report["long_tail"] = long_tail_bin_metrics(y_true, y_pred, train_counts, bins)
    return report


def _run_attn_epoch(model, loader, criterion, device, optimizer=None, desc=None, heavy_aug=None):
    """Epoch runner for the attention-pooled part-fusion arm (crop-stack batches)."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    # Same as _run_fusion_epoch: no autograd graph on the validation pass. It
    # matters most here -- this arm forwards K+1 crops per image (12 on CUB),
    # so the retained activations were the largest of any arm.
    for crops, z_p, y in tqdm(loader, desc=desc, leave=False):
        crops, z_p, y = crops.to(device), z_p.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        soft = None
        if train and heavy_aug is not None:
            # `crops` is (B, K+1, C, H, W). HeavyAug erases per crop and mixes
            # along the batch axis under one permutation, so every stream of
            # sample i mixes with the same partner and the part-to-stream
            # correspondence the attention head pools over is preserved.
            plan = heavy_aug.plan_for(crops)
            crops = heavy_aug.transform_images(crops, plan)
            z_p = heavy_aug.mix_vector(z_p, plan)
            soft = heavy_aug.soft_targets(y, plan)
        with torch.set_grad_enabled(train):
            logits = model(part_crops=crops, z_p=z_p)
            loss = criterion(logits, y) if soft is None else heavy_aug.soft_ce(logits, soft)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _eval_attn(model, loader, device, train_counts, bins):
    model.eval()
    ys, preds, probas = [], [], []
    for crops, z_p, y in loader:
        crops, z_p = crops.to(device), z_p.to(device)
        logits = model(part_crops=crops, z_p=z_p)
        ys.append(y.numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        probas.append(torch.softmax(logits, 1).cpu().numpy())
    y_true, y_pred, y_proba = map(np.concatenate, (ys, preds, probas))
    report = classification_report_dict(
        y_true, y_pred, y_proba, labels=range(y_proba.shape[1])
    )
    report["long_tail"] = long_tail_bin_metrics(y_true, y_pred, train_counts, bins)
    return report


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _config_path_from_argv(default: str = "config.yaml") -> str:
    import sys

    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _loader_kwargs(cfg) -> dict:
    """DataLoader settings shared by every Stage B arm.

    `persistent_workers` matters here: without it each loader tears down and
    re-forks its workers every epoch, which on a 100-epoch budget is 100
    process-pool restarts per arm. `pin_memory` gives the H2D copy a
    page-locked staging buffer so it can overlap with compute.
    """
    nw = int(cfg["num_workers"])
    kw = {"num_workers": nw, "pin_memory": torch.cuda.is_available()}
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    return kw


def _strict_descriptors(cfg) -> bool:
    """Whether an all-NaN descriptor column should abort the pipeline."""
    return bool(cfg.get("descriptors", {}).get("strict_columns", True))


def _finalize_run(
    cfg,
    out_dir: Path,
    report: dict,
    *,
    ckpt_path: Path | None = None,
    extra_meta: dict | None = None,
    append_summary: bool = True,
) -> dict:
    """Write the two per-run artifacts every Stage B arm owes the pipeline.

    `run_meta.json` is not optional bookkeeping: `stage_report.py repro_pack`
    builds the reproducibility bundle by harvesting `**/run_meta.json`, so an
    arm that writes only `test_metrics.json` silently drops out of it. Before
    this was centralized, `baseline` (the whole-image reference every other arm
    is compared against) and both adversarial controls did exactly that, while
    the study contract asserted the opposite.

    The checkpoint hash is taken here, so the report records exactly what the
    run trained.
    """
    ensure_dir(out_dir)
    # Stamped on the report itself, not only on run_meta, because the summary
    # CSV that the comparison table reads is built from these rows.
    report.setdefault("code_version", code_version())
    report.setdefault("source_digest", source_digest())
    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    meta = {
        "dataset": cfg["dataset"],
        "mode": report.get("mode"),
        "group": report.get("group"),
        "mask_source": report.get("mask_source"),
        "backbone": report.get("backbone"),
        "seed": report.get("seed"),
        "n_params": report.get("n_params"),
        "backbone_passes_per_image": report.get("backbone_passes_per_image"),
        # `epochs` is the configured CEILING; with early stopping on it is no
        # longer what the arm ran. `epochs_ran` / `early_stopped` record what
        # actually happened, so repro_pack does not overstate the budget spent.
        "epochs": cfg["classification"]["epochs"],
        "patience": cls_patience(cfg),
        "epochs_ran": report.get("epochs_ran"),
        "early_stopped": report.get("early_stopped"),
        "best_epoch": report.get("best_epoch"),
        "best_val_loss": report.get("best_val_loss"),
        "batch_size": cfg["classification"]["batch_size"],
        "lr": cfg["classification"]["lr"],
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "checkpoint_sha256": (
            _sha256(ckpt_path) if ckpt_path is not None and ckpt_path.exists() else None
        ),
        # Provenance. `stage_report.py ablation_table` refuses to aggregate
        # across differing values without --allow_mixed, so a fix that changes
        # what an arm computes cannot silently be averaged with runs that
        # predate it. See config.code_version.
        "code_version": code_version(),
        "source_digest": source_digest(),
        "label_smoothing": label_smoothing(cfg),
        "cls_corpus": cls_corpus(cfg),
        # Which protocol produced this run, and the two things a protocol can
        # change. Recorded on every arm including the default, so a checkpoint
        # can never be mistaken for one trained at another resolution or under
        # another augmentation recipe -- the run directory's suffix says it too,
        # but a directory can be renamed and this cannot.
        #
        # The resolution lookup is guarded because this function runs *after*
        # training has finished and the checkpoint is on disk. A partial config
        # must not turn a completed run into a crash over a bookkeeping field:
        # the run would be lost and the arm would silently drop out of the
        # reproducibility bundle.
        "protocol": protocol(cfg),
        "cls_image_size": _safe_cls_image_size(cfg, report.get("backbone")),
        "heavy_augmentation": heavy_augmentation(cfg),
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if append_summary:
        _append_summary_rows(cfg, [report])
    print(json.dumps(report, indent=2))
    return report


def _resolve_seed(cfg, seed: int | None) -> int:
    return int(seed) if seed is not None else int(cfg["seed"])


def _summary_csv_path(cfg) -> Path:
    """The comparison-grid summary CSV, one file per protocol.

    A protocol gets its own file rather than a `protocol` column in the shared
    one. Two reasons, both about not being able to corrupt finished work: the
    merge here is a read-modify-write under an flock, so a protocol run that
    crashed mid-write could damage the default grid's summary; and every reader
    downstream (`ablation_table`, `recommend`) groups by
    (mode, group, mask_source, backbone) and would silently average a 320px arm
    together with its 384px counterpart if both lived in one file.
    """
    return stage_run_dir(cfg, "stage_b_descriptors", "fusion_ablation_summary.csv")


def _safe_cls_image_size(cfg, backbone: str | None) -> int | None:
    """Classification resolution for `run_meta.json`, or None if unresolvable.

    See the call site in `_finalize_run`: this runs after training completes, so
    it must never raise. A config without a resolution is not worth losing a
    finished run over.
    """
    try:
        return int(cls_image_size_for(cfg, backbone))
    except (KeyError, TypeError, ValueError):
        return None


def _make_heavy_aug(cfg, num_classes: int) -> "HeavyAug | None":
    """The augmenter every arm shares under `--protocol heavy_aug`, else None.

    Returns None for every other protocol, which is what keeps the default
    grid's training loops on exactly the path they were on before.
    """
    if not heavy_augmentation(cfg):
        return None
    return HeavyAug(num_classes, label_smoothing=label_smoothing(cfg))


def _append_summary_rows(cfg, rows: list[dict]) -> Path:
    """Merge rows into the shared comparison-grid summary CSV that
    stage_report.py's ablation_table reads, keyed by
    (mode, group, mask_source, backbone, seed).

    Every comparison arm's seed sweep writes here now, not just `ablate`'s
    5 fusion modes, so every arm can get the same bootstrap CI / paired
    significance treatment. Runs execute as separate parallel subprocesses
    (one per GPU) that can all reach this at once, so the read-modify-write
    is protected by an flock and the final write is atomic (temp + rename)
    to avoid two processes silently clobbering each other's rows.

    The re-read uses ``keep_default_na=False``. Two of the five key columns
    (``group``, ``mask_source``) carry the literal sentinel "n/a" for arms that
    have neither, and "n/a" is in pandas' default NA list -- so a plain
    ``read_csv`` here turned every stored sentinel into NaN and wrote it back as
    an empty field. That silently broke this function's whole contract: a
    re-run of `baseline`/`capacity_matched`/`heavy_aug`/`multitask` arrives with
    group="n/a" while the stored row now reads "", the two no longer match on
    the key, and `drop_duplicates` *appends* the re-run instead of replacing it.
    The duplicate then reads downstream as an extra seed, inflating `n_seeds`
    and narrowing the bootstrap CI. Preserving the file verbatim keeps the
    merge idempotent, which is what every rerun script depends on.
    """
    out = _summary_csv_path(cfg)
    ensure_dir(out.parent)
    new_df = pd.DataFrame(rows)
    key_cols = [c for c in ["mode", "group", "mask_source", "backbone", "seed"] if c in new_df.columns]
    lock_path = out.with_suffix(out.suffix + ".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            combined = (
                pd.concat([pd.read_csv(out, keep_default_na=False), new_df], ignore_index=True)
                if out.exists()
                else new_df
            )
            if key_cols:
                combined = combined.drop_duplicates(subset=key_cols, keep="last")
            tmp = out.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
            combined.to_csv(tmp, index=False)
            os.replace(tmp, out)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    return out


def _seed_cmds(cfg, subcmd: str, base_args: list[str], seeds: list[int] | None = None) -> list[list[str]]:
    """One argv per seed for a Stage B subcommand. Does not run anything."""
    import sys

    config_path = _config_path_from_argv(default_config_path())
    script = str(Path(__file__).resolve())
    seeds = list(seeds) if seeds else list(cfg.get("seeds") or [cfg["seed"]])
    # `dataset_cli_args`, not a bare `--dataset`: it carries the ACTIVE
    # PROTOCOL as well. Without it every child of a `--protocol heavy_aug`
    # sweep fell back to the default protocol, trained the default recipe and
    # wrote into the default grid's untagged run directories -- overwriting
    # published runs while the parent reported a heavy_aug sweep. stage_a.py
    # and stage_c.py have always forwarded it; this fan-out was the one that
    # did not.
    return [
        [
            sys.executable, script, subcmd,
            "--config", config_path, *dataset_cli_args(cfg),
            *base_args, "--seed", str(seed),
        ]
        for seed in seeds
    ]


def _job_run_dir(cfg, cmd: list[str]) -> Path | None:
    """The run directory one built seed-job argv will write to, or None.

    Mirrors the ``stage_run_dir(cfg, "stage_b", ...)`` name each ``cmd_*``
    chooses. Returning None for an unrecognised subcommand is deliberate: an
    unknown job is always run, never skipped.
    """
    if len(cmd) < 3:
        return None
    subcmd = cmd[2]
    flags = {
        cmd[i]: cmd[i + 1]
        for i in range(3, len(cmd) - 1)
        if str(cmd[i]).startswith("--")
    }
    if "--seed" not in flags:
        return None
    seed = flags["--seed"]
    bb = (flags.get("--backbone") or COMMON_BACKBONE).replace("/", "_")

    if subcmd == "multitask":
        arch = flags.get("--arch")
        encoder = flags.get("--encoder")
        if arch and encoder:
            tag = f"{arch}_{encoder}".replace("/", "_")
        else:
            tag = multitask_run_tag(cfg)
        name = f"multitask_{tag}_seed{seed}"
    elif subcmd in ("masked", "partcrop"):
        try:
            label = _mask_source_label(cfg, flags.get("--mask_source", "gt"))
        except SystemExit:
            return None
        name = f"{subcmd}_{label}_{bb}_seed{seed}"
    elif subcmd in ("baseline", "capacity_matched", "heavy_aug"):
        name = f"{subcmd}_{bb}_seed{seed}"
    else:
        return None
    return stage_run_dir(cfg, "stage_b", name)


def _drop_completed(cfg, cmds: list[list[str]]) -> list[list[str]]:
    """Filter out seed jobs whose run directory is already finished.

    Why this exists. A pooled sweep is one step to the runner, and
    ``_pipeline_lib.sh`` writes that step's resume marker only when the step
    exits 0. So interrupting a 15-job pool with 12 jobs finished used to
    re-train all 15 from scratch on the next ``--resume`` -- the 12 completed
    runs were discarded because nothing below the step level had a notion of
    "done". `run_meta.json` is that notion: `_finalize_run` writes it last, so
    its presence means the run trained, evaluated, and recorded its row.

    Set ``BEEMACHINE_FORCE_RETRAIN=1`` to retrain regardless.
    """
    if os.environ.get("BEEMACHINE_FORCE_RETRAIN") == "1":
        return list(cmds)
    keep, done = [], []
    for cmd in cmds:
        run_dir = _job_run_dir(cfg, cmd)
        if run_dir is not None and (run_dir / "run_meta.json").is_file():
            done.append(run_dir.name)
        else:
            keep.append(cmd)
    for name in done:
        print(f"SKIP {name}: run_meta.json present (set BEEMACHINE_FORCE_RETRAIN=1 to retrain)")
    return keep


def _seed_sweep(cfg, subcmd: str, base_args: list[str], seeds: list[int] | None = None) -> None:
    """Run one Stage B subcommand once per seed, in parallel across GPUs.

    Each child run appends its own row to the shared comparison summary CSV
    (`_append_summary_rows`), so after this returns, `stage_report.py
    ablation_table` can compute this arm's bootstrap CI / paired significance
    the same way it already does for the `ablate` fusion grid -- closing the
    gap where only the 5 fusion modes could ever get >=3 seeds.

    NOTE on utilization: with the default 3 seeds this fills 3 of 8 GPUs. The
    runner therefore prefers `sweep_controls`, which pools all seven of these
    arms into one 21-job submission; this entry point remains for running a
    single arm by hand.
    """
    cmds = _drop_completed(cfg, _seed_cmds(cfg, subcmd, base_args, seeds))
    n_gpus = resolve_num_gpus(cfg)
    cwd = str(Path(__file__).resolve().parent)
    if not cmds:
        print(f"Stage B {subcmd} seed sweep: every seed already complete; nothing to run")
        return
    print(f"Stage B {subcmd} seed sweep: {len(cmds)} seed(s) on {n_gpus} GPU(s)")
    codes = run_commands_parallel(cmds, n_gpus, cwd=cwd)
    if any(c != 0 for c in codes):
        raise SystemExit(f"{sum(c != 0 for c in codes)} {subcmd} seed job(s) failed")


def cmd_sweep_controls(
    cfg,
    backbone: str | None,
    capacity_backbone: str,
    seg_ckpt: str | None,
    device: int,
    arch: str | None = None,
    encoder: str | None = None,
    seeds: list[int] | None = None,
) -> None:
    """Every seed-swept Stage B arm as ONE pool, so all 8 GPUs stay busy.

    These seven arms -- capacity_matched, heavy_aug, masked x {gt,pred},
    partcrop x {gt,pred}, multitask -- used to be seven consecutive steps in
    `run_all_experiments.sh`, each launching `len(seeds)` = 3 jobs. Seven waves
    of 3 on an 8-GPU box leaves 5 devices idle throughout: measured at 20.9 h of
    pure idle across the three datasets.

    They are mutually independent -- different arms, different mask sources, no
    shared state beyond the append-only summary CSV -- and nothing reads any of
    their outputs until `stage_report.py ablation_table`. Pooling them into a
    single 21-job submission changes no number in any table; it only stops the
    box idling between waves.

    Kept per-arm: each job is still its own process with its own seed, writes to
    its own output directory, and appends its own summary row exactly as before.
    """
    bb = backbone or COMMON_BACKBONE
    mt_args: list[str] = []
    if arch:
        mt_args += ["--arch", arch]
    if encoder:
        mt_args += ["--encoder", encoder]
    pred = (["--seg_ckpt", seg_ckpt] if seg_ckpt else [])

    if cls_corpus(cfg) == "large_cls":
        # A large_cls dataset's classification arms already read
        # confidence-gated pseudo-masks baked into LargeClsPartsDataset --
        # there is no separate ground truth to predict from, so the gt/pred
        # contrast collapses to one cell (--mask_source gt selects the
        # direct-read path to those pseudo-masks; see _mask_source_label). A
        # --mask_source pred job here would be a second, redundant live
        # segmentation pass, and _mask_source_label rejects it outright.
        #
        # This branch used to key off `layout == "fish_vista"`, from when
        # Fish-Vista was the only large_cls dataset. Beemachine and CUB have
        # since moved to large_cls too, so they fell into the `else` plan
        # below and were scheduled 2 pred arms x len(seeds) jobs that could
        # only ever exit non-zero -- which then tripped the `any(c != 0)`
        # check at the end of this function and failed the whole step. The
        # condition now matches the thing it is actually about.
        plan: list[tuple[str, list[str]]] = [
            ("capacity_matched", ["--backbone", capacity_backbone]),
            ("heavy_aug", ["--backbone", bb]),
            ("masked", ["--backbone", bb, "--mask_source", "gt"]),
            ("partcrop", ["--backbone", bb, "--mask_source", "gt"]),
            ("multitask", mt_args),
        ]
    else:
        # part_set dataset: gt vs pred is a real axis, because every training
        # image has a hand-annotated mask to contrast the predicted one with.
        plan = [
            ("capacity_matched", ["--backbone", capacity_backbone]),
            ("heavy_aug", ["--backbone", bb]),
            ("masked", ["--backbone", bb, "--mask_source", "gt"]),
            ("masked", ["--backbone", bb, "--mask_source", "pred", *pred]),
            ("partcrop", ["--backbone", bb, "--mask_source", "gt"]),
            ("partcrop", ["--backbone", bb, "--mask_source", "pred", *pred]),
            ("multitask", mt_args),
        ]
    if not seg_ckpt and cls_corpus(cfg) != "large_cls":
        # The two pred arms resolve the segmenter themselves via resolve_seg_ckpt;
        # let them, rather than silently dropping half the mask-source axis.
        print("[sweep_controls] no --seg_ckpt given; pred arms will resolve it from stage_a")

    cmds: list[list[str]] = []
    for subcmd, base_args in plan:
        cmds.extend(_seed_cmds(cfg, subcmd, base_args, seeds))

    n_gpus = resolve_num_gpus(cfg)
    n_seeds = len(seeds) if seeds else len(cfg.get("seeds") or [cfg["seed"]])
    planned = len(cmds)
    cmds = _drop_completed(cfg, cmds)
    if not cmds:
        print(
            f"Stage B controls pool: all {planned} jobs "
            f"({len(plan)} arms x {n_seeds} seeds) already complete; nothing to run"
        )
        return
    print(
        f"Stage B controls pool: {len(cmds)} of {planned} jobs to run "
        f"({len(plan)} arms x {n_seeds} seeds) on {n_gpus} GPU(s)"
    )
    codes = run_commands_parallel(cmds, n_gpus, cwd=str(Path(__file__).resolve().parent))
    if any(c != 0 for c in codes):
        # Name the arms that failed, not just a count: with 21 jobs in one
        # submission a bare "3 failed" leaves you re-reading the whole log.
        failed = [
            _describe_job(cmd) for cmd, code in zip(cmds, codes) if code != 0
        ]
        raise SystemExit(
            f"{sum(c != 0 for c in codes)}/{len(cmds)} control job(s) failed: "
            + ", ".join(failed)
        )


def _describe_job(cmd: list[str]) -> str:
    """`masked/pred/seed13` -- enough to find the arm in a 21-job pool."""
    parts = [cmd[2]]
    for flag in ("--mask_source", "--seed"):
        if flag in cmd:
            parts.append(cmd[cmd.index(flag) + 1])
    return "/".join(parts)


def cmd_baseline(cfg, backbone: str, device: int, seed: int | None = None):
    """Whole-image classifier.

    The reference arm of the controlled grid: the abstraction every published
    model in the BeeMachine lineage uses, and the arm every anatomy-consuming
    arm is measured against. It runs on the common comparison backbone at the
    corpus resolution, like every other arm.

    `seed` lets `sweep_baselines`/`--seed` repeat this arm under several
    seeds, the same way `ablate` does for the fusion modes, so it can get a
    bootstrap CI / paired significance row in `stage_report.py
    ablation_table` instead of only ever a single-seed point estimate.
    """
    run_seed = _resolve_seed(cfg, seed)
    seed_everything(run_seed)
    img_size = cls_image_size_for(cfg, backbone)
    ref, train_ds, val_ds, test_ds, train_counts = _part_cls_splits(cfg, img_size)
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier(
        backbone, pretrained=True, num_classes=len(ref.classes),
        image_size=cls_image_size_for(cfg, backbone),
    ).to(device_t)
    n_params = sum(p.numel() for p in model.parameters())

    bs = auto_batch_size(
        device, mib_per_cls_sample(img_size), cfg["classification"]["batch_size"]
    )
    loader_kw = _loader_kwargs(cfg)

    def _build_baseline_loaders(bs_: int):
        return (
            DataLoader(train_ds, batch_size=bs_, shuffle=True, **loader_kw),
            DataLoader(val_ds, batch_size=bs_, shuffle=False, **loader_kw),
        )

    train_loader, val_loader = _build_baseline_loaders(bs)

    safe_bb = backbone.replace("/", "_")
    out_dir = ensure_dir(
        stage_run_dir(cfg, "stage_b", f"baseline_{safe_bb}_seed{run_seed}")
    )
    print(f"baseline dataset={cfg['dataset']} device={device} batch_size={bs} backbone={backbone} seed={run_seed}")
    fit_history: dict = {}
    fit_cls_model(
        model,
        train_loader,
        val_loader,
        device_t,
        cfg["classification"]["lr"],
        cfg["classification"]["epochs"],
        out_dir / "best.pt",
        patience=cls_patience(cfg),
        criterion=make_cls_criterion(cfg),
        history=fit_history,
        rebuild_loaders=_build_baseline_loaders,
        batch_size=bs,
        heavy_aug=_make_heavy_aug(cfg, len(ref.classes)),
    )
    test_loader = DataLoader(
        test_ds, batch_size=fit_history.get("final_batch_size") or bs, shuffle=False, **loader_kw
    )
    report = evaluate_cls_loader(
        model, test_loader, device_t, train_counts, cfg["metrics"]["long_tail_bins"]
    )
    report.update(
        {
            "mode": "baseline", "group": "n/a", "mask_source": "n/a",
            "backbone": backbone, "seed": run_seed, "n_params": int(n_params),
            "backbone_passes_per_image": 1,
        }
    )
    report.update(fit_history)
    _finalize_run(cfg, out_dir, report, ckpt_path=out_dir / "best.pt")


def cmd_sweep_baselines(cfg, device: int, seeds: list[int] | None = None):
    """Train all config classification backbones x seeds — parallel across GPUs."""
    import sys

    n_gpus = resolve_num_gpus(cfg)
    config_path = _config_path_from_argv(default_config_path())
    script = str(Path(__file__).resolve())
    seeds = list(seeds) if seeds else list(cfg.get("seeds") or [cfg["seed"]])
    cmds = []
    for bb in cfg["classification"]["backbones"]:
        for seed in seeds:
            cmds.append(
                [
                    sys.executable,
                    script,
                    "baseline",
                    "--config",
                    config_path,
                    "--dataset",
                    cfg["dataset"],
                    "--backbone",
                    bb,
                    "--seed",
                    str(seed),
                ]
            )
    print(f"Stage B baselines: {len(cmds)} jobs ({len(cfg['classification']['backbones'])} backbones x {len(seeds)} seeds) on {n_gpus} GPU(s)")
    codes = run_commands_parallel(cmds, n_gpus, cwd=str(Path(script).parent))
    if any(c != 0 for c in codes):
        raise SystemExit(f"{sum(c != 0 for c in codes)} baseline job(s) failed")


def cmd_capacity_matched(cfg, backbone: str, device: int, seed: int | None = None):
    """Whole-image, capacity-matched control.

    Isolates a confound raised repeatedly in the fusion-strategy literature:
    part-aware arms add parameters (a shape encoder, a descriptor head, or a
    K-stream late-fusion backbone). This arm trains a whole-image classifier
    on the *same backbone family, scaled up* (pass a larger `--backbone`,
    e.g. convnext_small instead of convnext_nano) so its parameter count can
    be compared directly against the fusion arms via the `n_params` field
    every arm already logs. It answers: is a fusion-arm's edge attributable
    to anatomy, or would the same parameter budget spent on a bigger
    whole-image backbone buy the same accuracy?
    """
    run_seed = _resolve_seed(cfg, seed)
    seed_everything(run_seed)
    _img_size_cap = cls_image_size_for(cfg, backbone)
    ref, train_ds, val_ds, test_ds, train_counts = _part_cls_splits(cfg, _img_size_cap)
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier(
        backbone, pretrained=True, num_classes=len(ref.classes),
        image_size=_img_size_cap,
    ).to(device_t)
    n_params = sum(p.numel() for p in model.parameters())

    bs = auto_batch_size(
        device, mib_per_cls_sample(_img_size_cap), cfg["classification"]["batch_size"]
    )
    loader_kw = _loader_kwargs(cfg)

    def _build_cap_loaders(bs_: int):
        return (
            DataLoader(train_ds, batch_size=bs_, shuffle=True, **loader_kw),
            DataLoader(val_ds, batch_size=bs_, shuffle=False, **loader_kw),
        )

    train_loader, val_loader = _build_cap_loaders(bs)

    safe_bb = backbone.replace("/", "_")
    out_dir = ensure_dir(
        stage_run_dir(cfg, "stage_b", f"capacity_matched_{safe_bb}_seed{run_seed}")
    )
    print(f"capacity_matched dataset={cfg['dataset']} backbone={backbone} n_params={n_params:,} seed={run_seed}")
    fit_history: dict = {}
    fit_cls_model(
        model, train_loader, val_loader, device_t,
        cfg["classification"]["lr"], cfg["classification"]["epochs"], out_dir / "best.pt",
        patience=cls_patience(cfg),
        criterion=make_cls_criterion(cfg), history=fit_history,
        rebuild_loaders=_build_cap_loaders, batch_size=bs,
        heavy_aug=_make_heavy_aug(cfg, len(ref.classes)),
    )
    test_loader = DataLoader(
        test_ds, batch_size=fit_history.get("final_batch_size") or bs, shuffle=False, **loader_kw
    )
    report = evaluate_cls_loader(model, test_loader, device_t, train_counts, cfg["metrics"]["long_tail_bins"])
    report.update(
        {
            "mode": "capacity_matched", "group": "n/a", "mask_source": "n/a",
            "backbone": backbone, "seed": run_seed, "n_params": int(n_params),
            "backbone_passes_per_image": 1,
        }
    )
    report.update(fit_history)
    _finalize_run(cfg, out_dir, report, ckpt_path=out_dir / "best.pt")


def cmd_sweep_capacity_matched(cfg, backbone: str, device: int, seeds: list[int] | None = None):
    """capacity_matched across seeds — parallel across GPUs."""
    _seed_sweep(cfg, "capacity_matched", ["--backbone", backbone], seeds)


def _mixup_cutmix(imgs: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float = 0.2):
    """Batch-level Mixup/CutMix used only by the heavy-augmentation control arm."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    y_onehot = torch.zeros(imgs.size(0), num_classes, device=imgs.device).scatter_(1, y.unsqueeze(1), 1.0)
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[perm]
    if np.random.rand() < 0.5:
        imgs_mix = lam * imgs + (1 - lam) * imgs[perm]
    else:
        h, w = imgs.shape[-2:]
        cut_w, cut_h = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
        imgs_mix = imgs.clone()
        imgs_mix[:, :, y1:y2, x1:x2] = imgs[perm][:, :, y1:y2, x1:x2]
        lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        y_mix = lam_adj * y_onehot + (1 - lam_adj) * y_onehot[perm]
    return imgs_mix, y_mix


def cmd_heavy_aug(cfg, backbone: str, device: int, seed: int | None = None):
    """Whole-image, heavy-augmentation control.

    A second adversarial baseline: instead of adding anatomical structure,
    regularize a plain whole-image classifier with RandomErasing +
    Mixup/CutMix under the same epoch budget as every fusion arm. If this
    control closes most of the gap to the fusion arms, the source of their
    advantage is regularization rather than anatomical evidence; if it does
    not, that strengthens the case that anatomy is doing real work.
    """
    from torchvision import transforms as T

    run_seed = _resolve_seed(cfg, seed)
    seed_everything(run_seed)
    _img_size_ha = cls_image_size_for(cfg, backbone)
    ref, train_ds, val_ds, test_ds, train_counts = _part_cls_splits(cfg, _img_size_ha)
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier(
        backbone, pretrained=True, num_classes=len(ref.classes),
        image_size=_img_size_ha,
    ).to(device_t)
    n_params = sum(p.numel() for p in model.parameters())
    num_classes = len(ref.classes)
    erase = T.RandomErasing(p=0.5, scale=(0.02, 0.2))

    bs = auto_batch_size(
        device, mib_per_cls_sample(_img_size_ha), cfg["classification"]["batch_size"]
    )
    loader_kw = _loader_kwargs(cfg)

    def _build_heavy_aug_loaders(bs_: int):
        return (
            DataLoader(train_ds, batch_size=bs_, shuffle=True, **loader_kw),
            DataLoader(val_ds, batch_size=bs_, shuffle=False, **loader_kw),
        )

    train_loader, val_loader = _build_heavy_aug_loaders(bs)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["classification"]["lr"])
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    safe_bb = backbone.replace("/", "_")
    out_dir = ensure_dir(
        stage_run_dir(cfg, "stage_b", f"heavy_aug_{safe_bb}_seed{run_seed}")
    )
    best = float("inf")
    total_epochs = cfg["classification"]["epochs"]
    stopper = EarlyStopper(cls_patience(cfg))
    epochs_ran = 0
    epoch = 0
    while epoch < total_epochs:
        oom_hit = False
        try:
            model.train()
            tr_loss, n = 0.0, 0
            for batch in tqdm(train_loader, desc=f"[heavy_aug] epoch {epoch + 1}/{total_epochs} train", leave=False):
                imgs, y = batch[0], batch[-1]
                imgs, y = imgs.to(device_t), y.to(device_t)
                imgs = erase(imgs)
                imgs_mix, y_soft = _mixup_cutmix(imgs, y, num_classes)
                # Label smoothing composes with Mixup/CutMix by shrinking the soft
                # target toward uniform, which is what nn.CrossEntropyLoss does
                # internally for hard targets. Applying it here keeps this control
                # arm on the same loss as every other arm -- the arm is meant to
                # isolate *augmentation*, so leaving it on an unsmoothed loss would
                # have confounded the two.
                eps = label_smoothing(cfg)
                if eps > 0.0:
                    y_soft = y_soft * (1.0 - eps) + eps / num_classes
                opt.zero_grad()
                logits = model(imgs_mix)
                loss = -(torch.log_softmax(logits, 1) * y_soft).sum(1).mean()
                loss.backward()
                opt.step()
                tr_loss += loss.item() * imgs.size(0)
                n += imgs.size(0)
            va_loss, va_acc = 0.0, 0.0
            model.eval()
            vn, vcorrect = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    imgs, y = batch[0], batch[-1]
                    imgs, y = imgs.to(device_t), y.to(device_t)
                    logits = model(imgs)
                    va_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
                    vcorrect += (logits.argmax(1) == y).sum().item()
                    vn += imgs.size(0)
        except torch.cuda.OutOfMemoryError:
            oom_hit = True
        if oom_hit:
            # See cmd_fusion for why this self-corrects at runtime rather than
            # trusting the offline `auto_batch_size` estimate, and why the
            # cleanup below must run outside the `except` block.
            if bs <= 1:
                raise RuntimeError("[heavy_aug] CUDA out of memory even at batch_size=1")
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            print(f"[heavy_aug] CUDA OOM; retrying epoch {epoch + 1} at batch_size={bs}", flush=True)
            train_loader, val_loader = _build_heavy_aug_loaders(bs)
            continue
        va_loss /= max(vn, 1)
        va_acc = vcorrect / max(vn, 1)
        sch.step(va_loss)
        print(f"[heavy_aug] epoch {epoch + 1}: train_loss={tr_loss / max(n, 1):.4f} val_acc={va_acc:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
        epochs_ran = epoch + 1
        if stopper.step(va_loss, epoch):
            print(f"[heavy_aug] early stop: {stopper.summary(total_epochs)}")
            break
        epoch += 1

    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kw)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device_t, weights_only=True))
    report = evaluate_cls_loader(model, test_loader, device_t, train_counts, cfg["metrics"]["long_tail_bins"])
    report.update(
        {
            "mode": "heavy_aug", "group": "n/a", "mask_source": "n/a",
            "epochs_ran": epochs_ran,
            "early_stopped": stopper.stopped_epoch is not None,
            "best_epoch": stopper.best_epoch + 1,
            "best_val_loss": round(stopper.best, 6),
            "backbone": backbone, "seed": run_seed, "n_params": int(n_params),
            "backbone_passes_per_image": 1,
        }
    )
    _finalize_run(cfg, out_dir, report, ckpt_path=out_dir / "best.pt")


def cmd_sweep_heavy_aug(cfg, backbone: str, device: int, seeds: list[int] | None = None):
    """heavy_aug across seeds — parallel across GPUs."""
    _seed_sweep(cfg, "heavy_aug", ["--backbone", backbone], seeds)


def cmd_masked(cfg, backbone: str, mask_source: str, seg_ckpt: str | None, device: int, seed: int | None = None):
    run_seed = _resolve_seed(cfg, seed)
    seed_everything(run_seed)
    mask_source_label = _fish_mask_source_label(cfg, mask_source)
    _img_size_mk = cls_image_size_for(cfg, backbone)
    ref, train_ds, val_ds, test_ds, train_counts = _part_cls_splits(cfg, _img_size_mk)
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    seg = None
    pred_cache = None
    if mask_source == "pred":
        resolved = resolve_seg_ckpt(cfg, seg_ckpt)
        seg = load_segmenter(resolved, device_t)
        pred_cache = _prediction_cache_dir(cfg, resolved)

    def mk(fold_ds):
        base, idxs = _as_index_dataset(fold_ds)
        if seg is not None:
            _ensure_pred_mask_cache(cfg, base, idxs, seg, device_t, pred_cache)
        return MaskedBeeDataset(base, idxs, mask_source, pred_cache, cfg=cfg)

    bs = auto_batch_size(
        device, mib_per_cls_sample(_img_size_mk), cfg["classification"]["batch_size"]
    )
    loader_kw = _loader_kwargs(cfg)
    train_data, val_data, test_data = mk(train_ds), mk(val_ds), mk(test_ds)

    def _build_masked_loaders(bs_: int):
        return (
            DataLoader(train_data, batch_size=bs_, shuffle=True, **loader_kw),
            DataLoader(val_data, batch_size=bs_, shuffle=False, **loader_kw),
        )

    train_loader, val_loader = _build_masked_loaders(bs)

    model = TimmClassifier(
        backbone, pretrained=True, num_classes=len(ref.classes),
        image_size=cls_image_size_for(cfg, backbone),
    ).to(device_t)
    n_params = sum(p.numel() for p in model.parameters())
    safe_bb = backbone.replace("/", "_")
    out_dir = ensure_dir(
        stage_run_dir(cfg, "stage_b", f"masked_{mask_source_label}_{safe_bb}_seed{run_seed}")
    )
    fit_history: dict = {}
    fit_cls_model(
        model,
        train_loader,
        val_loader,
        device_t,
        cfg["classification"]["lr"],
        cfg["classification"]["epochs"],
        out_dir / "best.pt",
        patience=cls_patience(cfg),
        criterion=make_cls_criterion(cfg),
        history=fit_history,
        rebuild_loaders=_build_masked_loaders,
        batch_size=bs,
        heavy_aug=_make_heavy_aug(cfg, len(ref.classes)),
    )
    test_loader = DataLoader(
        test_data, batch_size=fit_history.get("final_batch_size") or bs, shuffle=False, **loader_kw
    )
    report = evaluate_cls_loader(
        model, test_loader, device_t, train_counts, cfg["metrics"]["long_tail_bins"]
    )
    report.update(
        {
            "mode": "masked", "group": "n/a", "mask_source": mask_source_label,
            "backbone": backbone, "seed": run_seed, "n_params": int(n_params),
            "backbone_passes_per_image": 1,
        }
    )
    report.update(fit_history)
    _finalize_run(cfg, out_dir, report, ckpt_path=out_dir / "best.pt")


def cmd_sweep_masked(
    cfg, mask_source: str, seg_ckpt: str | None, device: int,
    seeds: list[int] | None = None, backbone: str | None = None,
):
    """masked across seeds on ONE backbone — the controlled grid's pin.

    This used to sweep all `classification.backbones` x seeds (27 jobs, run
    twice for gt/pred). Nothing could be done with 8/9 of that: every arm it
    is compared against -- concat, gated_residual, attention_parts, and both
    confound controls -- is pinned to the single common backbone, and
    `stage_report.py ablation_table` groups by backbone, so the eight extra
    backbones produced rows with no reference arm in their cell and therefore
    a blank paired-significance column. The 9-backbone cross-architecture
    check is `sweep_baselines`, which is where it belongs.
    """
    _seed_sweep(
        cfg,
        "masked",
        ["--backbone", backbone or COMMON_BACKBONE, "--mask_source", mask_source]
        + (["--seg_ckpt", seg_ckpt] if seg_ckpt else []),
        seeds,
    )


def cmd_partcrop(cfg, backbone: str, mask_source: str, seg_ckpt: str | None, device: int, seed: int | None = None):
    run_seed = _resolve_seed(cfg, seed)
    seed_everything(run_seed)
    mask_source_label = _fish_mask_source_label(cfg, mask_source)
    _img_size_pc = cls_image_size_for(cfg, backbone)
    ref, train_ds, val_ds, test_ds, train_counts = _part_cls_splits(cfg, _img_size_pc)
    part_ids = foreground_part_ids(cfg["part_labels"])
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    seg = None
    pred_cache = None
    if mask_source == "pred":
        resolved = resolve_seg_ckpt(cfg, seg_ckpt)
        seg = load_segmenter(resolved, device_t)
        pred_cache = _prediction_cache_dir(cfg, resolved)

    def mk(fold_ds):
        base, idxs = _as_index_dataset(fold_ds)
        if seg is not None:
            _ensure_pred_mask_cache(cfg, base, idxs, seg, device_t, pred_cache)
        return PartCropDataset(base, idxs, mask_source, part_ids, pred_cache, cfg=cfg)

    # Each DataLoader item contains K+1 crop streams (K foreground parts + whole
    # body). auto_batch_size accounts for that via n_streams so the effective
    # memory per item is estimated correctly without double-dividing.
    n_streams = len(part_ids) + 1
    bs = max(8, auto_batch_size(
        device, mib_per_cls_sample(_img_size_pc, n_streams), cfg["classification"]["batch_size"]
    ))
    loader_kw = _loader_kwargs(cfg)
    train_data, val_data, test_data = mk(train_ds), mk(val_ds), mk(test_ds)

    def _build_partcrop_loaders(bs_: int):
        return (
            DataLoader(train_data, batch_size=bs_, shuffle=True, **loader_kw),
            DataLoader(val_data, batch_size=bs_, shuffle=False, **loader_kw),
        )

    train_loader, val_loader = _build_partcrop_loaders(bs)

    n_streams = len(part_ids) + 1
    model = LateFusionModel(
        backbone,
        len(ref.classes),
        n_streams=n_streams,
        image_size=cls_image_size_for(cfg, backbone),
    ).to(device_t)
    n_params = sum(p.numel() for p in model.parameters())
    safe_bb = backbone.replace("/", "_")
    out_dir = ensure_dir(
        stage_run_dir(cfg, "stage_b", f"partcrop_{mask_source_label}_{safe_bb}_seed{run_seed}")
    )
    fit_history: dict = {}
    fit_cls_model(
        model,
        train_loader,
        val_loader,
        device_t,
        cfg["classification"]["lr"],
        cfg["classification"]["epochs"],
        out_dir / "best.pt",
        patience=cls_patience(cfg),
        criterion=make_cls_criterion(cfg),
        history=fit_history,
        rebuild_loaders=_build_partcrop_loaders,
        batch_size=bs,
        heavy_aug=_make_heavy_aug(cfg, len(ref.classes)),
    )
    test_loader = DataLoader(
        test_data, batch_size=fit_history.get("final_batch_size") or bs, shuffle=False, **loader_kw
    )
    report = evaluate_cls_loader(
        model, test_loader, device_t, train_counts, cfg["metrics"]["long_tail_bins"]
    )
    report.update(
        {
            "mode": "partcrop", "group": "n/a", "mask_source": mask_source_label,
            "backbone": backbone, "seed": run_seed, "n_params": int(n_params),
            "backbone_passes_per_image": n_streams,
        }
    )
    report.update(fit_history)
    _finalize_run(cfg, out_dir, report, ckpt_path=out_dir / "best.pt")


def cmd_sweep_partcrop(
    cfg, mask_source: str, seg_ckpt: str | None, device: int,
    seeds: list[int] | None = None, backbone: str | None = None,
):
    """partcrop across seeds on ONE backbone. See cmd_sweep_masked for why.

    Doubly worth pinning here: this arm costs K+1 backbone passes per image
    (12 on CUB, 10 on Fish-Vista), so the eight uncomparable backbones were
    the most expensive uncomparable rows in the whole study.
    """
    _seed_sweep(
        cfg,
        "partcrop",
        ["--backbone", backbone or COMMON_BACKBONE, "--mask_source", mask_source]
        + (["--seg_ckpt", seg_ckpt] if seg_ckpt else []),
        seeds,
    )


def cmd_multitask(
    cfg, arch: str | None, encoder: str | None, device: int,
    num_gpus: int | None = None, seed: int | None = None,
):
    """Multi-task (shared encoder + auxiliary segmentation loss).

    Kept on the same protocol as every other Stage B arm -- AdamW +
    ReduceLROnPlateau on validation loss, no early stopping, the full
    `cfg['classification']['epochs']` budget, and `cls_image_size` input --
    instead of the Lightning-default cosine schedule / IoU-based early
    stopping / segmentation resolution it used previously. Those were
    accidental extra degrees of freedom for an arm meant to be
    protocol-identical to its siblings, differing only in the auxiliary
    segmentation loss.
    """
    run_seed = _resolve_seed(cfg, seed)
    pl.seed_everything(run_seed, workers=True)
    # `classification.multitask`, not `segmentation`: the encoder this arm
    # shares between its two heads IS its visual backbone, so it is pinned to
    # the grid's common backbone like the other six arms. See
    # config.multitask_seg for why reading `segmentation` here was the bug.
    default_arch, default_encoder = multitask_seg(cfg)
    arch = arch or default_arch
    encoder = encoder or default_encoder
    n_gpus = 1 if num_gpus is None else max(1, int(num_gpus))

    img_size = int(cfg["cls_image_size"])
    ref, train_ds, val_ds, test_ds, train_counts = _part_cls_splits(cfg, img_size)

    global_bs = auto_batch_size(
        device, mib_per_cls_sample(img_size), cfg["classification"]["batch_size"]
    )
    bs = per_device_batch_size(global_bs, n_gpus)
    loader_kw = _loader_kwargs(cfg)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kw)

    model = MultiTaskBeeModel(
        arch=arch,
        encoder_name=encoder,
        num_parts=ref.num_parts,
        num_classes=len(ref.classes),
        label_smoothing_value=label_smoothing(cfg),
        learning_rate=cfg["classification"]["lr"],
    )
    model.heavy_aug = _make_heavy_aug(cfg, len(ref.classes))
    n_params = sum(p.numel() for p in model.parameters())
    # Same formula as config.multitask_run_tag, which every reader of this
    # directory uses; kept here as an f-string because --arch/--encoder can
    # override the configured pair for a one-off run.
    safe_tag = f"{arch}_{encoder}".replace("/", "_")
    out_dir = ensure_dir(
        stage_run_dir(cfg, "stage_b", f"multitask_{safe_tag}_seed{run_seed}")
    )
    # Early stopping on valid_loss with the classification patience, matching
    # the EarlyStopper the three hand-rolled classification loops use. This was
    # `early_stop=False` back when no Stage B arm stopped early; leaving it off
    # now would make multitask the one arm still burning its full budget.
    trainer, ckpt_cb = make_seg_trainer(
        out_dir,
        epochs=cfg["classification"]["epochs"],
        patience=cls_patience(cfg),
        device=device,
        num_gpus=n_gpus,
        monitor="valid_loss",
        mode="min",
        early_stop=cls_patience(cfg) > 0,
    )
    print(
        f"multitask dataset={cfg['dataset']} device={device} num_gpus={n_gpus} "
        f"global_batch_size={global_bs} per_device_batch_size={bs} seed={run_seed}"
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    if ckpt_cb.best_model_path:
        model = MultiTaskBeeModel.load_from_checkpoint(ckpt_cb.best_model_path, map_location=device_t)
    model.to(device_t).eval()

    ys, preds, probas = [], [], []
    with torch.no_grad():
        for imgs, _, y in test_loader:
            _, cls_logits = model(imgs.to(device_t))
            ys.append(y.numpy())
            preds.append(cls_logits.argmax(1).cpu().numpy())
            probas.append(torch.softmax(cls_logits, 1).cpu().numpy())
    y_true, y_pred, y_proba = map(np.concatenate, (ys, preds, probas))
    report = classification_report_dict(
        y_true, y_pred, y_proba, labels=range(len(ref.classes))
    )
    report["long_tail"] = long_tail_bin_metrics(
        y_true, y_pred, train_counts, cfg["metrics"]["long_tail_bins"]
    )
    report.update(
        {
            # On a large_cls dataset the auxiliary segmentation loss is
            # supervised by the confidence-gated pseudo-masks baked into
            # LargeClsPartsDataset, not by real annotations, and the row has
            # to say so or the provenance check downstream cannot see it.
            # This used to test `layout == "fish_vista"`, from when Fish-Vista
            # was the only large_cls dataset and Beemachine/CUB really were
            # supervised by annotations. Both have since moved to large_cls,
            # so their multitask rows were recording "n/a" for supervision
            # that is in fact pseudo-mask -- the exact mislabel the original
            # comment was written to prevent.
            "mode": "multitask", "group": "n/a",
            "mask_source": "pseudo" if cls_corpus(cfg) == "large_cls" else "n/a",
            # `backbone` is the VISUAL backbone -- the trunk the top-1 is
            # credited to -- because that is what every downstream grouping
            # means by it: `stage_report.ablation_table` groups on
            # (mode, group, mask_source, backbone) and pairs an arm only
            # against a reference sharing its backbone, and the paper's
            # comparison tables print this column as "Backbone". It used to
            # hold `safe_tag` ("deeplabv3plus_resnext50_32x4d"), which was
            # honest about the old encoder but put this arm in a backbone
            # class of its own. Now that the encoder IS the common backbone,
            # naming it as such is what makes the grid a controlled
            # comparison. The decoder pair is not lost: it stays in `arch` /
            # `encoder` here and in the run directory's own name.
            "backbone": encoder.removeprefix("tu-"),
            "arch": arch, "encoder": encoder,
            "seed": run_seed, "n_params": int(n_params),
            "backbone_passes_per_image": 1,
            # Lightning counts from 0 and reports the epoch it stopped ON, so
            # +1 makes this the same "epochs actually run" the other arms record.
            "epochs_ran": int(trainer.current_epoch) + 1,
            "early_stopped": int(trainer.current_epoch) + 1 < cfg["classification"]["epochs"],
        }
    )
    _finalize_run(
        cfg,
        out_dir,
        report,
        ckpt_path=Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None,
        extra_meta={"arch": arch, "encoder": encoder},
    )


def cmd_sweep_multitask(
    cfg, arch: str | None, encoder: str | None, device: int, seeds: list[int] | None = None,
):
    """multitask across seeds — parallel across GPUs."""
    base_args = []
    if arch:
        base_args += ["--arch", arch]
    if encoder:
        base_args += ["--encoder", encoder]
    _seed_sweep(cfg, "multitask", base_args, seeds)


@torch.no_grad()
def cmd_extract_desc(
    cfg,
    source: str,
    seg_ckpt: str | None,
    device: int,
    limit: int,
    shard_id: int = 0,
    num_shards: int = 1,
    confidence_weighted: bool | None = None,
):
    """Extract phi_s/phi_q/phi_c (+ learned IQA) for every image.

    ``confidence_weighted`` (falling back to ``descriptors.confidence_weighted``
    in config when not given on the CLI) applies `confidence_weight_vector`
    to every row before it's written — RQ4's "does confidence-weighted
    descriptor aggregation help relative to unweighted concatenation?"
    ablation (see docs/research_plan.md). Confidence is the segmenter's mean
    foreground softmax probability for `source='pred'` (identical formula to
    `stage_c.py`'s pseudo-label confidence), or 1.0 for `source='gt'` (ground
    truth is fully trusted, so weighting is a no-op there). Comparing the two
    aggregations means running this twice (flag off, then on) and pointing
    `ablate --desc_csv` at each resulting CSV in turn -- the weighting is
    baked into the descriptor values, so no downstream code needs to know
    which one it's reading.
    """
    seed_everything(cfg["seed"])
    use_conf_weight = (
        bool(cfg.get("descriptors", {}).get("confidence_weighted", False))
        if confidence_weighted is None
        else bool(confidence_weighted)
    )
    # Learned NR-IQA defaults to CPU; on this node that is ~11x slower per
    # image than an A40 and leaves every GPU idle. Bind it to this shard's
    # device, then load every configured metric up front so a broken one
    # aborts here instead of writing an all-NaN column.
    iqa_names = list(cfg.get("descriptors", {}).get("learned_iqa_metrics") or [])
    if iqa_names:
        set_iqa_device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        validate_iqa_metrics(iqa_names)
    if cls_corpus(cfg) == "large_cls":
        raise SystemExit(
            f"{cfg['dataset']} trains classification on the large corpus "
            "(config datasets.*.cls_corpus: large_cls), whose descriptors come "
            "from `stage_c.py extract_desc` over the pseudo-masked corpus. "
            "Running stage_b extract_desc here would write a CSV covering the "
            "part set, which no arm of this dataset reads."
        )

    # Extract over exactly the folds Stage B will index -- including the 6x
    # train-aug fold when the dataset uses it. Anything narrower produces a CSV
    # that does not cover the training split, and the descriptor join then has
    # to either drop most of the training set or fail; see
    # `_join_descriptor_indices`.
    entry = cfg.get("dataset_entry") or {}
    use_aug = bool(entry.get("use_train_aug", True))
    _, train_ds, val_ds, test_ds = part_train_val_test(
        cfg, cfg["image_size"], use_train_aug=use_aug
    )
    from torch.utils.data import ConcatDataset

    ds = ConcatDataset([train_ds, val_ds, test_ds])
    # Flatten names/species for ConcatDataset access. Folds arrive wrapped
    # in Subset, and Subset forwards no attributes -- read them through
    # fold_names so the lists stay aligned with `ds`.
    image_names, species_names = [], []
    for fold in (train_ds, val_ds, test_ds):
        fold_species, fold_images = fold_names(fold)
        species_names += fold_species
        image_names += fold_images
    print(
        f"descriptor extraction over {len(image_names):,} images "
        f"(train_aug={'on' if use_aug else 'off'}: "
        f"{len(fold_names(train_ds)[1]):,} train / {len(fold_names(val_ds)[1]):,} val / "
        f"{len(fold_names(test_ds)[1]):,} test)"
    )

    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    seg = None
    if source == "pred":
        seg = load_segmenter(resolve_seg_ckpt(cfg, seg_ckpt), device_t)

    records = []
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    if limit <= 0 or limit >= len(ds):
        indices = list(range(len(ds)))
    else:
        # Same folds as `ds` above, augmentation included -- a `limit` that
        # sampled from the unaugmented folds could not index into `ds`.
        fold_train, fold_val, fold_test = train_ds, val_ds, test_ds

        def _fold_image_names(fold_ds):
            # Distinct name: a local `fold_names` would shadow the shared
            # data.fold_names imported above for this whole function.
            return fold_names(fold_ds)[1]

        name_to_index = {name: i for i, name in enumerate(image_names)}
        selected_names = []
        per_fold = max(1, limit // 3)
        for fold_ds in (fold_train, fold_val, fold_test):
            names = _fold_image_names(fold_ds)
            take = min(per_fold, len(names))
            positions = np.linspace(0, len(names) - 1, num=take, dtype=int)
            selected_names.extend(names[i] for i in positions)
        indices = [name_to_index[name] for name in selected_names if name in name_to_index]
        if len(indices) < n:
            selected = set(indices)
            indices.extend(i for i in range(len(ds)) if i not in selected)
        indices = indices[:n]
    # This shard's slice of the work. Sharding is by stride so every shard
    # sees a comparable mix of the (sorted) corpus rather than one contiguous
    # block of a single species.
    if num_shards > 1:
        indices = indices[shard_id::num_shards]

    iqa_bs = int(cfg.get("descriptors", {}).get("iqa_batch_size", 16))
    iqa_names = list(cfg.get("descriptors", {}).get("learned_iqa_metrics") or [])

    for start in tqdm(
        range(0, len(indices), iqa_bs),
        desc=f"descriptors-{source}[{shard_id}/{num_shards}]",
    ):
        chunk = indices[start : start + iqa_bs]
        imgs, masks, metas = [], [], []
        for i in chunk:
            item = ds[i]
            if len(item) == 4:
                img, mask, class_id, name = item
            else:
                img, mask, class_id = item
                name = image_names[i]
            imgs.append(img)
            masks.append(mask)
            metas.append((name, int(class_id), species_names[i]))

        confs: list[float] | None = None
        if source == "pred":
            # One segmenter forward for the whole chunk instead of per image.
            batch = torch.stack(imgs).to(device_t)
            with torch.no_grad():
                logits = seg(batch)
                pred = logits.argmax(1)
                if use_conf_weight:
                    # Same formula as stage_c.py's pseudo-label confidence:
                    # mean max-softmax probability over foreground pixels (or
                    # the whole image when nothing was predicted foreground).
                    probs = torch.softmax(logits, dim=1)
                    confs = []
                    for k in range(len(chunk)):
                        p = probs[k]
                        fg = pred[k] > 0
                        conf_vals = p.max(0).values
                        confs.append(float(conf_vals[fg].mean() if fg.any() else conf_vals.mean()))
            mask_nps = [pred[k].cpu().numpy() for k in range(len(chunk))]
        else:
            mask_nps = [m.numpy() for m in masks]
            if use_conf_weight:
                confs = [1.0] * len(chunk)  # ground truth is fully trusted

        # Learned NR-IQA for the whole chunk in one forward per metric.
        iqa_batch = np.stack([normalize_image_for_iqa(im) for im in imgs])
        iqa_rows = extract_learned_iqa_features_batch(iqa_batch, names=iqa_names)

        for k, (name, class_id, species) in enumerate(metas):
            feats = extract_all_features(
                imgs[k],
                mask_nps[k],
                part_labels=cfg["part_labels"],
                learned_iqa=iqa_rows[k],
            )
            if use_conf_weight:
                keys = list(feats.keys())
                vec = np.asarray([feats[key] for key in keys], dtype=np.float64)
                weighted = confidence_weight_vector(vec, confs[k], feature_names=keys)
                feats = dict(zip(keys, weighted.tolist()))
            rec = {"image": name, "class_id": class_id, "species": species}
            rec.update(feats)
            records.append(rec)

    out_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_b_descriptors")
    weight_tag = "_confweighted" if use_conf_weight else ""
    suffix = f"_shard{shard_id}" if num_shards > 1 else ""
    out_csv = out_dir / f"descriptors_{source}{weight_tag}{suffix}.csv"
    frame = pd.DataFrame(records)
    frame.to_csv(out_csv, index=False)
    if num_shards == 1:
        report_dead_descriptor_columns(frame, strict=_strict_descriptors(cfg))
    exp = int(cfg.get("descriptors", {}).get("expected_dim") or 0)
    if exp > 0 and records:
        feat_dim = len(records[0]) - 3
        if feat_dim != exp:
            print(f"WARNING: descriptor dim={feat_dim} vs expected_dim={exp}")
    print(f"Saved {len(records)} rows → {out_csv}")


def _merge_descriptor_shards(
    cfg, source: str, num_shards: int, confidence_weighted: bool = False
) -> Path:
    """Concatenate per-shard CSVs into the single file downstream steps read."""
    out_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_b_descriptors")
    weight_tag = "_confweighted" if confidence_weighted else ""
    parts = []
    missing = []
    for s in range(num_shards):
        p = out_dir / f"descriptors_{source}{weight_tag}_shard{s}.csv"
        if not p.exists():
            missing.append(str(p))
            continue
        # An empty shard is legitimate -- it just drew no images (small corpus,
        # or `--limit` trimmed its slice). See data.read_shard_csv.
        df = read_shard_csv(p)
        if df is not None and not df.empty:
            parts.append(df)
    if missing:
        raise SystemExit(
            f"Descriptor shard(s) missing, refusing to write a partial "
            f"descriptors_{source}{weight_tag}.csv:\n- " + "\n- ".join(missing)
        )
    if not parts:
        raise SystemExit(
            f"All {num_shards} descriptor shards were empty; refusing to write an "
            f"empty descriptors_{source}{weight_tag}.csv."
        )
    merged = pd.concat(parts, ignore_index=True).drop_duplicates("image", keep="last")
    merged = merged.sort_values("image").reset_index(drop=True)
    out_csv = out_dir / f"descriptors_{source}{weight_tag}.csv"
    merged.to_csv(out_csv, index=False)
    # Written before the check so a long extraction is never thrown away.
    report_dead_descriptor_columns(merged, strict=_strict_descriptors(cfg))
    for s in range(num_shards):
        (out_dir / f"descriptors_{source}{weight_tag}_shard{s}.csv").unlink(missing_ok=True)
    print(f"Merged {num_shards} shards → {out_csv} ({len(merged)} rows)")
    return out_csv


def cmd_extract_desc_all_gpus(
    cfg, source: str, seg_ckpt: str | None, limit: int, confidence_weighted: bool | None = None
):
    """Fan descriptor extraction out over every GPU, then merge.

    Descriptor extraction is the most expensive serial step in the pipeline
    (MANIQA's 20-crop protocol on GPU, plus CPU-bound SIFT/ORB/Zernike/BRISQUE),
    and it runs six times in the full protocol. Measured s/image/GPU scales with
    part count, not just image count: 1.48 (beemachine, 4 parts), 2.59
    (fish_vista, 10), 2.97 (cub, 12). Spread over 8 A40s that is 12-33 min per
    dataset per mask source instead of hours on one device.
    """
    import sys

    n_gpus = resolve_num_gpus(cfg)
    if n_gpus <= 1:
        return cmd_extract_desc(cfg, source, seg_ckpt, 0, limit, confidence_weighted=confidence_weighted)

    use_conf_weight = (
        bool(cfg.get("descriptors", {}).get("confidence_weighted", False))
        if confidence_weighted is None
        else bool(confidence_weighted)
    )
    config_path = _config_path_from_argv(default_config_path())
    script = str(Path(__file__).resolve())
    cmds = []
    for shard in range(n_gpus):
        cmd = [
            sys.executable, script, "extract_desc",
            "--config", config_path,
            "--dataset", cfg["dataset"],
            "--source", source,
            "--shard_id", str(shard),
            "--num_shards", str(n_gpus),
            "--confidence_weighted" if use_conf_weight else "--no-confidence_weighted",
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        if seg_ckpt:
            cmd += ["--seg_ckpt", seg_ckpt]
        cmds.append(cmd)

    print(f"Stage B extract_desc: {n_gpus} shards across {n_gpus} GPU(s)")
    codes = run_commands_parallel(cmds, n_gpus, cwd=str(Path(script).parent))
    if any(c != 0 for c in codes):
        raise SystemExit(
            f"{sum(c != 0 for c in codes)} descriptor shard(s) failed; not merging."
        )
    _merge_descriptor_shards(cfg, source, n_gpus, confidence_weighted=use_conf_weight)


FUSION_MODES = (
    "backbone_only",
    "descriptors_only",
    "concat",
    "gated_residual",
    "attention_parts",
)


def cmd_fusion(
    cfg,
    mode: str,
    backbone: str,
    desc_csv: str,
    group: str,
    mask_source: str,
    seg_ckpt: str | None,
    device: int,
    seed: int | None = None,
):
    """Train and evaluate one (fusion arm x mask source x descriptor group)
    cell of the comparison grid. All arms share this same loop, optimizer,
    epoch budget, and evaluation code — they differ only in `mode`, i.e. in
    how visual evidence (and optionally descriptors) is combined before the
    classification head. `seed` lets the caller repeat a cell under several
    seeds for the confidence-interval / significance analysis in Stage E.
    """
    run_seed = int(seed) if seed is not None else int(cfg["seed"])
    seed_everything(run_seed)
    if mode not in FUSION_MODES:
        raise SystemExit(f"Unsupported fusion mode: {mode}")
    if mask_source not in {"gt", "pred"}:
        raise SystemExit("--mask_source must be gt or pred")
    mask_source_label = _fish_mask_source_label(cfg, mask_source)
    ref, train_ds, val_ds, test_ds, train_counts = _part_cls_splits(
        cfg, cls_image_size_for(cfg, backbone)
    )
    # Fusion ImageDescDataset needs a base dataset with image_names + indices
    if isinstance(train_ds, Subset):
        ds = ref
        train_idx, val_idx, test_idx = (
            list(train_ds.indices),
            list(val_ds.indices),
            list(test_ds.indices),
        )
    else:
        # Fish: build a concatenated indexable dataset for descriptor join
        from torch.utils.data import ConcatDataset

        ds = ConcatDataset([train_ds, val_ds, test_ds])
        # Patch attributes expected by ImageDescDataset
        _img_names, _sp_names = [], []
        for fold in (train_ds, val_ds, test_ds):
            fold_species, fold_images = fold_names(fold)
            _sp_names += fold_species
            _img_names += fold_images
        ds.image_names = _img_names  # type: ignore[attr-defined]
        ds.species_names = _sp_names  # type: ignore[attr-defined]
        # `_pred_mask_ram_cache` sizes the predicted-mask RAM cache from
        # `ds.image_size`, falling back to the segmentation resolution
        # (`part_image_size`) when that attribute is absent -- which it
        # always was here, since `ConcatDataset` forwards no attributes from
        # the folds beneath it. On Fish-Vista that fallback is 320 while
        # classification runs at 224 (cls_image_size_for), so the cache was
        # built at the wrong resolution for every mask_source="pred" fusion
        # arm: the disk fallback in `_read_pred_mask` reads each PNG at its
        # correct native size, so this only broke the RAM-cached path, which
        # is why it surfaced as a hard crash in `_crop_part` (attention_parts)
        # and as silently-misaligned masks feeding the other fusion arms
        # rather than a symptom present under mask_source="gt" too. Same bug,
        # same fix, as the one already handled for Masked/PartCropDataset --
        # those receive a `Subset`-derived single dataset with `.image_size`
        # already set, which is why only this ConcatDataset branch needed it.
        ds.image_size = cls_image_size_for(cfg, backbone)  # type: ignore[attr-defined]
        n_tr, n_va, n_te = len(train_ds), len(val_ds), len(test_ds)
        train_idx = list(range(0, n_tr))
        val_idx = list(range(n_tr, n_tr + n_va))
        test_idx = list(range(n_tr + n_va, n_tr + n_va + n_te))
        # ImageDescDataset indexes into ds[i] which works for ConcatDataset

    desc_df = pd.read_csv(desc_csv)
    meta_cols = {"image", "class_id", "species"}
    feat_cols = [c for c in desc_df.columns if c not in meta_cols]
    gmask = descriptor_group_mask(feat_cols, group)
    # Fit the z-scoring statistics on the training fold only, then share them
    # across train/val/test. Without this the raw `*_area` columns dominate the
    # head's LayerNorm and the backbone half of the fused vector is discarded --
    # see `fit_descriptor_standardizer` for the measurement.
    desc_stats = fit_descriptor_standardizer(
        desc_df, feat_cols, train_images=[ds.image_names[i] for i in train_idx]
    )

    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = PartAwareFusionClassifier(
        num_classes=len(ref.classes),
        backbone_name=backbone,
        descriptor_dim=len(feat_cols),
        fusion_mode=mode,
        shape_embed_dim=shape_embed_dim(cfg),
        use_descriptors=mode != "backbone_only",
        n_parts=ref.num_parts,
        image_size=cls_image_size_for(cfg, backbone),
    ).to(device_t)

    seg = None
    pred_cache = None
    if mask_source == "pred":
        resolved = resolve_seg_ckpt(cfg, seg_ckpt)
        seg = load_segmenter(resolved, device_t)
        for p in seg.parameters():
            p.requires_grad = False
        pred_cache = _prediction_cache_dir(cfg, resolved)
        _ensure_pred_mask_cache(
            cfg,
            ds,
            sorted(set(train_idx + val_idx + test_idx)),
            seg,
            device_t,
            pred_cache,
        )
        del seg
        seg = None

    is_attn = mode == "attention_parts"
    _img_size_fu = cls_image_size_for(cfg, backbone)
    if is_attn:
        part_ids = foreground_part_ids(cfg["part_labels"])
        n_streams = len(part_ids) + 1
        bs = max(8, auto_batch_size(
            device, mib_per_cls_sample(_img_size_fu, n_streams), cfg["classification"]["batch_size"]
        ))
        mk = lambda idxs: PartCropDescDataset(
            ds, idxs, mask_source, part_ids, desc_df, feat_cols, gmask,
            pred_cache=pred_cache, cfg=cfg, desc_stats=desc_stats,
        )
    else:
        bs = auto_batch_size(
            device, mib_per_cls_sample(_img_size_fu), cfg["classification"]["batch_size"]
        )
        mk = lambda idxs: ImageDescDataset(
            ds, idxs, desc_df, feat_cols, gmask, pred_cache=pred_cache, cfg=cfg,
            desc_stats=desc_stats,
        )
    train_data, val_data, test_data = mk(train_idx), mk(val_idx), mk(test_idx)
    if not train_data or not val_data or not test_data:
        raise SystemExit(
            "Descriptor CSV does not overlap every frozen split "
            f"(train={len(train_data)}, val={len(val_data)}, test={len(test_data)})"
        )

    def _build_fusion_loaders(bs_: int):
        return (
            DataLoader(train_data, batch_size=bs_, shuffle=True, **_loader_kwargs(cfg)),
            DataLoader(val_data, batch_size=bs_, shuffle=False, **_loader_kwargs(cfg)),
        )

    train_loader, val_loader = _build_fusion_loaders(bs)

    criterion = make_cls_criterion(cfg)
    heavy_aug = _make_heavy_aug(cfg, len(ref.classes))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["classification"]["lr"])
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    safe_bb = backbone.replace("/", "_")
    tag = f"{mode}_{mask_source_label}_{group}_{safe_bb}_seed{run_seed}"
    out_dir = ensure_dir(stage_run_dir(cfg, "stage_b_descriptors", tag))
    best = float("inf")
    total_epochs = cfg["classification"]["epochs"]
    stopper = EarlyStopper(cls_patience(cfg))
    epochs_ran = 0
    epoch = 0
    while epoch < total_epochs:
        tr_desc = f"[{tag}] epoch {epoch + 1}/{total_epochs} train"
        va_desc = f"[{tag}] epoch {epoch + 1}/{total_epochs} val"
        oom_hit = False
        try:
            if is_attn:
                tr_loss, tr_acc = _run_attn_epoch(
                    model, train_loader, criterion, device_t, opt, desc=tr_desc,
                    heavy_aug=heavy_aug,
                )
                va_loss, va_acc = _run_attn_epoch(model, val_loader, criterion, device_t, None, desc=va_desc)
            else:
                tr_loss, tr_acc = _run_fusion_epoch(
                    model, train_loader, criterion, device_t, opt, seg, ref.num_parts, cfg["image_size"],
                    desc=tr_desc, heavy_aug=heavy_aug,
                )
                va_loss, va_acc = _run_fusion_epoch(
                    model, val_loader, criterion, device_t, None, seg, ref.num_parts, cfg["image_size"],
                    desc=va_desc,
                )
        except torch.cuda.OutOfMemoryError:
            oom_hit = True
        if oom_hit:
            # The up-front `auto_batch_size` estimate is a flat per-sample
            # multiplier that does not know about this model's actual shape
            # (attention_parts forwards n_streams crops through a fusion head
            # whose memory does not scale purely linearly) and cannot see
            # allocator fragmentation building up over many epochs. Rather
            # than trust the offline estimate to be right, halve the batch on
            # a real OOM and retry the same epoch -- self-correcting on the
            # actual GPU instead of crashing the whole job.
            #
            # The cleanup must run *outside* the `except` block: Python keeps
            # an exception's traceback (and every tensor its frames
            # reference, e.g. `logits`) alive while the exception is
            # "currently being handled", so `empty_cache()` called from
            # inside `except` frees nothing and every retry OOMs again at a
            # smaller and smaller free amount -- confirmed by instrumenting a
            # synthetic OOM. `gc.collect()` is also needed, since the
            # autograd graph's reference cycle needs a cyclic-GC pass, not
            # just refcounting, to actually drop.
            if bs <= 1:
                raise RuntimeError(f"[{tag}] CUDA out of memory even at batch_size=1")
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            print(f"[{tag}] CUDA OOM; retrying epoch {epoch + 1} at batch_size={bs}", flush=True)
            train_loader, val_loader = _build_fusion_loaders(bs)
            continue
        sch.step(va_loss)
        print(f"[{tag}] epoch {epoch + 1}: train={tr_acc:.4f} val={va_acc:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
        epochs_ran = epoch + 1
        if stopper.step(va_loss, epoch):
            print(f"[{tag}] early stop: {stopper.summary(total_epochs)}")
            break
        epoch += 1

    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, **_loader_kwargs(cfg))

    model.load_state_dict(
        torch.load(out_dir / "best.pt", map_location=device_t, weights_only=True)
    )
    if is_attn:
        report = _eval_attn(model, test_loader, device_t, train_counts, cfg["metrics"]["long_tail_bins"])
    else:
        report = _eval_fusion(
            model,
            test_loader,
            device_t,
            train_counts,
            cfg["metrics"]["long_tail_bins"],
            seg,
            ref.num_parts,
            cfg["image_size"],
        )
    n_params = sum(p.numel() for p in model.parameters())
    n_streams = (len(foreground_part_ids(cfg["part_labels"])) + 1) if is_attn else 1
    report.update(
        {
            "mode": mode,
            "group": group,
            "backbone": backbone,
            "mask_source": mask_source_label,
            "seed": run_seed,
            "n_params": int(n_params),
            "backbone_passes_per_image": n_streams,
            "epochs_ran": epochs_ran,
            "early_stopped": stopper.stopped_epoch is not None,
            "best_epoch": stopper.best_epoch + 1,
            "best_val_loss": round(stopper.best, 6),
            # Provenance: rows without this flag predate the descriptor
            # z-scoring fix and are not comparable with rows that carry it.
            "desc_standardized": True,
        }
    )
    # `ablate` harvests these per-cell test_metrics.json files and merges the
    # summary rows itself, so this arm must not also append them.
    return _finalize_run(
        cfg,
        out_dir,
        report,
        ckpt_path=out_dir / "best.pt",
        extra_meta={"batch_size": bs},
        append_summary=False,
    )


def cmd_ablate(
    cfg,
    desc_csv: str,
    backbone: str,
    mask_source: str,
    device: int,
    seg_ckpt: str | None,
    seeds: list[int] | None = None,
):
    """Run the full fusion-strategy comparison grid in parallel: one
    (mode, group, seed) job per GPU. This is the core comparison the study is
    built around — every arm shares this loop and differs only in `mode`.
    Multiple seeds populate the confidence-interval / significance analysis
    consumed by `stage_report.py ablation_table`.
    """
    import sys

    mask_source_label = _fish_mask_source_label(cfg, mask_source)

    n_gpus = resolve_num_gpus(cfg)
    config_path = _config_path_from_argv(default_config_path())
    script = str(Path(__file__).resolve())
    modes = ["descriptors_only", "concat", "gated_residual", "attention_parts"]
    if cfg.get("layout") == "fish_vista":
        # Matching the prior work's own scope rather than this repo's full grid: it
        # never sub-ablates a descriptor group (shape/appearance/interpart)
        # on Fish-Vista, only the complete per-part descriptor vector ("all").
        # The classification corpus is ~14x the part-set (60k vs 4.3k train
        # images), so repeating the 3-group sub-ablation here would be the
        # single most expensive step in the whole pipeline for a comparison
        # the prior work's own experiments never ran either.
        groups = ["all"]
    else:
        groups = ["all", "shape", "appearance", "interpart"]
    # Modes that run only on the complete descriptor vector instead of the full
    # per-group sub-ablation. attention_parts costs k+1 backbone passes per
    # image, so sweeping it across every descriptor group was ~64% of this
    # step's GPU time (12 of 48 jobs, ~137 of ~215 GPU-hours on Beemachine) --
    # spent sub-ablating an arm the inference-cost analysis already rules out
    # of deployment. It keeps its "all" cell, so it still appears in the
    # mechanism comparison and in the aggregation-operator contrast against
    # part-crop late fusion; only the per-group breakdown is dropped, and that
    # breakdown is retained in full for the single-pass arms that are actual
    # deployment candidates.
    all_group_only = set(
        (cfg.get("descriptors") or {}).get("all_group_only_modes") or []
    )
    seeds = list(seeds) if seeds else list(cfg.get("seeds") or [cfg["seed"]])
    cmds = []
    per_mode_groups: dict[str, list[str]] = {}
    for mode in modes:
        if mode in all_group_only and "all" in groups:
            per_mode_groups[mode] = ["all"]
        else:
            per_mode_groups[mode] = list(groups)
    for mode in modes:
        for group in per_mode_groups[mode]:
            for seed in seeds:
                # `dataset_cli_args`, not a bare `--dataset`: it carries the
                # ACTIVE PROTOCOL too. This is the same defect `_seed_cmds`
                # had, and it is worse here, because a fusion child that loses
                # the protocol writes to the DEFAULT grid's untagged run
                # directory -- so `ablate --protocol heavy_aug` trained the
                # default recipe and overwrote the published default-recipe
                # cells while reporting a heavy_aug sweep.
                cmd = [
                    sys.executable,
                    script,
                    "fusion",
                    "--config",
                    config_path,
                    *dataset_cli_args(cfg),
                    "--mode",
                    mode,
                    "--group",
                    group,
                    "--backbone",
                    backbone,
                    "--desc_csv",
                    desc_csv,
                    "--mask_source",
                    mask_source,
                    "--seed",
                    str(seed),
                ]
                if seg_ckpt:
                    cmd += ["--seg_ckpt", seg_ckpt]
                cmds.append(cmd)
    restricted = [m for m in modes if len(per_mode_groups[m]) < len(groups)]
    shape = " + ".join(
        f"{m}x{len(per_mode_groups[m])}g" for m in modes
    )
    print(
        f"Stage B ablate: {len(cmds)} jobs "
        f"({len(modes)} modes x up to {len(groups)} groups x {len(seeds)} seeds; "
        f"{shape}) on {n_gpus} GPU(s)"
    )
    if restricted:
        print(
            f"[ablate] all-group-only (descriptors.all_group_only_modes): "
            f"{', '.join(restricted)} -- per-group sub-ablation skipped for these"
        )
    codes = run_commands_parallel(cmds, n_gpus, cwd=str(Path(script).parent))
    if any(c != 0 for c in codes):
        # Hard failure, not a warning. The harvest loop below skips any cell
        # whose test_metrics.json is missing, so a failed cell simply vanished
        # from the comparison grid: the summary CSV, the resume marker and the
        # published table all came out looking complete while describing fewer
        # cells than the grid claims. A comparison with silently absent cells
        # is not a comparison.
        failed = [_describe_job(cmd) for cmd, c in zip(cmds, codes) if c != 0]
        shown = "\n  ".join(failed[:10])
        more = f"\n  ... and {len(failed) - 10} more" if len(failed) > 10 else ""
        raise SystemExit(
            f"{len(failed)} of {len(codes)} fusion cell(s) failed:\n  {shown}{more}\n"
            "The grid is incomplete; not merging the summary. Fix the cause and "
            "re-run this step -- finished cells are overwritten in place, so a "
            "re-run is safe and only costs the cells that had already passed."
        )

    out_root = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_b_descriptors")
    rows = []
    safe_bb = backbone.replace("/", "_")
    for mode in modes:
        for group in groups:
            for seed in seeds:
                tag = f"{mode}_{mask_source_label}_{group}_{safe_bb}_seed{seed}"
                # `stage_run_dir`, not `out_root / tag`: under a non-default
                # protocol the jobs above write tagged directories, so a raw
                # join finds nothing and the summary silently reports zero
                # runs for a sweep that trained every cell.
                mp = stage_run_dir(cfg, "stage_b_descriptors", tag) / "test_metrics.json"
                if not mp.exists():
                    continue
                report = json.loads(mp.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "mode": mode,
                        "group": group,
                        "backbone": backbone,
                        "mask_source": mask_source_label,
                        "seed": seed,
                        "top1": report.get("top1"),
                        "top3": report.get("top3"),
                        "macro_f1": report.get("macro_f1"),
                        "n_params": report.get("n_params"),
                        "backbone_passes_per_image": report.get("backbone_passes_per_image"),
                        # cmd_fusion already writes these into test_metrics.json;
                        # dropping them here left the four fusion arms as the only
                        # arms in the summary CSV with no training-curve columns --
                        # exactly the arms most likely to fail, and exactly the
                        # signal ("ran the full budget, no early stop, val loss
                        # stuck near ln(num_classes)") that diagnosed the
                        # 2026-08-11 descriptor blow-up in the first place.
                        "epochs_ran": report.get("epochs_ran"),
                        "early_stopped": report.get("early_stopped"),
                        "best_epoch": report.get("best_epoch"),
                        "best_val_loss": report.get("best_val_loss"),
                        "desc_standardized": report.get("desc_standardized"),
                        "code_version": report.get("code_version"),
                        "source_digest": report.get("source_digest"),
                    }
                )
    # Merge (not overwrite): baseline/capacity_matched/heavy_aug/masked/partcrop
    # seed sweeps write their own rows into this same file, and re-running
    # ablate for a different --backbone or --mask_source must not erase them
    # (or an earlier ablate call's rows for a different cell).
    out = _append_summary_rows(cfg, rows) if rows else out_root / "fusion_ablation_summary.csv"
    print(f"Summary ({len(rows)} runs) → {out}")


def main():
    ap = argparse.ArgumentParser(description="Stage B / B+ — classification + descriptors")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_stage_parser(name: str):
        parser = sub.add_parser(name)
        add_global_stage_args(parser)
        return parser

    def _add_seed_arg(parser, help="Override cfg['seed'] for this run."):
        parser.add_argument("--seed", type=int, default=None, help=help)

    def _add_seeds_arg(parser):
        parser.add_argument(
            "--seeds", type=int, nargs="+", default=None,
            help="Seeds to repeat this arm under (default: cfg['seeds']).",
        )

    p = add_stage_parser("baseline")
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--device", type=int, default=0)
    _add_seed_arg(p)

    p = add_stage_parser("sweep_baselines")
    p.add_argument("--device", type=int, default=0)
    _add_seeds_arg(p)

    p = add_stage_parser("capacity_matched")
    p.add_argument("--backbone", default="convnext_small.in12k", help="A larger backbone than the standard baseline, for parameter-matched comparison.")
    p.add_argument("--device", type=int, default=0)
    _add_seed_arg(p)

    p = add_stage_parser("sweep_capacity_matched")
    p.add_argument("--backbone", default="convnext_small.in12k")
    p.add_argument("--device", type=int, default=0)
    _add_seeds_arg(p)

    p = add_stage_parser("heavy_aug")
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--device", type=int, default=0)
    _add_seed_arg(p)

    p = add_stage_parser("sweep_heavy_aug")
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--device", type=int, default=0)
    _add_seeds_arg(p)

    p = add_stage_parser("masked")
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--mask_source", choices=["gt", "pred"], default="gt")
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--device", type=int, default=0)
    _add_seed_arg(p)

    p = add_stage_parser("sweep_masked")
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--mask_source", choices=["gt", "pred"], default="gt")
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--device", type=int, default=0)
    _add_seeds_arg(p)

    # All seven seed-swept arms in one submission -- see cmd_sweep_controls.
    p = add_stage_parser("sweep_controls")
    p.add_argument("--backbone", default=COMMON_BACKBONE,
                   help="Common comparison backbone for heavy_aug/masked/partcrop.")
    p.add_argument("--capacity_backbone", default="convnext_small.in12k",
                   help="Larger backbone for the capacity-matched control.")
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--arch", default=None,
                   help="Auxiliary segmentation decoder for the multitask arm "
                        "(default: classification.multitask.arch).")
    p.add_argument("--encoder", default=None,
                   help="Shared encoder for the multitask arm, i.e. its visual "
                        "backbone (default: classification.multitask.encoder, "
                        "the grid's common backbone). Overriding this takes the "
                        "arm off the common backbone and reintroduces the "
                        "confound the config default exists to remove.")
    p.add_argument("--device", type=int, default=0)
    _add_seeds_arg(p)

    p = add_stage_parser("partcrop")
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--mask_source", choices=["gt", "pred"], default="gt")
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--device", type=int, default=0)
    _add_seed_arg(p)

    p = add_stage_parser("sweep_partcrop")
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--mask_source", choices=["gt", "pred"], default="gt")
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--device", type=int, default=0)
    _add_seeds_arg(p)

    p = add_stage_parser("multitask")
    p.add_argument("--arch", default=None)
    p.add_argument("--encoder", default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=None)
    _add_seed_arg(p)

    p = add_stage_parser("sweep_multitask")
    p.add_argument("--arch", default=None)
    p.add_argument("--encoder", default=None)
    p.add_argument("--device", type=int, default=0)
    _add_seeds_arg(p)

    p = add_stage_parser("extract_desc")
    p.add_argument("--source", choices=["gt", "pred"], default="gt")
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument(
        "--all_gpus",
        action="store_true",
        help="Shard extraction across compute.num_gpus GPUs, then merge (recommended).",
    )
    p.add_argument(
        "--confidence_weighted",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="RQ4 ablation: scale every descriptor row by the segmenter's "
        "foreground confidence (1.0, i.e. a no-op, for --source gt). Default: "
        "descriptors.confidence_weighted in config. Writes to a "
        "descriptors_{source}_confweighted.csv sibling file, so a plain and a "
        "weighted extraction never clobber each other -- run this flag off "
        "then on and point `ablate --desc_csv` at each in turn to compare.",
    )

    p = add_stage_parser("fusion")
    p.add_argument(
        "--mode",
        default="gated_residual",
        choices=list(FUSION_MODES),
        help="Which fusion arm to train: backbone_only/descriptors_only are "
        "reference points; masked/partcrop/multitask are separate subcommands; "
        "concat, gated_residual, and attention_parts are the deep-fusion arms.",
    )
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--desc_csv", required=True)
    p.add_argument("--group", default="all", choices=["all", "shape", "appearance", "interpart"])
    p.add_argument("--mask_source", default="gt", choices=["gt", "pred"])
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=None, help="Override cfg['seed'] for this run.")

    p = add_stage_parser("ablate")
    p.add_argument("--desc_csv", required=True)
    p.add_argument("--backbone", default=COMMON_BACKBONE)
    p.add_argument("--mask_source", default="gt", choices=["gt", "pred"])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seg_ckpt", default="")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to repeat every (mode, group) cell under (default: cfg['seeds']).",
    )

    args = ap.parse_args()
    cfg = load_config(args.config, dataset=args.dataset, protocol=getattr(args, "protocol", None))

    if args.cmd == "baseline":
        cmd_baseline(cfg, args.backbone, args.device, seed=args.seed)
    elif args.cmd == "sweep_baselines":
        cmd_sweep_baselines(cfg, args.device, seeds=args.seeds)
    elif args.cmd == "masked":
        cmd_masked(cfg, args.backbone, args.mask_source, args.seg_ckpt, args.device, seed=args.seed)
    elif args.cmd == "sweep_masked":
        cmd_sweep_masked(
            cfg, args.mask_source, args.seg_ckpt, args.device,
            seeds=args.seeds, backbone=args.backbone,
        )
    elif args.cmd == "partcrop":
        cmd_partcrop(cfg, args.backbone, args.mask_source, args.seg_ckpt, args.device, seed=args.seed)
    elif args.cmd == "sweep_partcrop":
        cmd_sweep_partcrop(
            cfg, args.mask_source, args.seg_ckpt, args.device,
            seeds=args.seeds, backbone=args.backbone,
        )
    elif args.cmd == "multitask":
        cmd_multitask(cfg, args.arch, args.encoder, args.device, num_gpus=args.num_gpus, seed=args.seed)
    elif args.cmd == "sweep_multitask":
        cmd_sweep_multitask(cfg, args.arch, args.encoder, args.device, seeds=args.seeds)
    elif args.cmd == "sweep_controls":
        cmd_sweep_controls(
            cfg, args.backbone, args.capacity_backbone, args.seg_ckpt, args.device,
            arch=args.arch, encoder=args.encoder, seeds=args.seeds,
        )
    elif args.cmd == "extract_desc":
        if args.all_gpus:
            cmd_extract_desc_all_gpus(
                cfg, args.source, args.seg_ckpt, args.limit,
                confidence_weighted=args.confidence_weighted,
            )
        else:
            cmd_extract_desc(
                cfg,
                args.source,
                args.seg_ckpt,
                args.device,
                args.limit,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
                confidence_weighted=args.confidence_weighted,
            )
    elif args.cmd == "fusion":
        cmd_fusion(
            cfg,
            args.mode,
            args.backbone,
            args.desc_csv,
            args.group,
            args.mask_source,
            args.seg_ckpt,
            args.device,
            seed=args.seed,
        )
    elif args.cmd == "ablate":
        cmd_ablate(
            cfg,
            args.desc_csv,
            args.backbone,
            args.mask_source,
            args.device,
            args.seg_ckpt or None,
            seeds=args.seeds,
        )
    elif args.cmd == "capacity_matched":
        cmd_capacity_matched(cfg, args.backbone, args.device, seed=args.seed)
    elif args.cmd == "sweep_capacity_matched":
        cmd_sweep_capacity_matched(cfg, args.backbone, args.device, seeds=args.seeds)
    elif args.cmd == "heavy_aug":
        cmd_heavy_aug(cfg, args.backbone, args.device, seed=args.seed)
    elif args.cmd == "sweep_heavy_aug":
        cmd_sweep_heavy_aug(cfg, args.backbone, args.device, seeds=args.seeds)


if __name__ == "__main__":
    main()
