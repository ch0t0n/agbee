#!/usr/bin/env python3
"""Stage D CLI: long_tail | confusion | calibration | selective | robustness | demo."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    add_global_stage_args,
    cls_image_size_for,
    foreground_part_ids,
    load_config,
    resolve_seg_ckpt,
    shape_embed_dim,
    stage_run_dir,
)
from distributed_utils import auto_batch_size, mib_per_inf_sample
from fusion import PartAwareFusionClassifier, TimmClassifier
from stage_b import PartCropDataset
from metrics import (
    CORRUPTION_KINDS,
    aurc,
    corrupt_batch,
    expected_calibration_error,
    long_tail_bin_metrics,
    reliability_by_frequency_bin,
    selective_prediction_curve,
)
from segmenters import load_segmenter

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Part overlay colors — values must stay in sync with stage_a._PALETTE[1:].
COLORS = {
    1: (220, 60, 60),
    2: (60, 140, 255),
    3: (60, 200, 100),
    4: (255, 180, 40),
    5: (180, 60, 200),
    6: (40, 200, 200),
    7: (200, 100, 40),
    8: (100, 100, 255),
    9: (255, 100, 180),
    10: (140, 220, 80),
    11: (80, 80, 160),
    12: (200, 200, 60),
    13: (160, 80, 80),
    14: (80, 160, 120),
    15: (120, 120, 120),
}


def _resolve_bins(cfg, cli_bins):
    """CLI edges > config `metrics.long_tail_bins`; `quantile` passes through.

    argparse gives us strings so that the single token `quantile` remains
    expressible alongside integer edges.
    """
    value = cli_bins if cli_bins else cfg["metrics"]["long_tail_bins"]
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and value[0] == "quantile":
        return "quantile"
    return [int(v) for v in value]


def cmd_long_tail(pred_csv: str, bins: list[int], out: str):
    df = pd.read_csv(pred_csv)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    train_counts = {}
    if "class_id" in df.columns and "train_count" in df.columns:
        train_counts = df.groupby("y_true")["train_count"].first().astype(int).to_dict()
    elif "train_count" in df.columns:
        train_counts = dict(zip(df["y_true"], df["train_count"]))
    report = long_tail_bin_metrics(y_true, y_pred, train_counts, bins)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cmd_confusion(pred_csv: str, top_k: int, out: str):
    df = pd.read_csv(pred_csv)
    if "species_true" in df.columns:
        a, b = df["species_true"], df["species_pred"]
    else:
        a, b = df["y_true"], df["y_pred"]
    misses = [(t, p) for t, p in zip(a, b) if t != p]
    counts = Counter(misses).most_common(top_k)
    out_df = pd.DataFrame([{"true": t, "pred": p, "count": c} for (t, p), c in counts])
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))


def cmd_calibration(pred_csv: str, out: str):
    df = pd.read_csv(pred_csv)
    conf = df["confidence"].to_numpy(dtype=float)
    correct = (df["y_true"] == df["y_pred"]).to_numpy(dtype=float)
    val, rows = expected_calibration_error(conf, correct)
    report = {"ece": val, "bins": rows}
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ECE={val:.4f} → {out_path}")

    if plt is not None and rows:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot([r["conf"] for r in rows], [r["acc"] for r in rows], marker="o")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability (ECE={val:.3f})")
        fig.savefig(out_path.with_suffix(".png"), dpi=140)
        plt.close(fig)


def cmd_selective(pred_csv: str, out: str, coverages: list[float] | None = None):
    """Risk-coverage / abstention analysis for one arm's test predictions.

    Complements `calibration`: ECE asks whether confidence numbers are
    well-calibrated in absolute terms, this asks the more deployment-relevant
    question of "if this arm abstains on its least-confident k%, how much
    does error drop" — the AURC summary lets arms be ranked on selective
    prediction quality, not only on unconditional accuracy.
    """
    df = pd.read_csv(pred_csv)
    conf = df["confidence"].to_numpy(dtype=float)
    correct = (df["y_true"] == df["y_pred"]).to_numpy(dtype=float)
    rows = selective_prediction_curve(conf, correct, coverages=coverages)
    report = {"aurc": aurc(rows), "curve": rows}
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"AURC={report['aurc']:.4f} → {out_path}")

    if plt is not None and rows:
        fig, ax = plt.subplots()
        ax.plot([r["coverage"] for r in rows], [r["risk"] for r in rows], marker="o")
        ax.set_xlabel("coverage (fraction accepted)")
        ax.set_ylabel("risk (error rate among accepted)")
        ax.set_title(f"Risk-coverage (AURC={report['aurc']:.4f})")
        fig.savefig(out_path.with_suffix(".png"), dpi=140)
        plt.close(fig)


#: Arms whose checkpoint is a plain whole-image classifier. `masked` differs
#: from `whole` only in what reaches the model -- the body-masked image rather
#: than the raw photograph -- so it needs no model code of its own, only the
#: matching input transform in `_ArmEvaluator.forward`.
_WHOLE_LIKE_MODES = ("whole", "masked")

#: Arms that consume no descriptor vector, so `--desc_csv` is not read for them.
#: `masked`, `partcrop` and `multitask` join this set alongside the reference:
#: all three take anatomy as pixels or as a training signal, never as phi.
_NO_DESCRIPTOR_MODES = ("whole", "backbone_only", "masked", "partcrop", "multitask")

#: Arms that need a mask at inference. Under `--mask_source pred` these re-run
#: the frozen segmenter, so their reported robustness includes their own
#: upstream segmentation noise. `multitask` is absent on purpose: its anatomy
#: signal is training-only and it takes a bare photograph at inference.
_MASK_CONSUMING_MODES = ("gated_residual", "attention_parts", "masked", "partcrop")


def _load_multitask(cfg, ckpt: str, device_t):
    """Restore the multi-task arm from its Lightning checkpoint.

    This arm is the one that does not write a bare ``best.pt``: it trains under
    Lightning and leaves a ``checkpoints/*.ckpt`` whose ``state_dict`` is
    prefixed and whose hyperparameters are stored alongside.

    Rebuilt from ``hyper_parameters``, never from the caller's ``--backbone``.
    The arm now shares the grid's common encoder (``tu-convnext_nano.in12k``,
    the same ConvNeXt-Nano weights every other arm loads through timm), but
    ``--backbone`` names a plain timm classifier, whereas this checkpoint holds
    an encoder-decoder plus two heads. Only the checkpoint's own hyperparameters
    say which decoder wraps that encoder, and a checkpoint trained before the
    encoder was pinned carries the old ResNeXt-50 pair -- so this stays the
    single source of truth rather than a CLI default.
    """
    from stage_b import MultiTaskBeeModel

    payload = torch.load(ckpt, map_location=device_t, weights_only=False)
    hparams = dict(payload.get("hyper_parameters") or {})
    state = payload.get("state_dict") or payload
    if not hparams:
        raise SystemExit(
            f"{ckpt} carries no hyper_parameters; cannot rebuild the multi-task "
            "encoder/decoder without guessing its architecture."
        )
    model = MultiTaskBeeModel(**hparams)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Buffers (mean/std) are registered, so a clean load leaves both empty.
    # Anything else means the checkpoint and this class have drifted apart, and
    # a partially-initialised encoder would score as a plausible-looking but
    # meaningless arm rather than crashing.
    real_missing = [k for k in missing if not k.endswith(("mean", "std"))]
    if real_missing or unexpected:
        raise SystemExit(
            f"Multi-task checkpoint does not match MultiTaskBeeModel: "
            f"{len(real_missing)} missing, {len(unexpected)} unexpected "
            f"(first missing: {real_missing[:3]}, first unexpected: {list(unexpected)[:3]})"
        )
    num_classes = int(model.cls_head[-1].out_features)
    return model.to(device_t).eval(), num_classes


def _load_arm(cfg, ckpt: str, backbone: str, mode: str, descriptor_dim: int, device_t):
    """Rebuild any comparison arm from its checkpoint.

    Covers all seven comparison arms plus the two controls:

    * ``whole`` / ``masked`` / the two controls -- a plain ``TimmClassifier``.
    * ``partcrop`` -- the ``LateFusionModel`` that concatenates K+1 crop
      features.
    * ``multitask`` -- the Lightning ``MultiTaskBeeModel`` (see above).
    * every fusion mode -- a ``PartAwareFusionClassifier``.

    ``partcrop`` and ``multitask`` used to fall through to the
    ``PartAwareFusionClassifier`` branch, where they raised ``ValueError`` on an
    unknown fusion mode. That failure was at least loud; the point of adding
    them properly is that reliability and robustness could previously only be
    reported for four of the seven arms.
    """
    if mode == "multitask":
        return _load_multitask(cfg, ckpt, device_t)

    state = torch.load(ckpt, map_location=device_t, weights_only=True)
    num_classes = None
    for k, v in reversed(list(state.items())):
        if k.endswith("weight") and v.ndim == 2 and v.shape[0] < 5000:
            num_classes = v.shape[0]
            break
    if num_classes is None:
        raise SystemExit("Could not infer num_classes from checkpoint")

    if mode in _WHOLE_LIKE_MODES:
        model = TimmClassifier(backbone, pretrained=False, num_classes=num_classes)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            model.model.load_state_dict(state, strict=False)
    elif mode == "partcrop":
        from stage_b import LateFusionModel

        # n_streams = K foreground parts + the whole-body crop, which is what
        # PartCropDataset stacks and therefore what the trained head's input
        # width was sized against.
        n_streams = len(foreground_part_ids(cfg["part_labels"])) + 1
        model = LateFusionModel(
            backbone,
            num_classes=num_classes,
            n_streams=n_streams,
            image_size=cls_image_size_for(cfg, backbone),
        )
        model.load_state_dict(state)
    else:
        model = PartAwareFusionClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            descriptor_dim=descriptor_dim,
            fusion_mode=mode,
            shape_embed_dim=shape_embed_dim(cfg),
            pretrained=False,
            use_descriptors=mode != "backbone_only",
            n_parts=len(cfg["part_labels"]),
        )
        model.load_state_dict(state, strict=False)
    return model.to(device_t).eval(), num_classes


class _ArmEvaluator:
    """Everything needed to score one trained comparison arm on its test fold.

    Shared by ``cmd_robustness`` and ``cmd_dump_preds`` rather than duplicated:
    the descriptor join, the descriptor *scale*, the mask source, and the
    per-mode forward signature all have to match how the arm was trained, and
    every one of those has already been the subject of a correctness bug. Two
    copies of this logic is how they drift apart.
    """

    def __init__(
        self,
        cfg,
        ckpt: str,
        backbone: str,
        mode: str,
        desc_csv: str,
        device_t,
        mask_source: str = "gt",
        seg_ckpt: str | None = None,
    ):
        from descriptors import fit_descriptor_standardizer, standardize_descriptor_vector
        from stage_b import _part_cls_splits

        self.cfg = cfg
        self.mode = mode
        self.mask_source = mask_source
        self.device_t = device_t
        self.part_ids = foreground_part_ids(cfg["part_labels"])

        # One `_part_cls_splits` call for both the descriptor statistics and the
        # test loader. Calling it twice would rebuild the dataset -- and on
        # Fish-Vista rebuild LargeClsPartsDataset's whole pseudo-mask cache.
        # Evaluate at the backbone's configured resolution so each arm is
        # corrupted and scored at the size it is actually trained on.
        # `_part_cls_splits` (not `part_train_val_test` directly) so Fish-Vista
        # evaluates against the same classification-corpus test split every
        # Stage B arm trained on -- see its docstring for why Fish-Vista's
        # segmentation part-set cannot serve as a classification test set.
        ref, train_ds, _, test_ds, train_counts = _part_cls_splits(
            cfg, cls_image_size_for(cfg, backbone)
        )
        self.ref = ref
        self.train_counts = train_counts

        self.desc_dim = 0
        self.desc_by_image: dict[str, np.ndarray] = {}
        if mode not in _NO_DESCRIPTOR_MODES and desc_csv:
            desc_df = pd.read_csv(desc_csv)
            feat_cols = [
                c for c in desc_df.columns if c not in {"image", "class_id", "species"}
            ]
            self.desc_dim = len(feat_cols)
            # Descriptors must reach the model on exactly the scale it was
            # trained on. Stage B z-scores them against training-fold statistics
            # (see descriptors.fit_descriptor_standardizer); feeding the raw
            # columns here would present a `full_area` of ~5e4 to a head whose
            # weights were fit against unit-variance input -- a worse
            # train/evaluation mismatch than the zero-vector bug fixed below.
            # Statistics come from the *training* fold, so the transform is
            # identical to fit time and no test data informs it.
            train_names = (
                [str(ref.image_names[i]) for i in train_ds.indices]
                if isinstance(train_ds, Subset)
                else [str(n) for n in train_ds.image_names]
            )
            desc_stats = fit_descriptor_standardizer(
                desc_df, feat_cols, train_images=train_names
            )
            # Look the real descriptors up per image. This table was built and
            # then never read: every batch was scored with a zero vector, so a
            # gated or concat arm was measured with its entire descriptor branch
            # switched off -- a model that was never trained, whose corruption
            # robustness says nothing about the arm the grid actually reports.
            values = desc_df[feat_cols].to_numpy(dtype=np.float32)
            self.desc_by_image = {
                str(name): standardize_descriptor_vector(values[i], desc_stats)
                for i, name in enumerate(desc_df["image"].astype(str))
            }

        self.model, _ = _load_arm(cfg, ckpt, backbone, mode, self.desc_dim, device_t)

        # Masks feed both gated_residual (shape channel) and attention_parts (crop
        # geometry). A checkpoint trained with --mask_source pred should be
        # re-scored against the *same frozen segmenter's* predictions here, not
        # silently against ground truth -- GT masks are always perfect and would
        # under-report how much the arm's own upstream segmentation noise hurts
        # it under corruption.
        self.seg = None
        if mode in _MASK_CONSUMING_MODES and mask_source == "pred":
            if not seg_ckpt:
                raise SystemExit("--seg_ckpt is required when --mask_source pred")
            self.seg = load_segmenter(seg_ckpt, device_t)

        _dev_idx = (device_t.index or 0) if device_t.type == "cuda" else 0
        _inf_bs = auto_batch_size(
            _dev_idx, mib_per_inf_sample(cls_image_size_for(cfg, backbone)),
            cfg["classification"]["batch_size"],
        )
        self.loader = DataLoader(
            test_ds,
            batch_size=_inf_bs,
            shuffle=False,
            num_workers=int(cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )
        # Test-fold image names in loader order (shuffle=False), so each batch
        # can be joined to its descriptor row by position.
        if isinstance(test_ds, Subset):
            self.test_names = [str(test_ds.dataset.image_names[i]) for i in test_ds.indices]
        else:
            self.test_names = [str(n) for n in test_ds.image_names]

    def descriptors_for(self, start: int, count: int):
        """(count, desc_dim) real descriptors for this batch, zeros if absent."""
        if not self.desc_dim:
            return None
        z = np.zeros((count, self.desc_dim), dtype=np.float32)
        for j in range(count):
            row = self.desc_by_image.get(self.test_names[start + j])
            if row is not None:
                z[j] = row
        return torch.from_numpy(z).to(self.device_t)

    def _resolve_masks(self, imgs, masks_gt):
        if self.mask_source == "gt" or self.seg is None:
            return masks_gt
        seg_size = int(self.cfg["image_size"])
        original_size = imgs.shape[-2:]
        inp = F.interpolate(
            imgs, size=(seg_size, seg_size), mode="bilinear", align_corners=False
        )
        pred = self.seg(inp).argmax(1, keepdim=True).float()
        return F.interpolate(pred, size=original_size, mode="nearest").squeeze(1).long()

    def _build_part_crops(self, imgs, masks):
        """(B, K+1, C, H, W): same crop stack PartCropDataset builds at train time."""
        per_image = []
        for b in range(imgs.shape[0]):
            crops = [
                PartCropDataset._crop_part(imgs[b], masks[b] == pid) for pid in self.part_ids
            ]
            crops.append(PartCropDataset._crop_part(imgs[b], masks[b] > 0))
            per_image.append(torch.stack(crops, dim=0))
        return torch.stack(per_image, dim=0)

    def forward(self, imgs, masks, z_p):
        if self.mode in ("whole", "backbone_only"):
            return self.model(imgs)
        if self.mode == "masked":
            # Exactly MaskedBeeDataset.__getitem__: the image times its binary
            # foreground. Scoring this arm on raw photographs would evaluate it
            # on an input distribution it never saw in training.
            masks = self._resolve_masks(imgs, masks)
            return self.model(imgs * (masks > 0).float().unsqueeze(1))
        if self.mode == "partcrop":
            masks = self._resolve_masks(imgs, masks)
            return self.model(self._build_part_crops(imgs, masks))
        if self.mode == "multitask":
            # Training-only anatomy: the auxiliary segmentation head shapes the
            # encoder but is not part of inference, so only the class logits
            # are scored -- the same Psi(I) = z_b the paper reports for it.
            return self.model(imgs)[1]
        if self.mode == "gated_residual":
            masks = self._resolve_masks(imgs, masks)
            b, h, w = masks.shape
            ml = torch.zeros(b, self.ref.num_parts, h, w, device=imgs.device)
            ml.scatter_(1, masks.unsqueeze(1), 1.0)
            return self.model(x=imgs, z_p=z_p, mask_logits=ml)
        if self.mode == "attention_parts":
            masks = self._resolve_masks(imgs, masks)
            crops = self._build_part_crops(imgs, masks)
            return self.model(part_crops=crops, z_p=z_p)
        return self.model(x=imgs, z_p=z_p)


def cmd_dump_preds(
    cfg,
    ckpt: str,
    backbone: str,
    mode: str,
    desc_csv: str,
    device: int,
    out: str,
    mask_source: str = "gt",
    seg_ckpt: str | None = None,
):
    """Per-image test predictions for one arm: the missing input to
    ``calibration`` / ``selective`` / ``reliability_bins`` / ``long_tail`` /
    ``confusion``.

    All five of those subcommands require a ``--pred_csv`` carrying
    ``y_true,y_pred,confidence`` (plus ``train_count`` for the frequency-stratified
    ones), and nothing in the pipeline emitted such a file -- which is why the
    calibration/selective/reliability table has never had data behind it. This
    closes that gap using the same ``_ArmEvaluator`` the robustness sweep uses,
    so an arm's predictions here are produced exactly as they are scored there.

    Writes one row per test image on clean (uncorrupted) input.
    """
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    ev = _ArmEvaluator(
        cfg, ckpt, backbone, mode, desc_csv, device_t, mask_source, seg_ckpt
    )

    classes = list(getattr(ev.ref, "classes", []) or [])
    rows: list[dict] = []
    offset = 0
    with torch.no_grad():
        for batch in ev.loader:
            imgs = batch[0].to(device_t)
            masks = batch[1].to(device_t) if len(batch) > 2 else None
            y = batch[-1]
            z_p = ev.descriptors_for(offset, imgs.size(0))
            prob = torch.softmax(ev.forward(imgs, masks, z_p), 1)
            conf, pred = prob.max(1)
            for j in range(imgs.size(0)):
                y_true = int(y[j])
                y_pred = int(pred[j].item())
                row = {
                    "image": ev.test_names[offset + j],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "confidence": float(conf[j].item()),
                    # long_tail / reliability_bins stratify by how often the
                    # true class was seen in training.
                    "train_count": int(ev.train_counts.get(y_true, 0)),
                }
                if classes:
                    row["species_true"] = classes[y_true] if y_true < len(classes) else ""
                    row["species_pred"] = classes[y_pred] if y_pred < len(classes) else ""
                rows.append(row)
            offset += imgs.size(0)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    acc = float((df["y_true"] == df["y_pred"]).mean()) if len(df) else float("nan")
    print(f"[dump_preds] {mode}: {len(df)} rows, top1={acc:.4f} → {out_path}")
    return out_path


@torch.no_grad()
def cmd_robustness(
    cfg,
    ckpt: str,
    backbone: str,
    mode: str,
    desc_csv: str,
    device: int,
    out: str,
    mask_source: str = "gt",
    seg_ckpt: str | None = None,
):
    """Accuracy AND reliability under synthetic corruption, for any arm.

    Two things are measured at every corruption/severity, not one:

    1. top-1, i.e. does the arm still classify correctly under shift; and
    2. ECE + AURC, i.e. does the arm still *know when it is wrong* under
       shift. The second is the operationally decisive one for a triage
       deployment -- an arm whose accuracy degrades gracefully but whose
       confidence stays high is more dangerous than one that degrades
       further but flags its own failures, because the first silently
       routes bad predictions past the reviewer.

    This directly answers the reviewer question of whether anatomy-guided
    arms abstain more reliably on corrupted / out-of-distribution input,
    which accuracy-only robustness reporting cannot address.
    """
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    ev = _ArmEvaluator(
        cfg, ckpt, backbone, mode, desc_csv, device_t, mask_source, seg_ckpt
    )
    loader, forward, descriptors_for = ev.loader, ev.forward, ev.descriptors_for

    rows = []
    kinds = cfg.get("robustness", {}).get("kinds") or list(CORRUPTION_KINDS)
    for kind in ["clean"] + list(kinds):
        severities = [0] if kind == "clean" else list(cfg.get("robustness", {}).get("severities", [1, 3, 5]))
        for sev in severities:
            confs, corrects = [], []
            offset = 0
            for batch in loader:
                imgs = batch[0].to(device_t)
                masks = batch[1].to(device_t) if len(batch) > 2 else None
                y = batch[-1]
                if kind != "clean":
                    imgs = corrupt_batch(imgs, kind, severity=sev)
                z_p = descriptors_for(offset, imgs.size(0))
                offset += imgs.size(0)
                logits = forward(imgs, masks, z_p)
                prob = torch.softmax(logits, 1)
                conf, pred = prob.max(1)
                confs.append(conf.cpu().numpy())
                corrects.append((pred.cpu() == y).numpy().astype(float))
            conf_a = np.concatenate(confs)
            corr_a = np.concatenate(corrects)
            ece_val, _ = expected_calibration_error(conf_a, corr_a, n_bins=cfg["metrics"].get("ece_bins", 15))
            rc = selective_prediction_curve(
                conf_a, corr_a, coverages=cfg["metrics"].get("selective_coverages")
            )
            rows.append(
                {
                    "arm": mode,
                    "corruption": kind,
                    "severity": sev,
                    "top1": float(corr_a.mean()),
                    "ece": ece_val,
                    "aurc": aurc(rc),
                    "n": int(len(corr_a)),
                }
            )
            print(
                f"[robustness] {mode} {kind} sev={sev}: "
                f"top1={corr_a.mean():.4f} ece={ece_val:.4f} aurc={aurc(rc):.4f}"
            )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def cmd_reliability_bins(pred_csv: str, bins: list[int], out: str):
    """ECE and AURC stratified by training-frequency bin.

    Overall calibration is dominated by common species; this reports whether
    an arm is also calibrated on the rare classes where deferral matters
    most, which is where a global ECE number can be quietly misleading.
    """
    df = pd.read_csv(pred_csv)
    conf = df["confidence"].to_numpy(dtype=float)
    correct = (df["y_true"] == df["y_pred"]).to_numpy(dtype=float)
    y_true = df["y_true"].to_numpy()
    if "train_count" in df.columns:
        train_counts = df.groupby("y_true")["train_count"].first().astype(int).to_dict()
    else:
        raise SystemExit("pred_csv needs a train_count column for frequency binning")
    report = reliability_by_frequency_bin(conf, correct, y_true, train_counts, bins)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _overlay_parts(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for k, c in COLORS.items():
        m = mask == k
        out[m] = (1 - alpha) * out[m] + alpha * np.array(c, dtype=np.float32)
    return out.astype(np.uint8)


def cmd_demo(cfg, seg_ckpt: str | None, cls_ckpt: str, backbone: str, port: int):
    ckpt = resolve_seg_ckpt(cfg, seg_ckpt)
    try:
        import gradio as gr
    except ImportError as e:
        raise SystemExit(f"Install gradio for the demo: {e}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seg = load_segmenter(ckpt, device)

    cls = None
    if cls_ckpt:
        state = torch.load(cls_ckpt, map_location=device, weights_only=True)
        num_classes = None
        for k, v in reversed(list(state.items())):
            if k.endswith("weight") and v.ndim == 2 and v.shape[0] < 5000:
                num_classes = v.shape[0]
                break
        if num_classes is None:
            raise SystemExit("Could not infer num_classes from cls_ckpt")
        cls = TimmClassifier(backbone, pretrained=False, num_classes=num_classes)
        try:
            cls.load_state_dict(state)
        except RuntimeError:
            # Compatibility with checkpoints produced before normalization was
            # wrapped with the timm model.
            cls.model.load_state_dict(state, strict=False)
        cls.to(device).eval()

    tf_seg = transforms.Compose(
        [
            transforms.Resize(
                (cfg["image_size"], cfg["image_size"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    tf_cls = transforms.Compose(
        [
            transforms.Resize(
                (cfg["cls_image_size"], cfg["cls_image_size"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )

    @torch.no_grad()
    def predict(pil_img: Image.Image):
        rgb = np.asarray(pil_img.convert("RGB").resize((cfg["image_size"], cfg["image_size"])))
        x = tf_seg(pil_img.convert("RGB")).unsqueeze(0).to(device)
        mask = seg(x).argmax(1)[0].cpu().numpy()
        vis = _overlay_parts(rgb, mask)
        label = "segmentation only"
        if cls is not None:
            xc = tf_cls(pil_img.convert("RGB")).unsqueeze(0).to(device)
            logits = cls(xc)
            prob = torch.softmax(logits, 1)[0]
            conf, pred = prob.max(0)
            label = f"class_id={int(pred)}  conf={float(conf):.3f}"
        return Image.fromarray(vis), label

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Image(type="pil", label="parts"), gr.Textbox(label="prediction")],
        title=f"{cfg['dataset']} anatomy-guided identification demo",
        description="Image → part overlay → species (if classifier provided)",
    )
    demo.launch(server_port=port)


def main():
    ap = argparse.ArgumentParser(description="Stage D — analysis + demo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("long_tail")
    add_global_stage_args(p)
    p.add_argument("--pred_csv", required=True)
    p.add_argument(
        "--bins",
        nargs="+",
        default=None,
        help="Frequency-bin edges in training images per class, or the "
        "single token `quantile` to derive them from the corpus. "
        "Default: config metrics.long_tail_bins.",
    )
    p.add_argument("--out", default=None)

    p = sub.add_parser("confusion")
    add_global_stage_args(p)
    p.add_argument("--pred_csv", required=True)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--out", default=None)

    p = sub.add_parser("calibration")
    add_global_stage_args(p)
    p.add_argument("--pred_csv", required=True, help="needs y_true,y_pred,confidence")
    p.add_argument("--out", default=None)

    p = sub.add_parser("selective")
    add_global_stage_args(p)
    p.add_argument("--pred_csv", required=True, help="needs y_true,y_pred,confidence")
    p.add_argument("--out", default=None)

    p = sub.add_parser("robustness")
    add_global_stage_args(p)
    p.add_argument("--ckpt", required=True, help="Checkpoint for the arm being evaluated")
    p.add_argument("--backbone", default="convnext_nano.in12k")
    p.add_argument(
        "--mode",
        default="whole",
        choices=["whole", "backbone_only", "masked", "partcrop", "multitask",
                 "concat", "gated_residual", "attention_parts"],
        help="Which comparison arm this checkpoint belongs to.",
    )
    p.add_argument("--desc_csv", default="", help="Descriptor CSV (needed for descriptor-consuming arms)")
    p.add_argument(
        "--mask_source",
        default="gt",
        choices=["gt", "pred"],
        help="Masks for gated_residual/attention_parts. Use 'pred' with --seg_ckpt "
        "when the checkpoint was itself trained on predicted masks, so "
        "robustness is scored against the same upstream segmentation noise.",
    )
    p.add_argument("--seg_ckpt", default=None, help="Required when --mask_source pred")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", default=None)

    p = sub.add_parser(
        "dump_preds",
        help="Per-image test predictions for one arm -- the --pred_csv that "
        "calibration/selective/reliability_bins/long_tail/confusion consume.",
    )
    add_global_stage_args(p)
    p.add_argument("--ckpt", required=True, help="Checkpoint for the arm being evaluated")
    p.add_argument("--backbone", default="convnext_nano.in12k")
    p.add_argument(
        "--mode",
        default="whole",
        choices=["whole", "backbone_only", "masked", "partcrop", "multitask",
                 "concat", "gated_residual", "attention_parts"],
        help="Which comparison arm this checkpoint belongs to.",
    )
    p.add_argument("--desc_csv", default="", help="Descriptor CSV (needed for descriptor-consuming arms)")
    p.add_argument("--mask_source", default="gt", choices=["gt", "pred"])
    p.add_argument("--seg_ckpt", default=None, help="Required when --mask_source pred")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", default=None)

    p = sub.add_parser("reliability_bins")
    add_global_stage_args(p)
    p.add_argument("--pred_csv", required=True, help="needs y_true,y_pred,confidence,train_count")
    p.add_argument(
        "--bins",
        nargs="+",
        default=None,
        help="Frequency-bin edges in training images per class, or the "
        "single token `quantile` to derive them from the corpus. "
        "Default: config metrics.long_tail_bins.",
    )
    p.add_argument("--out", default=None)

    p = sub.add_parser("demo")
    add_global_stage_args(p)
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--cls_ckpt", default="", help="Optional timm classifier state_dict")
    p.add_argument("--backbone", default="convnext_nano.in12k")
    p.add_argument("--port", type=int, default=7860)

    args = ap.parse_args()
    cfg = load_config(args.config, dataset=args.dataset, protocol=getattr(args, "protocol", None))
    out_root = stage_run_dir(cfg, "stage_d")

    if args.cmd == "long_tail":
        cmd_long_tail(
            args.pred_csv,
            _resolve_bins(cfg, args.bins),
            args.out or str(out_root / "long_tail.json"),
        )
    elif args.cmd == "confusion":
        cmd_confusion(args.pred_csv, args.top_k, args.out or str(out_root / "confusion_pairs.csv"))
    elif args.cmd == "calibration":
        cmd_calibration(args.pred_csv, args.out or str(out_root / "calibration.json"))
    elif args.cmd == "selective":
        cmd_selective(
            args.pred_csv,
            args.out or str(out_root / "selective_prediction.json"),
            coverages=cfg["metrics"].get("selective_coverages"),
        )
    elif args.cmd == "robustness":
        cmd_robustness(
            cfg,
            args.ckpt,
            args.backbone,
            args.mode,
            args.desc_csv,
            args.device,
            args.out or str(out_root / f"robustness_{args.mode}.csv"),
            mask_source=args.mask_source,
            seg_ckpt=args.seg_ckpt,
        )
    elif args.cmd == "dump_preds":
        cmd_dump_preds(
            cfg,
            args.ckpt,
            args.backbone,
            args.mode,
            args.desc_csv,
            args.device,
            args.out or str(out_root / f"preds_{args.mode}.csv"),
            mask_source=args.mask_source,
            seg_ckpt=args.seg_ckpt,
        )
    elif args.cmd == "reliability_bins":
        cmd_reliability_bins(
            args.pred_csv,
            _resolve_bins(cfg, args.bins),
            args.out or str(out_root / "reliability_by_bin.json"),
        )
    elif args.cmd == "demo":
        cmd_demo(cfg, args.seg_ckpt, args.cls_ckpt, args.backbone, args.port)


if __name__ == "__main__":
    main()
