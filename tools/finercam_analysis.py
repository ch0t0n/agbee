#!/usr/bin/env python3
"""Qualitative Finer-CAM saliency comparison for the BeeMachine arms.

Standalone tool, not wired into run_all_experiments.sh. It answers a
different question than the accuracy/calibration grid: does each arm's
decision attend to the bee's anatomy, or to context?

Finer-CAM is due to Zhang et al., "Finer-CAM: Spotting the Difference Reveals
Finer Details for Visual Explanation", CVPR 2025 (arXiv:2501.11309), whose
official implementation is at https://github.com/Imageomics/Finer-CAM and
which has since been merged into `jacobgil/pytorch-grad-cam`
(https://github.com/jacobgil/pytorch-grad-cam), the `grad-cam` package on
PyPI. This script uses that merged implementation.

Method follows the reference qualitative analysis: the `pytorch_grad_cam`
package's `FinerCAM` wrapper over a GradCAM base method (Selvaraju et al.,
arXiv:1610.02391), hooked on each backbone's last spatial stage. Unlike a single fixed target class
(`ClassifierOutputTarget(true_label)`, ordinary Grad-CAM behaviour), this
script uses FinerCAM's distinguishing mode: `targets=None` with
`target_idx=true_label` auto-selects the classes closest in logit score to
the true class and highlights what visually separates the true class from
those look-alikes.

Ten arms, one CAM row each, on six fixed test-set indices (matching PADC's
own arbitrary-index convention: 0, 101, 245, 440, 550, 760 into the frozen
Beemachine test split). Two arms -- part-crop late fusion and
attention-pooled fusion -- do not take a single whole photograph as input;
they take K+1 crops (whole-body-masked view last, K part crops first, see
``stage_b.PartCropDataset``). For those two, the CAM shown is for the
K-th ("full-body") crop stream specifically, not literally the raw
photograph -- flagged in the row label and this docstring rather than
glossed over.

No result interpretation is written here or in the paper by design: the
point of this script is to produce the images for manual inspection, not to
pre-judge which arm's saliency looks more trustworthy.

Usage:
    ./run.sh tools/finercam_analysis.py --config config.yaml --dataset beemachine
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Subset

from pytorch_grad_cam import FinerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import (add_global_stage_args, cls_image_size_for, foreground_part_ids,
                    load_config, multitask_run_tag, shape_embed_dim)
from descriptors import fit_descriptor_standardizer, standardize_descriptor_vector
from fusion import PartAwareFusionClassifier, TimmClassifier
from stage_b import LateFusionModel, MultiTaskBeeModel, PartCropDataset, _part_cls_splits

EXAMPLE_INDICES = [0, 101, 245, 440, 550, 760]

#: Suffix `config.yaml` gives `protocols.heavy_aug.run_tag`. A protocol run's
#: directory is the default one with this inserted before the first dot of the
#: last path component -- the rule `config.stage_run_dir` applies when it writes
#: the run, mirrored here because this tool reads those directories back and the
#: two must not drift.
HAUG_TAG = "__haug"


def haug_dir(run_dir: str) -> str:
    """`run_dir` as the heavy-augmentation protocol writes it.

    Applied only after MULTITASK_TAG has been substituted: the dot the tag goes
    in front of comes from the encoder name that substitution supplies, so
    tagging the raw template would append to the end of the path instead.
    """
    head, _, name = run_dir.rpartition("/")
    stem, dot, ext = name.partition(".")
    return f"{head}/{stem}{HAUG_TAG}{dot}{ext}"


# (key, mode, backbone_or_arch, run_dir, ckpt_relpath, row_label, protocol)
# `key` names the output file and `mode` selects the loading path, so an arm can
# appear under two training recipes without either row overwriting the other's
# images or needing a second loader branch. `protocol` is "heavy_aug" for a run
# written under that protocol's tag, and "default" otherwise.
ARM_SPECS = [
    ("whole", "whole", "convnext_nano.in12k", "stage_b/baseline_convnext_nano.in12k_seed{seed}", "best.pt", "Whole-image reference\n(default recipe)", "default"),
    ("capacity_matched", "capacity_matched", "convnext_small.in12k", "stage_b/capacity_matched_convnext_small.in12k_seed{seed}", "best.pt", "Capacity-matched control", "default"),
    # The heavily augmented reference predates the seven-arm heavy-augmentation
    # grid: it was run as a control arm under the default protocol, so its
    # directory carries the mode name and NOT the `__haug` tag.
    ("heavy_aug", "heavy_aug", "convnext_nano.in12k", "stage_b/heavy_aug_convnext_nano.in12k_seed{seed}", "best.pt", "Whole-image reference\n(heavy augmentation)", "default"),
    ("masked", "masked", "convnext_nano.in12k", "stage_b/masked_pseudo_convnext_nano.in12k_seed{seed}", "best.pt", "Body-masked input", "default"),
    # Run directory resolved from config (classification.multitask), not spelled
    # out: this arm's directory name carries its arch/encoder pair, and that pair
    # changed when the arm was pinned to the grid's common backbone.
    ("multitask", "multitask", "MULTITASK_TAG", "stage_b/multitask_MULTITASK_TAG_seed{seed}", None, "Multi-task supervision\n(default recipe)", "default"),
    ("multitask_haug", "multitask", "MULTITASK_TAG", "stage_b/multitask_MULTITASK_TAG_seed{seed}", None, "Multi-task supervision\n(heavy augmentation)", "heavy_aug"),
    ("partcrop", "partcrop", "convnext_nano.in12k", "stage_b/partcrop_pseudo_convnext_nano.in12k_seed{seed}", "best.pt", "Part-crop late fusion\n(full-body stream)", "default"),
    ("attention_parts", "attention_parts", "convnext_nano.in12k", "stage_b_descriptors/attention_parts_pseudo_all_convnext_nano.in12k_seed{seed}", "best.pt", "Attention-pooled fusion\n(full-body stream)", "default"),
    ("concat", "concat", "convnext_nano.in12k", "stage_b_descriptors/concat_pseudo_all_convnext_nano.in12k_seed{seed}", "best.pt", "Descriptor concatenation", "default"),
    ("gated_residual", "gated_residual", "convnext_nano.in12k", "stage_b_descriptors/gated_residual_pseudo_all_convnext_nano.in12k_seed{seed}", "best.pt", "Gated residual fusion", "default"),
]


class ClsOnlyWrapper(nn.Module):
    """Reduce MultiTaskBeeModel's (seg_logits, cls_logits) forward to cls_logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[1]


class FixedSideInputWrapper(nn.Module):
    """Present a PartAwareFusionClassifier's multi-input forward as forward(x).

    z_p / mask_logits are the same for every call this wrapper makes (they
    describe one fixed example image), so they are closed over here rather
    than threaded through pytorch_grad_cam's single-tensor call convention.
    """

    def __init__(self, model, z_p=None, mask_logits=None, crops_mode=False):
        super().__init__()
        self.model = model
        self.z_p = z_p
        self.mask_logits = mask_logits
        self.crops_mode = crops_mode

    def forward(self, x):
        if self.crops_mode:
            # x arrives as (K+1, C, H, W) -- see the crops_mode branch in
            # main() for why -- but PartAwareFusionClassifier's part_crops
            # arg wants (B, K, C, H, W).
            return self.model(part_crops=x.unsqueeze(0), z_p=self.z_p)
        if self.mask_logits is not None:
            return self.model(x=x, z_p=self.z_p, mask_logits=self.mask_logits)
        return self.model(x=x, z_p=self.z_p)


class CropBatchWrapper(nn.Module):
    """LateFusionModel.forward wants (B, K, C, H, W); see FixedSideInputWrapper's
    crops_mode docstring for why the CAM-side tensor omits the batch dim."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.unsqueeze(0))


def _multitask_target_layer(model):
    """Deepest encoder stage of the multi-task arm, reached through ClsOnlyWrapper.

    The encoder is `smp`'s timm-universal wrapper around the same ConvNeXt-Nano
    every other arm uses, so its deepest stage is `model.stages_3` -- the
    counterpart of `_convnext_target_layer`'s `stages[-1]`, flattened by timm's
    `features_only` builder into a top-level attribute. The ResNeXt-50 path
    (`encoder.layer4[-1]`) is kept as a fallback so a checkpoint trained before
    the encoder was pinned still renders instead of raising.
    """
    enc = model.model.seg.encoder
    inner = getattr(enc, "model", None)
    if inner is not None:
        stages = [m for n, m in inner.named_children() if n.startswith("stages")]
        if stages:
            return [stages[-1]]
    if hasattr(enc, "layer4"):
        return [enc.layer4[-1]]
    raise SystemExit(
        f"Cannot locate a CAM target layer on multi-task encoder {type(enc).__name__}"
    )


def _convnext_target_layer(model_holder, attr_path: str):
    """attr_path is one of 'model.model' (TimmClassifier) or 'model.backbone'
    (PartAwareFusionClassifier) or 'model.backbone.model' (LateFusionModel)."""
    obj = model_holder
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return [obj.stages[-1]]


def load_arm_model(cfg, mode: str, backbone: str, run_dir: Path, ckpt_name, num_classes: int,
                    descriptor_dim: int, device_t):
    if mode == "multitask":
        ckpt_glob = list((run_dir / "checkpoints").glob("*.ckpt"))
        if not ckpt_glob:
            raise SystemExit(f"No multitask checkpoint under {run_dir/'checkpoints'}")
        model = MultiTaskBeeModel.load_from_checkpoint(str(ckpt_glob[0]), map_location=device_t)
        model.to(device_t).eval()
        return model, ClsOnlyWrapper(model), _multitask_target_layer, "cls_only"

    ckpt_path = run_dir / ckpt_name
    state = torch.load(ckpt_path, map_location=device_t, weights_only=True)

    if mode in ("whole", "capacity_matched", "heavy_aug", "masked"):
        model = TimmClassifier(backbone, num_classes=num_classes, pretrained=False)
        model.load_state_dict(state)
        model.to(device_t).eval()
        return model, model, lambda m: _convnext_target_layer(m, "model"), "whole"

    if mode == "partcrop":
        part_ids = foreground_part_ids(cfg["part_labels"])
        model = LateFusionModel(backbone, num_classes, n_streams=len(part_ids) + 1,
                                 image_size=cls_image_size_for(cfg, backbone))
        model.load_state_dict(state)
        model.to(device_t).eval()
        return model, model, lambda m: _convnext_target_layer(m, "backbone.model"), "crops"

    # concat / gated_residual / attention_parts
    model = PartAwareFusionClassifier(
        num_classes=num_classes, backbone_name=backbone, descriptor_dim=descriptor_dim,
        fusion_mode=mode, shape_embed_dim=shape_embed_dim(cfg), pretrained=False,
        use_descriptors=True, n_parts=len(cfg["part_labels"]),
        image_size=cls_image_size_for(cfg, backbone),
    )
    model.load_state_dict(state, strict=False)
    model.to(device_t).eval()
    kind = "crops" if mode == "attention_parts" else "whole"
    return model, model, lambda m: _convnext_target_layer(m, "backbone"), kind


def build_mask_logits(mask_ids: torch.Tensor, num_parts: int, device_t) -> torch.Tensor:
    """One-hot mask_logits, matching stage_b._fusion_mask_logits's seg=None branch
    (the branch large_cls training actually takes -- see the docstring)."""
    b = 1
    h, w = mask_ids.shape[-2:]
    ml = torch.zeros(b, num_parts, h, w, device=device_t)
    ml.scatter_(1, mask_ids.view(1, 1, h, w).to(device_t), 1.0)
    return ml


def run_cam(wrapper: nn.Module, target_layers, input_tensor: torch.Tensor, label: int,
            crops_mode: bool, crop_take_index: int | None):
    cam = FinerCAM(model=wrapper, target_layers=target_layers)
    grayscale = cam(input_tensor=input_tensor, targets=None, target_idx=label)
    if crops_mode:
        return grayscale[crop_take_index]
    return grayscale[0]


def overlay(rgb01: np.ndarray, grayscale_cam: np.ndarray) -> np.ndarray:
    resized = cv2.resize(grayscale_cam, (rgb01.shape[1], rgb01.shape[0]))
    return show_cam_on_image(rgb01, resized, use_rgb=True)


def main():
    p = argparse.ArgumentParser()
    add_global_stage_args(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--arms", default=None,
                   help="Comma-separated ARM_SPECS keys to render, in the order "
                        "given. Default: every arm. The paper's grid is "
                        "'whole,heavy_aug,multitask,multitask_haug'.")
    args = p.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    device_t = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    out_root = Path(cfg["paths"]["output_root"])
    out_dir = Path(args.out_dir) if args.out_dir else out_root / "finercam" / "bee_subplots"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_key = {sp[0]: sp for sp in ARM_SPECS}
    if args.arms:
        wanted = [k.strip() for k in args.arms.split(",") if k.strip()]
        unknown = [k for k in wanted if k not in by_key]
        if unknown:
            raise SystemExit(f"Unknown arm(s) {unknown}; choose from {list(by_key)}")
        specs = [by_key[k] for k in wanted]
    else:
        specs = list(ARM_SPECS)

    backbone_for_split = "convnext_nano.in12k"
    ref, train_ds, val_ds, test_ds, _ = _part_cls_splits(cfg, cls_image_size_for(cfg, backbone_for_split))
    num_classes = len(ref.classes)
    num_parts = len(cfg["part_labels"])
    part_ids = foreground_part_ids(cfg["part_labels"])

    test_indices = list(test_ds.indices) if isinstance(test_ds, Subset) else list(range(len(test_ds)))
    max_idx = max(EXAMPLE_INDICES)
    if max_idx >= len(test_indices):
        raise SystemExit(f"Test split has only {len(test_indices)} images; need index {max_idx}")

    # ---- pull the six example items once, save the "Original" row ----
    examples = []
    for ex_i, pos in enumerate(EXAMPLE_INDICES):
        real_idx = test_indices[pos]
        img, mask_ids, label = ref[real_idx]
        name = str(ref.image_names[real_idx])
        rgb01 = img.permute(1, 2, 0).numpy()
        Image.fromarray((rgb01 * 255).astype(np.uint8)).save(out_dir / f"original_idx_{pos}.png")
        examples.append({
            "pos": pos, "img": img, "mask_ids": mask_ids, "label": int(label),
            "name": name, "rgb01": rgb01,
        })
        species = ref.classes[label] if label < len(ref.classes) else "?"
        print(f"[example] idx={pos} image={name} label={label} ({species})")

    # Descriptor CSV is ~171k rows / 2.8GB -- fit the standardizer on the full
    # training fold (as training did), but only standardize the six rows we
    # actually need. An earlier version called `.to_numpy()` on the whole
    # frame inside a 171k-iteration dict comprehension (O(n^2)); this doesn't.
    desc_csv = out_root / "stage_c" / "pseudo_descriptors.csv"
    desc_df = pd.read_csv(desc_csv)
    meta_cols = {"image", "class_id", "species"}
    feat_cols = [c for c in desc_df.columns if c not in meta_cols]
    descriptor_dim = len(feat_cols)
    train_indices = list(train_ds.indices) if isinstance(train_ds, Subset) else list(range(len(train_ds)))
    train_names = [str(ref.image_names[i]) for i in train_indices]
    desc_stats = fit_descriptor_standardizer(desc_df, feat_cols, train_images=train_names)
    wanted_names = {ex["name"] for ex in examples}
    needed_rows = desc_df[desc_df["image"].astype(str).isin(wanted_names)]
    needed_values = needed_rows[feat_cols].to_numpy(dtype=np.float32)
    desc_by_image = {
        str(name): standardize_descriptor_vector(needed_values[i], desc_stats)
        for i, name in enumerate(needed_rows["image"].astype(str))
    }
    del desc_df, needed_rows, needed_values  # 2.8GB frame, drop it once extracted

    # ---- one arm at a time: load, generate all 6 CAMs, free memory ----
    # The multi-task row's `{arch}_{encoder}` fragment comes from the config it
    # was trained under, so this follows the arm when its encoder is repinned
    # instead of pointing at a directory that no longer exists.
    mt_tag = multitask_run_tag(cfg)
    for key, mode, backbone, run_dir_tmpl, ckpt_name, row_label, prot in specs:
        backbone = backbone.replace("MULTITASK_TAG", mt_tag)
        rel = run_dir_tmpl.replace("MULTITASK_TAG", mt_tag).format(seed=args.seed)
        if prot == "heavy_aug":
            rel = haug_dir(rel)
        run_dir = out_root / rel
        print(f"\n=== {key} ({' / '.join(row_label.splitlines())}) — {run_dir} ===")
        model, wrapper_base, target_layer_fn, kind = load_arm_model(
            cfg, mode, backbone, run_dir, ckpt_name, num_classes, descriptor_dim, device_t
        )

        for ex in examples:
            img_t = ex["img"].unsqueeze(0).to(device_t)
            mask_ids = ex["mask_ids"]
            label = ex["label"]
            z_row = desc_by_image.get(ex["name"])
            z_p = (
                torch.from_numpy(z_row).unsqueeze(0).to(device_t)
                if z_row is not None else torch.zeros(1, descriptor_dim, device=device_t)
            )

            if mode == "masked":
                bar_m = (mask_ids > 0).float().unsqueeze(0).unsqueeze(0).to(device_t)
                input_tensor = img_t * bar_m
                wrapper = wrapper_base
                crops_mode, crop_take = False, None
            elif mode in ("whole", "capacity_matched", "heavy_aug", "multitask"):
                input_tensor = img_t
                wrapper = wrapper_base if mode != "multitask" else ClsOnlyWrapper(model)
                crops_mode, crop_take = False, None
            elif mode == "concat":
                input_tensor = img_t
                wrapper = FixedSideInputWrapper(model, z_p=z_p)
                crops_mode, crop_take = False, None
            elif mode == "gated_residual":
                input_tensor = img_t
                ml = build_mask_logits(mask_ids, num_parts, device_t)
                wrapper = FixedSideInputWrapper(model, z_p=z_p, mask_logits=ml)
                crops_mode, crop_take = False, None
            elif mode in ("partcrop", "attention_parts"):
                crops = [
                    PartCropDataset._crop_part(ex["img"], mask_ids == pid) for pid in part_ids
                ]
                crops.append(PartCropDataset._crop_part(ex["img"], mask_ids > 0))
                # (K+1, C, H, W), no batch dim: pytorch_grad_cam's
                # get_target_width_height treats a 5D input as a single 3D
                # volume, not a batch of 2D crops, so batch=1 has to be
                # implicit here and added back inside the wrapper instead.
                input_tensor = torch.stack(crops, dim=0).to(device_t)
                wrapper = (
                    CropBatchWrapper(model) if mode == "partcrop"
                    else FixedSideInputWrapper(model, z_p=z_p, crops_mode=True)
                )
                crops_mode, crop_take = True, len(part_ids)  # last stream = full-body
            else:
                raise ValueError(mode)

            # target_layer_fn's attribute paths (see load_arm_model) are all
            # written against the *raw* reconstructed model, not whichever
            # CAM-facing wrapper happens to sit in front of it this
            # iteration -- except multitask, whose target layer lives on
            # ClsOnlyWrapper.model.seg, i.e. needs the wrapper itself.
            target_layers = target_layer_fn(wrapper) if mode == "multitask" else target_layer_fn(model)
            grayscale = run_cam(wrapper, target_layers, input_tensor, label, crops_mode, crop_take)
            vis = overlay(ex["rgb01"], grayscale)
            out_path = out_dir / f"{key}_idx_{ex['pos']}.png"
            Image.fromarray(vis).save(out_path)
            print(f"  idx={ex['pos']}: wrote {out_path}")

        del model, wrapper_base
        torch.cuda.empty_cache()

    print(f"\nDone. Images under {out_dir}")
    print("Row order for the LaTeX grid:", ["Original"] + [sp[5].splitlines()[0] for sp in specs])


if __name__ == "__main__":
    main()
