#!/usr/bin/env python3
"""Write the six-fold train-only geometric augmentation used by Stage A.

Val and test folds are never augmented. Names match the frozen train split
so the count check in data.part_train_val_test holds: N_train x 6.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

CODES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODES))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config  # noqa: E402
from data import frozen_split_path, load_splits  # noqa: E402
from aug_utils import (  # noqa: E402
    apply_transform,
    combine_cub_part_masks,
    load_rgb_and_mask,
    transform_pairs,
)


def _reset_aug_outputs(out_dir: Path, csv_name: str | None = None) -> None:
    out_dir = out_dir.resolve()
    for sub in ("train_aug_images", "train_aug_masks"):
        path = out_dir / sub
        if path.is_dir():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    if csv_name:
        csv_path = out_dir / csv_name
        if csv_path.is_file():
            csv_path.unlink()


def _pairs(cfg: dict):
    return transform_pairs(cfg.get("augmentation"))


def _write_pair(img, mask, out_img: Path, out_mask: Path) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_mask), mask.astype("uint8"))


def _aug_beemachine(cfg: dict) -> None:
    entry = cfg["dataset_entry"]
    root = Path(cfg["paths"]["partwhole_root"])
    splits = load_splits(frozen_split_path(cfg, kind="part"))
    train_names = set(splits["train"])
    labels = pd.read_csv(root / entry.get("class_filename", "species_labels.csv"))
    img_col = "images" if "images" in labels.columns else "filename"
    if img_col == "filename":
        labels = labels.rename(columns={"filename": "images"})
    labels = labels[labels["images"].isin(train_names)].copy()
    _reset_aug_outputs(root, entry.get("train_aug_csv", "segmentation_train_aug.csv"))
    img_root = root / entry.get("train_aug_images_subdir", "train_aug_images")
    mask_root = root / entry.get("train_aug_masks_subdir", "train_aug_masks")
    records = []
    for row in tqdm(labels.itertuples(index=False), total=len(labels), desc="beemachine aug"):
        fname = str(getattr(row, "images"))
        species = str(row.species)
        stem = Path(fname).stem
        img_path = root / entry.get("images_subdir", "images") / fname
        mask_path = root / entry.get("masks_subdir", "masks") / f"{stem}_m.png"
        if not img_path.is_file() or not mask_path.is_file():
            continue
        img, mask = load_rgb_and_mask(img_path, mask_path)
        for suffix, tf in _pairs(cfg):
            img_t, mask_t = apply_transform(img, mask, tf)
            # Keep the original filename inside the stem, matching the paper run.
            new_img = f"{fname}_{suffix}.jpg"
            new_mask = f"{fname}_{suffix}_m.png"
            _write_pair(img_t, mask_t, img_root / new_img, mask_root / new_mask)
            records.append({"filename": new_img, "mask_filename": new_mask, "species": species})
    csv_path = root / entry.get("train_aug_csv", "segmentation_train_aug.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"wrote {len(records)} rows -> {csv_path}")


def _cub_mask_dict(class_mask_dir: Path, stem: str, part_names: list[str]) -> dict[str, str]:
    masks: dict[str, str] = {}
    if not class_mask_dir.is_dir():
        return masks
    for f in class_mask_dir.iterdir():
        if stem not in f.name:
            continue
        for lbl in part_names:
            if lbl in f.name:
                masks[lbl] = str(f)
    return masks


def _aug_cub(cfg: dict) -> None:
    entry = cfg["dataset_entry"]
    root = Path(cfg["paths"]["partwhole_root"])
    splits = load_splits(frozen_split_path(cfg, kind="part"))
    train_names = set(splits["train"])
    labels = list(cfg["part_labels"])
    part_names = [l for l in labels if l != "background"]
    classes = pd.read_csv(root / "classes.txt", sep=" ", header=None, names=["id", "name"])
    name_to_cid = {str(r.name): int(r.id) for r in classes.itertuples(index=False)}
    images = pd.read_csv(root / "images.txt", sep=" ", header=None, names=["id", "name"])
    _reset_aug_outputs(root)
    img_root = root / entry.get("train_aug_images_subdir", "train_aug_images")
    mask_root = root / entry.get("train_aug_masks_subdir", "train_aug_masks")
    n = 0
    for rel in tqdm(images["name"].astype(str).tolist(), desc="cub aug"):
        if rel not in train_names:
            continue
        img_path = root / entry.get("images_subdir", "images") / rel
        if not img_path.is_file():
            continue
        class_name = img_path.parent.name
        cid = name_to_cid.get(class_name)
        if cid is None:
            continue
        stem = img_path.stem
        mask_dict = _cub_mask_dict(
            root / entry.get("masks_subdir", "AnnotationMasksPerclass") / str(cid),
            stem,
            part_names,
        )
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = combine_cub_part_masks(mask_dict, labels, h, w)
        for suffix, tf in _pairs(cfg):
            img_t, mask_t = apply_transform(img, mask, tf)
            new_img = f"{stem}_{suffix}.jpg"
            new_mask = f"{stem}_{suffix}_m.png"
            _write_pair(
                img_t,
                mask_t,
                img_root / class_name / new_img,
                mask_root / class_name / new_mask,
            )
            n += 1
    print(f"wrote {n} cub train-aug files under {img_root}")


def _aug_fish(cfg: dict) -> None:
    entry = cfg["dataset_entry"]
    root = Path(cfg["paths"]["partwhole_root"])
    train_csv = root / entry.get("seg_train_csv", "segmentation_train.csv")
    df = pd.read_csv(train_csv)
    _reset_aug_outputs(root, entry.get("seg_train_aug_csv", "segmentation_train_aug.csv"))
    img_root = root / entry.get("train_aug_images_subdir", "train_aug_images")
    mask_root = root / entry.get("train_aug_masks_subdir", "train_aug_masks")
    img_dir = root / entry.get("images_subdir", "Images")
    mask_dir = root / entry.get("masks_subdir", "segmentation_masks/images")
    records = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="fish_vista aug"):
        fname = str(row.filename)
        species = str(row.standardized_species)
        stem = Path(fname).stem
        img_path = img_dir / fname
        mask_path = mask_dir / f"{stem}.png"
        if not img_path.is_file() or not mask_path.is_file():
            continue
        img, mask = load_rgb_and_mask(img_path, mask_path)
        for suffix, tf in _pairs(cfg):
            img_t, mask_t = apply_transform(img, mask, tf)
            new_img = f"{fname}_{suffix}.jpg"
            new_mask = f"{fname}_{suffix}_m.png"
            _write_pair(img_t, mask_t, img_root / new_img, mask_root / new_mask)
            records.append(
                {
                    "filename": new_img,
                    "mask_filename": new_mask,
                    "standardized_species": species,
                }
            )
    csv_path = root / entry.get("seg_train_aug_csv", "segmentation_train_aug.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"wrote {len(records)} rows -> {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CODES / "config.yaml"))
    ap.add_argument("--dataset", required=True, choices=("beemachine", "cub", "fish_vista"))
    args = ap.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)
    fn = {"beemachine": _aug_beemachine, "cub": _aug_cub, "fish_vista": _aug_fish}[args.dataset]
    fn(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
