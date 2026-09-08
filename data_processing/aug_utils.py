"""Shared train-only geometric augmentations (reference protocol).

Six versions per train image — same transform on RGB and integer mask:
original, horizontal_flip, vertical_flip, 90/180/270 rotation.

Used by ``data_processing/make_train_aug.py``. Do **not** augment val/test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

CODES_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = CODES_ROOT / "config.yaml"

# Canonical transform ids → OpenCV ops (None = identity copy)
TRANSFORM_OPS = {
    "original": None,
    "horizontal_flip": "flip_h",
    "vertical_flip": "flip_v",
    "90_rotation": "rot90",
    "180_rotation": "rot180",
    "270_rotation": "rot270",
}


def load_augmentation_cfg(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    aug = cfg.get("augmentation") or {}
    transforms = list(aug.get("transforms") or list(TRANSFORM_OPS.keys()))
    suffixes = dict(aug.get("suffixes") or {})
    # Fill missing suffixes from defaults
    defaults = {
        "original": "orig",
        "horizontal_flip": "flip_h",
        "vertical_flip": "flip_v",
        "90_rotation": "rot90",
        "180_rotation": "rot180",
        "270_rotation": "rot270",
    }
    for k, v in defaults.items():
        suffixes.setdefault(k, v)
    return {"transforms": transforms, "suffixes": suffixes}


def apply_transform(img: np.ndarray, mask: np.ndarray, transform: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Apply a named geometric transform to uint8 RGB image and integer mask."""
    if transform is None or transform in {"original", "orig", ""}:
        return img.copy(), mask.copy()

    op = TRANSFORM_OPS.get(transform, transform)
    if op == "rot90":
        return (
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE),
        )
    if op == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180)
    if op == "rot270":
        return (
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE),
        )
    if op == "flip_h":
        return cv2.flip(img, 1), cv2.flip(mask, 1)
    if op == "flip_v":
        return cv2.flip(img, 0), cv2.flip(mask, 0)
    raise ValueError(f"Unknown transform: {transform}")


def transform_pairs(aug_cfg: dict[str, Any] | None = None) -> list[tuple[str, str | None]]:
    """Return list of (filename_suffix, transform_key_or_None)."""
    aug_cfg = aug_cfg or load_augmentation_cfg()
    out: list[tuple[str, str | None]] = []
    for name in aug_cfg["transforms"]:
        suffix = aug_cfg["suffixes"][name]
        tf = None if name == "original" else name
        out.append((suffix, tf))
    return out


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_rgb_jpg(path: str | Path, rgb: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8))


def stem_with_suffix(filename: str, suffix: str) -> str:
    """``foo.jpg`` + ``rot90`` → ``foo_rot90.jpg``; mask uses ``foo_rot90_m.png``."""
    p = Path(filename)
    return f"{p.stem}_{suffix}{p.suffix}"


def mask_name_with_suffix(stem_or_filename: str, suffix: str) -> str:
    stem = Path(stem_or_filename).stem
    # Drop trailing _m if present on stem
    if stem.endswith("_m"):
        stem = stem[:-2]
    return f"{stem}_{suffix}_m.png"


def expected_aug_count(n_train: int, aug_cfg: dict[str, Any] | None = None) -> int:
    aug_cfg = aug_cfg or load_augmentation_cfg()
    return int(n_train) * len(aug_cfg["transforms"])


def find_freeze_json(
    codes_root: str | Path | None = None,
    dataset: str = "beemachine",
    filename: str = "splits_seed42.json",
) -> Path:
    """Locate frozen part-set splits (namespaced or legacy Beemachine path)."""
    codes_root = Path(codes_root) if codes_root else CODES_ROOT
    candidates = [
        codes_root / "outputs" / dataset / "frozen_splits" / filename,
        codes_root / "outputs" / "frozen_splits" / filename,  # legacy beemachine
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No freeze JSON for {dataset}. Tried: "
        + ", ".join(str(c) for c in candidates)
        + f". Run: ./run.sh stage_a.py freeze --dataset {dataset}"
    )


def load_rgb_and_mask(image_path: str | Path, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB uint8 image and single-channel integer mask (same H×W)."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, mask


def emit_six_augs(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    image_stem: str,
    image_ext: str,
    out_img_dir: str | Path,
    out_mask_dir: str | Path,
    aug_cfg: dict[str, Any] | None = None,
    image_rel_prefix: str = "",
    mask_rel_prefix: str = "",
) -> list[dict[str, str]]:
    """Write six geometric variants; return CSV row dicts (filename, mask_filename).

    Filenames: ``{stem}_{suffix}{ext}`` and ``{stem}_{suffix}_m.png``.
    Optional ``*_rel_prefix`` (e.g. ``ClassName/``) is prepended to CSV paths.
    """
    out_img_dir = Path(out_img_dir)
    out_mask_dir = Path(out_mask_dir)
    ensure_dirs(out_img_dir, out_mask_dir)
    rows: list[dict[str, str]] = []
    for suffix, tf in transform_pairs(aug_cfg):
        aug_img, aug_mask = apply_transform(img, mask, tf)
        img_name = f"{image_stem}_{suffix}{image_ext}"
        mask_name = f"{image_stem}_{suffix}_m.png"
        save_rgb_jpg(out_img_dir / img_name, aug_img)
        save_mask_png(out_mask_dir / mask_name, aug_mask)
        rows.append(
            {
                "filename": f"{image_rel_prefix}{img_name}",
                "mask_filename": f"{mask_rel_prefix}{mask_name}",
            }
        )
    return rows


def combine_cub_part_masks(
    part_mask_paths: dict[str, str],
    part_labels: Iterable[str],
    height: int,
    width: int,
) -> np.ndarray:
    """Combine per-part binary PNGs into a single integer mask (label index = id)."""
    labels = list(part_labels)
    # labels[0] is background
    out = np.zeros((height, width), dtype=np.uint8)
    for pid, name in enumerate(labels):
        if pid == 0:
            continue
        path = part_mask_paths.get(name)
        if not path:
            continue
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        out[m > 0] = np.uint8(pid)
    return out
