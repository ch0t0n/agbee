"""Datasets, stratified splits, and freeze helpers (Beemachine / CUB / Fish-Vista)."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import ensure_dir
from image_cache import build_or_load as _cache_build_or_load
from image_cache import read_mask_array as _read_mask_array

PART_LABELS = ["background", "abdomen", "head", "thorax"]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_shard_csv(path) -> "pd.DataFrame | None":
    """A shard CSV, or None when that shard produced no rows.

    A shard with nothing to do writes `pd.DataFrame([]).to_csv(...)`, which is a
    single newline -- 1 byte, not 0, so a size check does not catch it -- and
    `pd.read_csv` raises EmptyDataError on it. `names[gpu_id::n]` comes up empty
    whenever the corpus is smaller than the shard count or `--limit` trims a
    slice to nothing, so this is a normal outcome, not a failure. Letting it
    propagate threw away the work every other shard had just finished.

    Lives here because both stage_b and stage_c merge sharded descriptor CSVs
    and already import this module; a second copy is how the two drift.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def frozen_split_path(cfg: dict[str, Any], kind: str = "part") -> Path:
    """Path to frozen part-set or large-cls split JSON."""
    sub = cfg.get("splits", {}).get("freeze_subdir", "frozen_splits")
    key = "cls_filename" if kind == "cls" else "filename"
    default = "splits_cls_seed42.json" if kind == "cls" else "splits_seed42.json"
    filename = cfg.get("splits", {}).get(key, default)
    primary = Path(cfg["paths"]["output_root"]) / sub / filename
    if not primary.exists():
        production = (
            Path(cfg.get("_config_path", ".")).resolve().parent
            / "outputs"
            / str(cfg.get("dataset", "beemachine"))
            / sub
            / filename
        )
        if production.exists():
            return production

    # Legacy Beemachine location (pre-namespaced outputs/)
    if (
        kind == "part"
        and cfg.get("dataset") == "beemachine"
        and not primary.exists()
    ):
        legacy = Path(cfg.get("_config_path", ".")).resolve().parent / "outputs" / "frozen_splits" / filename
        if not legacy.exists():
            legacy = Path("outputs") / "frozen_splits" / filename
        if legacy.exists():
            return legacy
    return primary


def load_splits(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_frozen_splits(
    image_names: list[str],
    payload: dict[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    """Reject split overlap, duplicates, unknown names, and dataset drift."""
    available = set(image_names)
    folds: dict[str, list[str]] = {}
    for fold in ("train", "val", "test"):
        values = list(payload.get(fold) or [])
        if len(values) != len(set(values)):
            raise ValueError(f"Frozen split '{fold}' contains duplicate filenames")
        folds[fold] = values

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(folds[left]) & set(folds[right])
        if overlap:
            raise ValueError(
                f"Frozen splits {left}/{right} overlap ({len(overlap)} names; "
                f"example={next(iter(overlap))})"
            )

    frozen = set().union(*(set(v) for v in folds.values()))
    unknown = frozen - available
    if unknown:
        raise ValueError(
            f"Frozen splits reference {len(unknown)} missing images; "
            f"example={next(iter(unknown))}"
        )
    if require_complete:
        omitted = available - frozen
        if omitted:
            raise ValueError(
                f"Frozen splits omit {len(omitted)} current images; "
                f"example={next(iter(omitted))}"
            )
        expected = payload.get("n_images")
        if expected is not None and int(expected) != len(image_names):
            raise ValueError(
                f"Frozen split n_images={expected}, current dataset={len(image_names)}"
            )


def indices_from_frozen(
    image_names: list[str],
    split_path: str | Path,
    fold: str,
) -> list[int]:
    payload = load_splits(split_path)
    validate_frozen_splits(image_names, payload)
    name_to_idx = {n: i for i, n in enumerate(image_names)}
    return [name_to_idx[n] for n in payload[fold]]


def stratified_split_indices(
    species_ids: np.ndarray,
    ratios: tuple[float, float, float] = (0.75, 0.15, 0.10),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for sid in np.unique(species_ids):
        idxs = np.where(species_ids == sid)[0]
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_idx.append(int(idxs[0]))
            continue
        if n == 2:
            train_idx.append(int(idxs[0]))
            val_idx.append(int(idxs[1]))
            continue
        n_test = max(1, int(round(n * ratios[2])))
        n_val = max(1, int(round(n * ratios[1])))
        while n_test + n_val >= n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_test = 0
                n_val = 1
                break
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test : n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val :].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def freeze_splits(
    image_names: list[str],
    species_ids: np.ndarray,
    species_names: list[str],
    out_path: str | Path,
    ratios: tuple[float, float, float] = (0.75, 0.15, 0.10),
    seed: int = 42,
    extra_meta: dict[str, Any] | None = None,
    train_names: list[str] | None = None,
    val_names: list[str] | None = None,
    test_names: list[str] | None = None,
) -> dict[str, Any]:
    if train_names is None:
        train_idx, val_idx, test_idx = stratified_split_indices(species_ids, ratios, seed)
        train_names = [image_names[i] for i in train_idx]
        val_names = [image_names[i] for i in val_idx]
        test_names = [image_names[i] for i in test_idx]
    payload = {
        "seed": seed,
        "ratios": list(ratios),
        "n_images": len(image_names),
        "n_species": int(len(set(species_names))),
        "train": train_names,
        "val": val_names,
        "test": test_names,
        "species_by_image": {
            image_names[i]: species_names[i] for i in range(len(image_names))
        },
        "dataset_fingerprint": hashlib.sha256(
            "\n".join(
                f"{image_names[i]}\t{species_names[i]}" for i in range(len(image_names))
            ).encode("utf-8")
        ).hexdigest(),
        "meta": extra_meta or {},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def train_counts_from_indices(species_ids: np.ndarray, train_idx: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for i in train_idx:
        sid = int(species_ids[i])
        counts[sid] = counts.get(sid, 0) + 1
    return counts


def _list_image_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        f
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in _IMG_EXTS
    )


def _read_part_labels_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _load_fish_trait_labels(root: Path, trait_map_file: str) -> list[str]:
    path = root / trait_map_file
    text = path.read_text(encoding="utf-8")
    try:
        mapping = json.loads(text)
    except json.JSONDecodeError:
        mapping = ast.literal_eval(text)
    # Keys may be str or int
    items = sorted(((int(k), str(v)) for k, v in mapping.items()), key=lambda x: x[0])
    return [v for _, v in items]


class CachedSamplesMixin:
    """Serve decoded uint8 images/masks from a shared RAM cache when available.

    Datasets opt in by calling `_attach_caches` at the end of `__init__` and
    reading through `_cached_image` / `_cached_mask`, which fall back to
    decoding from disk whenever the cache is disabled or was not built. The
    cached arrays hold exactly what the on-disk path would have produced --
    a bilinear (image) or nearest (mask) resize to `image_size` -- so the two
    paths are interchangeable.
    """

    _img_cache = None
    _mask_cache = None

    def _attach_caches(
        self,
        key: str,
        image_paths,
        mask_paths=None,
        *,
        cfg=None,
        image_size: int | None = None,
        mask_decode_index=None,
    ) -> None:
        size = int(image_size if image_size is not None else self.image_size)
        self._img_cache = _cache_build_or_load(
            f"{key}_img", list(image_paths), size, mode="rgb", cfg=cfg
        )
        if mask_paths is not None:
            self._mask_cache = _cache_build_or_load(
                f"{key}_mask", list(mask_paths), size, mode="mask", cfg=cfg,
                decode_index=mask_decode_index,
            )

    def _cached_image(self, idx: int):
        """CHW float tensor in [0, 1], matching `_img_transform`'s output.

        Returns None -- so the caller decodes from disk -- when the cache does
        not cover ``idx``. Some datasets are assembled by appending folds to an
        already-constructed dataset (Fish-Vista's classification corpus stitches
        val/test onto train), which leaves the cache shorter than `samples`.
        Falling back is always correct; indexing past the memmap is an
        IndexError inside a DataLoader worker.
        """
        if self._img_cache is None or idx >= len(self._img_cache):
            return None
        arr = np.asarray(self._img_cache[idx])
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255.0)

    def _cached_mask(self, idx: int):
        """HW int64 tensor of part ids, or None when uncached (see above)."""
        if self._mask_cache is None or idx >= len(self._mask_cache):
            return None
        return torch.from_numpy(np.asarray(self._mask_cache[idx]).astype(np.int64))


def _img_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
        ]
    )


# ---------------------------------------------------------------------------
# Part–whole datasets
# ---------------------------------------------------------------------------


class PartWholeDataset(CachedSamplesMixin, Dataset):
    """Beemachine V6: flat images/ + masks/ + species_labels.csv."""

    def __init__(
        self,
        root: str,
        class_filename: str = "species_labels.csv",
        images_dir: str = "images",
        masks_dir: str = "masks",
        image_size: int = 320,
        part_labels: Optional[list[str]] = None,
        split_names: Optional[set[str]] = None,
        return_name: bool = False,
        cfg: Optional[dict] = None,
    ):
        self.root = root
        self.images_dir = os.path.join(root, images_dir)
        self.masks_dir = os.path.join(root, masks_dir)
        self.image_size = image_size
        self.return_name = return_name
        self.labels = list(part_labels) if part_labels is not None else list(PART_LABELS)
        self.num_parts = len(self.labels)

        df = pd.read_csv(os.path.join(root, class_filename))
        df["species"] = df["species"].astype("category")
        self.classes = list(df["species"].cat.categories)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        df["class_id"] = df["species"].cat.codes.astype(np.int64)

        existing_images = set(os.listdir(self.images_dir))
        existing_masks = set(os.listdir(self.masks_dir))
        df = df[df["images"].isin(existing_images)].copy()
        df["base_name"] = df["images"].str.rsplit(".", n=1).str[0]
        df["mask_name"] = df["base_name"] + "_m.png"
        df = df[df["mask_name"].isin(existing_masks)].copy()
        if split_names is not None:
            df = df[df["images"].isin(split_names)].copy()
        if df.empty:
            raise FileNotFoundError(
                f"No matched image/mask pairs under {self.images_dir} and {self.masks_dir}"
            )
        df = df.sort_values("images").reset_index(drop=True)
        df["img_path"] = self.images_dir + os.sep + df["images"]
        df["mask_path"] = self.masks_dir + os.sep + df["mask_name"]
        self.image_names = df["images"].tolist()
        self.species_names = df["species"].astype(str).tolist()
        self.samples = list(
            zip(
                df["img_path"].tolist(),
                df["mask_path"].tolist(),
                df["class_id"].tolist(),
                df["images"].tolist(),
            )
        )
        self.species_ids = df["class_id"].to_numpy(dtype=np.int64)
        self.img_transform = _img_transform(image_size)
        self._attach_caches(
            f"beemachine_part_{Path(root).name}",
            df["img_path"].tolist(),
            df["mask_path"].tolist(),
            cfg=cfg,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, class_id, name = self.samples[idx]
        img = self._cached_image(idx)
        if img is None:
            img = self.img_transform(Image.open(img_path).convert("RGB"))
        mask = self._cached_mask(idx)
        if mask is None:
            mask = torch.from_numpy(
                _read_mask_array(mask_path, self.image_size).astype(np.int64)
            )
        if self.return_name:
            return img, mask, int(class_id), name
        return img, mask, int(class_id)


class CubPartWholeDataset(CachedSamplesMixin, Dataset):
    """CUB part–whole pairs (reference-protocol-compatible).

    Two mask layouts:
    - **AnnotationMasksPerclass** (GT): per-part PNGs under numeric class-id folders;
      combined on the fly (same as the reference CUB protocol).
    - **Combined** (``train_aug_masks`` / ``part_masks``): class-name folders with ``*_m.png``.
    """

    def __init__(
        self,
        root: str,
        images_dir: str = "images",
        masks_dir: str = "AnnotationMasksPerclass",
        image_size: int = 320,
        part_labels: Optional[list[str]] = None,
        part_labels_file: str = "part_labels.txt",
        split_names: Optional[set[str]] = None,
        return_name: bool = False,
        cfg: Optional[dict] = None,
    ):
        self.root = root
        self.images_dir = os.path.join(root, images_dir)
        self.masks_dir = os.path.join(root, masks_dir)
        self.image_size = image_size
        self.return_name = return_name
        self._cfg = cfg

        if part_labels is not None:
            self.labels = list(part_labels)
        else:
            labels_path = Path(root) / part_labels_file
            self.labels = ["background"] + _read_part_labels_file(labels_path)
        self.num_parts = len(self.labels)

        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"CUB masks missing: {self.masks_dir}")

        mask_basename = os.path.basename(os.path.normpath(self.masks_dir))
        self._annotation_mode = mask_basename == "AnnotationMasksPerclass"

        if self._annotation_mode:
            self._init_annotation_masks(split_names)
        else:
            self._init_combined_masks(split_names)

        self.img_transform = _img_transform(image_size)

    def _init_annotation_masks(self, split_names: Optional[set[str]]) -> None:
        """reference-protocol: images.txt + AnnotationMasksPerclass/{class_id}/."""
        classes = pd.read_csv(
            os.path.join(self.root, "classes.txt"),
            sep=" ",
            header=None,
            names=["id", "name"],
        )
        name_to_cid = {str(r.name): int(r.id) for r in classes.itertuples(index=False)}

        images = pd.read_csv(
            os.path.join(self.root, "images.txt"),
            sep=" ",
            header=None,
            names=["id", "name"],
        )

        part_names = [l for l in self.labels if l != "background"]
        samples: list[tuple[str, dict[str, str], int, str]] = []
        species_names: list[str] = []

        for rel in images["name"].astype(str).tolist():
            img_path = os.path.join(self.images_dir, rel)
            if not os.path.isfile(img_path):
                continue
            class_name = os.path.basename(os.path.dirname(img_path))
            if class_name not in name_to_cid:
                continue
            cid_ann = name_to_cid[class_name]
            class_mask_dir = os.path.join(self.masks_dir, str(cid_ann))
            if not os.path.isdir(class_mask_dir):
                continue
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_dict: dict[str, str] = {}
            for f in os.listdir(class_mask_dir):
                if stem not in f:
                    continue
                for lbl in part_names:
                    if lbl in f:
                        mask_dict[lbl] = os.path.join(class_mask_dir, f)
            if not mask_dict:
                continue
            if split_names is not None and rel not in split_names and os.path.basename(rel) not in split_names:
                continue
            samples.append((img_path, mask_dict, cid_ann, rel))
            species_names.append(class_name)

        if not samples:
            raise FileNotFoundError(
                f"No CUB AnnotationMasksPerclass pairs under {self.images_dir} / {self.masks_dir}"
            )

        present_classes = sorted(set(species_names))
        self.classes = present_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        remapped: list[tuple[str, dict[str, str], int, str]] = []
        for img_path, mask_dict, _cid_ann, rel in samples:
            sp = os.path.basename(os.path.dirname(img_path))
            cid = self.class_to_idx[sp]
            remapped.append((img_path, mask_dict, cid, rel))

        order = sorted(range(len(remapped)), key=lambda i: remapped[i][3])
        self.samples = [remapped[i] for i in order]
        self.image_names = [self.samples[i][3] for i in range(len(self.samples))]
        self.species_names = [
            os.path.basename(os.path.dirname(self.samples[i][0])) for i in range(len(self.samples))
        ]
        self.species_ids = np.asarray(
            [self.samples[i][2] for i in range(len(self.samples))], dtype=np.int64
        )
        self._sample_kind = "annotation"
        # Identity per sample must cover every per-part PNG that feeds the
        # combined mask, so regenerating any of them invalidates the cache.
        self._attach_caches(
            f"cub_part_ann_{Path(self.root).name}_{Path(self.masks_dir).name}",
            [s_[0] for s_ in self.samples],
            ["|".join(sorted(s_[1].values())) for s_ in self.samples],
            cfg=getattr(self, "_cfg", None),
            mask_decode_index=self._combine_annotation_mask,
        )

    def _init_combined_masks(self, split_names: Optional[set[str]]) -> None:
        """Class-folder ``*_m.png`` pairs (train_aug_masks / optional part_masks)."""
        self.classes = sorted(
            d
            for d in os.listdir(self.masks_dir)
            if os.path.isdir(os.path.join(self.masks_dir, d)) and not d.startswith(".")
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        samples: list[tuple[str, str, int, str]] = []
        species_names: list[str] = []
        image_names: list[str] = []
        species_ids: list[int] = []
        for cls in self.classes:
            cid = self.class_to_idx[cls]
            cls_mask_dir = os.path.join(self.masks_dir, cls)
            cls_img_dir = os.path.join(self.images_dir, cls)
            for mname in sorted(os.listdir(cls_mask_dir)):
                if not mname.endswith("_m.png"):
                    continue
                stem = mname[: -len("_m.png")]
                img_path = None
                fname = None
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG"):
                    cand = os.path.join(cls_img_dir, stem + ext)
                    if os.path.isfile(cand):
                        img_path = cand
                        fname = stem + ext
                        break
                if img_path is None:
                    continue
                rel = f"{cls}/{fname}"
                if split_names is not None and rel not in split_names and fname not in split_names:
                    continue
                samples.append((img_path, os.path.join(cls_mask_dir, mname), cid, rel))
                species_names.append(cls)
                image_names.append(rel)
                species_ids.append(cid)

        if not samples:
            raise FileNotFoundError(
                f"No CUB image/mask pairs under {self.images_dir} / {self.masks_dir}"
            )
        order = sorted(range(len(samples)), key=lambda i: samples[i][3])
        self.samples = [samples[i] for i in order]
        self.image_names = [image_names[i] for i in order]
        self.species_names = [species_names[i] for i in order]
        self.species_ids = np.asarray([species_ids[i] for i in order], dtype=np.int64)
        self._sample_kind = "combined"
        self._attach_caches(
            f"cub_part_comb_{Path(self.root).name}_{Path(self.masks_dir).name}",
            [s_[0] for s_ in self.samples],
            [s_[1] for s_ in self.samples],
            cfg=getattr(self, "_cfg", None),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _combine_annotation_mask(self, idx: int, size: int) -> np.ndarray:
        """Fuse the per-part PNGs for sample ``idx`` into one label map.

        Up to one decode per part label (12 for CUB), which is why the result
        is worth caching rather than redoing every epoch.
        """
        _img_path, mask_dict, _class_id, _name = self.samples[idx]
        mask = np.zeros((size, size), dtype=np.uint8)
        for label_idx, label in enumerate(self.labels):
            path = mask_dict.get(label)
            if not path:
                continue
            pm = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
            mask[np.asarray(pm) > 127] = label_idx
        return mask

    def __getitem__(self, idx: int):
        if self._sample_kind == "annotation":
            img_path, mask_dict, class_id, name = self.samples[idx]
            img = self._cached_image(idx)
            if img is None:
                img = self.img_transform(Image.open(img_path).convert("RGB"))
            mask_t = self._cached_mask(idx)
            if mask_t is None:
                mask_t = torch.from_numpy(
                    self._combine_annotation_mask(idx, self.image_size).astype(np.int64)
                )
            if self.return_name:
                return img, mask_t, int(class_id), name
            return img, mask_t, int(class_id)

        img_path, mask_path, class_id, name = self.samples[idx]
        img = self._cached_image(idx)
        if img is None:
            img = self.img_transform(Image.open(img_path).convert("RGB"))
        mask = self._cached_mask(idx)
        if mask is None:
            mask = torch.from_numpy(
                _read_mask_array(mask_path, self.image_size).astype(np.int64)
            )
        if self.return_name:
            return img, mask, int(class_id), name
        return img, mask, int(class_id)


class TrainAugPartDataset(CachedSamplesMixin, Dataset):
    """Flat or class-folder train-aug image/mask pairs from ``segmentation_train_aug.csv``.

    CSV columns: ``filename``, ``mask_filename``, ``species`` (or ``standardized_species``).
    ``filename`` may be a bare name (Fish/Beemachine flat) or ``Class/name.jpg`` (CUB).
    """

    def __init__(
        self,
        root: str,
        csv_path: str,
        images_subdir: str,
        masks_subdir: str,
        image_size: int = 320,
        part_labels: Optional[list[str]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
        return_name: bool = False,
        species_col: str | None = None,
        cfg: Optional[dict] = None,
    ):
        self.root = root
        self.image_dir = os.path.join(root, images_subdir)
        self.mask_dir = os.path.join(root, masks_subdir)
        self.image_size = image_size
        self.return_name = return_name
        self.labels = list(part_labels) if part_labels is not None else []
        self.num_parts = len(self.labels)

        df = pd.read_csv(csv_path)
        if species_col is None:
            if "standardized_species" in df.columns:
                species_col = "standardized_species"
            elif "species" in df.columns:
                species_col = "species"
            else:
                raise ValueError(f"{csv_path} needs species or standardized_species")
        if "filename" not in df.columns or "mask_filename" not in df.columns:
            raise ValueError(f"{csv_path} needs filename and mask_filename")

        species = df[species_col].astype(str)
        if class_to_idx is None:
            class_to_idx = {c: i for i, c in enumerate(sorted(species.unique().tolist()))}
        self.class_to_idx = class_to_idx
        self.classes = [None] * len(class_to_idx)
        for c, i in class_to_idx.items():
            self.classes[i] = c

        samples: list[tuple[str, str, int, str]] = []
        species_names: list[str] = []
        image_names: list[str] = []
        species_ids: list[int] = []
        for row in df.itertuples(index=False):
            fname = getattr(row, "filename")
            mname = getattr(row, "mask_filename")
            sp = str(getattr(row, species_col))
            if sp not in class_to_idx:
                continue
            img_path = os.path.join(self.image_dir, fname)
            mask_path = os.path.join(self.mask_dir, mname)
            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                continue
            cid = int(class_to_idx[sp])
            samples.append((img_path, mask_path, cid, fname))
            species_names.append(sp)
            image_names.append(fname)
            species_ids.append(cid)
        if not samples:
            raise FileNotFoundError(f"No train-aug pairs from {csv_path}")
        self.samples = samples
        self.image_names = image_names
        self.species_names = species_names
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
        self.img_transform = _img_transform(image_size)
        self._attach_caches(
            f"trainaug_part_{Path(self.root).name}_{Path(self.image_dir).name}",
            [s_[0] for s_ in self.samples],
            [s_[1] for s_ in self.samples],
            cfg=cfg,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, class_id, name = self.samples[idx]
        img = self._cached_image(idx)
        if img is None:
            img = self.img_transform(Image.open(img_path).convert("RGB"))
        mask = self._cached_mask(idx)
        if mask is None:
            mask = torch.from_numpy(
                _read_mask_array(mask_path, self.image_size).astype(np.int64)
            )
        if self.return_name:
            return img, mask, int(class_id), name
        return img, mask, int(class_id)


class FishPartWholeDataset(CachedSamplesMixin, Dataset):
    """Fish-Vista CSV-driven image/mask pairs (reference layout)."""

    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        image_dir: str,
        mask_dir: str,
        image_size: int = 320,
        part_labels: Optional[list[str]] = None,
        mask_sfx: str = ".png",
        class_to_idx: Optional[dict[str, int]] = None,
        return_name: bool = False,
        cfg: Optional[dict] = None,
    ):
        self.root = root
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.mask_sfx = mask_sfx
        self.return_name = return_name
        self.labels = list(part_labels) if part_labels is not None else []
        self.num_parts = len(self.labels)

        df = df.copy().reset_index(drop=True)
        if "standardized_species" not in df.columns:
            raise ValueError("Fish CSV must contain standardized_species")
        species = df["standardized_species"].astype(str)
        if class_to_idx is None:
            cats = sorted(species.unique().tolist())
            class_to_idx = {c: i for i, c in enumerate(cats)}
        self.class_to_idx = class_to_idx
        self.classes = [None] * len(class_to_idx)
        for c, i in class_to_idx.items():
            self.classes[i] = c

        samples: list[tuple[str, str, int, str]] = []
        species_names: list[str] = []
        image_names: list[str] = []
        species_ids: list[int] = []
        for row in df.itertuples(index=False):
            fname = getattr(row, "filename")
            sp = str(getattr(row, "standardized_species"))
            if sp not in class_to_idx:
                continue
            img_path = os.path.join(image_dir, fname)
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, stem + mask_sfx)
            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                continue
            cid = int(class_to_idx[sp])
            samples.append((img_path, mask_path, cid, fname))
            species_names.append(sp)
            image_names.append(fname)
            species_ids.append(cid)

        if not samples:
            raise FileNotFoundError(
                f"No Fish image/mask pairs (image_dir={image_dir}, mask_dir={mask_dir})"
            )
        self.samples = samples
        self.image_names = image_names
        self.species_names = species_names
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
        self.img_transform = _img_transform(image_size)
        self._attach_caches(
            f"fish_part_{Path(self.root).name}_{Path(self.image_dir).name}",
            [s_[0] for s_ in self.samples],
            [s_[1] for s_ in self.samples],
            cfg=cfg,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, class_id, name = self.samples[idx]
        img = self._cached_image(idx)
        if img is None:
            img = self.img_transform(Image.open(img_path).convert("RGB"))
        mask = self._cached_mask(idx)
        if mask is None:
            mask = torch.from_numpy(
                _read_mask_array(mask_path, self.image_size).astype(np.int64)
            )
        if self.return_name:
            return img, mask, int(class_id), name
        return img, mask, int(class_id)


# ---------------------------------------------------------------------------
# Large classification corpora
# ---------------------------------------------------------------------------


class SpeciesImageDataset(CachedSamplesMixin, Dataset):
    """Species-labeled images only (large corpus).

    Layouts (auto-detected):
    - **species folders**: ``root/<species>/*.jpg``
    - **flat + CSV**: ``root/images/`` + ``species_labels.csv``
    """

    def __init__(
        self,
        root: str,
        labels_csv: str = "species_labels.csv",
        images_dir: str = "images",
        image_size: int = 224,
        split_names: Optional[set[str]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
        cfg: Optional[dict] = None,
    ):
        self.root = root
        self.image_size = image_size
        flat_images = os.path.join(root, images_dir)
        csv_path = os.path.join(root, labels_csv)
        use_flat = os.path.isdir(flat_images) and os.path.isfile(csv_path)

        if use_flat:
            self.images_dir = flat_images
            df = pd.read_csv(csv_path)
            img_col = "images" if "images" in df.columns else "image"
            df = df.rename(columns={img_col: "images"})
            existing = set(os.listdir(self.images_dir))
            df = df[df["images"].isin(existing)].copy()
            if split_names is not None:
                df = df[df["images"].isin(split_names)].copy()
            if class_to_idx is None:
                cats = sorted(df["species"].astype(str).unique().tolist())
                class_to_idx = {c: i for i, c in enumerate(cats)}
            self.class_to_idx = class_to_idx
            self.classes = [None] * len(class_to_idx)
            for c, i in class_to_idx.items():
                self.classes[i] = c
            df["class_id"] = df["species"].astype(str).map(class_to_idx)
            df = df.dropna(subset=["class_id"]).reset_index(drop=True)
            df["class_id"] = df["class_id"].astype(np.int64)
            self.samples = [
                (os.path.join(self.images_dir, row.images), int(row.class_id), row.images)
                for row in df.itertuples()
            ]
            self.species_ids = df["class_id"].to_numpy(dtype=np.int64)
            self.image_names = df["images"].tolist()
            self.layout = "flat_csv"
        else:
            self.images_dir = root
            species_dirs = sorted(
                d
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
            )
            if class_to_idx is None:
                class_to_idx = {name: i for i, name in enumerate(species_dirs)}
            self.class_to_idx = class_to_idx
            self.classes = [None] * len(class_to_idx)
            for c, i in class_to_idx.items():
                self.classes[i] = c

            samples: list[tuple[str, int, str]] = []
            species_ids: list[int] = []
            image_names: list[str] = []
            for species in species_dirs:
                if species not in class_to_idx:
                    continue
                cid = int(class_to_idx[species])
                sdir = os.path.join(root, species)
                for fname in _list_image_files(sdir):
                    rel = f"{species}/{fname}"
                    if split_names is not None and rel not in split_names and fname not in split_names:
                        continue
                    samples.append((os.path.join(sdir, fname), cid, rel))
                    species_ids.append(cid)
                    image_names.append(rel)
            if not samples:
                raise FileNotFoundError(
                    f"No images found under species folders in {root}"
                )
            self.samples = samples
            self.species_ids = np.asarray(species_ids, dtype=np.int64)
            self.image_names = image_names
            self.layout = "species_folders"

        self.img_transform = _img_transform(image_size)
        self._attach_caches(
            f"species_{Path(self.root).name}",
            [s_[0] for s_ in self.samples],
            cfg=cfg,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, class_id, name = self.samples[idx]
        img = self._cached_image(idx)
        if img is None:
            img = self.img_transform(Image.open(path).convert("RGB"))
        return img, class_id, name


class CsvSpeciesDataset(CachedSamplesMixin, Dataset):
    """Flat Images/ + classification CSV (Fish-Vista scale corpus)."""

    def __init__(
        self,
        root: str,
        csv_path: str,
        images_subdir: str = "Images",
        image_size: int = 224,
        class_to_idx: Optional[dict[str, int]] = None,
        species_col: str = "standardized_species",
        filename_col: str = "filename",
        cfg: Optional[dict] = None,
        attach_cache: bool = True,
    ):
        self.root = root
        self.image_size = image_size
        self.images_dir = os.path.join(root, images_subdir)
        df = pd.read_csv(csv_path)
        if filename_col not in df.columns or species_col not in df.columns:
            raise ValueError(f"CSV {csv_path} missing {filename_col}/{species_col}")
        species = df[species_col].astype(str)
        if class_to_idx is None:
            cats = sorted(species.unique().tolist())
            class_to_idx = {c: i for i, c in enumerate(cats)}
        self.class_to_idx = class_to_idx
        self.classes = [None] * len(class_to_idx)
        for c, i in class_to_idx.items():
            self.classes[i] = c

        samples: list[tuple[str, int, str]] = []
        species_ids: list[int] = []
        image_names: list[str] = []
        for row in df.itertuples(index=False):
            fname = getattr(row, filename_col)
            sp = str(getattr(row, species_col))
            if sp not in class_to_idx:
                continue
            path = os.path.join(self.images_dir, fname)
            if not os.path.isfile(path):
                continue
            cid = int(class_to_idx[sp])
            samples.append((path, cid, fname))
            species_ids.append(cid)
            image_names.append(fname)
        if not samples:
            raise FileNotFoundError(f"No images for {csv_path} under {self.images_dir}")
        self.samples = samples
        self.species_ids = np.asarray(species_ids, dtype=np.int64)
        self.image_names = image_names
        self.layout = "csv_flat"
        self.img_transform = _img_transform(image_size)
        # `attach_cache=False` for folds that are only built to be stitched
        # into a combined dataset (see build_large_cls_dataset, fold=None).
        # Those per-fold caches were built in full and then never read once the
        # combined cache replaced them: 8.5 GB of Fish-Vista images decoded and
        # held in /dev/shm twice over, for one usable copy.
        if attach_cache:
            self._attach_caches(
                f"csvspecies_{Path(self.root).name}",
                [s_[0] for s_ in self.samples],
                cfg=cfg,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, class_id, name = self.samples[idx]
        img = self._cached_image(idx)
        if img is None:
            img = self.img_transform(Image.open(path).convert("RGB"))
        return img, class_id, name


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_part_dataset(
    cfg: dict[str, Any],
    fold: str | None = None,
    image_size: int | None = None,
    return_name: bool = False,
) -> Dataset:
    """Build the part-annotated dataset for the active config dataset.

    For ``fish_vista`` with ``fold`` set, returns that official fold only
    (train uses aug when ``use_train_aug``). For Beemachine/CUB, ``fold`` is
    ignored here — use :func:`split_indices` with frozen splits.
    """
    entry = cfg.get("dataset_entry") or {}
    layout = cfg.get("layout", entry.get("layout", "beemachine_v6"))
    root = cfg["paths"]["partwhole_root"]
    size = int(image_size if image_size is not None else cfg["image_size"])
    labels = cfg["part_labels"]

    if layout == "beemachine_v6":
        return PartWholeDataset(
            root=root,
            class_filename=entry.get("class_filename", "species_labels.csv"),
            images_dir=entry.get("images_subdir", "images"),
            masks_dir=entry.get("masks_subdir", "masks"),
            image_size=size,
            part_labels=labels,
            return_name=return_name,
            cfg=cfg,
        )

    if layout == "cub":
        return CubPartWholeDataset(
            root=root,
            images_dir=entry.get("images_subdir", "images"),
            masks_dir=entry.get("masks_subdir", "part_masks"),
            image_size=size,
            part_labels=labels,
            part_labels_file=entry.get("part_labels_file", "part_labels.txt"),
            return_name=return_name,
            cfg=cfg,
        )

    if layout == "fish_vista":
        return _build_fish_part_fold(cfg, fold=fold or "train", image_size=size, return_name=return_name)

    raise ValueError(f"Unknown layout: {layout}")


def _fish_class_to_idx(cfg: dict[str, Any]) -> dict[str, int]:
    """Union of species across seg CSVs for stable class ids.

    Segmentation-only: this id space is 3,338 species over 6,132 images
    (median 1 image/species, assigned to train/val/test independently per
    official fold with no species stratification), which is why Stage
    B/C classification for Fish-Vista does not use it -- see
    ``build_large_cls_dataset`` / ``stage_c.large_cls_parts_splits`` for the
    classification-corpus id space every classification arm actually trains
    against.
    """
    entry = cfg.get("dataset_entry") or {}
    root = Path(cfg["paths"]["partwhole_root"])
    names: set[str] = set()
    for key in ("seg_train_csv", "seg_train_aug_csv", "seg_val_csv", "seg_test_csv"):
        csv_name = entry.get(key)
        if not csv_name:
            continue
        path = root / csv_name
        if path.is_file():
            df = pd.read_csv(path)
            names.update(df["standardized_species"].astype(str).tolist())
    return {c: i for i, c in enumerate(sorted(names))}


def _build_fish_part_fold(
    cfg: dict[str, Any],
    fold: str,
    image_size: int,
    return_name: bool = False,
) -> FishPartWholeDataset:
    entry = cfg.get("dataset_entry") or {}
    root = Path(cfg["paths"]["partwhole_root"])
    labels = cfg["part_labels"]
    if not labels and entry.get("trait_map_file"):
        labels = _load_fish_trait_labels(root, entry["trait_map_file"])
        cfg["part_labels"] = labels

    class_to_idx = _fish_class_to_idx(cfg)
    use_aug = bool(entry.get("use_train_aug", True)) and fold == "train"

    if use_aug:
        csv_path = root / entry.get("seg_train_aug_csv", "segmentation_train_aug.csv")
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Fish train-aug CSV missing: {csv_path}. "
                "Run data_processing/make_train_aug.py --dataset fish_vista first."
            )
        image_dir = str(root / entry.get("train_aug_images_subdir", "train_aug_images"))
        mask_dir = str(root / entry.get("train_aug_masks_subdir", "train_aug_masks"))
        mask_sfx = "_m.png"
    else:
        csv_key = {"train": "seg_train_csv", "val": "seg_val_csv", "test": "seg_test_csv"}[fold]
        csv_path = root / entry.get(csv_key, f"segmentation_{fold}.csv")
        image_dir = str(root / entry.get("images_subdir", "Images"))
        mask_dir = str(root / entry.get("masks_subdir", "segmentation_masks/images"))
        mask_sfx = ".png"

    df = pd.read_csv(csv_path)
    return FishPartWholeDataset(
        root=str(root),
        df=df,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        part_labels=labels,
        mask_sfx=mask_sfx,
        class_to_idx=class_to_idx,
        return_name=return_name,
        cfg=cfg,
    )


def build_fish_part_splits(
    cfg: dict[str, Any], image_size: int | None = None
) -> tuple[Dataset, Dataset, Dataset]:
    """Official Fish train(aug)/val/test part datasets."""
    size = int(image_size if image_size is not None else cfg["image_size"])
    train_ds = _build_fish_part_fold(cfg, "train", size)
    val_ds = _build_fish_part_fold(cfg, "val", size)
    test_ds = _build_fish_part_fold(cfg, "test", size)
    return train_ds, val_ds, test_ds


def split_indices(ds, cfg: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    """Load frozen train/val/test indices for a PartWholeDataset-like object."""
    if cfg.get("split_policy") == "official_csv":
        raise SystemExit(
            "official_csv datasets should use part_train_val_test / build_fish_part_splits, "
            "not split_indices on a single dataset"
        )
    split_path = frozen_split_path(cfg)
    if not split_path.exists():
        raise SystemExit(f"Frozen splits required: {split_path} (python stage_a.py freeze)")
    payload = load_splits(split_path)
    validate_frozen_splits(ds.image_names, payload)
    species_by_image = payload.get("species_by_image") or {}
    mismatched = [
        name
        for name, species in zip(ds.image_names, ds.species_names)
        if name in species_by_image and str(species_by_image[name]) != str(species)
    ]
    if mismatched:
        raise ValueError(
            f"Frozen split species labels drifted for {len(mismatched)} images; "
            f"example={mismatched[0]}"
        )
    expected_fingerprint = payload.get("dataset_fingerprint")
    if expected_fingerprint:
        current_fingerprint = hashlib.sha256(
            "\n".join(
                f"{name}\t{species}"
                for name, species in zip(ds.image_names, ds.species_names)
            ).encode("utf-8")
        ).hexdigest()
        if current_fingerprint != expected_fingerprint:
            raise ValueError("Frozen split dataset fingerprint does not match current labels/order")
    return (
        indices_from_frozen(ds.image_names, split_path, "train"),
        indices_from_frozen(ds.image_names, split_path, "val"),
        indices_from_frozen(ds.image_names, split_path, "test"),
    )


def _build_train_aug_dataset(
    cfg: dict[str, Any],
    image_size: int,
    class_to_idx: Optional[dict[str, int]] = None,
    return_name: bool = False,
) -> Dataset:
    entry = cfg.get("dataset_entry") or {}
    root = cfg["paths"]["partwhole_root"]
    layout = cfg.get("layout", "beemachine_v6")
    images_subdir = entry.get("train_aug_images_subdir", "train_aug_images")
    masks_subdir = entry.get("train_aug_masks_subdir", "train_aug_masks")

    # CUB: class folders under train_aug_images/train_aug_masks (no CSV; reference-protocol writer)
    if layout == "cub":
        aug_img = Path(root) / images_subdir
        if not aug_img.is_dir():
            raise FileNotFoundError(
                f"CUB train_aug_images missing: {aug_img}. "
                "Run data_processing/make_train_aug.py --dataset cub first."
            )
        return CubPartWholeDataset(
            root=root,
            images_dir=images_subdir,
            masks_dir=masks_subdir,
            image_size=image_size,
            part_labels=cfg["part_labels"],
            part_labels_file=entry.get("part_labels_file", "part_labels.txt"),
            return_name=return_name,
            cfg=cfg,
        )

    csv_name = entry.get("train_aug_csv") or entry.get(
        "seg_train_aug_csv", "segmentation_train_aug.csv"
    )
    csv_path = str(Path(root) / csv_name)
    if not Path(csv_path).is_file():
        raise FileNotFoundError(
            f"Train-aug CSV missing: {csv_path}. "
            "Run data_processing/make_train_aug.py for this dataset first."
        )
    return TrainAugPartDataset(
        root=root,
        csv_path=csv_path,
        images_subdir=images_subdir,
        masks_subdir=masks_subdir,
        image_size=image_size,
        part_labels=cfg["part_labels"],
        class_to_idx=class_to_idx,
        return_name=return_name,
        cfg=cfg,
    )


def fold_names(fold) -> tuple[list[str], list[str]]:
    """(species_names, image_names) for a fold that may be a ``Subset``.

    `part_train_val_test` returns folds wrapped in a ``Subset``, and ``Subset``
    forwards no attributes to the dataset beneath it. Reading ``.image_names`` /
    ``.species_names`` straight off a fold raises AttributeError. Indices are
    followed in order so the returned names stay aligned with the fold's own
    indexing.
    """
    from torch.utils.data import Subset

    if isinstance(fold, Subset):
        base = fold.dataset
        return (
            [base.species_names[i] for i in fold.indices],
            [base.image_names[i] for i in fold.indices],
        )
    return (list(fold.species_names), list(fold.image_names))


def part_train_val_test(
    cfg: dict[str, Any],
    image_size: int | None = None,
    *,
    use_train_aug: bool | None = None,
) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    """Return ``(ref_ds, train_ds, val_ds, test_ds)``.

    ``ref_ds`` exposes ``classes``, ``labels``, ``num_parts`` for metrics.
    When ``use_train_aug`` is True, ``train_ds`` is the 6× train-aug set; val/test
    remain original GT folds (frozen or official CSV).

    Fish-Vista: this is Stage A (segmentation) infrastructure only. Stage B/C
    classification does not use it -- see ``build_large_cls_dataset`` /
    ``stage_c.large_cls_parts_splits`` for why: the 6,132-image segmentation
    part-set spans 3,338 species (median 1 image/species) and was never
    curated for classification, unlike Beemachine/CUB's part-sets. Matching
    the reference protocol (every variant of the prior work trains classification
    against ``classification_{train,val,test}.csv`` with predicted masks,
    never the segmentation split), every Fish-Vista classification arm trains
    on the classification corpus instead.
    """
    from torch.utils.data import Subset

    size = int(image_size if image_size is not None else cfg["image_size"])
    layout = cfg.get("layout", "beemachine_v6")
    entry = cfg.get("dataset_entry") or {}

    if use_train_aug is None:
        use_train_aug = bool(entry.get("use_train_aug", True))

    if layout == "fish_vista":
        cfg_train = {**cfg, "dataset_entry": {**entry, "use_train_aug": use_train_aug}}
        cfg_eval = {**cfg, "dataset_entry": {**entry, "use_train_aug": False}}
        train_ds = _build_fish_part_fold(cfg_train, "train", size)
        val_ds = _build_fish_part_fold(cfg_eval, "val", size)
        test_ds = _build_fish_part_fold(cfg_eval, "test", size)
        if use_train_aug:
            original_train = _build_fish_part_fold(cfg_eval, "train", size)
            _validate_train_aug_dataset(cfg, train_ds, len(original_train))
        ref_ds = train_ds
        return ref_ds, train_ds, val_ds, test_ds

    # Beemachine / CUB: original GT for freeze indices; optional train_aug for Stage A
    ds = build_part_dataset(cfg, image_size=size)
    train_idx, val_idx, test_idx = split_indices(ds, cfg)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)
    if use_train_aug:
        train_ds = _build_train_aug_dataset(
            cfg, size, class_to_idx=ds.class_to_idx
        )
        full_train_count = len(load_splits(frozen_split_path(cfg))["train"])
        _validate_train_aug_dataset(cfg, train_ds, full_train_count)
    else:
        train_ds = Subset(ds, train_idx)
    return ds, train_ds, val_ds, test_ds


def _validate_train_aug_dataset(cfg: dict[str, Any], train_ds: Dataset, n_original: int) -> None:
    """Verify the generated Stage A set is exactly train-only N×augmentation."""
    suffixes = list((cfg.get("augmentation") or {}).get("suffixes", {}).values())
    n_transforms = len((cfg.get("augmentation") or {}).get("transforms") or suffixes)
    expected = int(n_original) * n_transforms
    if len(train_ds) != expected:
        raise ValueError(
            f"Train augmentation has {len(train_ds)} samples; expected "
            f"{n_original} × {n_transforms} = {expected} from the frozen train fold"
        )
    if suffixes and hasattr(train_ds, "image_names"):
        counts = {suffix: 0 for suffix in suffixes}
        for name in train_ds.image_names:
            stem = Path(str(name)).stem
            for suffix in suffixes:
                if stem.endswith(f"_{suffix}"):
                    counts[suffix] += 1
                    break
        wrong = {suffix: count for suffix, count in counts.items() if count != n_original}
        if wrong:
            raise ValueError(
                f"Train augmentation suffix counts do not match frozen train size "
                f"{n_original}: {wrong}"
            )


def build_large_cls_dataset(
    cfg: dict[str, Any],
    fold: str | None = None,
    image_size: int | None = None,
    class_to_idx: Optional[dict[str, int]] = None,
) -> Dataset:
    """Large species-labeled corpus for Stage C."""
    entry = cfg.get("dataset_entry") or {}
    layout = cfg.get("layout", entry.get("layout", "beemachine_v6"))
    root = cfg["paths"]["large_cls_root"]
    size = int(image_size if image_size is not None else cfg["cls_image_size"])

    if layout == "fish_vista":
        csv_key = {
            "train": "cls_train_csv",
            "val": "cls_val_csv",
            "test": "cls_test_csv",
        }.get(fold or "train", "cls_train_csv")
        csv_name = entry.get(csv_key, f"classification_{fold or 'train'}.csv")
        # For fold=None, concatenate all classification CSVs under one class map
        if fold is None:
            paths = [
                Path(root) / entry.get("cls_train_csv", "classification_train.csv"),
                Path(root) / entry.get("cls_val_csv", "classification_val.csv"),
                Path(root) / entry.get("cls_test_csv", "classification_test.csv"),
            ]
            dfs = [pd.read_csv(p) for p in paths if p.is_file()]
            if not dfs:
                raise FileNotFoundError("No Fish classification CSVs found")
            df_all = pd.concat(dfs, ignore_index=True)
            species = df_all["standardized_species"].astype(str)
            if class_to_idx is None:
                class_to_idx = {c: i for i, c in enumerate(sorted(species.unique().tolist()))}
            # Write nothing — use CsvSpeciesDataset on train and merge manually
            # No per-fold caches here: the combined `_attach_caches` below
            # covers all three folds in one array, and building them anyway
            # doubled both the decode time and the /dev/shm footprint.
            train = CsvSpeciesDataset(
                root,
                str(paths[0]),
                images_subdir=entry.get("images_subdir", "Images"),
                image_size=size,
                class_to_idx=class_to_idx,
                cfg=cfg,
                attach_cache=False,
            )
            # Reuse class map; stitch samples from all folds
            all_ds = train
            for p in paths[1:]:
                if not p.is_file():
                    continue
                part = CsvSpeciesDataset(
                    root,
                    str(p),
                    images_subdir=entry.get("images_subdir", "Images"),
                    image_size=size,
                    class_to_idx=class_to_idx,
                    cfg=cfg,
                    attach_cache=False,
                )
                all_ds.samples.extend(part.samples)
                all_ds.image_names.extend(part.image_names)
                all_ds.species_ids = np.concatenate(
                    [all_ds.species_ids, part.species_ids]
                )
            # `all_ds` is the train dataset with val/test stitched on afterwards,
            # so the cache attached during its __init__ only covers the train
            # rows. Re-attach over the full sample list or every index past the
            # train fold reads off the end of the memmap.
            all_ds._attach_caches(
                f"csvspecies_{Path(root).name}_all",
                [s_[0] for s_ in all_ds.samples],
                cfg=cfg,
                image_size=size,
            )
            return all_ds
        return CsvSpeciesDataset(
            root,
            str(Path(root) / csv_name),
            images_subdir=entry.get("images_subdir", "Images"),
            image_size=size,
            class_to_idx=class_to_idx,
            cfg=cfg,
        )

    if layout == "cub":
        images_root = os.path.join(root, entry.get("large_images_subdir", "images"))
        ds = SpeciesImageDataset(
            root=images_root,
            image_size=size,
            class_to_idx=class_to_idx,
            cfg=cfg,
        )
        if fold is not None:
            split_path = frozen_split_path(cfg, kind="cls")
            if not split_path.exists():
                raise SystemExit(
                    f"Frozen CLS splits required: {split_path} (python stage_a.py freeze)"
                )
            names = set(load_splits(split_path)[fold])
            return SpeciesImageDataset(
                root=images_root,
                image_size=size,
                split_names=names,
                class_to_idx=ds.class_to_idx,
                cfg=cfg,
            )
        return ds

    # Beemachine default
    return SpeciesImageDataset(
        root=root,
        labels_csv=entry.get("class_filename", "species_labels.csv"),
        images_dir=entry.get("images_subdir", "images"),
        image_size=size,
        class_to_idx=class_to_idx,
        cfg=cfg,
    )


def list_large_cls_relpaths(
    cfg: dict[str, Any],
    images_subdir: str | None = None,
) -> tuple[Path, list[str]]:
    """Return (image_root, relative paths) for Stage C pseudo-labeling."""
    entry = cfg.get("dataset_entry") or {}
    layout = cfg.get("layout", "beemachine_v6")
    root = Path(cfg["paths"]["large_cls_root"])

    if layout == "fish_vista":
        img_root = root / (images_subdir or entry.get("images_subdir", "Images"))
        ds = build_large_cls_dataset(cfg, fold=None, image_size=cfg["cls_image_size"])
        return img_root, list(ds.image_names)

    if layout == "cub":
        img_root = root / (images_subdir or entry.get("large_images_subdir", "images"))
        ds = SpeciesImageDataset(root=str(img_root), image_size=cfg["cls_image_size"], cfg=cfg)
        return img_root, list(ds.image_names)

    # Beemachine: flat images/ or species folders
    flat = root / (images_subdir or entry.get("images_subdir", "images"))
    if flat.is_dir():
        names = sorted(
            p.name for p in flat.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if names:
            return flat, names
    names: list[str] = []
    for sp in sorted(root.iterdir()):
        if not sp.is_dir() or sp.name.startswith("."):
            continue
        for p in sorted(sp.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                names.append(f"{sp.name}/{p.name}")
    if not names:
        raise FileNotFoundError(f"No images under {root}")
    return root, names


def cmd_freeze_splits(cfg: dict[str, Any], force: bool = False) -> None:
    """CLI body for freezing splits (also used by stage_a)."""
    layout = cfg.get("layout", "beemachine_v6")
    policy = cfg.get("split_policy", "stratified_freeze")
    existing_path = frozen_split_path(cfg, kind="part")
    if existing_path.exists() and not force:
        print(f"Reusing frozen part splits -> {existing_path}")
        _maybe_freeze_cls(cfg, force=False)
        return
    # For legacy fallback path, still write into namespaced output_root
    out_path = Path(cfg["paths"]["output_root"]) / cfg.get("splits", {}).get(
        "freeze_subdir", "frozen_splits"
    ) / cfg.get("splits", {}).get("filename", "splits_seed42.json")
    ensure_dir(out_path.parent)

    if out_path.exists() and not force:
        print(f"Frozen splits already exist → {out_path}")
        print("Refusing to overwrite (pass --force to regenerate).")
        _maybe_freeze_cls(cfg, force=force)
        return

    if policy == "official_csv" and layout == "fish_vista":
        entry = cfg.get("dataset_entry") or {}
        root = Path(cfg["paths"]["partwhole_root"])
        train_df = pd.read_csv(root / entry.get("seg_train_csv", "segmentation_train.csv"))
        val_df = pd.read_csv(root / entry.get("seg_val_csv", "segmentation_val.csv"))
        test_df = pd.read_csv(root / entry.get("seg_test_csv", "segmentation_test.csv"))
        train_names = train_df["filename"].astype(str).tolist()
        val_names = val_df["filename"].astype(str).tolist()
        test_names = test_df["filename"].astype(str).tolist()
        all_names = train_names + val_names + test_names
        species = (
            train_df["standardized_species"].astype(str).tolist()
            + val_df["standardized_species"].astype(str).tolist()
            + test_df["standardized_species"].astype(str).tolist()
        )
        payload = freeze_splits(
            image_names=all_names,
            species_ids=np.zeros(len(all_names), dtype=np.int64),
            species_names=species,
            out_path=out_path,
            ratios=tuple(cfg["splits"]["ratios"]),
            seed=cfg["seed"],
            train_names=train_names,
            val_names=val_names,
            test_names=test_names,
            extra_meta={
                "dataset": cfg["dataset"],
                "layout": layout,
                "split_policy": "official_csv",
                "partwhole_root": str(root),
                "part_labels": cfg["part_labels"],
                "note": "Official Fish-Vista segmentation CSVs (non-aug train listed)",
            },
        )
        print(f"Recorded official Fish seg splits -> {out_path}")
        print(
            f"train={len(payload['train'])} val={len(payload['val'])} "
            f"test={len(payload['test'])} species={payload['n_species']}"
        )
        _maybe_freeze_cls(cfg, force=True)
        return

    ds = build_part_dataset(cfg, image_size=cfg["image_size"])
    payload = freeze_splits(
        image_names=ds.image_names,
        species_ids=ds.species_ids,
        species_names=ds.species_names,
        out_path=out_path,
        ratios=tuple(cfg["splits"]["ratios"]),
        seed=cfg["seed"],
        extra_meta={
            "dataset": cfg["dataset"],
            "layout": layout,
            "partwhole_root": cfg["paths"]["partwhole_root"],
            # Named to match the classification splits. The part set has no
            # leakage exclusion applied to it, so pool and post-check agree.
            "n_classes_pool": len(ds.classes),
            "part_labels": cfg["part_labels"],
        },
    )
    print(f"Froze splits -> {out_path}")
    print(
        f"train={len(payload['train'])} val={len(payload['val'])} "
        f"test={len(payload['test'])} species={payload['n_species']}"
    )
    _maybe_freeze_cls(cfg, force=force)


def _maybe_freeze_cls(cfg: dict[str, Any], force: bool = False) -> None:
    """Freeze / record large-corpus splits where applicable."""
    layout = cfg.get("layout", "beemachine_v6")
    out_path = Path(cfg["paths"]["output_root"]) / cfg.get("splits", {}).get(
        "freeze_subdir", "frozen_splits"
    ) / cfg.get("splits", {}).get("cls_filename", "splits_cls_seed42.json")
    ensure_dir(out_path.parent)
    if out_path.exists() and not force:
        print(f"Frozen CLS splits already exist → {out_path}")
        return

    if layout == "cub":
        entry = cfg.get("dataset_entry") or {}
        images_root = os.path.join(
            cfg["paths"]["large_cls_root"], entry.get("large_images_subdir", "images")
        )
        ds = SpeciesImageDataset(root=images_root, image_size=cfg["cls_image_size"], cfg=cfg)
        keep, n_overlap = _large_cls_nonoverlap_indices(cfg, ds.image_names)
        payload = freeze_splits(
            image_names=[ds.image_names[i] for i in keep],
            species_ids=ds.species_ids[keep],
            species_names=[ds.classes[int(ds.species_ids[i])] for i in keep],
            out_path=out_path,
            ratios=tuple(cfg["splits"]["ratios"]),
            seed=cfg["seed"],
            extra_meta={
                "dataset": cfg["dataset"],
                "kind": "large_cls",
                "large_cls_root": images_root,
                # Two different class counts used to be reported as one.
                # `n_classes` was the PRE-exclusion class list, while the
                # sibling `n_species` field counts the species that actually
                # survive into the split -- 147 vs 137 on BeeMachine. Nothing
                # reads this field, but a reader comparing it with the paper's
                # corpus size had no way to tell which number was which.
                "n_classes_pool": len(ds.classes),
                "n_classes_after_leakage_check": len(
                    {ds.classes[int(ds.species_ids[i])] for i in keep}),
                "excluded_part_overlap": n_overlap,
            },
        )
        print(f"Froze CUB CLS splits -> {out_path}")
        print(
            f"train={len(payload['train'])} val={len(payload['val'])} "
            f"test={len(payload['test'])} species={payload['n_species']}"
        )
        return

    if layout == "fish_vista":
        entry = cfg.get("dataset_entry") or {}
        root = Path(cfg["paths"]["large_cls_root"])
        train_df = pd.read_csv(root / entry.get("cls_train_csv", "classification_train.csv"))
        val_df = pd.read_csv(root / entry.get("cls_val_csv", "classification_val.csv"))
        test_df = pd.read_csv(root / entry.get("cls_test_csv", "classification_test.csv"))
        part_root = Path(cfg["paths"]["partwhole_root"])
        part_names: set[str] = set()
        for key in ("seg_train_csv", "seg_val_csv", "seg_test_csv"):
            path = part_root / entry.get(key, "")
            if path.is_file():
                part_names.update(pd.read_csv(path)["filename"].astype(str).tolist())
        overlap_mask = train_df["filename"].astype(str).isin(part_names)
        n_overlap = int(overlap_mask.sum())
        train_df = train_df.loc[~overlap_mask].copy()
        train_names = train_df["filename"].astype(str).tolist()
        val_names = val_df["filename"].astype(str).tolist()
        test_names = test_df["filename"].astype(str).tolist()
        all_names = train_names + val_names + test_names
        species = (
            train_df["standardized_species"].astype(str).tolist()
            + val_df["standardized_species"].astype(str).tolist()
            + test_df["standardized_species"].astype(str).tolist()
        )
        payload = freeze_splits(
            image_names=all_names,
            species_ids=np.zeros(len(all_names), dtype=np.int64),
            species_names=species,
            out_path=out_path,
            ratios=tuple(cfg["splits"]["ratios"]),
            seed=cfg["seed"],
            train_names=train_names,
            val_names=val_names,
            test_names=test_names,
            extra_meta={
                "dataset": cfg["dataset"],
                "kind": "large_cls",
                "split_policy": "official_csv",
                "large_cls_root": str(root),
                "excluded_part_overlap_from_train": n_overlap,
            },
        )
        print(f"Recorded official Fish CLS splits -> {out_path}")
        print(
            f"train={len(payload['train'])} val={len(payload['val'])} "
            f"test={len(payload['test'])} species={payload['n_species']}"
        )
        return

    if layout == "beemachine_v6":
        ds = build_large_cls_dataset(cfg, fold=None)
        keep, n_overlap = _large_cls_nonoverlap_indices(cfg, ds.image_names)
        payload = freeze_splits(
            image_names=[ds.image_names[i] for i in keep],
            species_ids=ds.species_ids[keep],
            species_names=[ds.classes[int(ds.species_ids[i])] for i in keep],
            out_path=out_path,
            ratios=tuple(cfg["splits"]["ratios"]),
            seed=cfg["seed"],
            extra_meta={
                "dataset": cfg["dataset"],
                "kind": "large_cls",
                "large_cls_root": cfg["paths"]["large_cls_root"],
                # Two different class counts used to be reported as one.
                # `n_classes` was the PRE-exclusion class list, while the
                # sibling `n_species` field counts the species that actually
                # survive into the split -- 147 vs 137 on BeeMachine. Nothing
                # reads this field, but a reader comparing it with the paper's
                # corpus size had no way to tell which number was which.
                "n_classes_pool": len(ds.classes),
                "n_classes_after_leakage_check": len(
                    {ds.classes[int(ds.species_ids[i])] for i in keep}),
                "excluded_part_overlap": n_overlap,
            },
        )
        print(f"Froze Beemachine CLS splits -> {out_path}")
        print(
            f"train={len(payload['train'])} val={len(payload['val'])} "
            f"test={len(payload['test'])} excluded_overlap={n_overlap}"
        )
        return


def _large_cls_nonoverlap_indices(
    cfg: dict[str, Any],
    large_names: list[str],
) -> tuple[list[int], int]:
    """Exclude part-annotated images from the Stage C scale corpus.

    Two mechanisms, because one is not sufficient for every dataset:

    * **By name.** Works for CUB (1,888 excluded) and Fish-Vista (1,129), whose
      two corpora share a naming scheme.
    * **By content.** Beemachine's part set is Roboflow-renamed
      (`…_jpg.rf.<hash>.jpg`) while its 195k corpus keeps original names, so no
      filename can ever match and the name check removes exactly 0 -- which was
      being recorded as `excluded_part_overlap: 0`, i.e. as though the corpora
      had been checked and found disjoint. They had not been checked at all.
      `tools/check_part_scale_leakage.py` measures it with SHA-256 + dHash and
      writes `frozen_splits/part_scale_leakage.json`; this reads that file when
      it exists.

    The content file is optional so a freeze never *requires* an expensive hash
    pass, but its absence for a rename-based corpus is reported rather than
    passed over in silence.
    """
    part_ds = build_part_dataset(cfg, image_size=cfg["image_size"])
    part_exact = set(part_ds.image_names)
    part_basenames = {Path(name).name for name in part_ds.image_names}
    overlap = {
        i
        for i, name in enumerate(large_names)
        if name in part_exact or Path(name).name in part_basenames
    }
    n_by_name = len(overlap)

    overlap_file = (
        Path(cfg["paths"]["output_root"])
        / cfg.get("splits", {}).get("freeze_subdir", "frozen_splits")
        / "part_scale_leakage.json"
    )
    n_by_content = 0
    if overlap_file.is_file():
        flagged = set(json.loads(overlap_file.read_text(encoding="utf-8")).get(
            "overlapping_scale_relpaths", []
        ))
        content_hits = {
            i
            for i, name in enumerate(large_names)
            if name in flagged or Path(name).name in {Path(f).name for f in flagged}
        }
        n_by_content = len(content_hits - overlap)
        overlap |= content_hits
        print(
            f"[freeze] content-hash overlap file: {overlap_file} "
            f"({n_by_content} additional image(s) excluded)"
        )
    elif n_by_name == 0:
        print(
            f"[freeze] WARNING: filename matching excluded 0 images and no "
            f"{overlap_file.name} exists. Part/scale overlap for "
            f"{cfg['dataset']} is UNMEASURED, not zero. Run "
            f"tools/check_part_scale_leakage.py --dataset {cfg['dataset']}."
        )

    keep = [i for i in range(len(large_names)) if i not in overlap]
    if not keep:
        raise ValueError("Part/scale overlap exclusion removed the entire large corpus")
    return keep, len(overlap)
