#!/usr/bin/env python3
"""Stage C CLI: pseudo | filter | extract_desc | ddp | plot_curve.

`filter --conf TAU` is the confidence-gated pseudo-mask regime: masks below
TAU are dropped before Stage B+-style fusion training on the scale corpus,
so the scale-set comparison spans three mask regimes end to end — ground
truth (Stage B), Stage-A-predicted (Stage B, `--mask_source pred`), and
confidence-gated pseudo (`filter` + this module) — letting every fusion arm's
ranking be checked as localization quality degrades from oracle to noisy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
# `subprocess` and `sys` must be imported HERE, not inside the functions that
# fan out. They used to be function-local (in `cmd_pseudo` and
# `cmd_extract_desc_all_gpus`), which was invisible until the module-level
# helper `_shard_popen` was factored out of both: a top-level function cannot
# see another function's local import, so every `--all_gpus` Stage C run died
# with `NameError: name 'subprocess' is not defined` after ~25 h of Stage A/B.
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from config import (
    add_global_stage_args,
    cls_image_size,
    code_version,
    cls_patience,
    dataset_cli_args,
    default_config_path,
    ensure_dir,
    label_smoothing,
    load_config,
    source_digest,
    belongs_to_protocol,
    resolve_seg_ckpt,
    shape_embed_dim,
    stage_run_dir,
)
from data import (
    build_large_cls_dataset,
    frozen_split_path,
    list_large_cls_relpaths,
    load_splits,
    read_shard_csv,
    train_counts_from_indices,
    validate_frozen_splits,
)
from image_cache import build_or_load as cache_build_or_load
from descriptors import (
    extract_all_features,
    extract_learned_iqa_features_batch,
    fit_descriptor_standardizer,
    normalize_image_for_iqa,
    set_iqa_device,
    standardize_descriptor_vector,
    validate_iqa_metrics,
)
from distributed_utils import (
    _write_job_meta,
    auto_batch_size,
    cleanup_distributed,
    device_for_rank,
    ensure_torchrun,
    init_distributed,
    is_main_process,
    job_log_dir,
    job_slug,
    make_loader,
    mib_per_inf_sample,
    per_device_batch_size,
    resolve_num_gpus,
    seed_everything,
    wrap_ddp,
)
from fusion import PartAwareFusionClassifier, TimmClassifier
from metrics import evaluate_cls_loader, fit_cls_model, make_cls_criterion
from segmenters import load_segmenter

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_mask_cache(key: str, paths: list[str], size: int, cfg: dict | None = None):
    """Shared uint8 memmap of pseudo-masks, or None when unavailable.

    Missing masks decode to zeros (the same fallback the on-disk path used),
    so a filtered-out image still yields an all-background mask rather than
    dropping the sample.
    """

    def decode(path: str, s: int) -> np.ndarray:
        try:
            im = Image.open(path).convert("L").resize((s, s), Image.NEAREST)
            return np.asarray(im, dtype=np.uint8)
        except (OSError, ValueError):
            return np.zeros((s, s), dtype=np.uint8)

    return cache_build_or_load(key, paths, size, mode="mask", cfg=cfg, decoder=decode)


def _mask_filename(rel_image: str) -> str:
    """Unique mask name for flat or nested image paths."""
    p = Path(rel_image)
    if p.parent.name and p.parent.name not in (".",):
        return f"{p.parent.name}__{p.stem}_m.png"
    return f"{p.stem}_m.png"


class ImageFolderNames(Dataset):
    """Scale-corpus images by relative name, for the pseudo-labeling pass.

    Deliberately NOT wired to the shared RAM cache, unlike every training
    dataset in data.py. Pseudo-labeling makes exactly one sequential pass over
    the corpus, so a cache would pay the full decode cost to build and then be
    thrown away: no epoch ever reads it twice. For Beemachine that would be
    ~60 GB of /dev/shm bought for nothing, and each of the 8 shards holds a
    different slice of `names`, so they would not even share one array. The
    corpus reads that *are* repeated -- Stage C `ddp`, 30 epochs over the same
    images -- go through SpeciesImageDataset / CsvSpeciesDataset, which are
    cached. Decode here is also not the bottleneck: 8 shards decode ~5,600
    img/s between them while the segmenter forward pass is the limiting stage.
    """

    def __init__(self, root: str, image_size: int, names: list[str]):
        self.root = Path(root)
        self.names = names
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(self.root / name).convert("RGB")
        return self.tf(img), name


class LargeImageDescriptorDataset(Dataset):
    """Large-corpus images joined to pseudo descriptors and optional masks."""

    def __init__(
        self,
        dataset,
        indices: list[int],
        desc_df: pd.DataFrame,
        feature_cols: list[str],
        masks_dir: str | None,
        image_size: int | None = None,
        cfg: dict | None = None,
        desc_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.feature_cols = feature_cols
        self.desc_stats = desc_stats
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.name_to_features = {
            str(row.image): np.asarray(
                [getattr(row, col) for col in feature_cols],
                dtype=np.float32,
            )
            for row in desc_df.itertuples(index=False)
        }

        # Pseudo-mask PNGs were re-opened, converted, and resized on every
        # __getitem__ -- 146k decodes per epoch for Beemachine, 30 epochs deep,
        # under 8 DDP ranks. Cache them the same way images are cached: one
        # shared uint8 memmap in /dev/shm, so all ranks and their workers read
        # the same pages and nothing decodes twice.
        self.mask_size = int(image_size) if image_size else None
        self._mask_cache = None
        if self.masks_dir is not None and self.mask_size:
            paths = [
                str(self.masks_dir / _mask_filename(str(dataset.image_names[j])))
                for j in self.indices
            ]
            self._mask_cache = build_mask_cache(
                f"pseudomask_{self.masks_dir.name}",
                paths,
                self.mask_size,
                cfg=cfg,
            )

    def __len__(self):
        return len(self.indices)

    def _mask_for(self, i: int, name: str, h: int, w: int) -> torch.Tensor:
        if self._mask_cache is not None and i < len(self._mask_cache):
            return torch.from_numpy(np.asarray(self._mask_cache[i]).astype(np.int64))
        if self.masks_dir is not None:
            path = self.masks_dir / _mask_filename(name)
            if path.is_file():
                pil_mask = Image.open(path).convert("L").resize((w, h), Image.NEAREST)
                return torch.from_numpy(np.asarray(pil_mask, dtype=np.int64))
        return torch.zeros((h, w), dtype=torch.long)

    def __getitem__(self, i):
        img, y, name = self.dataset[self.indices[i]]
        features = self.name_to_features.get(str(name))
        if features is None:
            # Image with no descriptor row: sit at the training mean, which is
            # exactly 0.0 once standardized.
            features = np.zeros(len(self.feature_cols), dtype=np.float32)
        else:
            features = standardize_descriptor_vector(features, self.desc_stats)
        mask = self._mask_for(i, str(name), img.shape[-2], img.shape[-1])
        return img, int(y), name, torch.from_numpy(features), mask


class LargeClsPartsDataset(Dataset):
    """Fish-Vista's classification corpus, presented as the same
    ``(image, mask, label)`` triple ``FishPartWholeDataset`` returns for
    Beemachine/CUB -- so every Stage B consumer (``MaskedBeeDataset``,
    ``PartCropDataset``, ``ImageDescDataset``, ``PartCropDescDataset``,
    ``MultiTaskBeeModel``, ``cmd_baseline``, ``cmd_fusion``, ``cmd_partcrop``,
    ``cmd_multitask``) runs against it completely unmodified.

    Fish-Vista's classification corpus (~60k images / ~1.7k species) carries
    no ground-truth part annotations -- only the unrelated, and for
    classification purposes unusable, 6,132-image segmentation corpus does
    (see ``data.part_train_val_test``'s Fish-Vista docstring). Every mask
    here is therefore the frozen segmenter's confidence-gated prediction
    (``stage_c.py filter``'s output), matching the reference protocol:
    every variant of the prior work (Part/Full/Red/Zero) trains classification against
    ``classification_{train,val,test}.csv`` with predicted masks, never
    against the segmentation split. There is deliberately no
    ``mask_source="gt"`` for Fish-Vista's classification grid -- ground
    truth does not exist at this scale.

    For the multi-task arm specifically, this also means its auxiliary
    segmentation loss supervises against the same confidence-gated
    pseudo-masks rather than true annotations -- a real difference from
    Beemachine/CUB's multi-task arm that should be stated as such wherever
    Fish-Vista's multi-task numbers are reported.
    """

    def __init__(self, base, masks_dir: str, image_size: int, num_parts: int, cfg=None):
        self.base = base
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.image_names = base.image_names
        self.species_ids = base.species_ids
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx
        self.num_parts = num_parts
        self.layout = "fish_large_cls_parts"
        paths = [str(self.masks_dir / _mask_filename(n)) for n in self.image_names]
        self._mask_cache = build_mask_cache(
            f"fishclsparts_{self.masks_dir.name}", paths, self.image_size, cfg=cfg
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, class_id, name = self.base[idx]
        if self._mask_cache is not None and idx < len(self._mask_cache):
            mask = torch.from_numpy(np.asarray(self._mask_cache[idx]).astype(np.int64))
        else:
            path = self.masks_dir / _mask_filename(str(name))
            if path.is_file():
                pil_mask = Image.open(path).convert("L").resize(
                    (self.image_size, self.image_size), Image.NEAREST
                )
                mask = torch.from_numpy(np.asarray(pil_mask, dtype=np.int64))
            else:
                mask = torch.zeros((self.image_size, self.image_size), dtype=torch.int64)
        return img, mask, int(class_id)


def large_cls_parts_splits(cfg, image_size: int, masks_dir: str):
    """``(ref, train, val, test)`` Subsets of ``LargeClsPartsDataset``, on the
    same frozen classification split ``_large_cls_splits``/Stage C DDP use --
    so Stage B's grid and Stage C's at-scale run share one label space and
    one train/val/test assignment for Fish-Vista, the way ``_part_cls_splits``
    already keeps every arm on one split for Beemachine/CUB.
    """
    from torch.utils.data import Subset as _Subset

    ds_all, train_idx, val_idx, test_idx = _large_cls_splits(cfg, image_size=image_size)
    ds = LargeClsPartsDataset(
        ds_all, masks_dir, image_size, num_parts=len(cfg["part_labels"]), cfg=cfg
    )
    return ds, _Subset(ds, train_idx), _Subset(ds, val_idx), _Subset(ds, test_idx)


def _shard_popen(cmd: list[str], cwd: str, env: dict | None = None, *, idx: int):
    """Launch one shard, redirecting to a per-job log when monitored.

    The two `--all_gpus` fan-outs here spawn 8 processes that all inherit this
    terminal. Under `tools/run_monitor.py` they get one log each, exactly like
    the jobs `run_commands_parallel` launches; unmonitored, behaviour is
    unchanged.
    """
    log_dir = job_log_dir()
    if log_dir is None:
        return subprocess.Popen(cmd, cwd=cwd, env=env), None
    stem = f"{idx:03d}_{job_slug(cmd)}"
    meta = log_dir / f"{stem}.json"
    _write_job_meta(meta, idx=idx, gpu=idx, argv=cmd, started=time.time())
    fh = open(log_dir / f"{stem}.log", "w", encoding="utf-8", buffering=1)
    fh.write(f"$ {' '.join(cmd)}\n")
    fh.flush()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return proc, (fh, meta, idx, cmd)


def _shard_wait(procs) -> list[int]:
    """Wait for `_shard_popen` results, closing logs and stamping the sidecars."""
    codes = []
    for proc, handle in procs:
        rc = int(proc.wait())
        codes.append(rc)
        if handle is not None:
            fh, meta, idx, cmd = handle
            fh.close()
            _write_job_meta(meta, idx=idx, argv=cmd, finished=time.time(), rc=rc)
    return codes


@torch.no_grad()
def cmd_pseudo(
    cfg,
    seg_ckpt: str | None,
    gpu_id: int,
    num_shards: int,
    images_subdir: str,
    all_gpus: bool = False,
    config_path: str | None = None,
    limit: int = 0,
):
    if all_gpus:
        n = resolve_num_gpus(cfg)
        cfg_path = config_path or default_config_path()
        script = str(Path(__file__).resolve())
        procs = []
        for gid in range(n):
            cmd = [
                sys.executable,
                script,
                "pseudo",
                "--config",
                cfg_path,
                *dataset_cli_args(cfg),
                "--gpu_id",
                str(gid),
                "--num_shards",
                str(n),
            ]
            # Only forward --images_subdir when one was actually given. Its
            # default is None so each dataset falls back to its configured
            # `images_subdir`; passing None through to Popen is a TypeError
            # ("expected str, bytes or os.PathLike"), which is how every
            # --all_gpus invocation from run_all_experiments.sh died before
            # a single image was labeled.
            if images_subdir:
                cmd += ["--images_subdir", images_subdir]
            if seg_ckpt:
                cmd += ["--seg_ckpt", seg_ckpt]
            if limit > 0:
                cmd += ["--limit", str(limit)]
            procs.append(_shard_popen(cmd, cwd=str(Path(script).parent), idx=gid))
        codes = _shard_wait(procs)
        if any(c != 0 for c in codes):
            raise SystemExit(max(codes) if codes else 1)
        print(f"Finished pseudo-labeling on {n} GPUs")
        return

    ckpt = resolve_seg_ckpt(cfg, seg_ckpt)
    img_root, names = list_large_cls_relpaths(cfg, images_subdir=images_subdir)
    shard_names = names[gpu_id::num_shards]
    if limit > 0:
        shard_names = shard_names[:limit]
    print(
        f"[{cfg['dataset']}] Shard {gpu_id}/{num_shards}: "
        f"{len(shard_names)} images from {img_root}"
    )

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = load_segmenter(ckpt, device)

    ds = ImageFolderNames(str(img_root), cfg["image_size"], shard_names)
    pseudo_bs = auto_batch_size(
        gpu_id, mib_per_inf_sample(int(cfg["image_size"])),
        cfg["pseudo_labeling"]["batch_size"],
    )
    loader = DataLoader(
        ds, batch_size=pseudo_bs, num_workers=cfg["num_workers"]
    )

    out_masks = ensure_dir(
        Path(cfg["paths"]["output_root"]) / "stage_c" / "pseudo_masks" / f"shard{gpu_id}"
    )
    conf_rows = []
    for imgs, batch_names in tqdm(loader, desc=f"pseudo-parts-gpu{gpu_id}"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1)
        for i, name in enumerate(batch_names):
            p = probs[i]
            m = pred[i]
            fg = m > 0
            if fg.any():
                conf = float(p.max(0).values[fg].mean().cpu())
            else:
                conf = float(p.max(0).values.mean().cpu())
            mask_u8 = m.cpu().numpy().astype(np.uint8)
            Image.fromarray(mask_u8, mode="L").save(out_masks / _mask_filename(name))
            conf_rows.append({"image": name, "confidence": conf, "mask": _mask_filename(name)})

    (out_masks / "confidence.json").write_text(json.dumps(conf_rows, indent=2), encoding="utf-8")
    print(f"Wrote masks → {out_masks}")


def cmd_filter(cfg, conf: float) -> dict:
    shard_root = Path(cfg["paths"]["output_root"]) / "stage_c" / "pseudo_masks"
    rows = []
    for conf_file in sorted(shard_root.glob("shard*/confidence.json")):
        rows.extend(json.loads(conf_file.read_text(encoding="utf-8")))

    kept = [r for r in rows if r["confidence"] >= conf]
    out_dir = Path(cfg["paths"]["output_root"]) / "stage_c" / f"pseudo_masks_conf{conf}"
    tmp_dir = out_dir.with_name(f"{out_dir.name}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    ensure_dir(tmp_dir)
    copied = []
    for r in kept:
        mask_name = r.get("mask") or _mask_filename(r["image"])
        srcs = list(shard_root.glob(f"shard*/{mask_name}"))
        if not srcs:
            stem = Path(r["image"]).stem
            srcs = list(shard_root.glob(f"shard*/{stem}_m.png"))
        if not srcs:
            continue
        shutil.copy2(srcs[0], tmp_dir / mask_name)
        copied.append(r)

    meta = {
        "dataset": cfg["dataset"],
        "threshold": conf,
        "n_total": len(rows),
        "n_kept": len(copied),
        "n_missing_masks": len(kept) - len(copied),
        "kept_images": [r["image"] for r in copied],
    }
    (tmp_dir / "filter_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    print(f"Kept {len(copied)}/{len(rows)} → {out_dir}")
    return meta


def cmd_filter_sweep(cfg) -> dict:
    """Sweep every threshold in `pseudo_labeling.conf_thresholds` and report
    retention, so a discriminating tau can actually be picked (research plan
    RQ3: "confidence-gated pseudo (tau swept, selected on validation)").

    Runs `cmd_filter` once per threshold -- same masks copied into
    `pseudo_masks_conf{tau}/` as running `filter --conf` by hand at each
    value -- and additionally writes one consolidated report so the retention
    curve is visible without opening N `filter_meta.json` files. Selecting the
    final tau for `extract_desc --masks_dir .../pseudo_masks_conf{tau}` (and
    everything downstream of it) is a validation-based protocol decision, not
    something this sweep does automatically: at the currently configured
    thresholds this repo measured 100% retention at every one (a discriminating
    tau, if it exists on a given corpus, may need widening the sweep beyond the
    default [0.5, 0.7, 0.9] via `pseudo_labeling.conf_thresholds`).
    """
    thresholds = list(cfg.get("pseudo_labeling", {}).get("conf_thresholds") or [0.7])
    rows = []
    for tau in thresholds:
        meta = cmd_filter(cfg, float(tau))
        n_total = meta["n_total"]
        rows.append(
            {
                "threshold": float(tau),
                "n_total": n_total,
                "n_kept": meta["n_kept"],
                "retention": (meta["n_kept"] / n_total) if n_total else 0.0,
            }
        )

    out_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_c")
    report = {"dataset": cfg["dataset"], "thresholds": rows}
    (out_dir / "pseudo_masks_threshold_sweep.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = ["| tau | n_total | n_kept | retention |", "|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['threshold']} | {r['n_total']} | {r['n_kept']} | {r['retention']:.4f} |")
    md = "\n".join(lines) + "\n"
    (out_dir / "pseudo_masks_threshold_sweep.md").write_text(md, encoding="utf-8")
    print(md)
    spread = max(r["retention"] for r in rows) - min(r["retention"] for r in rows) if rows else 0.0
    if spread < 1e-6:
        print(
            f"[filter_sweep] retention is identical across all {len(rows)} threshold(s) "
            "-- the confidence gate is a no-op at these values; widen "
            "pseudo_labeling.conf_thresholds to find one that discriminates."
        )
    print(f"Wrote {out_dir / 'pseudo_masks_threshold_sweep.json'}")
    return report


def cmd_extract_desc(
    cfg,
    masks_dir: str,
    images_subdir: str,
    labels_csv: str,
    limit: int,
    shard_id: int = 0,
    num_shards: int = 1,
):
    # Learned NR-IQA defaults to CPU (~11x slower per image than an A40 here).
    # Bind it to this shard's GPU and load every configured metric up front so
    # a broken one aborts instead of writing an all-NaN column.
    iqa_names = list(cfg.get("descriptors", {}).get("learned_iqa_metrics") or [])
    iqa_bs = int(cfg.get("descriptors", {}).get("iqa_batch_size", 16))
    if iqa_names:
        set_iqa_device("cuda:0" if torch.cuda.is_available() else "cpu")
        validate_iqa_metrics(iqa_names)
    img_root, _ = list_large_cls_relpaths(cfg, images_subdir=images_subdir)
    masks_path = Path(masks_dir)
    root = Path(cfg["paths"]["large_cls_root"])
    entry = cfg.get("dataset_entry") or {}

    name_to_species: dict[str, str] = {}
    layout = cfg.get("layout", "beemachine_v6")
    if layout == "fish_vista":
        for key in ("cls_train_csv", "cls_val_csv", "cls_test_csv"):
            csv_path = root / entry.get(key, "")
            if csv_path.is_file():
                labels = pd.read_csv(csv_path)
                name_to_species.update(
                    dict(
                        zip(
                            labels["filename"].astype(str),
                            labels["standardized_species"].astype(str),
                        )
                    )
                )
    else:
        csv_path = root / labels_csv
        if csv_path.is_file():
            labels = pd.read_csv(csv_path)
            img_col = "images" if "images" in labels.columns else "image"
            labels = labels.rename(columns={img_col: "images"})
            name_to_species = dict(zip(labels["images"], labels["species"].astype(str)))

    tf = transforms.Compose(
        [
            transforms.Resize(
                (cfg["image_size"], cfg["image_size"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )

    mask_files = sorted(masks_path.glob("*_m.png"))
    if limit > 0:
        mask_files = mask_files[:limit]

    kept_images = []
    meta_path = masks_path / "filter_meta.json"
    if meta_path.exists():
        kept_images = json.loads(meta_path.read_text(encoding="utf-8")).get("kept_images", [])
    mask_to_rel = {_mask_filename(n): n for n in kept_images}

    if num_shards > 1:
        mask_files = mask_files[shard_id::num_shards]

    pending: list[tuple] = []
    records = []
    for mp in tqdm(mask_files, desc=f"pseudo-descriptors[{shard_id}/{num_shards}]"):
        rel = mask_to_rel.get(mp.name)
        if rel is None:
            stem = mp.name.replace("_m.png", "")
            if "__" in stem:
                species, base = stem.split("__", 1)
                rel_candidates = [f"{species}/{base}{ext}" for ext in (".jpg", ".jpeg", ".png", ".JPG")]
            else:
                rel_candidates = [f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".JPG")]
            img_path = None
            for cand in rel_candidates:
                if (img_root / cand).exists():
                    rel = cand
                    img_path = img_root / cand
                    break
            if img_path is None:
                continue
        else:
            img_path = img_root / rel
            if not img_path.exists():
                continue

        species = name_to_species.get(rel) or name_to_species.get(Path(rel).name)
        if species is None and "/" in rel:
            species = rel.split("/", 1)[0]
        species = species or "UNK"

        img = tf(Image.open(img_path).convert("RGB"))
        mask = Image.open(mp).convert("L").resize(
            (cfg["image_size"], cfg["image_size"]), Image.NEAREST
        )
        pending.append((rel, species, img, np.asarray(mask, dtype=np.int64)))
        if len(pending) >= iqa_bs:
            _flush_descriptor_chunk(cfg, pending, records, iqa_names)
            pending = []
    if pending:
        _flush_descriptor_chunk(cfg, pending, records, iqa_names)

    out_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_c")
    suffix = f"_shard{shard_id}" if num_shards > 1 else ""
    out = out_dir / f"pseudo_descriptors{suffix}.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"Saved {len(records)} → {out}")


def _flush_descriptor_chunk(cfg, pending, records, iqa_names) -> None:
    """Score one chunk's learned NR-IQA in a single forward pass per metric."""
    iqa_rows = extract_learned_iqa_features_batch(
        np.stack([normalize_image_for_iqa(img) for _rel, _sp, img, _m in pending]),
        names=iqa_names,
    )
    for k, (rel, species, img, mask_np) in enumerate(pending):
        feats = extract_all_features(
            img, mask_np, part_labels=cfg["part_labels"], learned_iqa=iqa_rows[k]
        )
        rec = {"image": rel, "species": species}
        rec.update(feats)
        records.append(rec)


def cmd_extract_desc_all_gpus(cfg, masks_dir, images_subdir, labels_csv, limit):
    """Shard pseudo-descriptor extraction over every GPU, then merge."""
    n = resolve_num_gpus(cfg)
    if n <= 1:
        return cmd_extract_desc(cfg, masks_dir, images_subdir, labels_csv, limit)

    script = str(Path(__file__).resolve())
    cmds = []
    for shard in range(n):
        cmd = [
            sys.executable, script, "extract_desc",
            "--config", cfg.get("_config_path", default_config_path()),
            *dataset_cli_args(cfg),
            "--masks_dir", str(masks_dir),
            "--labels_csv", labels_csv,
            "--shard_id", str(shard), "--num_shards", str(n),
        ]
        # See cmd_pseudo: --images_subdir defaults to None, and forwarding None
        # to Popen raises TypeError before any shard starts.
        if images_subdir:
            cmd += ["--images_subdir", images_subdir]
        if limit:
            cmd += ["--limit", str(limit)]
        cmds.append(cmd)

    print(f"Stage C extract_desc: {n} shards across {n} GPU(s)")
    procs = []
    for gid, cmd in enumerate(cmds):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gid))
        procs.append(_shard_popen(cmd, cwd=str(Path(script).parent), env=env, idx=gid))
    codes = _shard_wait(procs)
    if any(c != 0 for c in codes):
        raise SystemExit(f"{sum(c != 0 for c in codes)} descriptor shard(s) failed; not merging.")

    out_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_c")
    parts = []
    for shard in range(n):
        p = out_dir / f"pseudo_descriptors_shard{shard}.csv"
        if not p.exists():
            raise SystemExit(f"Missing shard output {p}; refusing to write a partial CSV.")
        df = read_shard_csv(p)
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        raise SystemExit(
            f"All {n} descriptor shards were empty; nothing to merge. Check that "
            f"the mask directory is populated and that --limit is not 0."
        )
    merged = pd.concat(parts, ignore_index=True).drop_duplicates("image", keep="last")
    merged = merged.sort_values("image").reset_index(drop=True)
    out = out_dir / "pseudo_descriptors.csv"
    merged.to_csv(out, index=False)
    for shard in range(n):
        (out_dir / f"pseudo_descriptors_shard{shard}.csv").unlink(missing_ok=True)
    print(f"Merged {n} shards → {out} ({len(merged)} rows)")


def _large_cls_splits(cfg, image_size: int | None = None):
    """Return (ds_all, train_idx, val_idx, test_idx) for Stage C DDP."""
    ds_all = build_large_cls_dataset(cfg, fold=None, image_size=image_size)
    split_path = frozen_split_path(cfg, kind="cls")
    if not split_path.exists():
        raise SystemExit(
            f"Frozen CLS splits required: {split_path}. "
            "Run: ./run.sh stage_a.py freeze --config config.yaml --dataset "
            f"{cfg['dataset']}"
        )
    payload = load_splits(split_path)
    validate_frozen_splits(ds_all.image_names, payload, require_complete=False)
    name_to_idx = {name: i for i, name in enumerate(ds_all.image_names)}
    train_idx = [name_to_idx[name] for name in payload["train"]]
    val_idx = [name_to_idx[name] for name in payload["val"]]
    test_idx = [name_to_idx[name] for name in payload["test"]]
    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            f"Frozen CLS split is empty: train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )
    return ds_all, train_idx, val_idx, test_idx


def cmd_ddp(
    cfg,
    mode: str,
    backbone: str,
    desc_csv: str,
    masks_dir: str,
    max_epochs: int,
    subset: int,
    device: int = 0,
    num_gpus: int | None = None,
    seed: int | None = None,
    frac: float = 1.0,
):
    """Train one large-corpus classifier across `compute.num_gpus` devices.

    This is the only true DDP job in the pipeline and by far the largest
    training set (Beemachine 146k / Fish-Vista 39k / CUB 7.4k images), so the
    default is every configured GPU, not one. It used to default to 1, and
    neither run_all_experiments.sh nor the README passed `--num_gpus`, so the
    single most expensive training step in the study ran on 1 of 8 A40s with
    the other 7 idle. `per_device_batch_size` already divides the configured
    global batch across ranks, so the effective batch is unchanged.
    """
    n_gpus = resolve_num_gpus(cfg) if num_gpus is None else max(1, int(num_gpus))
    if n_gpus > 1:
        ensure_torchrun(n_gpus)
        rank, local_rank, world = init_distributed()
        device_t = device_for_rank(local_rank)
    else:
        rank, local_rank, world = 0, 0, 1
        device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    run_seed = int(seed) if seed is not None else int(cfg["seed"])
    seed_everything(run_seed, rank)

    ds_all, train_idx, val_idx, test_idx = _large_cls_splits(cfg)
    # Scaling curve. `--frac` takes a stratified-by-position slice of the frozen
    # training split; `--subset` remains the absolute-count form. Both keep
    # val/test at full size, so points on the curve differ only in how much
    # training data the arm saw. Slicing with a stride rather than a prefix
    # matters because the frozen split is grouped by species: `train_idx[:N]`
    # would hand a 12.5% point only the alphabetically-first species.
    if 0.0 < float(frac) < 1.0:
        step = max(1, int(round(1.0 / float(frac))))
        train_idx = train_idx[::step]
    if subset > 0:
        train_idx = train_idx[:subset]

    bs = per_device_batch_size(cfg["classification"]["batch_size"], world)

    if mode == "whole":
        train_set = Subset(ds_all, train_idx)
        val_set = Subset(ds_all, val_idx)
        test_set = Subset(ds_all, test_idx)
        model = TimmClassifier(backbone, pretrained=True, num_classes=len(ds_all.classes))

        def batch_fn(batch):
            return batch[0], batch[1]

    else:
        if not desc_csv:
            raise SystemExit("--desc_csv is required for descriptor modes")
        if mode == "gated_residual" and not masks_dir:
            raise SystemExit("--masks_dir is required for gated_residual")
        desc_df = pd.read_csv(desc_csv)
        if "image" not in desc_df.columns:
            raise SystemExit(f"{desc_csv} must contain an image column")
        feature_cols = [
            c for c in desc_df.columns if c not in {"image", "species", "class_id"}
        ]
        if not feature_cols:
            raise SystemExit(f"{desc_csv} contains no descriptor columns")
        feat_dim = len(feature_cols)
        # Z-score on the training fold only, then share across all three folds --
        # without this the raw `*_area` columns dominate the fusion head's
        # LayerNorm and its backbone half is discarded (see
        # descriptors.fit_descriptor_standardizer).
        desc_stats = fit_descriptor_standardizer(
            desc_df, feature_cols,
            train_images=[ds_all.image_names[j] for j in train_idx],
        )
        train_set = LargeImageDescriptorDataset(
            ds_all, train_idx, desc_df, feature_cols, masks_dir or None,
            image_size=cls_image_size(cfg), cfg=cfg, desc_stats=desc_stats,
        )
        val_set = LargeImageDescriptorDataset(
            ds_all, val_idx, desc_df, feature_cols, masks_dir or None,
            image_size=cls_image_size(cfg), cfg=cfg, desc_stats=desc_stats,
        )
        test_set = LargeImageDescriptorDataset(
            ds_all, test_idx, desc_df, feature_cols, masks_dir or None,
            image_size=cls_image_size(cfg), cfg=cfg, desc_stats=desc_stats,
        )
        model = PartAwareFusionClassifier(
            num_classes=len(ds_all.classes),
            backbone_name=backbone,
            descriptor_dim=feat_dim,
            fusion_mode="gated_residual" if mode == "gated_residual" else "concat",
            shape_embed_dim=shape_embed_dim(cfg),
            use_descriptors=True,
            n_parts=len(cfg["part_labels"]),
        )

        def batch_fn(batch):
            if mode == "gated_residual":
                return (batch[0], batch[3], batch[4]), batch[1]
            return (batch[0], batch[3]), batch[1]

    train_loader, train_sampler = make_loader(
        train_set,
        bs,
        shuffle=True,
        num_workers=cfg["num_workers"],
        world_size=world,
    )
    val_loader, _ = make_loader(
        val_set,
        bs,
        shuffle=False,
        num_workers=cfg["num_workers"],
        world_size=world,
    )
    model = model.to(device_t)
    model = wrap_ddp(model, local_rank, world)

    safe_bb = backbone.replace("/", "_")
    out_dir = ensure_dir(
        stage_run_dir(
            cfg, "stage_c", "scaling",
            f"{mode}_n{len(train_idx)}_{safe_bb}_seed{run_seed}",
        )
    )
    if is_main_process():
        print(
            f"Stage C [{cfg['dataset']}] layout={getattr(ds_all, 'layout', '?')} "
            f"world={world} batch={bs} n_train={len(train_idx)} n_classes={len(ds_all.classes)}"
        )

    fit_cls_model(
        model,
        train_loader,
        val_loader,
        device_t,
        cfg["classification"]["lr"],
        max_epochs,
        out_dir / "best.pt",
        batch_fn=batch_fn,
        train_sampler=train_sampler,
        # Safe under DDP: run_cls_epoch all-reduces the validation loss, so
        # every rank sees the same number and breaks on the same epoch.
        patience=cls_patience(cfg),
        criterion=make_cls_criterion(cfg),
    )

    if is_main_process():
        test_loader = DataLoader(
            test_set,
            batch_size=bs,
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg["num_workers"] > 0,
        )
        train_counts = train_counts_from_indices(ds_all.species_ids, train_idx)
        report = evaluate_cls_loader(
            model,
            test_loader,
            device_t,
            train_counts,
            cfg["metrics"]["long_tail_bins"],
            batch_fn=batch_fn,
        )
        report.update(
            {
                "dataset": cfg["dataset"],
                "mode": mode,
                "backbone": backbone,
                "n_train": len(train_idx),
                "seed": run_seed,
                "code_version": code_version(),
                "source_digest": source_digest(),
            }
        )
        (out_dir / "test_metrics.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        split_path = frozen_split_path(cfg, kind="cls")
        split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
        (out_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "dataset": cfg["dataset"],
                    "mode": mode,
                    "n_train": len(train_idx),
                    "backbone": backbone,
                    "world_size": world,
                    "batch_size": bs,
                    "layout": getattr(ds_all, "layout", None),
                    "n_classes": len(ds_all.classes),
                    "split_path": str(split_path),
                    "split_sha256": split_hash,
                    "seed": run_seed,
                    "code_version": code_version(),
                    "source_digest": source_digest(),
                    "desc_csv": desc_csv or None,
                    "masks_dir": masks_dir or None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
    if world > 1:
        cleanup_distributed()


def cmd_plot_curve(metrics_glob: str, out: str, cfg: dict | None = None):
    """Aggregate the scaling runs into one curve.

    `cfg`, when given, restricts the glob's matches to the active protocol.
    Every protocol's scaling runs share the stage_c/scaling tree and differ only
    by a directory-name suffix, and the default protocol's suffix is empty --
    so an unfiltered glob would silently average a 320px default run together
    with its 384px capacity-matched counterpart.
    """
    rows = []
    # recursive=True is required for the `**` in the documented
    # `.../scaling/**/test_metrics.json` pattern to descend more than one
    # level; without it `**` silently degrades to a single-level `*`.
    for path in sorted(glob(metrics_glob, recursive=True)):
        if cfg is not None and not belongs_to_protocol(Path(path).parent.name, cfg):
            continue
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append({"path": path, **{k: m[k] for k in m if k != "long_tail"}})
    df = pd.DataFrame(rows)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df)
    print(f"Wrote {out_path}")

    if plt is not None and "n_train" in df.columns and "top1" in df.columns:
        # Aggregate over seeds at each (mode, n_train). With one seed this is
        # the old behaviour; with the >=3 the plan requires, the band is the
        # min-max spread across seeds, so a curve whose points are within seed
        # noise of each other is visible as such instead of reading as a trend.
        keys = ["mode", "n_train"] if "mode" in df.columns else ["n_train"]
        agg = (
            df.groupby(keys)["top1"]
            .agg(mean="mean", lo="min", hi="max", n="size")
            .reset_index()
            .sort_values("n_train")
        )
        fig, ax = plt.subplots()
        grouped = agg.groupby("mode") if "mode" in agg.columns else [("all", agg)]
        for mode, g in grouped:
            line, = ax.plot(g["n_train"], g["mean"], marker="o", label=str(mode))
            if (g["n"] > 1).any():
                ax.fill_between(
                    g["n_train"], g["lo"], g["hi"], alpha=0.15, color=line.get_color()
                )
        ax.set_xlabel("train size")
        ax.set_ylabel("top-1")
        ax.set_xscale("log")
        ax.legend()
        fig.savefig(out_path.with_suffix(".png"), dpi=140)
        plt.close(fig)
        agg.to_csv(out_path.with_name(out_path.stem + "_agg.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Stage C — scale / pseudo-parts")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pseudo")
    add_global_stage_args(p)
    p.add_argument("--seg_ckpt", default=None)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    # None, not "images": every consumer already falls back to the active
    # dataset's configured `images_subdir`, and a non-None CLI default
    # silently overrides it. Fish-Vista stores images under "Images", so the
    # old default sent Stage C looking for a lowercase directory that does
    # not exist on a case-sensitive filesystem.
    p.add_argument("--images_subdir", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--all_gpus",
        action="store_true",
        help="Launch one pseudo shard per GPU (uses compute.num_gpus).",
    )

    p = sub.add_parser("filter")
    add_global_stage_args(p)
    p.add_argument("--conf", type=float, default=0.7)

    p = sub.add_parser(
        "filter_sweep",
        help="Sweep pseudo_labeling.conf_thresholds and report retention per tau "
        "(RQ3's confidence-gating threshold sweep).",
    )
    add_global_stage_args(p)

    p = sub.add_parser("extract_desc")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument(
        "--all_gpus",
        action="store_true",
        help="Shard extraction across compute.num_gpus GPUs, then merge (recommended).",
    )
    add_global_stage_args(p)
    p.add_argument("--masks_dir", required=True)
    # None, not "images": every consumer already falls back to the active
    # dataset's configured `images_subdir`, and a non-None CLI default
    # silently overrides it. Fish-Vista stores images under "Images", so the
    # old default sent Stage C looking for a lowercase directory that does
    # not exist on a case-sensitive filesystem.
    p.add_argument("--images_subdir", default=None)
    p.add_argument("--labels_csv", default="species_labels.csv")
    p.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("ddp")
    add_global_stage_args(p)
    p.add_argument(
        "--mode",
        # Crop-based arms (partcrop late fusion, attention_parts) are
        # intentionally not offered at Stage C scale: they cost K+1 backbone
        # passes per image, which is prohibitive at ~195k/56k images. This
        # scalability gap is reported in the paper rather than worked around.
        #
        # `concat` is spelled the same as Stage B's arm on purpose. It used to
        # be `descriptor_concat` here, so the same mechanism carried two names
        # across the two stages and no pooled table could join them;
        # `descriptor_concat` is still accepted and normalised below.
        choices=["whole", "concat", "descriptor_concat", "gated_residual"],
        default="whole",
    )
    p.add_argument("--backbone", default="convnext_nano.in12k")
    p.add_argument("--desc_csv", default="")
    p.add_argument("--masks_dir", default="")
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="Fraction of the frozen training split to train on, for the "
        "scaling curve (e.g. 0.125 0.25 0.5 1.0). Val/test stay full size.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed. Default: config `seed`. Stage C ran single-seed "
        "through 2026-08, so no at-scale CI was computable; pass each of "
        "config `seeds` to get one.",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Devices for this job. Default: compute.num_gpus (all 8 A40s), "
        "which runs DDP via torchrun. Pass 1 to force a single GPU.",
    )

    p = sub.add_parser("plot_curve")
    add_global_stage_args(p)
    p.add_argument("--metrics_glob", required=True)
    p.add_argument("--out", default=None)

    args = ap.parse_args()
    cfg = load_config(args.config, dataset=args.dataset, protocol=getattr(args, "protocol", None))

    if args.cmd == "pseudo":
        cmd_pseudo(
            cfg,
            args.seg_ckpt,
            args.gpu_id,
            args.num_shards,
            args.images_subdir,
            all_gpus=args.all_gpus,
            config_path=args.config,
            limit=args.limit,
        )
    elif args.cmd == "filter":
        cmd_filter(cfg, args.conf)
    elif args.cmd == "filter_sweep":
        cmd_filter_sweep(cfg)
    elif args.cmd == "extract_desc":
        if args.all_gpus:
            cmd_extract_desc_all_gpus(
                cfg, args.masks_dir, args.images_subdir, args.labels_csv, args.limit
            )
        else:
            cmd_extract_desc(
                cfg, args.masks_dir, args.images_subdir, args.labels_csv, args.limit,
                shard_id=args.shard_id, num_shards=args.num_shards,
            )
    elif args.cmd == "ddp":
        cmd_ddp(
            cfg,
            # Legacy spelling of the same arm; normalised so Stage B and
            # Stage C rows join on one `mode` value.
            "concat" if args.mode == "descriptor_concat" else args.mode,
            args.backbone,
            args.desc_csv,
            args.masks_dir,
            args.max_epochs,
            args.subset,
            device=args.device,
            num_gpus=args.num_gpus,
            seed=args.seed,
            frac=args.frac,
        )
    elif args.cmd == "plot_curve":
        out = args.out or str(
            stage_run_dir(cfg, "stage_c", "curves", "scaling_curve.csv")
        )
        cmd_plot_curve(args.metrics_glob, out, cfg)


if __name__ == "__main__":
    main()
