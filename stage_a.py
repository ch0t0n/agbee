#!/usr/bin/env python3
"""Stage A CLI: freeze | qa | train | eval | gallery | sweep | pick_best."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import add_global_stage_args, dataset_cli_args, ensure_dir, load_config
from data import (
    build_part_dataset,
    cmd_freeze_splits,
    fold_names as _fold_names,
    part_train_val_test,
)
from distributed_utils import (
    auto_batch_size,
    mib_per_seg_sample,
    per_device_batch_size,
    resolve_num_gpus,
    run_commands_parallel,
)
from metrics import evaluate_seg_loader
from segmenters import (
    build_segmenter,
    load_segmenter,
    make_seg_trainer,
    resolve_model_spec,
    run_name,
)

# Distinct colors for up to 16 part ids (bg + parts)
_PALETTE = [
    (0, 0, 0),
    (220, 60, 60),
    (60, 140, 255),
    (60, 200, 100),
    (255, 180, 40),
    (180, 60, 200),
    (40, 200, 200),
    (200, 100, 40),
    (100, 100, 255),
    (255, 100, 180),
    (140, 220, 80),
    (80, 80, 160),
    (200, 200, 60),
    (160, 80, 80),
    (80, 160, 120),
    (120, 120, 120),
]

def _colorize(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k in np.unique(mask):
        rgb[mask == k] = _PALETTE[int(k) % len(_PALETTE)]
    return rgb


def _best_ckpt(run_dir: Path) -> Path | None:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        ckpt = json.loads(meta.read_text(encoding="utf-8")).get("best_ckpt")
        if ckpt and Path(ckpt).exists():
            return Path(ckpt)
    ckpts = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    return ckpts[0] if ckpts else None


def cmd_qa(cfg, max_check: int):
    from collections import Counter

    if cfg.get("layout") == "fish_vista":
        ref, train_ds, val_ds, test_ds = part_train_val_test(
            cfg, cfg["image_size"], use_train_aug=False
        )
        from torch.utils.data import ConcatDataset

        ds = ConcatDataset([train_ds, val_ds, test_ds])
        labels = cfg["part_labels"]
        # `part_train_val_test` returns each fold wrapped in a Subset, which
        # carries none of the dataset's metadata. Read the class list off the
        # reference dataset it returns, and map per-sample names through the
        # Subset indices so they stay aligned with `ds`. Reading them off the
        # folds directly raised "'Subset' object has no attribute 'classes'".
        n_species = len(ref.classes)
        species_names, image_names = [], []
        for fold in (train_ds, val_ds, test_ds):
            fold_species, fold_images = _fold_names(fold)
            species_names += fold_species
            image_names += fold_images
    else:
        ds = build_part_dataset(cfg, image_size=cfg["image_size"])
        labels = ds.labels
        n_species = len(ds.classes)
        species_names = ds.species_names
        image_names = ds.image_names

    empty, part_hist = [], Counter()
    n = len(ds) if max_check <= 0 else min(max_check, len(ds))
    for i in tqdm(range(n), desc="QA masks"):
        _, mask, _ = ds[i]
        m = mask.numpy()
        if (m > 0).sum() == 0:
            empty.append(image_names[i])
        for pid in np.unique(m):
            part_hist[int(pid)] += 1
    report = {
        "dataset": cfg["dataset"],
        "n_images": len(ds),
        "n_species": n_species,
        "species_lt5": sum(1 for _, c in Counter(species_names).items() if c < 5),
        "empty_masks": empty,
        "part_id_presence_counts": dict(sorted(part_hist.items())),
        "part_labels": labels,
    }
    out = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_a") / "qa_summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "empty_masks"}, indent=2))
    print(f"empty_masks={len(empty)} → {out}")


def cmd_train(cfg, backend, arch, encoder, device: int, num_gpus: int | None = None):
    """Train one segmenter on a single GPU (``device``)."""
    backend, arch, encoder = resolve_model_spec(cfg, arch, encoder, backend)
    pl.seed_everything(cfg["seed"], workers=True)

    ref, train_ds, val_ds, _ = part_train_val_test(
        cfg, cfg["image_size"]
    )
    entry = cfg.get("dataset_entry") or {}
    use_aug = bool(entry.get("use_train_aug", True))
    print(
        f"Using dataset={cfg['dataset']} layout={cfg.get('layout')} "
        f"split_policy={cfg.get('split_policy')} use_train_aug={use_aug} "
        f"n_train={len(train_ds)} n_val={len(val_ds)}"
    )

    n_gpus = 1 if num_gpus is None else max(1, int(num_gpus))
    nw = cfg["num_workers"]
    loader_kw = dict(
        num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0
    )
    global_bs = cfg["segmentation"]["batch_size"]
    # Scale the global batch down if current GPU free memory is insufficient.
    # Each subprocess runs with CUDA_VISIBLE_DEVICES pinned to one device so
    # device_index=0 always refers to the right physical GPU here.
    global_bs = auto_batch_size(
        device,
        mib_per_seg_sample(cfg["image_size"], backend=backend, arch=arch),
        global_bs,
    )
    # `batch_size` is the global (per-job) batch; Lightning's DDP strategy runs
    # one process per GPU, each pulling its own batch, so the per-process batch
    # must be the global batch divided by n_gpus or the effective batch size
    # silently multiplies by n_gpus.
    bs = per_device_batch_size(global_bs, n_gpus)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kw)
    model = build_segmenter(backend, arch, encoder, ref.num_parts, cfg)
    name = run_name(backend, arch, encoder)
    out_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_a" / name)
    trainer, ckpt_cb = make_seg_trainer(
        out_dir,
        cfg["segmentation"]["epochs"],
        cfg["segmentation"]["patience"],
        device=device,
        num_gpus=n_gpus,
    )
    print(
        f"Stage A train {name}: device={device}, num_gpus={n_gpus}, "
        f"global_batch_size={global_bs}, per_device_batch_size={bs}, n_parts={ref.num_parts}"
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if trainer.is_global_zero:
        meta = {
            "dataset": cfg["dataset"],
            "backend": backend,
            "arch": arch,
            "encoder": encoder,
            "run_name": name,
            "best_ckpt": ckpt_cb.best_model_path,
            "best_score": float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None
            else None,
            "device": device,
            "num_gpus": n_gpus,
            "global_batch_size": global_bs,
            "batch_size": bs,
            "num_parts": ref.num_parts,
            "part_labels": cfg["part_labels"],
            "use_train_aug": use_aug,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        }
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))


@torch.no_grad()
def cmd_eval(cfg, ckpt, split, backend, arch, encoder, run_dir, device: int):
    ref, train_ds, val_ds, test_ds = part_train_val_test(
        cfg, cfg["image_size"], use_train_aug=False
    )
    fold_ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    loader = DataLoader(
        fold_ds, batch_size=32, shuffle=False, num_workers=cfg["num_workers"]
    )
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = load_segmenter(ckpt, device_t, backend=backend)

    class _Tqdm:
        def __init__(self, loader, desc):
            self.loader, self.desc = loader, desc

        def __iter__(self):
            return iter(tqdm(self.loader, desc=self.desc))

    report = evaluate_seg_loader(
        model, _Tqdm(loader, f"eval-{split}"), device_t, ref.num_parts, ref.labels
    )
    report.update(
        {
            "dataset": cfg["dataset"],
            "split": split,
            "ckpt": str(ckpt),
            "backend": backend,
            "arch": arch,
            "encoder": encoder,
        }
    )
    if run_dir:
        out_dir = ensure_dir(Path(run_dir))
    elif arch and encoder:
        out_dir = ensure_dir(
            Path(cfg["paths"]["output_root"])
            / "stage_a"
            / run_name(backend or "smp", arch, encoder)
        )
    else:
        out_dir = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_a")
    out = out_dir / f"metrics_{split}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


@torch.no_grad()
def cmd_gallery(cfg, ckpt, backend, run_dir, n: int, device: int):
    ref, _, _, test_ds = part_train_val_test(
        cfg, cfg["image_size"], use_train_aug=False
    )
    idxs = list(range(min(n, len(test_ds))))
    device_t = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = load_segmenter(ckpt, device_t, backend=backend)
    out_dir = ensure_dir(
        Path(run_dir) / "gallery"
        if run_dir
        else Path(cfg["paths"]["output_root"]) / "stage_a" / "gallery"
    )
    for i, (img, mask, _) in enumerate(DataLoader(Subset(test_ds, idxs), batch_size=1)):
        pred = model(img.to(device_t)).argmax(1)[0].cpu().numpy()
        rgb = (img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(rgb)
        ax[0].set_title("image")
        ax[1].imshow(_colorize(mask[0].numpy()))
        ax[1].set_title("GT")
        ax[2].imshow(_colorize(pred))
        ax[2].set_title("pred")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:03d}.png", dpi=120)
        plt.close(fig)
    print(f"Wrote {n} panels → {out_dir}")


def cmd_sweep(
    cfg,
    device: int,
    skip_train: bool,
    skip_gallery: bool,
    n_gallery: int,
    num_gpus: int | None = None,
    models_key: str = "models",
):
    """Train one segmenter per GPU in parallel, then eval/gallery.

    `models_key` selects which list in `config.segmentation` to sweep. The
    default `models` holds the encoder-matched comparison (every decoder on
    resnext50_32x4d), which is the one the paper's Stage A table reports;
    `models_encoder_free` is the optional companion sweep that lets each
    decoder pick its own encoder.
    """
    import sys

    stage_root = ensure_dir(Path(cfg["paths"]["output_root"]) / "stage_a")
    n_gpus = resolve_num_gpus(cfg) if num_gpus is None else max(1, int(num_gpus))
    models = cfg["segmentation"].get(models_key)
    if not models:
        raise SystemExit(
            f"config.segmentation.{models_key!r} is empty or missing; "
            f"available keys: "
            f"{sorted(k for k, v in cfg['segmentation'].items() if isinstance(v, list))}"
        )
    config_path = str(Path(cfg.get("_config_path", "config.yaml")))
    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        if i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]

    if not skip_train:
        cmds = []
        # There are 9 models and 8 GPUs, so exactly one job waits for a free
        # device; `models` keeps config order so sweep_summary.json and the
        # Stage E table stay in the documented order.
        for spec in models:
            backend, arch, encoder = spec.get("backend", "smp"), spec["arch"], spec["encoder"]
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "train",
                "--config",
                config_path,
                *dataset_cli_args(cfg),
                "--backend",
                backend,
                "--arch",
                arch,
                "--encoder",
                encoder,
                "--num_gpus",
                "1",
            ]
            cmds.append(cmd)
        print(f"Stage A sweep ({cfg['dataset']}): {len(cmds)} models on {n_gpus} GPU(s)")
        codes = run_commands_parallel(cmds, n_gpus, cwd=str(Path(__file__).resolve().parent))
        if any(c != 0 for c in codes):
            # Hard failure, not a warning. A partial sweep still writes
            # sweep_summary.json, so the runner's artifact check passes, the
            # resume marker is written, and `pick_best` then selects the best
            # of whatever happened to survive -- silently freezing a segmenter
            # chosen from an incomplete comparison, which every predicted-mask
            # arm downstream then depends on.
            # `models`, not a separate launch_order: `cmds` is built by one
            # pass over `models` above, so `codes` is index-aligned with it.
            # This used to name a `launch_order` that is bound nowhere in this
            # module, so the branch that exists to report WHICH architectures
            # died raised NameError instead -- and only ever ran when a job had
            # already failed, which is why it survived every green sweep.
            failed = [
                f"{s.get('arch')}/{s.get('encoder')}"
                for s, c in zip(models, codes)
                if c != 0
            ]
            raise SystemExit(
                f"{len(failed)} of {len(codes)} Stage A train job(s) failed: "
                f"{', '.join(failed)}. Fix the cause and re-run this step; "
                "the sweep is not usable until every model in it has trained."
            )

    summary = []
    for spec in models:
        backend, arch, encoder = spec.get("backend", "smp"), spec["arch"], spec["encoder"]
        name = run_name(backend, arch, encoder)
        run_dir = stage_root / name
        print(f"\n=== Sweep post: {name} ===")
        ckpt = _best_ckpt(run_dir)
        if ckpt is None:
            print(f"WARNING: no checkpoint for {name}")
            continue
        for split in ("val", "test"):
            cmd_eval(cfg, str(ckpt), split, backend, arch, encoder, str(run_dir), device)
        if not skip_gallery:
            cmd_gallery(cfg, str(ckpt), backend, str(run_dir), n_gallery, device)
        row = {
            "dataset": cfg["dataset"],
            "backend": backend,
            "arch": arch,
            "encoder": encoder,
            "run_name": name,
            "ckpt": str(ckpt),
        }
        for split in ("val", "test"):
            mp = run_dir / f"metrics_{split}.json"
            if mp.exists():
                m = json.loads(mp.read_text(encoding="utf-8"))
                row[f"{split}_miou"] = m.get("miou", m.get("dataset_iou"))
                # Carried alongside macro mIoU because prior work (Choton et
                # al., 2026) reports this one -- smp's reduction="micro" IoU,
                # pooled over all pixels and therefore dominated by background.
                # Without both numbers the Stage A table looks like a 40-point
                # collapse against that paper on CUB when the models are in
                # fact comparable; see docs/padc_vs_new_report.md section 2.
                row[f"{split}_dataset_iou"] = m.get("dataset_iou")
                row[f"{split}_per_part_iou"] = m.get("per_part_iou", {})
        summary.append(row)
    out = stage_root / "sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSweep summary → {out}")


def cmd_pick_best(cfg, summary_path: str | None):
    stage_root = Path(cfg["paths"]["output_root"]) / "stage_a"
    summary_path = Path(summary_path) if summary_path else stage_root / "sweep_summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    ranked = [r for r in rows if r.get("val_miou") is not None]
    if not ranked:
        raise SystemExit("No val_miou in sweep summary")
    best = max(ranked, key=lambda r: float(r["val_miou"]))
    payload = {
        "dataset": cfg["dataset"],
        "selection_metric": "val_miou",
        "backend": best["backend"],
        "arch": best["arch"],
        "encoder": best["encoder"],
        "run_name": best["run_name"],
        "ckpt": best["ckpt"],
        "val_miou": best.get("val_miou"),
        "test_miou": best.get("test_miou"),
        "val_per_part_iou": best.get("val_per_part_iou"),
        "test_per_part_iou": best.get("test_per_part_iou"),
    }
    out = ensure_dir(stage_root) / "best_model.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Stage A — segmentation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("freeze")
    add_global_stage_args(p)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("qa")
    add_global_stage_args(p)
    p.add_argument("--max_check", type=int, default=0)

    p = sub.add_parser("train")
    add_global_stage_args(p)
    p.add_argument("--backend", choices=["smp"])
    p.add_argument("--arch")
    p.add_argument("--encoder")
    p.add_argument("--device", type=int, default=0, help="CUDA index for this single-model job")
    p.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Usually 1 (one model per GPU). Sweep fans out across compute.num_gpus.",
    )

    p = sub.add_parser("eval")
    add_global_stage_args(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--backend", choices=["smp"])
    p.add_argument("--arch")
    p.add_argument("--encoder")
    p.add_argument("--run_dir")
    p.add_argument("--device", type=int, default=0)

    p = sub.add_parser("gallery")
    add_global_stage_args(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--backend", choices=["smp"])
    p.add_argument("--run_dir")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--device", type=int, default=0)

    p = sub.add_parser("sweep")
    add_global_stage_args(p)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=None)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_gallery", action="store_true")
    p.add_argument("--n_gallery", type=int, default=16)
    p.add_argument(
        "--models_key",
        default="models",
        help="config.segmentation list to sweep: 'models' (encoder-matched, "
        "the reported table) or 'models_encoder_free' (companion sweep).",
    )

    p = sub.add_parser("pick_best")
    add_global_stage_args(p)
    p.add_argument("--summary")

    args = ap.parse_args()
    cfg = load_config(args.config, dataset=args.dataset)

    if args.cmd == "freeze":
        cmd_freeze_splits(cfg, force=args.force)
    elif args.cmd == "qa":
        cmd_qa(cfg, args.max_check)
    elif args.cmd == "train":
        cmd_train(cfg, args.backend, args.arch, args.encoder, args.device, num_gpus=args.num_gpus)
    elif args.cmd == "eval":
        cmd_eval(
            cfg,
            args.ckpt,
            args.split,
            args.backend,
            args.arch,
            args.encoder,
            args.run_dir,
            args.device,
        )
    elif args.cmd == "gallery":
        cmd_gallery(cfg, args.ckpt, args.backend, args.run_dir, args.n, args.device)
    elif args.cmd == "sweep":
        cmd_sweep(
            cfg,
            args.device,
            args.skip_train,
            args.skip_gallery,
            args.n_gallery,
            num_gpus=args.num_gpus,
            models_key=args.models_key,
        )
    elif args.cmd == "pick_best":
        cmd_pick_best(cfg, args.summary)


if __name__ == "__main__":
    main()
