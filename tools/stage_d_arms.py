#!/usr/bin/env python3
"""Run Stage D's per-arm reliability/robustness analyses for every comparison arm.

Two callers, one implementation:

* ``run_all_experiments.sh``'s ``d_arms`` step calls this for the dataset it is
  currently working through.
* Run it by hand to **backfill** arms that a finished campaign never produced.
  It is idempotent: an arm whose outputs already exist is skipped unless
  ``--force`` is given, so re-running it on a completed dataset costs nothing
  and cannot disturb numbers that are already published.

Why it exists at all. ``d_arms`` used to iterate a hardcoded list of six arms
-- the whole-image reference, the two controls, and three of the six
anatomy-consuming mechanisms. Body-masked input, part-crop late fusion and
multi-task supervision were absent, so the paper's calibration and robustness
tables covered four of the seven comparison arms on CUB and five of seven on
BeeMachine, while every accuracy table covered all seven. The missing three
were not a data problem: all three train fine and their checkpoints are on
disk. They were a loader problem, now fixed in ``stage_d._load_arm``.

Typical uses::

    # backfill just the three arms that were never covered, on a finished dataset
    ./run.sh tools/stage_d_arms.py --dataset cub --arms masked,partcrop,multitask

    # everything for one dataset, on two GPUs, leaving the rest for a live campaign
    ./run.sh tools/stage_d_arms.py --dataset beemachine --devices 2,3

    # see what would run, touching nothing
    ./run.sh tools/stage_d_arms.py --dataset fish_vista --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

CODES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODES))

from config import (  # noqa: E402
    DATASET_CHOICES,
    PROTOCOL_CHOICES,
    cls_corpus,
    default_config_path,
    load_config,
    multitask_run_tag,
    stage_run_dir,
)

#: The seven comparison arms the paper reports, then the two controls that are
#: reported only in the appendix. Order is the paper's, so console output reads
#: in the same order as the tables.
ALL_ARMS = (
    "whole",
    "masked",
    "multitask",
    "partcrop",
    "attention_parts",
    "concat",
    "gated_residual",
    "capacity_matched",
    "heavy_aug",
)

#: Analyses that consume the per-image prediction CSV rather than the model.
#: Cheap, CPU-only, and re-derivable, so they always re-run once preds exist.
_PRED_ANALYSES = (
    ("calibration", "calibration_{arm}.json"),
    ("selective", "selective_{arm}.json"),
    ("long_tail", "long_tail_{arm}.json"),
    ("reliability_bins", "reliability_by_bin_{arm}.json"),
    ("confusion", "confusion_{arm}.csv"),
)


class Arm:
    """One arm's Stage D inputs: where its checkpoint is and how to load it."""

    def __init__(self, name: str, rundir: Path, ckpt: Path | None, mode: str,
                 backbone: str, needs_desc: bool):
        self.name = name
        self.rundir = rundir
        self.ckpt = ckpt
        self.mode = mode
        self.backbone = backbone
        self.needs_desc = needs_desc


def _multitask_ckpt(rundir: Path) -> Path | None:
    """The multi-task arm's Lightning checkpoint.

    This arm is the reason the old shell loop could not have included it even
    with a working loader: it never writes ``best.pt``. Lightning leaves one or
    more ``checkpoints/*.ckpt``, and the gate in ``d_arms`` tested for
    ``best.pt`` and skipped the arm before anything else could run.
    """
    ckpt_dir = rundir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    cands = sorted(ckpt_dir.glob("*.ckpt"))
    if not cands:
        return None
    # Prefer an explicitly-named best checkpoint; otherwise the newest, which
    # is what ModelCheckpoint leaves when it tracks the monitored metric.
    for c in cands:
        if "best" in c.name.lower():
            return c
    return max(cands, key=lambda p: p.stat().st_mtime)


def resolve_arms(cfg, backbone: str, capacity_backbone: str, seed: int) -> list[Arm]:
    """Map each arm name to its run directory, checkpoint and stage_d mode."""
    label = "pseudo" if cls_corpus(cfg) == "large_cls" else "gt"
    safe_bb = backbone.replace("/", "_")
    safe_cap = capacity_backbone.replace("/", "_")
    # From `classification.multitask`, via the same helper `stage_b.cmd_multitask`
    # writes with. This used to be built from `cfg["segmentation"]`, which is
    # Stage A's segmenter pair and stopped naming the multi-task arm's run
    # directory the moment that arm was pinned to the common backbone.
    mt_tag = multitask_run_tag(cfg)

    # (subdir, dirname, mode, backbone, needs_desc)
    spec = {
        "whole":            ("stage_b", f"baseline_{safe_bb}_seed{seed}", "whole", backbone, False),
        "masked":           ("stage_b", f"masked_{label}_{safe_bb}_seed{seed}", "masked", backbone, False),
        "partcrop":         ("stage_b", f"partcrop_{label}_{safe_bb}_seed{seed}", "partcrop", backbone, False),
        "multitask":        ("stage_b", f"multitask_{mt_tag}_seed{seed}", "multitask", backbone, False),
        "capacity_matched": ("stage_b", f"capacity_matched_{safe_cap}_seed{seed}", "whole", capacity_backbone, False),
        "heavy_aug":        ("stage_b", f"heavy_aug_{safe_bb}_seed{seed}", "whole", backbone, False),
    }
    for mode in ("concat", "gated_residual", "attention_parts"):
        spec[mode] = (
            "stage_b_descriptors", f"{mode}_{label}_all_{safe_bb}_seed{seed}",
            mode, backbone, True,
        )

    arms: list[Arm] = []
    for name in ALL_ARMS:
        subdir, dirname, mode, bb, needs_desc = spec[name]
        rundir = stage_run_dir(cfg, subdir, dirname)
        ckpt = _multitask_ckpt(rundir) if name == "multitask" else rundir / "best.pt"
        if ckpt is not None and not ckpt.exists():
            ckpt = None
        arms.append(Arm(name, rundir, ckpt, mode, bb, needs_desc))
    return arms


def _descriptor_csv(cfg, out_root: Path) -> Path:
    """The descriptor vector the fusion arms were trained against.

    Protocol-independent on purpose: every protocol reuses the one descriptor
    CSV, so only the classification arms differ between protocols.
    """
    if cls_corpus(cfg) == "large_cls":
        return out_root / "stage_c" / "pseudo_descriptors.csv"
    return out_root / "stage_b" / "descriptors_gt.csv"


def _desc_provenance_ok(arm: Arm) -> bool:
    """Reject a fusion checkpoint that predates descriptor standardization.

    A checkpoint fit against raw descriptors but scored with z-scored input is
    a silent train/evaluation mismatch, not a crash: it was measured at
    top1=0.7277 for an arm whose own trained value was 0.2827. Cheaper to gate
    here than to discover in a table.
    """
    if not arm.needs_desc:
        return True
    meta = arm.rundir / "test_metrics.json"
    try:
        return '"desc_standardized": true' in meta.read_text(encoding="utf-8").lower()
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config_path())
    ap.add_argument("--dataset", default=None, choices=list(DATASET_CHOICES))
    ap.add_argument("--protocol", default=None, choices=list(PROTOCOL_CHOICES))
    ap.add_argument("--arms", default=None,
                    help=f"Comma-separated subset of: {','.join(ALL_ARMS)} (default: all)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Which seed's checkpoint to analyse. Outputs land in "
                         "stage_d/seed<SEED>/; `tools/stage_d_aggregate.py` then "
                         "averages the seeds into stage_d/ itself, which is what "
                         "the paper reads.")
    ap.add_argument("--backbone", default="convnext_nano.in12k")
    ap.add_argument("--capacity_backbone", default="convnext_small.in12k")
    ap.add_argument("--devices", default=None,
                    help="Comma-separated GPU ids to use, round-robin (e.g. 2,3). "
                         "Set CUDA_VISIBLE_DEVICES yourself to hard-restrict instead. "
                         "Default: device 0.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute arms whose outputs already exist.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset, protocol=args.protocol)
    out_root = Path(cfg["paths"]["output_root"])
    # One directory per seed. Every arm's reliability numbers are a property of
    # a single trained checkpoint, so they carry the same seed dimension the
    # accuracy tables do; writing them all into stage_d/ made the last seed run
    # silently overwrite the others. stage_d/ itself now holds the seed-averaged
    # files that `tools/stage_d_aggregate.py` writes.
    out = stage_run_dir(cfg, "stage_d") / f"seed{args.seed}"
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    wanted = [a.strip() for a in args.arms.split(",")] if args.arms else list(ALL_ARMS)
    unknown = [a for a in wanted if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"Unknown arm(s): {unknown}. Choose from {list(ALL_ARMS)}")

    devices = [int(d) for d in args.devices.split(",")] if args.devices else [0]
    device_cycle = itertools.cycle(devices)

    desc_csv = _descriptor_csv(cfg, out_root)
    arms = {a.name: a for a in resolve_arms(cfg, args.backbone, args.capacity_backbone, args.seed)}

    stage_d = str(CODES / "stage_d.py")
    common = ["--config", args.config, "--dataset", cfg["dataset"]]
    if cfg.get("protocol", "default") != "default":
        common += ["--protocol", cfg["protocol"]]

    def run(cmd: list[str]) -> None:
        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
            return
        subprocess.run(cmd, check=True, cwd=str(CODES))

    ran, skipped, missing = [], [], []
    for name in wanted:
        arm = arms[name]
        preds = out / f"preds_{name}.csv"

        if arm.ckpt is None:
            print(f"SKIP {name}: no checkpoint under {arm.rundir}")
            missing.append(name)
            continue
        if not _desc_provenance_ok(arm):
            print(f"SKIP {name}: {arm.rundir} predates the descriptor-standardization fix.")
            skipped.append(name)
            continue
        if preds.exists() and not args.force:
            print(f"SKIP {name}: already analysed ({preds.name}); pass --force to redo.")
            skipped.append(name)
            continue

        device = next(device_cycle)
        dflag = ["--desc_csv", str(desc_csv)] if arm.needs_desc else []
        print(f"--- stage_d arm={name} mode={arm.mode} backbone={arm.backbone} "
              f"device={device} ckpt={arm.ckpt.name}")

        model_args = ["--ckpt", str(arm.ckpt), "--backbone", arm.backbone,
                      "--mode", arm.mode, *dflag]
        run([sys.executable, stage_d, "dump_preds", *common, *model_args,
             "--device", str(device), "--out", str(preds)])
        run([sys.executable, stage_d, "robustness", *common, *model_args,
             "--device", str(device), "--out", str(out / f"robustness_{name}.csv")])
        for sub, pattern in _PRED_ANALYSES:
            run([sys.executable, stage_d, sub, *common,
                 "--pred_csv", str(preds), "--out", str(out / pattern.format(arm=name))])
        ran.append(name)

    if not args.dry_run:
        _merge_robustness(out)

    print(f"\nstage_d arms: {len(ran)} analysed, {len(skipped)} skipped, "
          f"{len(missing)} without a checkpoint")
    if ran:
        print(f"  analysed:  {', '.join(ran)}")
    if missing:
        print(f"  no ckpt:   {', '.join(missing)}  "
              f"(train these arms first, or pass --seed for a different seed)")
    return 0


def _merge_robustness(out: Path) -> None:
    """One combined robustness.csv across every arm analysed so far.

    The `arm` column is rewritten from each part's filename rather than kept
    from `stage_d robustness`, which writes the classifier *mode*. Three arms
    share mode `whole` -- the reference, the larger-backbone capacity control
    and the heavy-augmentation control -- so trusting `mode` filed all three
    under `whole` and made the combined CSV report one arm with 57 rows.
    """
    import pandas as pd

    parts = sorted(out.glob("robustness_*.csv"))
    if not parts:
        return
    frames = []
    for p in parts:
        frame = pd.read_csv(p)
        frame["arm"] = p.stem[len("robustness_"):]
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "robustness.csv", index=False)
    print(f"combined {len(parts)} arm(s) -> {out / 'robustness.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
