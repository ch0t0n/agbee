#!/usr/bin/env python3
"""Standalone runner for the five remaining arms under the heavy-augmentation
recipe, on the common ConvNeXt-Nano backbone.

Body-masked input, part-crop late fusion, attention-pooled part fusion,
descriptor concatenation and gated residual fusion, at three seeds on all three
datasets: 45 training jobs. Together with the whole-image reference and
multi-task supervision, which are already trained under this recipe, that
completes the seven-arm augmentation grid the paper states as unrun -- the grid
that can say whether a heavier recipe helps the anatomy-consuming mechanisms
more than it helps the arm with no anatomical input, rather than only whether it
helps the winner.

Why this is a separate script rather than a flag on the existing runners.

  * ``stage_b.py ablate`` trains the three fusion arms as one pool, but it also
    sweeps the descriptor GROUPS (shape / appearance / interpart), which this
    grid does not need: the paper's mechanism comparison reads the ``all`` cell
    only. Going through ``ablate`` would train 3x more fusion cells than the
    question needs, and on BeeMachine that is measured in GPU-days.
  * ``stage_b.py sweep_controls`` pools masked and partcrop, but it pools them
    with capacity_matched, heavy_aug and multitask, which are already done under
    this recipe.
  * Both pool within one dataset. The 45 jobs here are mutually independent
    across arm, dataset AND seed, so they are submitted as ONE pool over all
    eight GPUs, longest job first. That ordering matters more than usual: on
    BeeMachine the two crop arms make k+1 = 4 backbone passes per image and run
    for the better part of a day each, so leaving them until the end would
    strand seven idle GPUs behind them.

Nothing here writes into the default protocol's tree. Every trained run goes to
the ``__haug``-tagged run directory that ``config.stage_run_dir`` produces, and
every Stage D artifact to ``stage_d__haug/seed<N>/``. ``run_all_experiments.sh``,
``config.yaml`` and every published table are untouched; this script is additive.

Phases, run in order:

  train     45 jobs: 5 arms x {beemachine, cub, fish_vista} x seeds {13, 42, 77}.
  stage_d    9 jobs: one per (dataset, seed), covering all five arms, so
             calibration, selective prediction, long-tail, reliability bins,
             confusion and robustness exist for every new row.
  summary   Reads what the first two phases wrote and prints each new arm
            beside the same arm's default-recipe row.

Resumable, and that matters at this length. A training job whose run directory
already holds ``run_meta.json`` is skipped, so an interrupted run continues by
re-issuing the same command; ``tools/stage_d_arms.py`` skips its own completed
arms.

Usage:

    ./run.sh run_haug_arms.py                    # all three phases
    ./run.sh run_haug_arms.py --dry-run          # print the 54 commands
    ./run.sh run_haug_arms.py --arm concat       # one arm, repeatable
    ./run.sh run_haug_arms.py --phase summary    # re-print the table
    ./run.sh run_haug_arms.py --plain            # no dashboard, for piping
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CODES = Path(__file__).resolve().parent
sys.path.insert(0, str(CODES))

from config import (  # noqa: E402
    cls_corpus,
    default_config_path,
    load_config,
    stage_run_dir,
)
from distributed_utils import resolve_num_gpus  # noqa: E402

# The job pool, the dashboard and the small formatting helpers are imported
# rather than copied. `run_haug_multitask.py` guards its own `main()` behind
# `__main__`, so importing it runs nothing; and a second copy of `run_pool` or
# `Dashboard.draw` would be free to drift from the one that has already been
# used for a published run.
from run_haug_multitask import (  # noqa: E402
    BACKBONE,
    DATASETS,
    PROTOCOL,
    Dashboard,
    Job,
    _fmt,
    _read_json,
    run_pool,
)

#: The five arms this script trains, in the paper's row order. The other two
#: arms of the seven -- the whole-image reference and multi-task supervision --
#: are already trained under this recipe and are not touched here.
ARMS = ("masked", "partcrop", "attention_parts", "concat", "gated_residual")

#: How each arm is launched. `subcmd` is the `stage_b.py` subcommand; `fusion`
#: arms additionally pass `--mode` and need the descriptor CSV. Every arm is
#: pinned to the common backbone and to `--mask_source gt`, which on a
#: `large_cls` corpus is the direct-read path to the confidence-gated
#: pseudo-masks and is reported as "pseudo" (see `stage_b._mask_source_label`).
ARM_SPEC = {
    "masked": {"subcmd": "masked"},
    "partcrop": {"subcmd": "partcrop"},
    "attention_parts": {"subcmd": "fusion", "mode": "attention_parts"},
    "concat": {"subcmd": "fusion", "mode": "concat"},
    "gated_residual": {"subcmd": "fusion", "mode": "gated_residual"},
}

#: The descriptor group every fusion arm in the mechanism comparison uses. The
#: per-group sub-ablation is a separate question and is not repeated here.
GROUP = "all"

#: Relative cost of one job, used only to order the pool longest-first. The two
#: crop arms make k+1 backbone passes per image, with k = 3 on BeeMachine, 11 on
#: CUB-200-2011 and 9 on Fish-Vista; the other three make one. Multiplied by a
#: rough per-dataset epoch cost, this is enough to keep the eight GPUs from
#: finishing their short jobs and waiting on a BeeMachine crop arm.
CROP_ARMS = {"partcrop", "attention_parts"}
DATASET_COST = {"beemachine": 60.0, "fish_vista": 9.0, "cub": 2.5}
PARTS = {"beemachine": 3, "cub": 11, "fish_vista": 9}

DEFAULT_LOG_DIR = CODES / "logs" / "haug_arms"
SUMMARY_DIR = CODES / "outputs" / "pooled" / "haug_arms"


# --------------------------------------------------------------------------
# job construction
# --------------------------------------------------------------------------


def _cfg(config: str, dataset: str, protocol: str | None = PROTOCOL):
    return load_config(config, dataset=dataset, protocol=protocol)


def _seeds(config: str, dataset: str) -> list[int]:
    cfg = _cfg(config, dataset)
    return list(cfg.get("seeds") or [cfg["seed"]])


def _mask_label(cfg) -> str:
    """The run-directory label for ``--mask_source gt``.

    This mirrors ``stage_b._mask_source_label`` rather than importing it.
    Importing ``stage_b`` would pull torch, timm and
    segmentation_models_pytorch into the SCHEDULER process, which never runs a
    model -- it names directories and spawns children -- and on a busy array
    that import costs minutes before the first job starts.

    The rule being mirrored is two lines: a ``large_cls`` corpus has no ground
    truth for classification, so ``--mask_source gt`` selects the direct-read
    path to the confidence-gated pseudo-masks and is recorded as "pseudo";
    anything else keeps its own name. A mirror can drift, so
    ``assert_naming_matches_disk`` checks it against the default recipe's
    directories before any job is submitted.
    """
    return "pseudo" if cls_corpus(cfg) == "large_cls" else "gt"


def assert_naming_matches_disk(config: str, datasets, arms) -> None:
    """Fail now if this script names a run directory differently from Stage B.

    Every arm here is already trained under the DEFAULT recipe, so the default
    protocol's directory for each (arm, dataset, seed) must exist on disk. If
    one does not, this script's idea of the naming convention has drifted from
    ``stage_b``'s, and the consequence is silent and expensive: the
    skip-if-done check never fires, so a resumed run retrains everything, and
    the summary reads zero seeds for rows that trained fine. Checking against
    the completed default grid costs a few stat calls and catches that in a
    second rather than after a day of GPU time.
    """
    missing = []
    for ds in datasets:
        for arm in arms:
            for seed in _seeds(config, ds):
                d = _run_dir_for(config, ds, arm, seed, None)
                if not d.is_dir():
                    missing.append(str(d))
    if missing:
        raise SystemExit(
            "run-directory naming does not match what is on disk; this script "
            "would not recognise its own completed jobs. Missing default-recipe "
            "directories:\n  " + "\n  ".join(missing[:8])
            + (f"\n  ... and {len(missing) - 8} more" if len(missing) > 8 else "")
        )


def _desc_csv(config: str, dataset: str) -> Path:
    """The descriptor CSV the fusion arms read.

    Descriptors are a shared input, not a protocol artifact: they are extracted
    once from the frozen segmenter's pseudo-masks and every protocol reads the
    same file. So this is deliberately NOT `stage_run_dir` -- a `__haug`-tagged
    descriptor path would name a file that does not exist and never will.
    """
    cfg = _cfg(config, dataset, protocol=None)
    # Absolute: `output_root` is written relative to `codes/`, and the children
    # do run with that cwd, but a path this long-lived should not depend on it.
    root = (CODES / Path(cfg["paths"]["output_root"])).resolve()
    if cls_corpus(cfg) == "large_cls":
        return root / "stage_c" / "pseudo_descriptors.csv"
    return root / "stage_b_descriptors" / "descriptors_gt.csv"


def run_dir(config: str, dataset: str, arm: str, seed: int) -> Path:
    """Where one (arm, dataset, seed) job writes, under the active protocol."""
    cfg = _cfg(config, dataset)
    bb = BACKBONE.replace("/", "_")
    label = _mask_label(cfg)
    spec = ARM_SPEC[arm]
    if spec["subcmd"] == "fusion":
        tag = f"{spec['mode']}_{label}_{GROUP}_{bb}_seed{seed}"
        return stage_run_dir(cfg, "stage_b_descriptors", tag)
    tag = f"{spec['subcmd']}_{label}_{bb}_seed{seed}"
    return stage_run_dir(cfg, "stage_b", tag)


def _cost(dataset: str, arm: str) -> float:
    passes = PARTS[dataset] + 1 if arm in CROP_ARMS else 1
    return DATASET_COST[dataset] * passes


def train_jobs(config: str, datasets, arms, log_dir: Path, force: bool) -> list[Job]:
    """One training job per (arm, dataset, seed), longest first, done ones cut.

    `run_meta.json` is the completion mark because `stage_b._finalize_run`
    writes it last, after the checkpoint is saved, the test split is scored and
    the summary row is appended.
    """
    planned = []
    for ds in datasets:
        for arm in arms:
            for seed in _seeds(config, ds):
                out = run_dir(config, ds, arm, seed)
                if not force and (out / "run_meta.json").is_file():
                    print(f"SKIP train {arm}/{ds}/seed{seed}: already complete")
                    continue
                planned.append((_cost(ds, arm), ds, arm, seed))
    planned.sort(key=lambda r: -r[0])

    jobs: list[Job] = []
    for idx, (_c, ds, arm, seed) in enumerate(planned):
        spec = ARM_SPEC[arm]
        argv = [
            sys.executable, str(CODES / "stage_b.py"), spec["subcmd"],
            "--config", config, "--dataset", ds, "--protocol", PROTOCOL,
            "--backbone", BACKBONE, "--mask_source", "gt",
            "--seed", str(seed), "--device", "0",
        ]
        if spec["subcmd"] == "fusion":
            argv += [
                "--mode", spec["mode"], "--group", GROUP,
                "--desc_csv", str(_desc_csv(config, ds)),
            ]
        jobs.append(Job(
            label=f"{arm}/{ds}/{seed}",
            argv=argv,
            log=log_dir / "train.jobs" / f"{idx:03d}_{arm}_{ds}_seed{seed}.log",
        ))
    return jobs


def stage_d_jobs(config: str, datasets, arms, log_dir: Path, force: bool) -> list[Job]:
    """Reliability for these arms only, one job per (dataset, seed).

    `--arms` is what keeps this additive: `tools/stage_d_arms.py` would
    otherwise walk all nine arms, and under `--protocol heavy_aug` the ones this
    script does not train have no checkpoint in the tagged tree, so they would
    be reported as missing on every run. Restricting the arm list also means
    this phase cannot touch another arm's Stage D artifacts.
    """
    jobs: list[Job] = []
    idx = 0
    for ds in datasets:
        for seed in _seeds(config, ds):
            ready = [a for a in arms
                     if (run_dir(config, ds, a, seed) / "run_meta.json").is_file()]
            if not force and not ready:
                print(f"SKIP stage_d {ds}/seed{seed}: no trained checkpoint yet")
                continue
            argv = [
                sys.executable, str(CODES / "tools" / "stage_d_arms.py"),
                "--config", config, "--dataset", ds, "--protocol", PROTOCOL,
                "--arms", ",".join(ready or arms), "--seed", str(seed),
                "--backbone", BACKBONE, "--devices", "0",
            ]
            if force:
                argv.append("--force")
            jobs.append(Job(
                label=f"{ds}/seed{seed} ({len(ready or arms)} arms)",
                argv=argv,
                log=log_dir / "stage_d.jobs" / f"{idx:03d}_{ds}_seed{seed}.log",
            ))
            idx += 1
    return jobs


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def collect_row(config: str, dataset: str, arm: str, protocol: str | None) -> dict:
    """Seed-averaged metrics for one arm on one dataset, under one protocol.

    Accuracy comes from each run's `test_metrics.json`; the rare stratum, ECE,
    rare ECE and AURC come from Stage D, which is where the paper reads them
    (`tools/paper_sources.py`). Reading Stage D for the rare column rather than
    Stage B's own `long_tail` block matters: the two are cut by the same
    quantile rule, but only Stage D's is recut when that rule is revised.
    """
    cfg = _cfg(config, dataset, protocol=protocol)
    seeds = list(cfg.get("seeds") or [cfg["seed"]])
    d_root = stage_run_dir(cfg, "stage_d")
    top1, top3, f1, rare, ece, rare_ece, aurc = ([] for _ in range(7))
    for seed in seeds:
        m = _read_json(
            _run_dir_for(config, dataset, arm, seed, protocol) / "test_metrics.json"
        )
        if m:
            top1.append(100 * m["top1"])
            f1.append(100 * m["macro_f1"])
            if m.get("top3") is not None:
                top3.append(100 * m["top3"])
        sd = d_root / f"seed{seed}"
        lt = _read_json(sd / f"long_tail_{arm}.json")
        if lt:
            rare.append(100 * lt[next(iter(lt))]["top1"])
        rb = _read_json(sd / f"reliability_by_bin_{arm}.json")
        if rb:
            rare_ece.append(100 * rb[next(iter(rb))]["ece"])
        cal = _read_json(sd / f"calibration_{arm}.json")
        if cal:
            ece.append(100 * cal["ece"])
        sel = _read_json(sd / f"selective_{arm}.json")
        if sel:
            aurc.append(sel["aurc"])
    return {
        "dataset": dataset, "arm": arm, "protocol": protocol or "default",
        "n_seeds": len(top1),
        "top1": _mean(top1), "top3": _mean(top3), "macro_f1": _mean(f1),
        "rare": _mean(rare), "ece": _mean(ece), "rare_ece": _mean(rare_ece),
        "aurc": _mean(aurc),
    }


def _run_dir_for(config: str, dataset: str, arm: str, seed: int,
                 protocol: str | None) -> Path:
    """`run_dir`, but for either protocol, so the summary can read both rows."""
    cfg = _cfg(config, dataset, protocol=protocol)
    bb = BACKBONE.replace("/", "_")
    label = _mask_label(cfg)
    spec = ARM_SPEC[arm]
    if spec["subcmd"] == "fusion":
        tag = f"{spec['mode']}_{label}_{GROUP}_{bb}_seed{seed}"
        return stage_run_dir(cfg, "stage_b_descriptors", tag)
    return stage_run_dir(cfg, "stage_b", f"{spec['subcmd']}_{label}_{bb}_seed{seed}")


def summarize(config: str, datasets, arms) -> int:
    """Print each new arm beside the same arm's default-recipe row."""
    rows = []
    for ds in datasets:
        for arm in arms:
            rows.append(collect_row(config, ds, arm, PROTOCOL))
            rows.append(collect_row(config, ds, arm, None))
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["dataset", "arm", "protocol", "n_seeds", "top1", "top3",
            "macro_f1", "rare", "ece", "rare_ece", "aurc"]
    csv_path = SUMMARY_DIR / "haug_arms_summary.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                "" if r[c] is None else
                (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                for c in cols
            ) + "\n")

    lines = [
        "| Dataset | Arm | Recipe | Seeds | Top-1 | Top-3 | Macro-F1 | Rare | ECE | Rare ECE | AURC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        recipe = "heavy aug" if r["protocol"] == PROTOCOL else "default"
        lines.append(
            f"| {r['dataset']} | {r['arm']} | {recipe} | {r['n_seeds']} "
            f"| {_fmt(r['top1'])} | {_fmt(r['top3'])} | {_fmt(r['macro_f1'])} "
            f"| {_fmt(r['rare'])} | {_fmt(r['ece'])} | {_fmt(r['rare_ece'])} "
            f"| {_fmt(r['aurc'], 4)} |"
        )
    md = "\n".join(lines)
    md_path = SUMMARY_DIR / "haug_arms_summary.md"
    md_path.write_text(
        "# The five remaining arms under the heavy-augmentation recipe\n\n"
        "ConvNeXt-Nano, three seeds, protocol `heavy_aug`, descriptor group\n"
        "`all` for the fusion arms. Rare, ECE, Rare ECE and AURC are read from\n"
        "Stage D, as the paper's tables are. Each new row is printed beside the\n"
        "same arm's default-recipe row so the recipe increment is a subtraction\n"
        "of two lines. This file is written by `run_haug_arms.py` and is not\n"
        "read by any checker or table generator.\n\n" + md + "\n",
        encoding="utf-8",
    )
    print("\n" + md + "\n")
    print(f"csv: {csv_path}")
    print(f"md : {md_path}")
    incomplete = [r for r in rows
                  if r["protocol"] == PROTOCOL
                  and r["n_seeds"] < len(_seeds(config, r["dataset"]))]
    if incomplete:
        print("\nWARNING: incomplete under heavy augmentation: " + ", ".join(
            f"{r['arm']}/{r['dataset']} ({r['n_seeds']} seed(s))" for r in incomplete))
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and analyse the five remaining comparison arms "
                    "under the heavy-augmentation recipe.",
    )
    p.add_argument("--config", default=default_config_path())
    p.add_argument("--dataset", action="append", default=None,
                   help="Restrict to a dataset; repeatable. Default: all three.")
    p.add_argument("--arm", action="append", default=None, choices=list(ARMS),
                   help="Restrict to an arm; repeatable. Default: all five.")
    p.add_argument("--phase", action="append", default=None,
                   choices=["train", "stage_d", "summary"],
                   help="Run only this phase; repeatable. Default: all three, in order.")
    p.add_argument("--gpus", type=int, default=None,
                   help="Pool size (default: compute.num_gpus, i.e. all eight).")
    p.add_argument("--force", action="store_true",
                   help="Retrain and re-analyse even where results already exist.")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands each phase would run, then exit.")
    p.add_argument("--plain", action="store_true",
                   help="One line per state change instead of the dashboard.")
    p.add_argument("--no-color", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    datasets = args.dataset or list(DATASETS)
    arms = args.arm or list(ARMS)
    phases = args.phase or ["train", "stage_d", "summary"]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    num_gpus = args.gpus or resolve_num_gpus(_cfg(args.config, datasets[0]))

    assert_naming_matches_disk(args.config, datasets, arms)

    if args.dry_run:
        for phase, builder in (("train", train_jobs), ("stage_d", stage_d_jobs)):
            if phase not in phases:
                continue
            jobs = builder(args.config, datasets, arms, log_dir, args.force)
            print(f"\n=== {phase}: {len(jobs)} job(s) on {num_gpus} GPU(s) ===")
            for job in jobs:
                print("  " + " ".join(job.argv))
        if "summary" in phases:
            print("\n=== summary: reads results only ===")
        return 0

    interactive = sys.stdout.isatty() and not args.plain
    ui = Dashboard(
        title="Five remaining arms + heavy augmentation (ConvNeXt-Nano)",
        plain=not interactive, color=not args.no_color,
    )
    if interactive:
        sys.stdout.write("\033[?25l")
    failures = 0
    try:
        for phase, builder in (("train", train_jobs), ("stage_d", stage_d_jobs)):
            if phase not in phases:
                continue
            jobs = builder(args.config, datasets, arms, log_dir, args.force)
            if not jobs:
                print(f"{phase}: nothing to do")
                continue
            ui.phase, ui.jobs, ui.started = phase, jobs, time.time()
            failures += run_pool(jobs, num_gpus, ui.draw)
            if failures:
                # Stop before Stage D rather than analysing a partial grid: a
                # row averaged over two of three seeds reads as a result.
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to continue; "
              "completed jobs are skipped.")
        return 130
    finally:
        if interactive:
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()

    if failures:
        print(f"\n{failures} job(s) failed. Logs under {log_dir}")
        for job in ui.jobs:
            if job.rc not in (0, None):
                print(f"  {job.label}: rc={job.rc}  {job.log}")
        return 1

    if "summary" in phases:
        return summarize(args.config, datasets, arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
