#!/usr/bin/env python3
"""Standalone runner for ONE extra arm: multi-task supervision under the
heavy-augmentation recipe, on the common ConvNeXt-Nano backbone.

Why this is a separate script rather than a flag on the existing runners.
The question it answers is narrow -- Appendix "Controls" shows the
heavy-augmentation recipe is the largest single lever measured on the
whole-image reference, and this checks whether that lever composes with the
auxiliary segmentation loss instead of substituting for it. That is one new
row in one appendix table. It is not a change to the study's comparison grid,
so nothing here writes into the default protocol's tree:

  * every trained run goes to ``stage_b/multitask_..__haug..seed<N>``, and
    every Stage D artifact to ``stage_d__haug/seed<N>/`` -- the protocol tag
    ``__haug`` that ``config.stage_run_dir`` appends. The default grid's
    ``stage_b/multitask_..._seed<N>`` and ``stage_d/`` are never opened for
    writing by anything below.
  * ``run_all_experiments.sh``, ``tools/run_monitor.py``, ``config.yaml`` and
    every published table are untouched. This script is additive.
  * the summary it writes lands in its own directory
    (``outputs/pooled/haug_multitask/``), so no existing CSV or Markdown table
    is rewritten.

After the three phases below, two more commands turn this arm into the rows the
paper carries. ``tools/stage_d_aggregate.py --tree stage_d__haug`` collapses the
per-seed Stage D outputs the way the default grid's are collapsed, and
``tools/deployment_gate.py --gated-tree stage_d__haug --fallback heavy_aug
--suffix __haug`` measures the gated policy for this arm against the
heavy-augmentation whole-image control. Both paper checkers then read this arm
directly -- ``tools/paper_sources.py`` keys it by its ``run_meta.json``
``protocol`` field, apart from the default recipe's runs -- so the appendix and
control-section rows are verified rather than transcribed.

Why it does not reuse ``tools/run_monitor.py``. That driver walks datasets
sequentially, one step per dataset, and this arm is 3 seeds per dataset -- so
under the monitor it would occupy 3 of 8 GPUs and leave 5 idle for the whole
run. The nine training jobs here are mutually independent across BOTH seed and
dataset, so they are submitted as ONE pool of 9 over all 8 GPUs. That is the
only structural difference from how the campaign normally runs; the jobs
themselves are the ordinary ``stage_b.py multitask`` and
``tools/stage_d_arms.py`` entry points, with the ordinary config.

Phases, run in order:

  train     9 jobs: {beemachine, cub, fish_vista} x seeds {13, 42, 77}.
  stage_d   9 jobs: reliability for this arm only (--arms multitask), so
            calibration, selective prediction, long-tail, reliability bins,
            confusion and robustness all exist for the new row.
  summary   Reads what the first two phases wrote and prints the appendix row,
            beside the two rows it has to be read against: the default-recipe
            multi-task arm, and the heavy-augmentation whole-image control.

Resumable. A training job whose run directory already holds ``run_meta.json``
is skipped; ``tools/stage_d_arms.py`` skips its own completed arms. So an
interrupted run continues by re-issuing the same command.

Usage:

    ./run.sh run_haug_multitask.py                 # all three phases
    ./run.sh run_haug_multitask.py --dry-run       # print the 18 commands
    ./run.sh run_haug_multitask.py --phase summary # re-print the row
    ./run.sh run_haug_multitask.py --plain         # no dashboard, for piping
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

CODES = Path(__file__).resolve().parent
sys.path.insert(0, str(CODES))

from config import (  # noqa: E402
    default_config_path,
    load_config,
    multitask_run_tag,
    stage_run_dir,
)
from distributed_utils import resolve_num_gpus  # noqa: E402

#: The protocol whose recipe this arm is being retrained under. `config.yaml`
#: already defines it (`protocols.heavy_aug`), and every arm's training loop
#: already honours it through `stage_b._make_heavy_aug`; nothing new is
#: configured here.
PROTOCOL = "heavy_aug"
#: The one arm this script trains and analyses. Named once so the Stage D
#: filter and the summary reader cannot drift apart.
ARM = "multitask"
#: The grid's common visual backbone. Passed to Stage D explicitly because
#: `tools/stage_d_arms.py` defaults to it but resolves several arms by name;
#: being explicit keeps this readable next to the paper's Table 2.
BACKBONE = "convnext_nano.in12k"
DATASETS = ("beemachine", "cub", "fish_vista")

DEFAULT_LOG_DIR = CODES / "logs" / "haug_multitask"
SUMMARY_DIR = CODES / "outputs" / "pooled" / "haug_multitask"

_ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")


# --------------------------------------------------------------------------
# job pool
# --------------------------------------------------------------------------


@dataclass
class Job:
    """One subprocess in the pool, pinned to one GPU."""

    label: str
    argv: list[str]
    log: Path
    gpu: int | None = None
    started: float | None = None
    finished: float | None = None
    rc: int | None = None

    @property
    def running(self) -> bool:
        return self.started is not None and self.finished is None

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    @property
    def state(self) -> str:
        if self.finished is not None:
            return "ok" if self.rc == 0 else "failed"
        return "running" if self.started is not None else "pending"


def run_pool(jobs: list[Job], num_gpus: int, draw) -> int:
    """Run every job, at most ``num_gpus`` at a time, one physical GPU each.

    A private pool rather than ``distributed_utils.run_commands_parallel``
    for two reasons. That helper appends ``--device 0`` to every argv, which
    ``tools/stage_d_arms.py`` (it takes ``--devices``) would reject outright;
    and it pools within a single dataset, whereas the point here is to pool
    ACROSS datasets so all eight GPUs are busy. Everything else matches it:
    ``CUDA_VISIBLE_DEVICES`` is set to the one assigned device, so each child
    sees exactly one GPU and indexes it as 0.
    """
    if not jobs:
        return 0
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in range(num_gpus):
        gpu_q.put(g)

    def _run(job: Job) -> int:
        job.gpu = gpu_q.get()
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
            job.started = time.time()
            job.log.parent.mkdir(parents=True, exist_ok=True)
            with job.log.open("w", encoding="utf-8", buffering=1) as fh:
                fh.write(f"$ CUDA_VISIBLE_DEVICES={job.gpu} {' '.join(job.argv)}\n\n")
                fh.flush()
                proc = subprocess.run(
                    job.argv, cwd=str(CODES), env=env,
                    stdout=fh, stderr=subprocess.STDOUT,
                )
            job.rc = int(proc.returncode)
            return job.rc
        finally:
            job.finished = time.time()
            gpu_q.put(job.gpu)

    workers = min(num_gpus, len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, j) for j in jobs]
        while not all(f.done() for f in futs):
            draw()
            time.sleep(2.0)
        draw()
        for f in futs:
            f.result()
    return sum(1 for j in jobs if j.rc not in (0, None))


# --------------------------------------------------------------------------
# the two work phases
# --------------------------------------------------------------------------


def _cfg(config: str, dataset: str):
    return load_config(config, dataset=dataset, protocol=PROTOCOL)


def _run_dir(config: str, dataset: str, seed: int) -> Path:
    cfg = _cfg(config, dataset)
    return stage_run_dir(
        cfg, "stage_b", f"multitask_{multitask_run_tag(cfg)}_seed{seed}"
    )


def _seeds(config: str, dataset: str) -> list[int]:
    cfg = _cfg(config, dataset)
    return list(cfg.get("seeds") or [cfg["seed"]])


def train_jobs(config: str, datasets, log_dir: Path, force: bool) -> list[Job]:
    """One ``stage_b.py multitask`` per (dataset, seed), already-done ones cut.

    ``run_meta.json`` is the completion mark because ``stage_b._finalize_run``
    writes it last, after the checkpoint is saved, the test split is scored and
    the summary row is appended.
    """
    jobs: list[Job] = []
    idx = 0
    for ds in datasets:
        for seed in _seeds(config, ds):
            out = _run_dir(config, ds, seed)
            if not force and (out / "run_meta.json").is_file():
                print(f"SKIP train {ds}/seed{seed}: already complete ({out})")
                continue
            argv = [
                sys.executable, str(CODES / "stage_b.py"), "multitask",
                "--config", config, "--dataset", ds, "--protocol", PROTOCOL,
                "--seed", str(seed), "--device", "0",
            ]
            jobs.append(Job(
                label=f"{ds}/seed{seed}",
                argv=argv,
                log=log_dir / "train.jobs" / f"{idx:03d}_{ds}_seed{seed}.log",
            ))
            idx += 1
    return jobs


def stage_d_jobs(config: str, datasets, log_dir: Path, force: bool) -> list[Job]:
    """Reliability for this arm only, per (dataset, seed).

    ``--arms multitask`` is what keeps this additive: ``tools/stage_d_arms.py``
    would otherwise walk all nine arms, and under ``--protocol heavy_aug`` the
    other eight have no checkpoint, so they would be reported as missing on
    every run. Restricting the arm also means this phase cannot touch another
    arm's Stage D artifacts.
    """
    jobs: list[Job] = []
    idx = 0
    for ds in datasets:
        for seed in _seeds(config, ds):
            if not force and not (_run_dir(config, ds, seed) / "run_meta.json").is_file():
                print(f"SKIP stage_d {ds}/seed{seed}: no trained checkpoint yet")
                continue
            argv = [
                sys.executable, str(CODES / "tools" / "stage_d_arms.py"),
                "--config", config, "--dataset", ds, "--protocol", PROTOCOL,
                "--arms", ARM, "--seed", str(seed),
                "--backbone", BACKBONE, "--devices", "0",
            ]
            if force:
                argv.append("--force")
            jobs.append(Job(
                label=f"{ds}/seed{seed}",
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


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect_row(config: str, dataset: str, arm: str, protocol: str | None) -> dict:
    """Seed-averaged metrics for one arm on one dataset.

    Accuracy comes from each run's ``test_metrics.json``; the rare stratum,
    ECE and AURC come from Stage D, which is where the paper reads them from
    (``tools/paper_sources.py``). Reading Stage D for the rare column rather
    than Stage B's own ``long_tail`` block matters: the two are cut by the same
    quantile rule, but only Stage D's is recut when that rule is revised.
    """
    cfg = load_config(config, dataset=dataset, protocol=protocol)
    seeds = list(cfg.get("seeds") or [cfg["seed"]])
    d_root = stage_run_dir(cfg, "stage_d")
    if arm == ARM:
        run_name = f"multitask_{multitask_run_tag(cfg)}_seed{{seed}}"
    elif arm == "heavy_aug":
        run_name = f"heavy_aug_{BACKBONE.replace('/', '_')}_seed{{seed}}"
    else:
        run_name = f"baseline_{BACKBONE.replace('/', '_')}_seed{{seed}}"

    top1, top3, f1, rare, ece, rare_ece, aurc = ([] for _ in range(7))
    for seed in seeds:
        m = _read_json(
            stage_run_dir(cfg, "stage_b", run_name.format(seed=seed)) / "test_metrics.json"
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


def _fmt(v, places=2) -> str:
    return "--" if v is None else f"{v:.{places}f}"


def summarize(config: str, datasets) -> int:
    """Print the new appendix row and the two rows it is read against."""
    rows = []
    for ds in datasets:
        rows.append(collect_row(config, ds, ARM, PROTOCOL))       # the new arm
        rows.append(collect_row(config, ds, ARM, None))           # default recipe
        rows.append(collect_row(config, ds, "heavy_aug", None))   # heavy-aug reference
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = SUMMARY_DIR / "haug_multitask_summary.csv"
    cols = ["dataset", "arm", "protocol", "n_seeds", "top1", "top3",
            "macro_f1", "rare", "ece", "rare_ece", "aurc"]
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                "" if r[c] is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                for c in cols
            ) + "\n")

    label = {
        (ARM, PROTOCOL): "Multi-task + heavy aug  (NEW)",
        (ARM, "default"): "Multi-task, default recipe",
        ("heavy_aug", "default"): "Whole-image + heavy aug",
    }
    lines = [
        "| Dataset | Arm | Seeds | Top-1 | Top-3 | Macro-F1 | Rare | ECE | Rare ECE | AURC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {label[(r['arm'], r['protocol'])]} | {r['n_seeds']} "
            f"| {_fmt(r['top1'])} | {_fmt(r['top3'])} | {_fmt(r['macro_f1'])} "
            f"| {_fmt(r['rare'])} | {_fmt(r['ece'])} | {_fmt(r['rare_ece'])} "
            f"| {_fmt(r['aurc'], 4)} |"
        )
    md = "\n".join(lines)
    md_path = SUMMARY_DIR / "haug_multitask_summary.md"
    md_path.write_text(
        "# Multi-task supervision under the heavy-augmentation recipe\n\n"
        "ConvNeXt-Nano, three seeds, protocol `heavy_aug`. Rare, ECE, Rare ECE\n"
        "and AURC are read from Stage D, as the paper's tables are. This file is\n"
        "written by `run_haug_multitask.py` and is not read by any checker or\n"
        "table generator; transcribe the row into the appendix by hand.\n\n"
        + md + "\n",
        encoding="utf-8",
    )
    print("\n" + md + "\n")
    print(f"csv: {csv_path}")
    print(f"md : {md_path}")
    incomplete = [r for r in rows if r["arm"] == ARM and r["protocol"] == PROTOCOL
                  and r["n_seeds"] < len(_seeds(config, r["dataset"]))]
    if incomplete:
        print("\nWARNING: the new arm is incomplete on: "
              + ", ".join(f"{r['dataset']} ({r['n_seeds']} seed(s))" for r in incomplete))
    return 0


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------


def hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def tail_line(path: Path, limit: int = 8192) -> str:
    """Last meaningful line of a job log, with tqdm's carriage returns resolved."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > limit:
                fh.seek(-limit, os.SEEK_END)
            blob = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    fallback = ""
    for chunk in reversed(blob.replace("\r", "\n").splitlines()):
        text = _ANSI.sub("", chunk).strip()
        if not text:
            continue
        fallback = fallback or text
        if len(text) > 3 and any(ch.isalnum() for ch in text):
            return text
    return fallback


_GPU_CACHE: tuple[float, list[tuple[int, int, float]]] = (0.0, [])


def gpu_stats(ttl: float = 5.0) -> list[tuple[int, int, float]]:
    """[(index, util%, mem_used_GB)], cached -- nvidia-smi costs ~50 ms."""
    global _GPU_CACHE
    now = time.time()
    if now - _GPU_CACHE[0] < ttl:
        return _GPU_CACHE[1]
    rows: list[tuple[int, int, float]] = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        for line in out.strip().splitlines():
            i, util, mem = (p.strip() for p in line.split(","))
            rows.append((int(i), int(util), int(mem) / 1024.0))
    except Exception:
        rows = []
    _GPU_CACHE = (now, rows)
    return rows


MARK = {"pending": "·", "running": "▶", "ok": "✔", "failed": "✘"}
COLOR = {"pending": "\033[90m", "running": "\033[36m",
         "ok": "\033[32m", "failed": "\033[31m"}
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[90m"


@dataclass
class Dashboard:
    #: Heading line. A field rather than a literal so `run_haug_arms.py` can
    #: reuse this class as-is; a second copy of `draw` would drift from this one.
    title: str = "Multi-task + heavy augmentation (ConvNeXt-Nano)"
    phase: str = ""
    jobs: list[Job] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    plain: bool = False
    color: bool = True
    _seen: dict = field(default_factory=dict)

    def _c(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.color else text

    def draw(self) -> None:
        if self.plain:
            for job in self.jobs:
                if self._seen.get(job.label) != job.state:
                    self._seen[job.label] = job.state
                    print(f"[{self.phase}] {job.label}: {job.state}"
                          + (f" (rc={job.rc})" if job.rc not in (0, None) else ""),
                          flush=True)
            return
        width = shutil.get_terminal_size((120, 40)).columns
        rows = shutil.get_terminal_size((120, 40)).lines
        done = sum(1 for j in self.jobs if j.finished is not None)
        bad = sum(1 for j in self.jobs if j.rc not in (0, None))
        out = ["\033[H\033[J"]
        out.append(self._c(self.title, BOLD))
        out.append(
            f"phase {self._c(self.phase, BOLD)}   {done}/{len(self.jobs)} done"
            f"   failed {bad}   elapsed {hms(time.time() - self.started)}"
        )
        out.append("")
        for job in self.jobs[: max(4, rows - 10)]:
            mark = self._c(MARK[job.state], COLOR[job.state])
            gpu = "-" if job.gpu is None else str(job.gpu)
            head = f" {mark} gpu {gpu:>2}  {job.label:<22} {hms(job.elapsed):>9}  "
            tail = tail_line(job.log) if job.started is not None else ""
            out.append((head + self._c(tail, DIM))[: width + len(DIM) + len(RESET)])
        stats = gpu_stats()
        if stats:
            out.append("")
            out.append("GPU  " + "  ".join(
                f"{i}: {u:>3}% {m:4.1f}G" for i, u, m in stats
            )[: width - 5])
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and analyse ONE extra arm: multi-task supervision "
                    "under the heavy-augmentation recipe.",
    )
    p.add_argument("--config", default=default_config_path())
    p.add_argument("--dataset", action="append", default=None,
                   help="Restrict to a dataset; repeatable. Default: all three.")
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
    phases = args.phase or ["train", "stage_d", "summary"]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    num_gpus = args.gpus or resolve_num_gpus(_cfg(args.config, datasets[0]))

    if args.dry_run:
        for phase, builder in (("train", train_jobs), ("stage_d", stage_d_jobs)):
            if phase not in phases:
                continue
            jobs = builder(args.config, datasets, log_dir, args.force)
            print(f"\n=== {phase}: {len(jobs)} job(s) on {num_gpus} GPU(s) ===")
            for job in jobs:
                print("  " + " ".join(job.argv))
        if "summary" in phases:
            print("\n=== summary: reads results only ===")
        return 0

    interactive = sys.stdout.isatty() and not args.plain
    ui = Dashboard(plain=not interactive, color=not args.no_color)
    if interactive:
        sys.stdout.write("\033[?25l")
    failures = 0
    try:
        for phase, builder in (("train", train_jobs), ("stage_d", stage_d_jobs)):
            if phase not in phases:
                continue
            jobs = builder(args.config, datasets, log_dir, args.force)
            if not jobs:
                print(f"{phase}: nothing to do")
                continue
            ui.phase, ui.jobs, ui.started = phase, jobs, time.time()
            failures += run_pool(jobs, num_gpus, ui.draw)
            if failures:
                # Stop before Stage D rather than analysing a partial arm: a
                # summary averaged over two of three seeds reads as a result.
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
                print(f"\n--- {job.label} (rc={job.rc}) — {job.log}")
                try:
                    print("\n".join(
                        job.log.read_text(encoding="utf-8", errors="replace")
                        .splitlines()[-25:]
                    ))
                except OSError:
                    print("  (log unreadable)")
        return 1

    if "summary" in phases:
        summarize(args.config, datasets)
    print(f"\nlogs: {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
