#!/usr/bin/env python3
"""Average Stage D's per-seed reliability outputs into the files the paper reads.

`tools/stage_d_arms.py --seed S` writes one directory per seed,
`stage_d/seedS/`. This collapses those into `stage_d/` itself, keeping every
filename and schema byte-compatible with the single-seed layout that preceded
it, so the paper's checkers, `paper/figures_src/make_figures.py` and
`tools/deployment_gate.py` read seed-averaged numbers without knowing that a
seed dimension was added underneath them.

Why aggregate here rather than in each consumer. An arm's ECE, AURC and
corruption deltas are properties of a trained checkpoint exactly as its top-1
is, so they carry the same seed dimension the accuracy tables do. Six separate
consumers each re-deriving "mean over seeds, then bootstrap the seed-level
values" is six chances to disagree about it. They agree by reading one file.

The aggregation rules, and where they differ:

* Scalars the paper quotes (`ece`, `aurc`, per-bin `ece`/`aurc`/`accuracy`,
  corruption `top1`) are the plain mean of the S seed-level values, matching
  the convention Appendix "Bootstrap intervals" states for accuracy. Each also
  gains a `*_ci95` percentile-bootstrap interval over those same S values and
  an `n_seeds`, so a table can report an interval without recomputing one.
* Reliability-diagram BINS are pooled instead: a bin's `conf` and `acc` are
  n-weighted means across seeds, because a bin holds a different number of
  images in each seed and an unweighted mean would let a seed that put four
  images in a bin count as much as one that put four hundred there. Its `n` is
  the mean per-seed occupancy, so "images in overconfident bins" stays a
  per-seed count and remains comparable to the test-split size.
* `preds_*.csv` is NOT aggregated. It is one row per test image per seed, and
  averaging predictions across seeds would invent an ensemble that no reported
  arm is. Consumers that need per-image data (the deployment gate, the
  Fish-Vista leakage bound) iterate `stage_d/seed*/preds_*.csv` and average the
  scalar they compute, which is what `seed_dirs()` here is for.

Usage:
    ./run.sh tools/stage_d_aggregate.py                  # all three datasets
    ./run.sh tools/stage_d_aggregate.py --dataset cub
    ./run.sh tools/stage_d_aggregate.py --tree stage_d__haug   # heavy-aug arm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CODES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODES))

from config import DATASET_CHOICES  # noqa: E402

#: Campaign output root. Overridable so a second campaign (a different common
#: comparison backbone, its own ./outputs_new tree) can be aggregated without
#: editing this file or clobbering the published one. Unset, this is exactly
#: the historical ``codes/outputs``, so the ConvNeXt-Nano campaign's numbers
#: are reproduced byte-for-byte by the same command as before.
#:
#: `tools/deployment_gate.py` imports `seed_dirs` from here, so pointing this
#: at a tree moves BOTH tools at once -- they must never disagree about which
#: campaign they are scoring.
OUT = Path(os.environ.get("BEEMACHINE_OUTPUT_ROOT") or (CODES / "outputs"))


def set_output_root(config_path: str | None, dataset: str | None = None) -> None:
    """Point OUT at the campaign `config_path` writes to.

    `paths.output_root` resolves to ``<root>/<dataset>``, so the campaign root
    is its parent. A no-op when `config_path` is None.
    """
    global OUT
    if not config_path:
        return
    from config import load_config
    cfg = load_config(config_path, dataset=dataset or "beemachine")
    OUT = Path(cfg["paths"]["output_root"]).resolve().parent

#: Bootstrap settings. `N_BOOT` is READ from config.yaml `metrics.bootstrap_n`
#: rather than restated here, so an ECE interval and a top-1 interval in the
#: same table are the same estimator at the same resample count -- a literal
#: 2000 only claimed that, and would have gone on claiming it after the config
#: changed. `tools/deployment_gate.py` imports both from here.
def _bootstrap_n() -> int:
    from config import default_config_path, load_config
    return int(load_config(default_config_path())["metrics"]["bootstrap_n"])


N_BOOT = _bootstrap_n()
BOOT_SEED = 20260830


#: The Stage D subtree being aggregated. ``stage_d`` is the default protocol's
#: grid, the one every published table reads. A run under a non-default protocol
#: writes to a tagged sibling instead -- ``config.stage_run_dir`` appends the tag,
#: so the heavy-augmentation multi-task arm lands in ``stage_d__haug`` -- and that
#: tree has to be aggregated by the same code, or its seed-averaged ECE and AURC
#: would be computed a second time somewhere else and be free to disagree.
#: `tools/deployment_gate.py` imports `seed_dirs` from here, so setting this
#: moves both tools onto the same tree.
TREE = "stage_d"


def set_tree(tree: str) -> None:
    """Aggregate `tree` instead of `stage_d`. A no-op when `tree` is falsy."""
    global TREE
    if tree:
        TREE = tree


def seed_dirs(dataset: str) -> dict[int, Path]:
    """{seed: <TREE>/seedN} for every seed present on disk, ascending."""
    root = OUT / dataset / TREE
    found = {}
    for p in sorted(root.glob("seed*")):
        if p.is_dir() and p.name[4:].isdigit():
            found[int(p.name[4:])] = p
    return dict(sorted(found.items()))


def boot_ci(values: list[float]) -> list[float]:
    """Percentile bootstrap over the seed-level values themselves.

    Same estimator as `stage_report.py`'s accuracy intervals: resample the S
    seed scores with replacement, not the n test examples. An interval built
    from three values is wide and is meant to read as such.
    """
    if len(values) < 2:
        return [values[0], values[0]] if values else [float("nan")] * 2
    rng = random.Random(BOOT_SEED)
    means = []
    for _ in range(N_BOOT):
        means.append(statistics.mean(rng.choices(values, k=len(values))))
    means.sort()
    lo = means[int(0.025 * (N_BOOT - 1))]
    hi = means[int(0.975 * (N_BOOT - 1))]
    return [lo, hi]


def arms_in(dirs: dict[int, Path], pattern: str) -> list[str]:
    """Arm names for which EVERY seed has this output.

    Requiring all seeds is deliberate: an arm present at two seeds out of three
    would otherwise be averaged over a different seed set than the arm beside
    it in the same table, and the table would silently stop being a comparison.
    """
    per_seed = []
    for d in dirs.values():
        names = set()
        for p in d.glob(pattern.replace("{arm}", "*")):
            pre, suf = pattern.split("{arm}")
            names.add(p.name[len(pre):len(p.name) - len(suf)])
        per_seed.append(names)
    if not per_seed:
        return []
    common = set.intersection(*per_seed)
    missing = set.union(*per_seed) - common
    if missing:
        print(f"    skipping {pattern}: incomplete across seeds -> {sorted(missing)}")
    return sorted(common)


# ---------------------------------------------------------------- calibration
def agg_calibration(dirs, arm, dest):
    docs = [json.load(open(d / f"calibration_{arm}.json")) for d in dirs.values()]
    out = {"ece": statistics.mean(x["ece"] for x in docs),
           "ece_ci95": boot_ci([x["ece"] for x in docs]),
           "n_seeds": len(docs)}
    by_bin = defaultdict(list)
    for doc in docs:
        for b in doc["bins"]:
            by_bin[b["bin"]].append(b)
    bins = []
    for idx in sorted(by_bin):
        rows = by_bin[idx]
        tot = sum(r["n"] for r in rows)
        # n-weighted so a bin holding 4 images in one seed and 400 in another
        # is summarised by where its mass actually is.
        wmean = (lambda k: sum(r[k] * r["n"] for r in rows) / tot) if tot else (lambda k: 0.0)
        bins.append({"bin": idx, "acc": wmean("acc"), "conf": wmean("conf"),
                     "n": tot / len(dirs)})
    out["bins"] = bins
    dest.write_text(json.dumps(out, indent=2))


# ------------------------------------------------------------------ selective
def agg_selective(dirs, arm, dest):
    docs = [json.load(open(d / f"selective_{arm}.json")) for d in dirs.values()]
    out = {"aurc": statistics.mean(x["aurc"] for x in docs),
           "aurc_ci95": boot_ci([x["aurc"] for x in docs]),
           "n_seeds": len(docs)}
    by_cov = defaultdict(list)
    for doc in docs:
        for pt in doc["curve"]:
            by_cov[pt["coverage"]].append(pt)
    out["curve"] = [
        {"coverage": cov,
         "risk": statistics.mean(p["risk"] for p in pts),
         "risk_ci95": boot_ci([p["risk"] for p in pts]),
         "n": statistics.mean(p["n"] for p in pts)}
        for cov, pts in sorted(by_cov.items())
    ]
    dest.write_text(json.dumps(out, indent=2))


# ------------------------------------------- long_tail / reliability_by_bin
def agg_keyed(dirs, arm, dest, stem, fields):
    """Both files share a shape: {frequency_bin: {metric: value, 'n': int}}."""
    docs = [json.load(open(d / f"{stem}_{arm}.json")) for d in dirs.values()]
    keys = list(docs[0])
    if any(list(x) != keys for x in docs):
        raise SystemExit(f"{stem}_{arm}: frequency bins differ across seeds")
    out = {}
    for k in keys:
        rows = [x[k] for x in docs]
        entry = {}
        for f in fields:
            vals = [r[f] for r in rows]
            entry[f] = statistics.mean(vals)
            entry[f"{f}_ci95"] = boot_ci(vals)
        # The test split is frozen, so a bin holds the same images every seed.
        ns = {r["n"] for r in rows}
        if len(ns) != 1:
            raise SystemExit(f"{stem}_{arm} bin {k}: n differs across seeds {ns} "
                             "-- the arms were scored on different splits")
        entry["n"] = ns.pop()
        entry["n_seeds"] = len(rows)
        out[k] = entry
    dest.write_text(json.dumps(out, indent=2))


# ----------------------------------------------------------------- robustness
def agg_robustness(dirs, arm, dest):
    by_cell = defaultdict(list)
    for d in dirs.values():
        for r in csv.DictReader(open(d / f"robustness_{arm}.csv")):
            by_cell[(r["corruption"], int(r["severity"]))].append(r)
    rows = []
    for (corr, sev), rs in sorted(by_cell.items(), key=lambda kv: (kv[0][0] != "clean", kv[0])):
        if len(rs) != len(dirs):
            raise SystemExit(f"robustness_{arm}: {corr}/{sev} present in "
                             f"{len(rs)} of {len(dirs)} seeds")
        ns = {int(r["n"]) for r in rs}
        if len(ns) != 1:
            raise SystemExit(f"robustness_{arm} {corr}/{sev}: n differs across seeds {ns}")
        cell = {"arm": arm, "corruption": corr, "severity": sev}
        for f in ("top1", "ece", "aurc"):
            vals = [float(r[f]) for r in rs]
            cell[f] = statistics.mean(vals)
            lo, hi = boot_ci(vals)
            cell[f"{f}_lo"], cell[f"{f}_hi"] = lo, hi
        cell["n"] = ns.pop()
        cell["n_seeds"] = len(rs)
        rows.append(cell)
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ confusion
def agg_confusion(dirs, arm, dest):
    counts = defaultdict(float)
    for d in dirs.values():
        p = d / f"confusion_{arm}.csv"
        if not p.exists():
            return
        for r in csv.DictReader(open(p)):
            counts[(r["true"], r["pred"])] += int(r["count"])
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true", "pred", "count"])
        for (t, p_), c in sorted(counts.items(), key=lambda kv: -kv[1]):
            w.writerow([t, p_, round(c / len(dirs), 4)])


def merge_robustness(out: Path) -> None:
    """One combined robustness.csv, mirroring stage_d_arms._merge_robustness."""
    parts = sorted(p for p in out.glob("robustness_*.csv"))
    if not parts:
        return
    rows, header = [], None
    for p in parts:
        with open(p) as fh:
            rd = csv.DictReader(fh)
            header = header or rd.fieldnames
            rows.extend(rd)
    with open(out / "robustness.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def run(dataset: str) -> int:
    dirs = seed_dirs(dataset)
    out = OUT / dataset / TREE
    if not dirs:
        print(f"{dataset}: no {TREE}/seed*/ directories; nothing to aggregate")
        return 1
    print(f"{dataset}: aggregating seeds {sorted(dirs)}")
    if len(dirs) < 3:
        print(f"  WARNING: only {len(dirs)} seed(s); the paper reports three")

    n = 0
    for arm in arms_in(dirs, "calibration_{arm}.json"):
        agg_calibration(dirs, arm, out / f"calibration_{arm}.json"); n += 1
    for arm in arms_in(dirs, "selective_{arm}.json"):
        agg_selective(dirs, arm, out / f"selective_{arm}.json"); n += 1
    for arm in arms_in(dirs, "long_tail_{arm}.json"):
        agg_keyed(dirs, arm, out / f"long_tail_{arm}.json",
                  "long_tail", ("top1", "macro_f1")); n += 1
    for arm in arms_in(dirs, "reliability_by_bin_{arm}.json"):
        agg_keyed(dirs, arm, out / f"reliability_by_bin_{arm}.json",
                  "reliability_by_bin", ("ece", "aurc", "accuracy")); n += 1
    for arm in arms_in(dirs, "robustness_{arm}.csv"):
        agg_robustness(dirs, arm, out / f"robustness_{arm}.csv"); n += 1
    for arm in arms_in(dirs, "confusion_{arm}.csv"):
        agg_confusion(dirs, arm, out / f"confusion_{arm}.csv"); n += 1
    merge_robustness(out)
    print(f"  wrote {n} aggregated files + robustness.csv into {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, choices=list(DATASET_CHOICES),
                    help="Default: every dataset with a stage_d/seed*/ layout.")
    ap.add_argument("--config", default=None,
                    help="Aggregate the campaign this config writes to. "
                         "Default: codes/outputs (the published campaign).")
    ap.add_argument("--tree", default="stage_d",
                    help="Stage D subtree to aggregate, for arms trained under a "
                         "non-default protocol. Default: stage_d. The heavy-"
                         "augmentation multi-task arm is stage_d__haug.")
    args = ap.parse_args()
    set_output_root(args.config, args.dataset)
    set_tree(args.tree)
    targets = [args.dataset] if args.dataset else list(DATASET_CHOICES)
    rc = 0
    for ds in targets:
        if not (OUT / ds / TREE).is_dir():
            continue
        rc |= run(ds)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
