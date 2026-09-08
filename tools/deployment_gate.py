#!/usr/bin/env python3
"""Measure the confidence-gated deployment policy for the BeeMachine rollout.

The policy the deployment section proposes is: serve multi-task supervision's
prediction when its softmax confidence clears a threshold tau, and fall back to
the whole-image reference below it. This script measures that policy on the
frozen BeeMachine test split, because two things about it cannot be read off
Table 2 and are easy to get wrong.

1. WHERE THE ADVANTAGE LIVES. Multi-task has the better AURC on BeeMachine, but
   that is an average over the whole risk-coverage curve. Broken out by
   coverage, its advantage sits in the HIGH-coverage (low-confidence) region;
   in the most-confident decile the whole-image reference is marginally ahead.
   A gate set naively high therefore captures the region where multi-task is
   not better, which is the opposite of what it is meant to do.

2. THE SCALE OF tau IS NOT [0, 1] IN PRACTICE. Multi-task is underconfident on
   this corpus -- its maximum softmax over the test split is below 0.95 -- so a
   threshold chosen by intuition ("very high confidence", 0.95, 0.99) routes
   NOTHING to it and the policy silently degenerates to the reference with
   extra serving cost. tau has to be read off the measured distribution.

Seeds. The policy is measured independently at each training seed, pairing each
seed's gated arm against the SAME seed's fallback arm, and the reported numbers
are means over those seeds with a percentile bootstrap over the seed-level
values -- the convention the accuracy tables use. Pairing within a seed matters:
the two arms then share split, initialisation and data order, so the gate is
measured on two models that saw the same corpus in the same order rather than
on a mismatched pair.

Selection honesty. Only a test split exists on disk for these arms, so a tau
picked and reported on that split would be selection-inflated. The headline
number this writes is therefore a SPLIT-HALF estimate: tau is chosen on a
random half and scored on the held-out half, repeated `--trials` times, per
seed. The full sweep is reported alongside it, clearly as an in-sample curve.
Neither is a substitute for re-deriving tau on a real validation split before
rollout, which the paper says explicitly.

Protocols. Both arms of a gate must be trained under the same recipe, or the
fallback is worse than the gated arm at every operating point and the gate has
nothing to do. An arm trained under a non-default protocol writes to a tagged
Stage D tree (`stage_d__haug`), so `--gated-tree` and `--fallback-tree` say where
each arm's predictions live and `--suffix` keeps that measurement from
overwriting the default recipe's JSON.

Usage:
    ./run.sh tools/deployment_gate.py                     # writes JSON + prints
    ./run.sh tools/deployment_gate.py --dataset beemachine
    ./run.sh tools/deployment_gate.py --gated-tree stage_d__haug \
        --fallback heavy_aug --suffix __haug              # the adopted policy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("BEEMACHINE_ROOT", Path(__file__).resolve().parents[2]))
#: Campaign output root. Mirrors tools/stage_d_aggregate.OUT and is re-pointed
#: with it by `--config`, so the gate can never be measured on one campaign's
#: seed directories and written into another's.
OUT = Path(os.environ.get("BEEMACHINE_OUTPUT_ROOT") or (ROOT / "codes" / "outputs"))

#: The paper's bottom class-frequency stratum on BeeMachine, as an inclusive bound
#: on a class's training-image count. Declared as a literal so that a change to it
#: is visible in a diff, and checked against `metrics.quantile_long_tail_edges` at
#: run time by `_assert_rare_bound`, so it cannot quietly disagree with the stratum
#: every other table in the paper is cut on. The bound sits *on* the 25th percentile
#: rather than one below it because three BeeMachine classes are tied at exactly 93
#: training images and a strict bound would split that tie group.
RARE_MAX_TRAIN_COUNT = 93

#: Imported, not restated. An interval here and an interval in the reliability
#: tables must be the same estimator on the same seed-level values; two copies
#: of `boot_ci` and two `N_BOOT` literals only asserted that. `N_BOOT` now comes
#: from config.yaml `metrics.bootstrap_n` via stage_d_aggregate.
import stage_d_aggregate  # noqa: E402
from stage_d_aggregate import BOOT_SEED, N_BOOT, boot_ci, seed_dirs  # noqa: E402,F401


def load(seed_dir: Path, arm: str) -> dict[str, dict]:
    with open(seed_dir / f"preds_{arm}.csv") as fh:
        return {r["image"]: r for r in csv.DictReader(fh)}


GRID = [round(0.05 * i, 2) for i in range(21)]


def _assert_rare_bound(rows: dict) -> None:
    """Fail if RARE_MAX_TRAIN_COUNT no longer names the paper's bottom stratum.

    The gate reports a rare-species row beside Table 2's rare column, so the two
    have to be the same set of classes. They are cut by different code -- this
    module filters per-image rows, `metrics.quantile_long_tail_edges` places bin
    edges -- and nothing but this makes them agree.
    """
    sys.path.insert(0, str(ROOT / "codes"))
    from metrics import quantile_long_tail_edges

    counts = {int(r["y_true"]): int(r["train_count"]) for r in rows.values()}
    want = quantile_long_tail_edges(counts)[0] - 1
    if want != RARE_MAX_TRAIN_COUNT:
        raise SystemExit(
            f"RARE_MAX_TRAIN_COUNT is {RARE_MAX_TRAIN_COUNT} but the bottom stratum "
            f"now ends at {want} training images. Set it to {want} so the gate's rare "
            "row and the paper's rare column cover the same classes."
        )


def measure_seed(gated: dict, fallback: dict, trials: int, rng_seed: int) -> dict:
    """The whole policy measurement for one training seed."""
    keys = [k for k in gated if k in fallback]
    if len(keys) != len(gated) or len(keys) != len(fallback):
        raise SystemExit(
            f"prediction files do not cover the same images: {len(gated)} vs "
            f"{len(fallback)}, {len(keys)} shared -- the two arms were scored on "
            "different splits and must not be compared per-image"
        )

    _assert_rare_bound(gated)
    ok_g = {k: gated[k]["y_true"] == gated[k]["y_pred"] for k in keys}
    ok_f = {k: fallback[k]["y_true"] == fallback[k]["y_pred"] for k in keys}
    conf = {k: float(gated[k]["confidence"]) for k in keys}
    rare_keys = [k for k in keys if int(gated[k]["train_count"]) <= RARE_MAX_TRAIN_COUNT]

    def acc(ks, pick_gated):
        return statistics.mean(ok_g[k] if k in pick_gated else ok_f[k] for k in ks)

    sweep = []
    for tau in GRID:
        sel = {k for k in keys if conf[k] >= tau}
        sweep.append({"tau": tau, "routed_frac": len(sel) / len(keys),
                      "top1": acc(keys, sel), "rare_top1": acc(rare_keys, sel)})

    # --- split-half: choose tau on A, score on B ---
    rng = random.Random(rng_seed)
    order = list(keys)
    held, taus = [], []
    for _ in range(trials):
        rng.shuffle(order)
        half = len(order) // 2
        a, b = order[:half], order[half:]
        best_tau, best = None, -1.0
        for tau in GRID:
            v = acc(a, {k for k in a if conf[k] >= tau})
            if v > best:
                best, best_tau = v, tau
        held.append(acc(b, {k for k in b if conf[k] >= best_tau}))
        taus.append(best_tau)

    return {
        "n": len(keys),
        "n_rare": len(rare_keys),
        "max_confidence": max(conf.values()),
        "gated_alone_top1": statistics.mean(ok_g[k] for k in keys),
        "fallback_alone_top1": statistics.mean(ok_f[k] for k in keys),
        "gated_alone_rare": statistics.mean(ok_g[k] for k in rare_keys),
        "fallback_alone_rare": statistics.mean(ok_f[k] for k in rare_keys),
        "sweep": sweep,
        "split_half_held_out": statistics.mean(held),
        "split_half_taus": taus,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beemachine")
    ap.add_argument("--gated", default="multitask", help="arm the gate serves")
    ap.add_argument("--fallback", default="whole", help="arm served below tau")
    ap.add_argument("--gated-tree", default="stage_d",
                    help="Stage D tree holding the gated arm's per-seed "
                         "predictions. Default: stage_d.")
    ap.add_argument("--fallback-tree", default="stage_d",
                    help="Stage D tree holding the fallback arm's predictions. "
                         "Default: stage_d.")
    ap.add_argument("--suffix", default="",
                    help="Appended to the output filename, so a second policy "
                         "does not overwrite deployment_gate.json.")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the split-half resampling, distinct from "
                         "the training seeds it is applied to.")
    ap.add_argument("--config", default=None,
                    help="Measure the campaign this config writes to. "
                         "Default: codes/outputs (the published campaign).")
    args = ap.parse_args()
    # Move BOTH roots together: seed_dirs() reads stage_d_aggregate.OUT, this
    # module's OUT decides where deployment_gate.json lands.
    if args.config:
        global OUT
        stage_d_aggregate.set_output_root(args.config, args.dataset)
        OUT = stage_d_aggregate.OUT

    stage_d_aggregate.set_tree(args.gated_tree)
    gated_dirs = seed_dirs(args.dataset)
    stage_d_aggregate.set_tree(args.fallback_tree)
    fallback_dirs = seed_dirs(args.dataset)
    stage_d_aggregate.set_tree("stage_d")
    if not gated_dirs or not fallback_dirs:
        raise SystemExit(f"no {args.dataset}/{args.gated_tree}/seed*/ or "
                         f"{args.fallback_tree}/seed*/ directories; run "
                         "tools/stage_d_arms.py first")
    if sorted(gated_dirs) != sorted(fallback_dirs):
        raise SystemExit(
            f"the two arms were run at different seeds -- {sorted(gated_dirs)} "
            f"vs {sorted(fallback_dirs)}. The policy pairs a seed against "
            "itself, so it cannot be measured on a mismatched seed list.")

    per_seed = {}
    for s, d in gated_dirs.items():
        per_seed[s] = measure_seed(load(d, args.gated),
                                   load(fallback_dirs[s], args.fallback),
                                   args.trials, args.seed)

    seeds = sorted(per_seed)
    first = per_seed[seeds[0]]
    for s in seeds[1:]:
        if per_seed[s]["n"] != first["n"] or per_seed[s]["n_rare"] != first["n_rare"]:
            raise SystemExit("seeds were scored on different splits: "
                             f"n={[per_seed[x]['n'] for x in seeds]}")

    def across(field):
        return [per_seed[s][field] for s in seeds]

    def summarise(field):
        v = across(field)
        return {"mean": statistics.mean(v), "ci95": boot_ci(v), "per_seed": v}

    # Sweep averaged tau-by-tau; every seed shares the same grid.
    sweep = []
    for i, tau in enumerate(GRID):
        cells = [per_seed[s]["sweep"][i] for s in seeds]
        sweep.append({
            "tau": tau,
            "routed_frac": statistics.mean(c["routed_frac"] for c in cells),
            "top1": statistics.mean(c["top1"] for c in cells),
            "top1_ci95": boot_ci([c["top1"] for c in cells]),
            "rare_top1": statistics.mean(c["rare_top1"] for c in cells),
            "rare_top1_ci95": boot_ci([c["rare_top1"] for c in cells]),
        })

    all_taus = [t for s in seeds for t in per_seed[s]["split_half_taus"]]
    result = {
        "dataset": args.dataset,
        "gated_arm": args.gated,
        "fallback_arm": args.fallback,
        "gated_tree": args.gated_tree,
        "fallback_tree": args.fallback_tree,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n": first["n"],
        "n_rare": first["n_rare"],
        "max_confidence": statistics.mean(across("max_confidence")),
        "max_confidence_per_seed": across("max_confidence"),
        "gated_alone_top1": statistics.mean(across("gated_alone_top1")),
        "gated_alone_top1_ci95": boot_ci(across("gated_alone_top1")),
        "fallback_alone_top1": statistics.mean(across("fallback_alone_top1")),
        "fallback_alone_top1_ci95": boot_ci(across("fallback_alone_top1")),
        "gated_alone_rare": statistics.mean(across("gated_alone_rare")),
        "fallback_alone_rare": statistics.mean(across("fallback_alone_rare")),
        "sweep": sweep,
        "best_insample": max(sweep, key=lambda r: r["top1"]),
        "split_half": {
            "trials": args.trials,
            "trials_total": args.trials * len(seeds),
            "held_out_top1_mean": statistics.mean(across("split_half_held_out")),
            "held_out_top1_ci95": boot_ci(across("split_half_held_out")),
            "held_out_per_seed": across("split_half_held_out"),
            "tau_modal": Counter(all_taus).most_common(1)[0][0],
            "tau_median": statistics.median(all_taus),
        },
        "per_seed": {str(s): {k: v for k, v in per_seed[s].items()
                              if k not in ("sweep", "split_half_taus")}
                     for s in seeds},
    }
    dest = OUT / args.dataset / "stage_d" / f"deployment_gate{args.suffix}.json"
    dest.write_text(json.dumps(result, indent=2))

    base_g, base_f = result["gated_alone_top1"], result["fallback_alone_top1"]
    print(f"{args.dataset}: {args.gated} gated, falling back to {args.fallback}")
    print(f"  seeds={seeds}  n={result['n']}  n_rare={result['n_rare']}")
    print(f"  max confidence={result['max_confidence']:.4f} "
          f"(per seed: {', '.join(f'{v:.4f}' for v in result['max_confidence_per_seed'])})")
    print(f"  {args.gated} alone   top1={100*base_g:.2f}  rare={100*result['gated_alone_rare']:.2f}")
    print(f"  {args.fallback} alone top1={100*base_f:.2f}  rare={100*result['fallback_alone_rare']:.2f}")
    print(f"  {'tau':>5} {'routed':>8} {'top1':>7} {'vs gated':>9} {'vs fallb':>9} {'rare':>7}")
    for r in sweep:
        if r["tau"] * 100 % 10:
            continue
        print(f"  {r['tau']:5.2f} {100*r['routed_frac']:7.1f}% {100*r['top1']:7.2f} "
              f"{100*(r['top1']-base_g):+9.2f} {100*(r['top1']-base_f):+9.2f} "
              f"{100*r['rare_top1']:7.2f}")
    b = result["best_insample"]
    print(f"  best in-sample: tau={b['tau']:.2f} top1={100*b['top1']:.2f}")
    sh = result["split_half"]
    print(f"  split-half held-out top1={100*sh['held_out_top1_mean']:.2f} "
          f"(modal tau={sh['tau_modal']:.2f}, {sh['trials_total']} draws)")
    print(f"  wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
