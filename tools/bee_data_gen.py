#!/usr/bin/env python3
"""Build a capped, deduplicated BeeMachine classification corpus.

The classification archives cannot be merged naively.

1. Most of the apparent volume is synthetic. Contributor collections often
   ship, for each original photograph, dozens of derived variants (blur,
   color shift, noise, background replacement, each crossed with rotations).
   Counting those toward a per-class minimum puts rotations of the same
   photograph into train and test, so rare-species accuracy would measure
   memorization. This script keeps originals only. A derived file embeds an
   image extension mid-name; an original does not.

2. The source collections overlap and re-export the same photograph under
   different names. Filename union is not enough; exact content hashing
   collapses the remainder.

3. Species that still have fewer than ``--min-originals`` distinct
   photographs after that filtering are dropped rather than padded, because
   below five a species cannot appear in the test fold of a 75/15/10 split.

The paper's classification corpus is 147 species, capped at 3,000 images each.

Sources are opened read-only. The destination must not resolve inside any
source. Re-running is idempotent: files already present at the destination
with a matching size are left alone.

Usage:
    python tools/bee_data_gen.py --datasets-root PATH --dest PATH --dry-run
    python tools/bee_data_gen.py --datasets-root PATH --dest PATH --cap 3000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Synthetic / duplicate collections that must never be scanned as sources.
EXCLUDED = {
    "new_iqa_datasets",
    "new_iqa_datasets_splitted",
    "masked_onlyvgg_oct15",
    "robert_data",
}

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------
# Original-vs-derived classification
# --------------------------------------------------------------------------


def _strip_extensions(name_lower: str) -> str:
    """Strip every trailing image extension (`a.jpg.jpg` -> `a`)."""
    changed = True
    while changed:
        changed = False
        for ext in IMAGE_EXT:
            if name_lower.endswith(ext):
                name_lower = name_lower[: -len(ext)]
                changed = True
    return name_lower


def is_original(name: str) -> bool:
    """True for a source photograph, False for a synthetic variant.

    Derived files are built by appending to the full original filename, so they
    carry an image extension *inside* the stem. This is marker-independent: it
    catches `_blurr_back`, `_segmented_white`, `_rotated_90` and any future
    suffix without needing to enumerate them.
    """
    low = name.lower()
    if not low.endswith(IMAGE_EXT):
        return False
    return not any(ext in _strip_extensions(low) for ext in IMAGE_EXT)


def identity_key(name: str) -> str:
    """Cross-corpus identity for the same photograph, ignoring case and
    re-saved extensions (`X.JPG` and `x.jpg.jpg` are one photograph)."""
    return _strip_extensions(name.lower())


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def scan_sources(root: Path, sources: list[str]) -> dict[str, list[tuple[int, Path]]]:
    """species -> [(source_rank, path)] for every original photograph."""
    found: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for rank, corpus in enumerate(sources):
        base = root / corpus
        if not base.is_dir():
            print(f"[scan] SKIP missing source: {base}", file=sys.stderr)
            continue
        n = 0
        species_dirs = [p for p in sorted(base.iterdir())
                        if p.is_dir() and p.name.startswith("Bombus_")]
        for species_dir in tqdm(species_dirs, desc=f"scan {corpus[:24]}", unit="sp",
                                leave=False, disable=None):
            try:
                entries = sorted(species_dir.iterdir())
            except OSError as exc:
                print(f"[scan] unreadable {species_dir}: {exc}", file=sys.stderr)
                continue
            for path in entries:
                if path.is_file() and is_original(path.name):
                    found[species_dir.name].append((rank, path))
                    n += 1
        print(f"[scan] {corpus}: {n} originals")
    return found


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def deduplicate(candidates: list[tuple[int, Path]]) -> list[tuple[int, Path]]:
    """Collapse the same photograph appearing in several corpora.

    Two passes. Filename identity first, keeping the highest-priority source.
    Then content: files whose size is unique cannot be duplicates, so only
    size-collision groups are hashed. That keeps the expensive pass proportional
    to the genuine overlap rather than to the corpus.
    """
    by_key: dict[str, tuple[int, Path]] = {}
    for rank, path in candidates:
        key = identity_key(path.name)
        current = by_key.get(key)
        if current is None or rank < current[0]:
            by_key[key] = (rank, path)
    survivors = list(by_key.values())

    by_size: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for rank, path in survivors:
        try:
            by_size[path.stat().st_size].append((rank, path))
        except OSError:
            continue

    out: list[tuple[int, Path]] = []
    for group in by_size.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        seen: dict[str, tuple[int, Path]] = {}
        for rank, path in sorted(group):
            try:
                digest = _sha256(path)
            except OSError:
                continue
            if digest not in seen:
                seen[digest] = (rank, path)
        out.extend(seen.values())
    return out


def select(pool: list[tuple[int, Path]], cap: int, rng: random.Random) -> list[Path]:
    """Take at most `cap`, exhausting higher-priority sources first.

    Within a source the order is shuffled so a truncated species is not biased
    toward alphabetically early filenames, which in these corpora correlates
    with observation ID and therefore with date and locality.
    """
    grouped: dict[int, list[Path]] = defaultdict(list)
    for rank, path in pool:
        grouped[rank].append(path)

    chosen: list[Path] = []
    for rank in sorted(grouped):
        paths = sorted(grouped[rank])
        rng.shuffle(paths)
        room = cap - len(chosen)
        if room <= 0:
            break
        chosen.extend(paths[:room])
    return chosen


# --------------------------------------------------------------------------
# Copy
# --------------------------------------------------------------------------


def copy_one(src: Path, dst: Path) -> str:
    """Copy preserving mtime. Returns 'skipped' if already present intact."""
    if dst.exists():
        try:
            if dst.stat().st_size == src.stat().st_size:
                return "skipped"
        except OSError:
            pass
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def assert_destination_safe(dest: Path, root: Path, sources: list[str]) -> None:
    dest = dest.resolve()
    for corpus in sources:
        src = (root / corpus).resolve()
        if dest == src or src in dest.parents or dest in src.parents:
            raise SystemExit(
                f"refusing to build: destination {dest} overlaps source {src}"
            )
    if dest.name in EXCLUDED:
        raise SystemExit(f"refusing to build: {dest.name} is an excluded corpus")


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-root", type=Path, required=True,
                    help="directory containing one or more species-folder collections")
    ap.add_argument("--dest", type=Path, required=True,
                    help="output classification corpus (absolute, or a name under --datasets-root)")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="subdirectory names under --datasets-root to merge, in priority order. "
                         "If omitted, every immediate subdirectory is scanned.")
    ap.add_argument("--cap", type=int, default=3000,
                    help="maximum images per species (default 3000)")
    ap.add_argument("--min-originals", type=int, default=5,
                    help="drop species below this many distinct originals")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan, copy nothing")
    args = ap.parse_args()

    root: Path = args.datasets_root
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = root / dest
    if args.sources:
        sources = list(args.sources)
    else:
        sources = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and p.name not in EXCLUDED and p.resolve() != dest.resolve()
        )
    assert_destination_safe(dest, root, sources)

    print(f"[plan] sources : {', '.join(sources)}")
    print(f"[plan] excluded: {', '.join(sorted(EXCLUDED))}")
    print(f"[plan] dest    : {dest}")
    print(f"[plan] cap={args.cap} min_originals={args.min_originals} seed={args.seed}\n")

    found = scan_sources(root, sources)
    print(f"\n[scan] species directories seen: {len(found)}")

    rng = random.Random(args.seed)
    plan: dict[str, list[Path]] = {}
    dropped: dict[str, int] = {}
    # Deduplication hashes every size-collision group, so this is the slow phase
    # before any copying starts and needs its own bar.
    for species in tqdm(sorted(found), desc="dedup", unit="sp", disable=None):
        unique = deduplicate(found[species])
        if len(unique) < args.min_originals:
            dropped[species] = len(unique)
            continue
        plan[species] = select(unique, args.cap, rng)

    total = sum(len(v) for v in plan.values())
    capped = [s for s, v in plan.items() if len(v) == args.cap]
    print(f"[plan] species kept    : {len(plan)}")
    print(f"[plan] species dropped : {len(dropped)} (below {args.min_originals} originals)")
    print(f"[plan] images to write : {total}")
    print(f"[plan] species at cap  : {len(capped)}")
    if dropped:
        print("\n[plan] dropped species:")
        for species, n in sorted(dropped.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"    {species:34s} {n}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    counts = {"copied": 0, "skipped": 0, "failed": 0}

    # One pool across every species rather than one per species: a per-species
    # pool drains at each of the 147 boundaries, and species sizes are wildly
    # uneven (38 sit at the cap, others hold a handful), so the tail of each
    # small species would run single-threaded.
    jobs: list[tuple[str, Path, Path]] = []
    for species in sorted(plan):
        species_dir = dest / species
        species_dir.mkdir(exist_ok=True)
        jobs.extend((species, src, species_dir / src.name) for src in plan[species])

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(copy_one, s, d): (sp, s, d) for sp, s, d in jobs}
        # disable=None turns the bar off when stderr is not a terminal, so a
        # logged or piped run stays readable instead of emitting one refresh
        # line per image.
        bar = tqdm(as_completed(futures), total=len(futures), desc="copy",
                   unit="img", disable=None, mininterval=0.5)
        last_species = None
        for fut in bar:
            species, src, dst = futures[fut]
            try:
                status = fut.result()
            except OSError as exc:
                status = "failed"
                bar.write(f"[copy] FAILED {src}: {exc}")
            counts[status] += 1
            if status != "failed":
                manifest.append({
                    "species": species,
                    "dest": str(dst.relative_to(dest)),
                    "source": str(src),
                    "status": status,
                })
            if species != last_species:
                bar.set_postfix_str(species[7:][:20], refresh=False)
                last_species = species

    manifest.sort(key=lambda row: (row["species"], row["dest"]))

    (dest / "_manifest.json").write_text(
        json.dumps({
            "cap": args.cap,
            "min_originals": args.min_originals,
            "seed": args.seed,
            "sources": sources,
            "excluded": sorted(EXCLUDED),
            "species": len(plan),
            "images": total,
            "dropped": dropped,
            "counts": counts,
        }, indent=2),
        encoding="utf-8",
    )
    with (dest / "_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["species", "dest", "source", "status"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\n[done] copied={counts['copied']} skipped={counts['skipped']} "
          f"failed={counts['failed']}")
    print(f"[done] {dest}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
