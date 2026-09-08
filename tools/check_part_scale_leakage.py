#!/usr/bin/env python3
"""Content-based overlap check between a dataset's part set and its scale corpus.

Why this exists
---------------
`data.py`'s leakage guard matches on **filename**. That genuinely removes 1,888
images for CUB and 1,129 for Fish-Vista, whose two corpora share a naming
scheme. For Beemachine it removes exactly 0 -- not because the corpora are
disjoint, but because the part set was exported through Roboflow and renamed to
`…_jpg.rf.<hash>.jpg` while the 195k corpus kept its original names, so no
filename can ever match. Beemachine's overlap has therefore been *unmeasured*,
not absent, for the whole project, and `splits_cls_seed42.json` records
`excluded_part_overlap: 0` as though it had been checked.

That matters because Stage A trains on the part set and Stage C reports
whole-image top-1 on the scale corpus. Any image in both means the frozen
segmenter -- and the pseudo-masks it produces -- saw Stage C's test images.

What it does
------------
Two passes, cheapest first:

1. **Exact**: SHA-256 of the file bytes. Catches byte-identical re-exports.
2. **Near**: 64-bit dHash (row-wise gradient of a 9x8 grayscale thumbnail),
   compared by Hamming distance. Catches re-encodes, resizes, and quality
   changes -- which is what a Roboflow export actually produces. dHash is
   implemented here rather than pulled from `imagehash` to avoid adding a
   dependency for ~15 lines of numpy.

dHash is *not* rotation- or flip-invariant, and it should not be: the 6x
geometric train-aug set is generated from the part set on purpose, so treating a
rotation as a duplicate would flag intended augmentation. Only the unaugmented
part set is compared.

Output
------
`outputs/{dataset}/frozen_splits/part_scale_leakage.json`, listing the scale
corpus relpaths that collide and the evidence for each. `data.py`'s freeze reads
this file when present and excludes those images, so re-running `stage_a.py
freeze` after this turns the measurement into an actual exclusion.

Usage
-----
    ./run.sh tools/check_part_scale_leakage.py --config config.yaml --dataset beemachine
    ./run.sh tools/check_part_scale_leakage.py --dataset beemachine --hamming 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import add_global_stage_args, ensure_dir, load_config  # noqa: E402
from data import build_part_dataset, list_large_cls_relpaths  # noqa: E402

_HASH_SIZE = 8  # -> 8x8 = 64-bit dHash


def _dhash_and_sha(path: str) -> tuple[str, int, str] | None:
    """(`relpath`, dHash as uint64, sha256-hex) for one image, or None if unreadable."""
    try:
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        with Image.open(path) as im:
            g = im.convert("L").resize((_HASH_SIZE + 1, _HASH_SIZE), Image.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
        # Row-wise gradient: bit i is 1 where pixel i is brighter than pixel i+1.
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        value = 0
        for b in bits:
            value = (value << 1) | int(b)
        return path, value, sha
    except Exception:
        return None


def _hash_many(paths: list[str], workers: int, desc: str):
    out_paths, out_hashes, out_shas = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in tqdm(
            ex.map(_dhash_and_sha, paths, chunksize=64),
            total=len(paths),
            desc=desc,
        ):
            if res is None:
                continue
            p, h, s = res
            out_paths.append(p)
            out_hashes.append(h)
            out_shas.append(s)
    return out_paths, np.asarray(out_hashes, dtype=np.uint64), out_shas


def _hamming_matches(part_h: np.ndarray, scale_h: np.ndarray, max_dist: int, chunk: int = 256):
    """For each part hash, the scale indices within `max_dist` bits.

    Chunked XOR + popcount over uint64. 7.7k x 195k is 1.5e9 comparisons, which
    is a few seconds of numpy but 12 GB if materialised at once -- hence the
    chunking over the (smaller) part axis.
    """
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    scale_bytes = scale_h.view(np.uint8).reshape(-1, 8)
    matches: dict[int, list[int]] = {}
    for start in tqdm(range(0, len(part_h), chunk), desc="hamming"):
        block = part_h[start : start + chunk]
        block_bytes = block.view(np.uint8).reshape(-1, 8)
        # (block, scale, 8) bytes -> popcount -> (block, scale) distances
        xor = block_bytes[:, None, :] ^ scale_bytes[None, :, :]
        dist = lut[xor].sum(axis=2)
        for local, row in enumerate(dist):
            hit = np.flatnonzero(row <= max_dist)
            if hit.size:
                matches[start + local] = hit.tolist()
    return matches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_global_stage_args(ap)
    ap.add_argument(
        "--hamming",
        type=int,
        default=5,
        help="Max dHash Hamming distance counted as a near-duplicate (of 64 bits). "
        "5 is the conventional threshold; 0 means exact-hash only.",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit_scale", type=int, default=0, help="Debug: cap corpus size.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset)
    entry = cfg.get("dataset_entry") or {}

    part_ds = build_part_dataset(cfg, image_size=64)
    part_paths = [str(Path(part_ds.image_dir) / n) for n in part_ds.image_names]

    # Returns (image_root, relative paths) -- the root already has the
    # per-dataset images subdirectory applied.
    scale_root, scale_rel = list_large_cls_relpaths(cfg)
    if args.limit_scale:
        scale_rel = scale_rel[: args.limit_scale]
    scale_paths = [str(Path(scale_root) / r) for r in scale_rel]

    print(f"part set : {len(part_paths):,} images")
    print(f"scale set: {len(scale_paths):,} images")

    p_paths, p_hash, p_sha = _hash_many(part_paths, args.workers, "hash part")
    s_paths, s_hash, s_sha = _hash_many(scale_paths, args.workers, "hash scale")

    sha_index: dict[str, list[int]] = {}
    for i, sha in enumerate(s_sha):
        sha_index.setdefault(sha, []).append(i)

    findings: dict[str, dict] = {}
    n_exact = 0
    for i, sha in enumerate(p_sha):
        for j in sha_index.get(sha, []):
            rel = scale_rel[scale_paths.index(s_paths[j])] if s_paths[j] in scale_paths else s_paths[j]
            findings.setdefault(str(rel), {"evidence": [], "part_images": []})
            findings[str(rel)]["evidence"].append("sha256")
            findings[str(rel)]["part_images"].append(Path(p_paths[i]).name)
            n_exact += 1

    # Threshold justification, not decoration. dHash distance separates cleanly
    # here -- on a 1,500 x 4,000 Beemachine sample the per-part-image *minimum*
    # distance was 0 for the 1st percentile and 11 by the 5th, while the overall
    # pair distribution centred on 31, and the match count was identical at
    # every threshold from 2 to 8. A histogram makes that separation checkable
    # per dataset instead of assumed, and shows immediately if some corpus lacks
    # the gap (in which case no threshold is defensible and the result is
    # "inconclusive", not a number).
    min_hist: dict[str, int] = {}
    if len(p_hash) and len(s_hash):
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        sb = s_hash.view(np.uint8).reshape(-1, 8)
        mins = []
        for start in range(0, len(p_hash), 256):
            pb = p_hash[start : start + 256].view(np.uint8).reshape(-1, 8)
            mins.append(lut[pb[:, None, :] ^ sb[None, :, :]].sum(axis=2).min(axis=1))
        mins = np.concatenate(mins)
        for edge in (0, 2, 5, 8, 12, 16, 64):
            min_hist[f"<={edge}"] = int((mins <= edge).sum())

    n_near = 0
    if args.hamming > 0 and len(p_hash) and len(s_hash):
        rel_by_path = dict(zip(scale_paths, scale_rel))
        for i, hits in _hamming_matches(p_hash, s_hash, args.hamming).items():
            for j in hits:
                rel = rel_by_path.get(s_paths[j], s_paths[j])
                entry_f = findings.setdefault(str(rel), {"evidence": [], "part_images": []})
                if "dhash" not in entry_f["evidence"]:
                    entry_f["evidence"].append("dhash")
                name = Path(p_paths[i]).name
                if name not in entry_f["part_images"]:
                    entry_f["part_images"].append(name)
                n_near += 1

    out_path = Path(
        args.out
        or Path(cfg["paths"]["output_root"]) / "frozen_splits" / "part_scale_leakage.json"
    )
    ensure_dir(out_path.parent)
    payload = {
        "dataset": cfg["dataset"],
        "n_part": len(p_paths),
        "n_scale": len(s_paths),
        "hamming_threshold": args.hamming,
        "n_exact_pairs": n_exact,
        "n_near_pairs": n_near,
        "n_scale_images_flagged": len(findings),
        "part_images_by_min_distance": min_hist,
        "overlapping_scale_relpaths": sorted(findings),
        "detail": findings,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"\nexact (sha256) pairs      : {n_exact:,}\n"
        f"near  (dHash<={args.hamming}) pairs : {n_near:,}\n"
        f"distinct scale images flagged: {len(findings):,} "
        f"({len(findings) / max(1, len(s_paths)):.3%} of the corpus)"
    )
    if min_hist:
        print(
            "\npart images having ANY scale image within distance d "
            f"(of {len(p_paths):,}):"
        )
        for k, v in min_hist.items():
            print(f"  d {k:>4}: {v:,}")
        print(
            "  A flat count between d<=2 and d<=8 means duplicates and "
            "non-duplicates are cleanly separated and the threshold is safe; a "
            "steadily rising count means they are not, and the result should be "
            "reported as inconclusive rather than as an overlap count."
        )
    print(f"\nWrote {out_path}")
    if findings:
        print(
            "\nNon-zero overlap. Re-run `stage_a.py freeze --force` so the frozen "
            "classification split excludes these images, then re-run Stage C: the "
            "current at-scale numbers were measured with them in the test fold."
        )
    else:
        print("\nNo overlap found. `excluded_part_overlap: 0` is now a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
