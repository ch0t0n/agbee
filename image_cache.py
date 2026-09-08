"""RAM-resident decoded-image cache.

Why decoded and not raw bytes. Measured on this node (beemachine part set,
320px, 300 samples), a `__getitem__` costs:

    read file bytes    0.03 ms   <- already served from the page cache
    JPEG decode        2.01 ms
    resize + ToTensor  4.12 ms
    mask open + resize 1.29 ms
    ------------------------------
    total              6.66 ms   -> 150 img/s per core

Disk I/O is 0.5% of that: with 754 GB of RAM the kernel already holds the
datasets in page cache, so caching *file bytes* would buy nothing. The cost is
JPEG decode and resize, which is why this module caches **decoded, already
resized uint8 arrays** instead. Reading one back costs 0.43 ms -- a 16x
reduction, and the difference between starving and feeding 8 A40s from only
16 CPU cores.

Why memory-mapped files in /dev/shm. The pipeline runs up to 8 training jobs
concurrently, each with its own DataLoader workers. A per-process in-RAM array
would be duplicated 8x (19 GB -> 152 GB) and re-decoded 8x. A single .npy in
/dev/shm (tmpfs, i.e. RAM) opened with `mmap_mode='r'` is shared by every
process and every worker: one copy of the pages, no decode after the first
build. Being tmpfs it is lost on reboot, which only costs a rebuild.

Correctness. Only the deterministic part of the pipeline is cached: resize to
a fixed size plus dtype conversion. Random augmentation (RandomErasing,
Mixup/CutMix in the heavy-aug arm) is applied to batches after loading, and
the 6x geometric train-aug set is materialized on disk as separate files, so
caching changes no augmentation semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

DEFAULT_CACHE_ROOT = "/dev/shm/beemachine_cache"

# Bump whenever the decode path changes what bytes land in the cache. The
# fingerprint otherwise covers only the source paths and target size, so a
# corrected decoder would silently keep serving arrays built by the old one --
# which is exactly what happened when palette-mode masks were fixed: the cache
# kept returning the luminance-mangled ids until it was cleared by hand.
# v2: palette ("P") masks read as indices instead of `.convert("L")`.
DECODER_VERSION = 2

# How many files to stat when fingerprinting content (see `_fingerprint`).
_FINGERPRINT_SAMPLE = 512


def cache_root(cfg: dict | None = None) -> Path:
    """Where cache files live.

    Config `cache.root`, else env `BEEMACHINE_CACHE_ROOT`, else
    ``DEFAULT_CACHE_ROOT`` (``/dev/shm/beemachine_cache``, the tmpfs mount the
    mechanism above assumes). A disk-backed path still works and still avoids
    re-decoding; it only loses the "one copy of the pages" property.
    """
    if cfg is not None:
        configured = (cfg.get("cache") or {}).get("root")
        if configured:
            return Path(str(configured))
    return Path(os.environ.get("BEEMACHINE_CACHE_ROOT", DEFAULT_CACHE_ROOT))


def cache_enabled(cfg: dict | None = None) -> bool:
    if os.environ.get("BEEMACHINE_CACHE") in {"0", "false", "off"}:
        return False
    if cfg is None:
        return True
    return bool((cfg.get("cache") or {}).get("enabled", True))


def _fingerprint(key: str, paths: Sequence[str], image_size: int, mode: str) -> str:
    """Identify a cache by its content, not just its name.

    The digest covers every source path and the target size, so adding images,
    regenerating the augmentation set, or changing resolution all produce a
    different cache file rather than silently reusing a stale one.
    """
    h = hashlib.sha256()
    h.update(f"{key}|{image_size}|{mode}|{len(paths)}|v{DECODER_VERSION}".encode())
    for p in paths:
        h.update(str(p).encode())
        h.update(b"\0")

    # Paths alone are not enough: regenerating the augmentation set rewrites the
    # same filenames with different pixels, and a path-only fingerprint would
    # keep serving the pre-regeneration arrays. Mixing in (size, mtime) from a
    # bounded, evenly-spaced sample detects that -- a regeneration touches every
    # file, so any sample sees it -- without stat()-ing 195k files on every run.
    for i in _sample_indices(len(paths), _FINGERPRINT_SAMPLE):
        try:
            st = os.stat(paths[i])
            h.update(f"|{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            h.update(b"|missing")
    return h.hexdigest()[:16]


def _sample_indices(n: int, k: int) -> list[int]:
    """Up to ``k`` evenly-spaced indices into a sequence of length ``n``."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    step = n / float(k)
    return sorted({int(i * step) for i in range(k)})


def _decode_image(path: str, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR),
        dtype=np.uint8,
    )


def read_mask_array(path: str, size: int) -> np.ndarray:
    """Class-id mask at ``size``, honoring palette-mode PNGs.

    Fish-Vista's ground-truth segmentation masks are palette ("P") PNGs whose
    palette *indices* are the class ids. `.convert("L")` maps palette colours
    to luminance instead, so class 1 becomes 48 and class 7 becomes 157 --
    values far outside the label range. One-hot encoding those ids then fails
    with a CUDA device-side assert (`ScatterGatherKernel.cu ... index out of
    bounds`), and any code that survived it would be training against garbage
    labels. Palette images must be read as indices.

    Every other mask source here is mode "L" with ids already in range (or, for
    CUB's per-part masks, binary RGB that callers threshold), so those take the
    grayscale path unchanged.
    """
    im = Image.open(path)
    if im.mode != "P":
        im = im.convert("L")
    # NEAREST preserves exact ids in both modes; resizing a palette image keeps
    # it in palette space, so the indices survive.
    return np.asarray(im.resize((size, size), Image.NEAREST), dtype=np.uint8)


def _decode_mask(path: str, size: int) -> np.ndarray:
    return read_mask_array(path, size)


def build_or_load(
    key: str,
    paths: Sequence[str],
    image_size: int,
    *,
    mode: str = "rgb",
    cfg: dict | None = None,
    decoder: Callable[[str, int], np.ndarray] | None = None,
    decode_index: Callable[[int, int], np.ndarray] | None = None,
    verbose: bool = True,
) -> np.ndarray | None:
    """Return a read-only memmap of decoded images, building it if needed.

    ``mode`` is "rgb" (N, S, S, 3) or "mask" (N, S, S), both uint8. Returns
    None when caching is disabled or the cache cannot be created, so callers
    fall back to reading from disk rather than failing.

    ``paths`` always identifies the content for fingerprinting. Datasets whose
    samples are not a single file -- CUB combines per-part mask PNGs on the fly
    -- pass ``decode_index`` instead of ``decoder`` and build sample ``i``
    themselves; ``paths[i]`` is then just a stable identity string.

    Build is done under a lock file so that 8 jobs starting at once produce one
    cache, not eight half-written ones.
    """
    if not cache_enabled(cfg) or not paths:
        return None

    root = cache_root(cfg)
    digest = _fingerprint(key, paths, image_size, mode)
    npy = root / f"{key}_{mode}_{image_size}_{digest}.npy"
    meta = npy.with_suffix(".json")

    try:
        root.mkdir(parents=True, exist_ok=True)
        if npy.exists() and meta.exists():
            return np.load(npy, mmap_mode="r")

        import fcntl

        lock = root / f"{npy.name}.lock"
        with open(lock, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # Another process may have finished while we waited.
                if npy.exists() and meta.exists():
                    return np.load(npy, mmap_mode="r")

                decode = decoder or (_decode_mask if mode == "mask" else _decode_image)
                if decode_index is not None:
                    decode = None
                n = len(paths)
                shape = (n, image_size, image_size) if mode == "mask" else (
                    n, image_size, image_size, 3
                )
                nbytes = int(np.prod(shape))
                if verbose:
                    print(
                        f"[cache] building {npy.name}: {n:,} images, "
                        f"{nbytes / 1e9:.1f} GB at {image_size}px -> {root}"
                    )
                free = _free_bytes(root)
                if free is not None and nbytes > free * 0.9:
                    print(
                        f"[cache] SKIP {npy.name}: needs {nbytes / 1e9:.1f} GB but only "
                        f"{free / 1e9:.1f} GB free under {root}; reading from disk instead."
                    )
                    return None

                tmp = npy.with_suffix(f".{os.getpid()}.tmp")
                arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=shape)
                _fill(
                    arr, paths, image_size, decode,
                    decode_index=decode_index, verbose=verbose,
                )
                arr.flush()
                del arr
                os.replace(tmp, npy)
                meta.write_text(
                    json.dumps(
                        {"key": key, "n": n, "image_size": image_size, "mode": mode},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        return np.load(npy, mmap_mode="r")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[cache] disabled for {key} ({type(exc).__name__}: {exc}); reading from disk.")
        return None


def _fill(arr, paths, image_size, decode, decode_index=None, verbose: bool = True) -> None:
    """Decode every path into ``arr``, in parallel across CPU cores."""
    from concurrent.futures import ThreadPoolExecutor

    # PIL decode releases the GIL, so threads scale here and avoid the memory
    # cost of forking a process per worker.
    n_threads = max(1, min(16, (os.cpu_count() or 8)))

    def one(i_path):
        i, p = i_path
        arr[i] = decode_index(i, image_size) if decode_index is not None else decode(p, image_size)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        # Wrap the *result* iterator, not the input. ThreadPoolExecutor.map
        # pulls from its input to schedule ahead of the workers, so a bar on
        # the input reaches 100% while minutes of decoding remain -- on the
        # 195k-image scale corpus it read "done" with ~40 GB still to write.
        # The result iterator yields only as tasks actually finish.
        results = ex.map(one, enumerate(paths), chunksize=32)
        if verbose:
            try:
                from tqdm import tqdm

                results = tqdm(
                    results, total=len(paths), desc="[cache] decoding", unit="img"
                )
            except Exception:
                pass
        for _ in results:
            pass


def _free_bytes(path: Path) -> int | None:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except Exception:
        return None


def clear_cache(cfg: dict | None = None) -> int:
    """Remove every cache file. Returns how many were deleted."""
    root = cache_root(cfg)
    if not root.is_dir():
        return 0
    n = 0
    for p in list(root.glob("*.npy")) + list(root.glob("*.json")) + list(root.glob("*.lock")):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


# A tiny CLI, because `clear_cache` had no caller anywhere in the repo and the
# shared-memory cache under /dev/shm outlives the run that filled it. A stale
# cache is a real operational hazard -- it keys off image path and resolution,
# so a re-exported dataset at the same paths silently serves the old pixels --
# and there was no supported way to drop it. Deleting the function would have
# removed the only remedy along with the dead code.
#
#     ./run.sh image_cache.py --clear
#     ./run.sh image_cache.py --clear --config config.yaml --dataset cub
if __name__ == "__main__":
    import argparse

    from config import default_config_path, load_config

    ap = argparse.ArgumentParser(description="Inspect or clear the decoded-image cache.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--clear", action="store_true",
                    help="Delete every cache file, then report how many were removed.")
    _a = ap.parse_args()
    _cfg = load_config(_a.config or default_config_path(), dataset=_a.dataset)
    _root = cache_root(_cfg)
    if _a.clear:
        print(f"cleared {clear_cache(_cfg)} file(s) from {_root}")
    else:
        _n = len(list(_root.glob("*.npy"))) if _root.is_dir() else 0
        print(f"{_root}: {_n} cached array(s); pass --clear to remove them")
