"""Multi-GPU helpers for BeeMachine Parts (8× A40 default).

Default training style: **one model per GPU**, queued in parallel across jobs.
Optional DDP helpers remain for large single-job runs (e.g. Stage C).
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def seed_everything(seed: int, rank: int = 0) -> int:
    """Seed Python, NumPy, and torch consistently for one process."""
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective


def resolve_num_gpus(cfg: dict[str, Any] | None = None) -> int:
    """Return how many GPUs to use. Config `compute.num_gpus` (-1 = all available)."""
    if not torch.cuda.is_available():
        return 1
    avail = torch.cuda.device_count()
    if cfg is None:
        return avail
    n = cfg.get("compute", {}).get("num_gpus", -1)
    if n is None or int(n) < 0:
        return avail
    return max(1, min(int(n), avail))


def per_device_batch_size(global_batch: int, num_devices: int) -> int:
    """Config batch sizes are global; DDP loaders use per-device chunks."""
    return max(1, int(global_batch) // max(1, int(num_devices)))


# ---------------------------------------------------------------------------
# Memory-aware batch-size computation
# ---------------------------------------------------------------------------

# Empirical "MiB per training sample" multipliers derived from A40 profiles.
# Each covers the raw FP32 input tensor PLUS encoder/decoder feature maps,
# backward-pass gradient buffers, and AdamW's two moment arrays.
#
#   SMP segmentation @ 320px, AMP:   raw 1.17 MiB × 200 ≈ 234 MiB/sample
#   Classification @ 224px, fp32:    raw 0.57 MiB × 80  ≈  46 MiB/sample
#   Inference only (no backward):    ≈ 30 × raw input (no grad buffer, no optimizer)
_MULT_SEG_TRAIN = 200       # legacy flat SMP multiplier; see _MULT_SEG_TRAIN_BY_ARCH

# Per-decoder multipliers, measured under AMP at 320px with a ResNeXt-50 encoder
# (peak allocated / batch, then divided by image_size^2*3*4). Rounded up ~10%
# for cuDNN workspace and allocator fragmentation, which a 4-sample probe does
# not see. Re-measure if the encoder or AMP dtype changes.
_MULT_SEG_TRAIN_BY_ARCH = {
    "pspnet": 125,
    "pan": 165,
    "deeplabv3plus": 195,
    "fpn": 195,
    "linknet": 220,
    "segformer": 245,
    "upernet": 285,
    "unetplusplus": 415,
    "manet": 670,
}
_MULT_SEG_TRAIN_UNKNOWN = 415   # UNet++ level: a safe batch beats a dead run
_MULT_CLS_TRAIN = 80        # classification, training
_MULT_INFERENCE = 30        # inference-only (no gradient, no optimizer states)


def gpu_free_memory_mib(device_index: int = 0) -> int:
    """Free GPU memory in MiB on *device_index*, or 0 when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        return int(free_bytes) >> 20
    except Exception:
        return 0


def auto_batch_size(
    device_index: int,
    mib_per_sample: float,
    max_batch: int,
    *,
    min_batch: int = 1,
    safety_factor: float = 0.80,
) -> int:
    """Largest power-of-two batch size that fits in the GPU's current free memory.

    ``safety_factor`` (default 0.80) reserves headroom for the CUDA workspace
    allocator, cuDNN benchmark buffers, and any other processes that share the
    device.  Returns ``max_batch`` on CPU (no CUDA) or when ``mib_per_sample``
    is zero.  Always clamps to [``min_batch``, ``max_batch``].
    """
    if not torch.cuda.is_available() or mib_per_sample <= 0:
        return max_batch
    free = gpu_free_memory_mib(device_index)
    if free <= 0:
        return max_batch
    usable = free * safety_factor
    bs = int(usable / mib_per_sample)
    bs = max(min_batch, min(bs, max_batch))
    if bs > 1:
        bs = 2 ** int(math.log2(bs))   # round down to power of 2
    return max(min_batch, bs)


def mib_per_seg_sample(
    image_size: int, *, backend: str = "smp", arch: str | None = None
) -> float:
    """Estimated MiB per training sample for segmentation at *image_size* px.

    The decoder matters as much as resolution, which a single constant used
    to hide. Measured under AMP at 320px with a ResNeXt-50 encoder, the spread
    across the sweep is 5.4x: PSPNet needs 132 MiB/sample, UNet++ 438, and
    MAnet 710. The old flat multiplier of 200 sat near FPN, so `auto_batch_size`
    handed UNet++ and MAnet the same batch as the cheapest decoder -- at 320px
    that is batch 128, which UNet++ cannot fit on a 45 GiB A40 (it needs ~55
    GiB) and MAnet misses by more than double. Both OOMed; the cheap decoders
    did not, which is why the failure looked architecture-specific rather than
    like a sizing bug.

    Unknown architectures fall back to the UNet++ figure rather than the mean:
    over-estimating costs a smaller batch, under-estimating costs a dead run.
    """
    key = (arch or "").lower()
    mult = _MULT_SEG_TRAIN_BY_ARCH.get(key, _MULT_SEG_TRAIN_UNKNOWN)
    return image_size * image_size * 3 * 4 * mult / (1024 * 1024)


def mib_per_cls_sample(image_size: int, n_streams: int = 1) -> float:
    """Estimated MiB per training sample for classification at *image_size* px.

    *n_streams* is the number of parallel image streams per sample: 1 for
    whole-image arms, K+1 for part-crop arms with K foreground parts.
    """
    return n_streams * image_size * image_size * 3 * 4 * _MULT_CLS_TRAIN / (1024 * 1024)


def mib_per_inf_sample(image_size: int) -> float:
    """Estimated MiB per sample for inference-only passes (no backward graph)."""
    return image_size * image_size * 3 * 4 * _MULT_INFERENCE / (1024 * 1024)


def _write_job_meta(path: Path, **fields: Any) -> None:
    """Merge `fields` into a job's sidecar JSON. Never raises.

    The dashboard polls these while jobs run, so a half-written file must not
    take down a training job. Written to a temp file and renamed so a reader
    never sees a partial document.
    """
    try:
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = {}
        existing.update({k: v for k, v in fields.items() if v is not None})
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(existing), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


JOB_LOG_DIR_ENV = "BEEMACHINE_JOB_LOG_DIR"


def job_log_dir() -> Path | None:
    """Where per-job logs go, or None to inherit the parent's stdout.

    Set by `tools/run_monitor.py`. Unset -- every direct shell invocation --
    keeps the original behaviour: children write straight to the terminal.
    """
    raw = os.environ.get(JOB_LOG_DIR_ENV)
    if not raw:
        return None
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


_SLUG_FLAGS = (
    "--mode", "--backbone", "--mask_source", "--group", "--seed",
    "--arch", "--encoder", "--source",
)


def job_slug(argv: Sequence[str]) -> str:
    """A short, unique-enough name for one parallel job, from its argv.

    `stage_b.py fusion --mode gated_residual --group shape --seed 77 ...` becomes
    `fusion_gated_residual_shape_77`. With 48 jobs in a single `ablate` submission,
    a log named by index alone is useless when one of them fails.
    """
    parts: list[str] = []
    argv = list(argv)
    for i, token in enumerate(argv):
        if token.endswith(".py"):
            rest = argv[i + 1:]
            if rest and not rest[0].startswith("-"):
                parts.append(rest[0])
            break
    for flag in _SLUG_FLAGS:
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                parts.append(str(argv[i + 1]).split("/")[-1])
    slug = "_".join(parts) or "job"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", slug)[:80]


def _force_device_arg(cmd: list[str], device: int = 0) -> list[str]:
    """Ensure ``--device N`` is present (replace existing)."""
    out = list(cmd)
    if "--device" in out:
        i = out.index("--device")
        if i + 1 < len(out):
            out[i + 1] = str(device)
        else:
            out.append(str(device))
    else:
        out.extend(["--device", str(device)])
    return out


def run_commands_parallel(
    commands: Sequence[Sequence[str]],
    num_gpus: int,
    *,
    cwd: str | None = None,
) -> list[int]:
    """
    Run one subprocess per command, assigning free GPUs from a pool.

    Each job gets ``CUDA_VISIBLE_DEVICES=<gpu>`` and ``--device 0`` so the
    process only sees that physical GPU. At most ``num_gpus`` jobs run at once.
    """
    commands = [list(c) for c in commands]
    if not commands:
        return []
    num_gpus = max(1, int(num_gpus))
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in range(num_gpus):
        gpu_q.put(g)
    codes = [1] * len(commands)

    log_dir = job_log_dir()

    def _run(idx: int, cmd: list[str]) -> tuple[int, int]:
        gpu = gpu_q.get()
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            argv = _force_device_arg(cmd, 0)
            if log_dir is None:
                # Unchanged default: children share this terminal.
                print(f"[parallel gpu={gpu}] {' '.join(argv)}", flush=True)
                proc = subprocess.run(argv, cwd=cwd, env=env)
                return idx, int(proc.returncode)

            # Monitored run: one log per job, plus a sidecar the dashboard
            # polls. Without this, 8-48 concurrent tqdm bars interleave into a
            # single stream and neither a human nor a tool can read it.
            stem = f"{idx:03d}_{job_slug(argv)}"
            meta = log_dir / f"{stem}.json"
            _write_job_meta(meta, idx=idx, gpu=gpu, argv=argv, started=time.time())
            with open(log_dir / f"{stem}.log", "w", encoding="utf-8", buffering=1) as fh:
                fh.write(f"$ CUDA_VISIBLE_DEVICES={gpu} {' '.join(argv)}\n")
                fh.flush()
                proc = subprocess.run(
                    argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT
                )
            rc = int(proc.returncode)
            _write_job_meta(
                meta, idx=idx, gpu=gpu, argv=argv, started=None,
                finished=time.time(), rc=rc,
            )
            return idx, rc
        finally:
            gpu_q.put(gpu)

    workers = min(num_gpus, len(commands))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, i, cmd) for i, cmd in enumerate(commands)]
        for fut in as_completed(futs):
            idx, code = fut.result()
            codes[idx] = code
            if code != 0:
                print(f"[parallel] job {idx} failed with code {code}", flush=True)
    return codes


def world_info() -> tuple[int, int, int]:
    """(rank, local_rank, world_size). Defaults to single-process when not launched under torchrun."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    if "RANK" in os.environ:
        return (
            int(os.environ["RANK"]),
            int(os.environ.get("LOCAL_RANK", 0)),
            int(os.environ.get("WORLD_SIZE", 1)),
        )
    return 0, 0, 1


def is_main_process() -> bool:
    return world_info()[0] == 0


def init_distributed() -> tuple[int, int, int]:
    """Initialize NCCL process group when launched via torchrun / Lightning env."""
    if not torch.cuda.is_available():
        return 0, 0, 1
    if dist.is_available() and dist.is_initialized():
        return world_info()
    if "RANK" not in os.environ:
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return world_info()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ensure_torchrun(num_gpus: int) -> None:
    """
    Re-exec the current CLI under torchrun when multi-GPU is requested and we are
    not already inside a distributed context. No-op for single-GPU / CPU.
    """
    if num_gpus <= 1 or not torch.cuda.is_available():
        return
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        return
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--standalone",
        *sys.argv,
    ]
    raise SystemExit(subprocess.call(cmd))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(
    model: torch.nn.Module,
    local_rank: int,
    world_size: int,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if world_size <= 1 or not torch.cuda.is_available():
        return model
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    world_size: int = 1,
    drop_last: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = DistributedSampler(dataset, shuffle=shuffle) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
    return loader, sampler


def device_for_rank(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")
