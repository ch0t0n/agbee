"""
Classical part descriptors (shape / appearance / inter-part).

Adapted from ref_codes/generate_groundtruth_descriptors and prior work (Choton et al., 2026).
Heavy optional deps (brisque, niqe, piqe, mahotas) degrade gracefully.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.measure import label, regionprops, shannon_entropy
from skimage.util import img_as_ubyte

try:
    from mahotas.features import zernike_moments
except Exception:  # pragma: no cover
    zernike_moments = None

try:
    from brisque import BRISQUE
except Exception:  # pragma: no cover
    BRISQUE = None

try:
    from pypiqe import piqe
except Exception:  # pragma: no cover
    piqe = None

# Classical NIQE via scikit-video. Broken in this environment for a reason
# worth spelling out: skvideo's implementation calls `scipy.misc.imresize`,
# removed in SciPy 1.3, so every call raises AttributeError and the `except`
# below turned that into NaN. The result was four all-NaN columns
# (abdomen_niqe / head_niqe / thorax_niqe / full_niqe) that looked like real
# descriptors in the column count. `_niqe_available()` probes it once so the
# failure is reported rather than absorbed; see
# `report_dead_descriptor_columns`.
try:
    from skvideo.measure import niqe
except Exception:  # pragma: no cover
    niqe = None

_niqe_probe: bool | None = None


def niqe_available() -> bool:
    """Whether classical NIQE actually computes, probed once."""
    global _niqe_probe
    if _niqe_probe is None:
        if niqe is None:
            _niqe_probe = False
        else:
            try:
                niqe(np.random.default_rng(0).random((1, 96, 96)).astype(np.float64))
                _niqe_probe = True
            except Exception:
                _niqe_probe = False
    return _niqe_probe

# Learned no-reference IQA metrics, aligned with the metric suite used for
# perceptual scoring in the NTIRE 2026 challenge report (Wang et al. 2026,
# "The Second Challenge on Real-World Face Restoration at NTIRE 2026",
# arXiv:2604.10532 Sec. 2.2.2): NIQE (statistical, handled above), CLIPIQA,
# MANIQA, MUSIQ, and Q-Align. Loaded lazily via the `pyiqa` (IQA-PyTorch)
# toolbox referenced in that report, so environments without a GPU / without
# `pyiqa` installed degrade to NaN exactly like the classical metrics above.
try:
    import pyiqa
except Exception:  # pragma: no cover
    pyiqa = None

# Q-Align is deliberately absent. It cannot load under this environment's
# transformers (4.57): its weights ship as remote code targeting the pre-4.49
# Llama attention API, and construction dies on
# `LlamaRotaryEmbedding.__init__() got an unexpected keyword argument
# 'max_position_embeddings'` even after the missing star-import symbols are
# backfilled. pyiqa 0.1.16 does not fix this -- it requires transformers>=5.0,
# which would break timm. Until that upgrade is worth doing, the learned
# branch is the three metrics below.
# This used to "work" only because a bare `except Exception` turned the failure
# into a silent NaN column for every image.
_LEARNED_IQA_NAMES = ("clipiqa", "maniqa", "musiq")
_iqa_metric_cache: dict[str, Any] = {}
_iqa_device = "cpu"
# Smallest side these networks tolerate before their pooling stages collapse.
_IQA_MIN_SIZE = 224


def set_iqa_device(device: str) -> None:
    """Select the device learned NR-IQA metrics run on (call once at startup).

    Defaults to CPU, which is ~11x slower per image than a GPU (measured:
    12.5 s vs 1.17 s on this machine's A40s). Every entry point that extracts
    descriptors must call this, or an 8-GPU node quietly does the work on 16
    CPU cores.
    """
    global _iqa_device
    _iqa_device = device
    _iqa_metric_cache.clear()


def _create_iqa_metric(name: str):
    if pyiqa is None:
        raise RuntimeError(
            "pyiqa is not installed, so the learned NR-IQA descriptors cannot be "
            "computed. Install it or remove `descriptors.learned_iqa_metrics` from the config."
        )
    return pyiqa.create_metric(name, device=_iqa_device)


def validate_iqa_metrics(names: Sequence[str] | None = None) -> list[str]:
    """Load every configured metric up front, raising on the first failure.

    Called once before a descriptor extraction run. A metric that fails to
    load would otherwise produce an all-NaN column that flows into every
    fusion arm and every ablation cell without anything noticing -- which is
    exactly what happened to Q-Align. Failing here costs seconds; failing
    silently costs the credibility of an entire descriptor group.
    """
    names = list(names) if names is not None else list(_LEARNED_IQA_NAMES)
    broken: list[str] = []
    for name in names:
        try:
            _iqa_metric_cache[name] = _create_iqa_metric(name)
        except Exception as exc:  # pragma: no cover - environment dependent
            broken.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
    if broken:
        raise SystemExit(
            "Learned NR-IQA metric(s) failed to load on device "
            f"'{_iqa_device}':\n- " + "\n- ".join(broken)
            + "\n\nFix the environment or drop them from "
            "`descriptors.learned_iqa_metrics` in the config. Refusing to run: a "
            "metric that cannot load yields an all-NaN descriptor column."
        )
    return names


def _get_iqa_metric(name: str):
    if name not in _iqa_metric_cache:
        _iqa_metric_cache[name] = _create_iqa_metric(name)
    return _iqa_metric_cache[name]


def report_dead_descriptor_columns(df, *, strict: bool = True) -> list[str]:
    """Flag descriptor columns that are NaN for every row.

    Distinguishes two very different causes, because only one is a defect:

    * **A broken metric** kills the same measurement in *every* region --
      e.g. `abdomen_niqe`, `head_niqe`, `full_niqe` all NaN because skvideo's
      NIQE calls a SciPy function removed years ago, or `image_iqa_qalign`
      NaN because the model cannot load. That inflates the reported
      dimensionality with columns carrying no information, and it went
      unnoticed for two metrics. This is fatal under ``strict``.

    * **A region that never appears** kills every measurement within that one
      region and leaves the others intact -- a rare part absent from a small
      sample, or a segmenter that has not learned to predict it yet. On CUB,
      whose segmenter is the weakest of the three, the predicted masks can
      contain no part5/6/11 at all. That is a property of the data or the
      model, not a bug in extraction, so it is reported and allowed.

    Returns the dead column names either way.
    """
    meta = {"image", "class_id", "species"}
    feature_cols = [c for c in df.columns if c not in meta]
    dead = [c for c in feature_cols if df[c].isna().all()]
    if not dead:
        return []

    def split(col: str) -> tuple[str, str]:
        """(region, measurement); region is "" for whole-image columns."""
        m = re.match(r"^(part\d+|full|image|abdomen|head|thorax)_(.+)$", col)
        return (m.group(1), m.group(2)) if m else ("", col)

    dead_set = set(dead)
    regions_for: dict[str, set[str]] = {}
    for col in feature_cols:
        region, measure = split(col)
        regions_for.setdefault(measure, set()).add(region)

    broken_metrics = []
    for measure, regions in regions_for.items():
        cols = [c for c in feature_cols if split(c)[1] == measure]
        if cols and all(c in dead_set for c in cols):
            broken_metrics.append(measure)

    dead_regions = sorted(
        {split(c)[0] for c in dead if split(c)[0]}
        - {split(c)[0] for c in feature_cols if c not in dead_set and split(c)[0]}
    )

    if dead_regions:
        print(
            f"NOTE: {len(dead_regions)} region(s) never appear in these "
            f"{len(df)} rows, so their descriptors are undefined: "
            + ", ".join(dead_regions)
            + ".\n  Expected when a part is rare, or when predicted masks come "
            "from an under-trained segmenter."
        )

    if broken_metrics:
        msg = (
            f"{len(broken_metrics)} descriptor measurement(s) are NaN in EVERY "
            f"region across all {len(df)} rows, so the underlying metric is "
            "failing for every image:\n  "
            + "\n  ".join(sorted(broken_metrics)[:20])
            + ("\n  ..." if len(broken_metrics) > 20 else "")
            + "\n\nFix the metric, or drop it from the descriptor set so the "
            "reported dimensionality reflects what was actually measured."
        )
        if strict:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")
    return dead


_sift = cv2.SIFT_create()
_orb = cv2.ORB_create()
_bri = BRISQUE(url=False) if BRISQUE is not None else None

# Column prefixes used when building group masks for ablations
SHAPE_KEYS_CORE = [
    "area",
    "perimeter",
    "aspect_ratio",
    "extent",
    "solidity",
    "eccentricity",
    "orientation",
    "circularity",
    "elongation",
    "compactness",
]
APPEARANCE_KEYS = [
    "brightness",
    "contrast",
    "sharpness",
    "colorfulness",
    "entropy",
    "brisque",
    "niqe",
    "piqe",
    # Learned NR-IQA metrics (NTIRE-2026-aligned suite); computed once per
    # image on the full-body region and broadcast into every part's feature
    # block under the "iqa_" prefix — see extract_learned_iqa_features().
    "iqa_clipiqa",
    "iqa_maniqa",
    "iqa_musiq",
]
# Legacy Beemachine ratio names (still recognized for group masks)
INTERPART_KEYS = [
    "head_to_thorax_area",
    "thorax_to_abdomen_area",
    "head_to_total_area",
    "thorax_to_total_area",
    "abdomen_to_total_area",
]


def _is_interpart_key(name: str) -> bool:
    return (
        name in INTERPART_KEYS
        or name.startswith("area_ratio_part")
        or name.startswith("area_ratio_")
    )


def _safe_ratio(numerator: float, denominator: float, cap: float = 1e3) -> float:
    """``numerator / denominator``, clamped to +/-``cap``.

    Every ratio feature in this module (circularity, compactness, and the
    inter-part area ratios below) divides by a quantity that is legitimately
    near-zero for a real image: a degenerate/thin region can report a
    near-zero perimeter, and a structurally absent part (routine on
    Fish-Vista, e.g. no adipose fin) has zero area. The previous code added a
    ``1e-6`` floor/epsilon to the denominator only, which does prevent a
    ZeroDivisionError but not a numerical explosion -- an absent-part area
    ratio measured as high as 2.5e13 was destabilizing training for every
    fusion arm that consumed the shape or inter-part descriptor group (see
    the 2026-08-11 experiment audit). A ratio feature beyond a few hundred
    carries no more information than "the denominator was ~0"; clamping
    preserves that signal without the blow-up.
    """
    if denominator <= 1e-6:
        return float(cap if numerator > 0 else 0.0)
    return float(np.clip(numerator / denominator, -cap, cap))


def extract_base_features(mask: np.ndarray) -> dict[str, float]:
    """Geometric descriptors for one part mask, measured on its LARGEST
    connected component.

    That choice is a deliberate, documented convention, not an accident, and it
    is worth stating because part masks fragment often. Measured over 1,500
    Beemachine ground-truth masks:

      | part    | >1 component | 2nd blob >= 25% of largest |
      |---------|-------------:|---------------------------:|
      | abdomen |        47.0% |                      13.2% |
      | head    |        33.4% |                       4.7% |
      | thorax  |        40.0% |                       2.7% |

    The fragmentation is *within* one bee, not between two. The union of all
    parts -- the whole body -- splits into more than one large blob in only
    0.1% of images (2 of 1,500), and per image the number of parts showing a
    large second blob is 0 for 80.3%, exactly one for 18.9%, and two for 0.7%.
    Two bees would split several parts at once and split the body union too;
    one part splitting while the body stays connected is an occluder (a wing,
    a leg, a petal) cutting a part in half while its neighbours bridge the gap.

    Consequence, accepted knowingly: on the ~19% of images with a fragmented
    part, `area` and `perimeter` describe a piece rather than the whole part,
    and the shape ratios below describe that piece's outline. The convention is
    kept because it is applied identically to ground-truth and predicted masks,
    so the mask-regime comparison (RQ3) stays fair, and because changing it
    would alter the descriptor vector for every arm mid-study.

    Note the scope: only the ten ratios computed from `p` below are affected.
    Hu moments and Zernike moments integrate over the whole mask and see every
    fragment; Fourier descriptors take the largest *contour*, matching this.

    Prior work (Choton et al., 2026) used `props[0]` here -- whichever fragment
    came first in raster order, which is not even reliably the biggest one.
    """
    features = SHAPE_KEYS_CORE
    if mask is None or mask.sum() == 0:
        return {f: 0.0 for f in features}

    labeled = label(mask.astype(np.uint8))
    props = regionprops(labeled)
    if not props:
        return {f: 0.0 for f in features}
    p = max(props, key=lambda region: region.area)
    major_axis = p.major_axis_length
    minor_axis = p.minor_axis_length
    area = float(p.area)
    perimeter = float(p.perimeter)
    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": major_axis / minor_axis if minor_axis > 0 else 0.0,
        "extent": float(p.extent),
        "solidity": float(p.solidity),
        "eccentricity": float(p.eccentricity),
        "orientation": float(p.orientation),
        # Circularity is theoretically <=~1 (a disc); clamp to a generous
        # cap of 4 rather than the 1000 used for the ratios below, since any
        # value that far outside the physically meaningful range is purely a
        # perimeter-measurement artifact, not signal.
        "circularity": _safe_ratio(4 * np.pi * area, perimeter**2, cap=4.0),
        "elongation": float(1 - (minor_axis / major_axis)) if major_axis > 0 else 0.0,
        "compactness": _safe_ratio(perimeter**2, 4 * np.pi * area, cap=100.0),
    }


def _mean_descriptor(descs: np.ndarray | None, dim: int) -> np.ndarray:
    if descs is None or len(descs) == 0:
        return np.full(dim, np.nan, dtype=np.float32)
    return np.nanmean(descs, axis=0).astype(np.float32)


def compute_hu_moments(mask: np.ndarray) -> np.ndarray:
    hu = cv2.HuMoments(cv2.moments(mask.astype(np.uint8))).flatten()
    return np.log(np.abs(hu) + 1e-12)


def compute_zernike(mask: np.ndarray, degree: int = 8) -> np.ndarray:
    if zernike_moments is None:
        # order-8 Zernike → 25 coeffs
        return np.full(25, np.nan, dtype=np.float32)
    radius = max(min(mask.shape) // 2, 1)
    mask_norm = mask / mask.max() if mask.max() > 0 else mask
    return np.asarray(zernike_moments(mask_norm, radius=radius, degree=degree), dtype=np.float32)


def compute_fourier_descriptors(mask: np.ndarray, n_harmonics: int = 20) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.full(n_harmonics, np.nan, dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return np.full(n_harmonics, np.nan, dtype=np.float32)
    complex_contour = cnt[:, 0, 0] + 1j * cnt[:, 0, 1]
    cnt_centered = complex_contour - np.mean(complex_contour)
    fd = np.fft.fft(cnt_centered)
    if len(fd) < 2 or np.abs(fd[1]) == 0:
        return np.full(n_harmonics, np.nan, dtype=np.float32)
    fd = np.abs(fd / np.abs(fd[1]))
    out = fd[:n_harmonics]
    if len(out) < n_harmonics:
        out = np.concatenate([out, np.full(n_harmonics - len(out), np.nan)])
    return out.astype(np.float32)


def extract_shape_features(image_u8: np.ndarray, mask_u8: np.ndarray) -> dict[str, float]:
    features = extract_base_features(mask_u8)

    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    sift_kp, sift_ds = _sift.detectAndCompute(gray, mask_u8)
    sift_mean = _mean_descriptor(sift_ds, 128)
    features["sift_kp_n"] = float(len(sift_kp) if sift_kp else 0)
    features["sift_kp_size"] = float(max((k.size for k in sift_kp), default=0.0))
    for i, v in enumerate(sift_mean):
        features[f"sift_ds{i + 1}"] = float(v)

    orb_kp, orb_ds = _orb.detectAndCompute(gray, mask_u8)
    orb_mean = _mean_descriptor(orb_ds, 32)
    features["orb_kp_n"] = float(len(orb_kp) if orb_kp else 0)
    for i, v in enumerate(orb_mean):
        features[f"orb_ds{i + 1}"] = float(v)

    for i, v in enumerate(compute_hu_moments(mask_u8)):
        features[f"hu{i + 1}"] = float(v)
    for i, v in enumerate(compute_zernike(mask_u8)):
        features[f"zernike_{i + 1}"] = float(v)
    for i, v in enumerate(compute_fourier_descriptors(mask_u8)):
        features[f"fourier_{i + 1}"] = float(v)
    return features


def extract_visual_features(image_f: np.ndarray, mask_u8: np.ndarray) -> dict[str, float]:
    img_cropped = np.zeros_like(image_f)
    img_cropped[mask_u8 == 1] = image_f[mask_u8 == 1]
    brightness = float(np.mean(img_cropped))
    gray = rgb2gray(img_cropped)
    contrast = float(np.std(gray))
    gray_8u = (gray * 255).astype(np.uint8)
    sharpness = float(cv2.Laplacian(gray_8u, cv2.CV_64F).var())
    r, g, b = cv2.split((img_cropped * 255).astype(np.uint8))
    rg, yb = np.abs(r.astype(float) - g), np.abs(0.5 * (r + g) - b)
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )
    entropy = float(shannon_entropy(gray))

    brisque_score = float(_bri.score(img_cropped)) if _bri is not None else float("nan")
    if piqe is not None:
        try:
            piqe_score, *_ = piqe(gray)
            piqe_score = float(piqe_score)
        except Exception:
            piqe_score = float("nan")
    else:
        piqe_score = float("nan")

    out = {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "colorfulness": colorfulness,
        "entropy": entropy,
        "brisque": brisque_score,
        "piqe": piqe_score,
    }
    # Emitted only when it can actually be computed. skvideo's NIQE is broken
    # against SciPy >= 1.3 (see `niqe_available`), and a column that is NaN for
    # every image is worse than an absent one: it inflates the descriptor
    # dimensionality and silently dilutes the appearance group in every
    # ablation. Restore it by installing a working NIQE.
    if niqe_available():
        try:
            out["niqe"] = float(np.asarray(niqe(gray)).reshape(-1)[0])
        except Exception:
            out["niqe"] = float("nan")
    return out


def extract_learned_iqa_features(image_f: np.ndarray) -> dict[str, float]:
    """Learned no-reference IQA scores for the whole (unmasked) image.

    Why these four. NIQE, BRISQUE, and PIQE are classical, statistics-based
    "completely blind" metrics: cheap, but they correlate weakly with human
    judgments of realism/naturalness on modern, diverse imagery. CLIPIQA,
    MANIQA, MUSIQ, and Q-Align are the learned counterparts used for
    perceptual ranking in the NTIRE 2026 challenge report (arXiv:2604.10532,
    Sec. 2.2.2) and are added here as an additional, non-classical branch of
    phi_q so the appearance-quality group is not limited to hand-designed
    statistics.

    - CLIPIQA: zero-shot IQA from a frozen CLIP image-text embedding, scored
      as the softmax similarity between the image and the antonym prompt
      pair ("Good photo." / "Bad photo."). Measures overall semantic
      naturalness rather than a specific distortion; interpreted as [0, 1],
      higher = more natural-looking crop.
    - MANIQA: a multi-dimension attention transformer trained with human
      MOS labels, predicting a scalar quality score directly from patch
      features via channel- and spatial-attention blocks. Captures texture
      and local-distortion cues (blur, compression, noise) that pure
      shape/statistical descriptors miss; interpreted as [0, 1], higher =
      better.
    - MUSIQ: a multi-scale image quality transformer that consumes the
      native-resolution image at several scales, so it does not require a
      fixed-size resize (relevant here because part crops vary widely in
      native size). Interpreted as roughly [0, 100], higher = better.
    - Q-Align: a large multimodal-model-based scorer that maps quality onto
      discrete text levels (excellent/good/fair/poor/bad) and returns their
      probability-weighted score. Included as a semantically grounded,
      language-model-based reference point distinct from the three
      regression-style scores above; interpreted as [1, 5], higher = better.

    How computed. Each metric is a frozen, pretrained network loaded lazily
    through the `pyiqa` (IQA-PyTorch) toolbox — the same toolbox the NTIRE
    report cites for its CLIPIQA computation — and evaluated once per image
    on the full (unmasked) crop; scores are NOT computed per tiny part crop,
    since these networks are trained on natural photographs at moderate
    resolution and are unreliable on small, heavily-cropped regions.

    How interpreted / integrated. These four scores are appended to phi_q
    like any other appearance descriptor (masked/scaled by
    `descriptor_group_mask`), so they participate in every fusion arm's
    "appearance" ablation on equal footing with BRISQUE/NIQE/PIQE. They also
    give the confidence-weighting ablation (see `confidence_weight_vector`)
    a second, complementary noise signal: segmenter confidence indicates
    "was the mask found", while these scores indicate "is the crop worth
    trusting once found" (blur, compression, and low resolution can make a
    correctly-localized crop unusable even when mask confidence is high).
    """
    return extract_learned_iqa_features_batch(image_f[None, ...])[0]


def learned_iqa_keys(names: Sequence[str] | None = None) -> list[str]:
    """Column names the learned branch contributes, in a stable order."""
    names = list(names) if names is not None else list(_LEARNED_IQA_NAMES)
    return [f"iqa_{n}" for n in names]


def extract_learned_iqa_features_batch(
    images_f: np.ndarray,
    names: Sequence[str] | None = None,
) -> list[dict[str, float]]:
    """Learned NR-IQA scores for a batch of images, one forward pass per metric.

    ``images_f`` is (N, H, W, 3) in [0, 1]. Scoring one image at a time leaves
    an A40 almost entirely idle -- these are small networks and the per-call
    Python/launch overhead dominates -- so the extraction loop feeds them in
    chunks instead. Metrics that do not support batching are detected once and
    fall back to a per-image loop rather than silently returning a wrong shape.

    Errors propagate. A metric that cannot run must not be reported as NaN;
    see `validate_iqa_metrics` for why.
    """
    import torch

    names = list(names) if names is not None else list(_LEARNED_IQA_NAMES)
    keys = learned_iqa_keys(names)
    arr = np.ascontiguousarray(np.asarray(images_f, dtype=np.float32))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) float images, got {arr.shape}")
    n = arr.shape[0]
    if not names:
        return [{} for _ in range(n)]

    batch = torch.from_numpy(arr.transpose(0, 3, 1, 2)).clamp(0, 1).to(_iqa_device)
    # These networks are trained at 224 and pool their feature maps several
    # times; anything smaller than that collapses to a 0x0 activation
    # ("Calculated output size: (128x0x0)"). Pipeline images arrive at
    # `image_size` (320) so this never fires in practice, but a degenerate
    # crop must not take down a 200k-image extraction job. Upsampling is the
    # honest fallback -- unlike the NaN this used to return, it still yields a
    # real, deterministic score, and it cannot hide a broken metric.
    if min(batch.shape[-2:]) < _IQA_MIN_SIZE:
        batch = torch.nn.functional.interpolate(
            batch, size=(_IQA_MIN_SIZE, _IQA_MIN_SIZE), mode="bilinear", align_corners=False
        )
    out: list[dict[str, float]] = [{} for _ in range(n)]
    for name, key in zip(names, keys):
        metric = _get_iqa_metric(name)
        with torch.no_grad():
            try:
                scores = metric(batch).reshape(-1)
                if scores.numel() != n:
                    raise RuntimeError(f"returned {scores.numel()} scores for {n} images")
            except Exception:
                # Some metrics (notably any with batch-size-1 preprocessing)
                # cannot take a stacked batch; fall back per image rather than
                # mis-assigning scores across the batch.
                scores = torch.stack(
                    [metric(batch[i : i + 1]).reshape(-1)[0] for i in range(n)]
                )
        vals = scores.detach().float().cpu().tolist()
        for i, v in enumerate(vals):
            out[i][key] = float(v)
    return out


def confidence_weight_vector(
    feature_vector: np.ndarray, confidence: float, feature_names: list[str] | None = None
) -> np.ndarray:
    """Optional confidence-weighted descriptor aggregation (evaluated, not proposed).

    Element-wise scales a descriptor vector by a scalar reliability signal
    (e.g. the segmenter's mean foreground pixel confidence, or a learned
    NR-IQA score for the region). This is one arm of the "weighted vs.
    unweighted descriptor aggregation" ablation (see phi_hat in the paper):
    Phi_hat = w_conf * [phi_s || phi_q || phi_c]. It is a standard,
    well-known reliability-weighting idea, not a contribution of this study;
    it is evaluated here as one more design choice alongside unweighted
    concatenation, so that the comparison covers "does noise-aware
    weighting help" as well as "which fusion mechanism is best."
    """
    w = float(np.clip(confidence, 0.0, 1.0))
    if feature_names is not None and len(feature_names) == len(feature_vector):
        out = feature_vector.copy()
        for i, name in enumerate(feature_names):
            if _is_interpart_key(name):
                continue  # ratios are already scale-free; leave unweighted
            out[i] = feature_vector[i] * w
        return out
    return feature_vector * w


def extract_combined_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask_u8 = mask.astype(np.uint8)
    image_f = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_u8 = img_as_ubyte(image_f)
    out = extract_shape_features(image_u8, mask_u8)
    out.update(extract_visual_features(image_f, mask_u8))
    return out


def normalize_image_for_iqa(image: Any) -> np.ndarray:
    """(H, W, 3) float in [0, 1], from a CHW tensor or an HWC array."""
    image_np = (
        image.detach().cpu().numpy().transpose(1, 2, 0) if hasattr(image, "detach") else image
    )
    image_np = np.asarray(image_np, dtype=np.float32)
    return (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)


def extract_all_features(
    image: Any,
    mask: Any,
    part_labels: list[str] | None = None,
    learned_iqa: dict[str, float] | None = None,
) -> dict[str, float]:
    """Descriptor vector over parts + full body + area ratios.

    Beemachine (abdomen/head/thorax labels): legacy named keys (~938 dims).
    CUB / Fish / other: reference-protocol ``part{id}_*`` + pairwise area ratios.

    ``learned_iqa`` lets a caller supply the NR-IQA scores it already computed
    for this image in a batched forward pass (see
    `extract_learned_iqa_features_batch`); when omitted they are computed here,
    one image at a time.
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)

    # Learned NR-IQA metrics are computed once per image, on the whole
    # (unmasked) frame, and broadcast into the record under "image_iqa_*"
    # (see extract_learned_iqa_features docstring for why per-part IQA is
    # not computed). Every downstream group mask picks these up as
    # ordinary phi_q columns.
    if learned_iqa is None:
        learned_iqa = extract_learned_iqa_features(normalize_image_for_iqa(image))
    iqa_prefix_record = {f"image_{k}": v for k, v in learned_iqa.items()}

    name_to_id: dict[str, int] = {}
    if part_labels is not None:
        name_to_id = {
            str(n).lower(): i
            for i, n in enumerate(part_labels)
            if str(n).lower() not in {"background", "bg"}
        }

    # Legacy Beemachine path (stable CSV schema / expected_dim=938)
    if {"abdomen", "head", "thorax"}.issubset(name_to_id.keys()) or (
        part_labels is None and set(int(x) for x in np.unique(mask) if int(x) > 0) <= {1, 2, 3}
    ):
        abd_id = name_to_id.get("abdomen", 1)
        head_id = name_to_id.get("head", 2)
        thorax_id = name_to_id.get("thorax", 3)
        parts = {
            "abdomen": mask == abd_id,
            "head": mask == head_id,
            "thorax": mask == thorax_id,
            "full": mask > 0,
        }
        record: dict[str, float] = {}
        part_feats = {}
        for name, m in parts.items():
            feats = extract_combined_features(image, m.astype(np.uint8))
            part_feats[name] = feats
            record.update({f"{name}_{k}": float(v) for k, v in feats.items()})
        area_sum = (
            part_feats["head"]["area"]
            + part_feats["thorax"]["area"]
            + part_feats["abdomen"]["area"]
        )
        record.update(
            {
                "head_to_thorax_area": _safe_ratio(
                    part_feats["head"]["area"], part_feats["thorax"]["area"]
                ),
                "thorax_to_abdomen_area": _safe_ratio(
                    part_feats["thorax"]["area"], part_feats["abdomen"]["area"]
                ),
                "head_to_total_area": _safe_ratio(part_feats["head"]["area"], area_sum),
                "thorax_to_total_area": _safe_ratio(part_feats["thorax"]["area"], area_sum),
                "abdomen_to_total_area": _safe_ratio(part_feats["abdomen"]["area"], area_sum),
            }
        )
        record.update(iqa_prefix_record)
        return record

    # Generalized N-part path (CUB / Fish-Vista)
    part_ids = sorted(int(pid) for pid in np.unique(mask) if int(pid) > 0)
    if part_labels is not None:
        labeled_ids = [
            i
            for i, name in enumerate(part_labels)
            if str(name).lower() not in {"background", "bg"}
        ]
        extra = [p for p in part_ids if p not in labeled_ids]
        part_ids = labeled_ids + extra

    record = {}
    part_feats = {}
    for pid in part_ids:
        feats = extract_combined_features(image, (mask == pid).astype(np.uint8))
        part_feats[pid] = feats
        for k, v in feats.items():
            record[f"part{pid}_{k}"] = float(v)

    full_feats = extract_combined_features(image, (mask > 0).astype(np.uint8))
    for k, v in full_feats.items():
        record[f"full_{k}"] = float(v)

    areas = {pid: float(part_feats[pid]["area"]) for pid in part_ids}
    total_area = sum(areas.values())
    for i in part_ids:
        for j in part_ids:
            if i != j:
                record[f"area_ratio_part{i}_to_part{j}"] = _safe_ratio(areas[i], areas[j])
    for pid in part_ids:
        record[f"area_ratio_part{pid}_to_total"] = _safe_ratio(areas[pid], total_area)
    record.update(iqa_prefix_record)
    return record


def fit_descriptor_standardizer(
    desc_df: Any,
    feature_cols: Sequence[str],
    train_images: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature ``(mean, std)`` for z-scoring descriptor vectors, fit on the
    training fold only.

    Descriptor columns span five orders of magnitude by construction: a raw
    ``*_area`` is ~1e4-1e5 px while ``solidity``/``circularity``/``eccentricity``
    are O(1). Feeding that raw into the fusion head made the head's single
    ``nn.LayerNorm`` over ``[z_img, z_p]`` derive its mean/variance almost
    entirely from the area columns, driving the backbone half of the vector to a
    near-constant (measured: image-block variance 1.45e-05x the descriptor
    block's) -- i.e. the visual pathway was discarded, and ``concat`` collapsed
    onto ``descriptors_only`` seed for seed. The 2026-08-11 ``_safe_ratio``
    clamp bounded the pathological ratios but could not fix this: the *raw*
    areas alone are four orders of magnitude above the pooled backbone
    features.

    Z-scoring per feature fixes both halves of that problem -- descriptor
    columns become mutually comparable, and the descriptor block as a whole
    becomes commensurate with the O(1) backbone features, so LayerNorm's
    statistics reflect both inputs.

    Statistics come from the training fold only (``train_images``); val/test
    reuse them, so no test-fold information reaches the model. Constant columns
    (std ~ 0, e.g. a part absent throughout training) get std 1.0 and so stay
    at 0.0 after centering rather than exploding.
    """
    cols = list(feature_cols)
    frame = desc_df
    if train_images is not None:
        wanted = set(str(n) for n in train_images)
        frame = desc_df[desc_df["image"].astype(str).isin(wanted)]
        if len(frame) == 0:
            raise ValueError(
                "Descriptor standardizer got no training rows: the descriptor CSV "
                "does not overlap the training split."
            )
    values = frame[cols].to_numpy(dtype=np.float64, copy=False)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_descriptor_vector(
    z: np.ndarray,
    stats: tuple[np.ndarray, np.ndarray] | None,
    group_mask: np.ndarray | None = None,
    clip: float = 10.0,
) -> np.ndarray:
    """Apply :func:`fit_descriptor_standardizer` output to one descriptor row,
    then zero the columns outside the active ablation group.

    NaNs are resolved *after* centering so a missing feature lands on the
    training mean (0.0 standardized) rather than on a raw 0.0, which for a
    column like ``full_area`` would be an extreme low outlier rather than a
    neutral value.

    ``clip`` bounds the z-scores at +/-10 sigma. Centering alone is not quite
    enough: a low-variance column with one far outlier can still reach z ~ 800,
    and a single entry that large re-creates the original failure for that one
    sample (measured: the fused vector's image block loses 26x of its spread on
    the worst row of a 512-row sample, versus none at all for the median row).
    +/-10 sits above the 95th percentile of per-row maxima, so it touches ~0.03%
    of entries while removing the tail entirely.
    """
    out = np.asarray(z, dtype=np.float32)
    if stats is not None:
        mean, std = stats
        out = (out - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if stats is not None and clip:
        out = np.clip(out, -clip, clip)
    if group_mask is not None:
        out = out * group_mask.astype(np.float32)
    return out


DESCRIPTOR_GROUPS = ("shape", "appearance", "interpart", "all")


def descriptor_group_mask(feature_names: list[str], group: str) -> np.ndarray:
    """
    Boolean mask over descriptor columns for ablation groups:
      shape | appearance | interpart | all

    The first three partition the vector by *descriptor family* and are
    mutually exclusive; ``all`` keeps everything.

    A fifth group, ``fullbody``, used to live here: it cut by *region* rather
    than by family, keeping every descriptor family but only for the
    whole-object mask, as a "does the part decomposition earn its keep?"
    control. It was never swept -- `ablate` has always defaulted to the four
    groups above -- so it produced no rows in any table, and it was removed
    along with the dead `descriptors.groups` config key that implied it ran.
    Restore it from git history if that control is wanted; it also needs
    adding to `ablate`'s group list, which is why it never ran.
    """
    group = group.lower()
    if group == "all":
        return np.ones(len(feature_names), dtype=bool)

    mask = np.zeros(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        base = name.split("_", 1)[-1] if "_" in name else name
        if group == "interpart":
            mask[i] = _is_interpart_key(name)
        elif group == "appearance":
            mask[i] = any(name.endswith(k) or base == k for k in APPEARANCE_KEYS)
        elif group == "shape":
            is_app = any(name.endswith(k) or base == k for k in APPEARANCE_KEYS)
            is_inter = _is_interpart_key(name)
            mask[i] = (not is_app) and (not is_inter)
        else:
            raise ValueError(
                f"Unknown descriptor group: {group!r} (expected one of {DESCRIPTOR_GROUPS})"
            )
    return mask
