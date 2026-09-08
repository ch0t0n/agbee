"""Classification + segmentation metrics and shared classification train/eval loops."""

from __future__ import annotations

import gc
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from tqdm import tqdm


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: Iterable[int] | None = None,
) -> dict:
    labels = list(labels) if labels is not None else None
    out = {
        "top1": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "n": int(len(y_true)),
    }
    if y_proba is not None:
        try:
            out["top3"] = float(top_k_accuracy_score(y_true, y_proba, k=3, labels=labels))
        except ValueError:
            out["top3"] = float("nan")
    return out


def quantile_long_tail_edges(
    train_counts: dict[int, int], n_bins: int = 4
) -> list[int]:
    """Frequency-bin edges at equal *class* quantiles of the training counts.

    Fixed absolute edges like (5, 30, 90) are calibrated for one corpus size
    and silently degenerate on another: on CUB's 1,439-image part-set training
    fold every class landed in the single `5-29` bin, so the long-tail table
    reported three bins of `n=0`/NaN and one bin identical to overall accuracy.
    A "long-tail analysis" that cannot separate head from tail is worse than
    none, because it looks like it ran.

    Quantile edges adapt to the corpus while keeping the bins interpretable
    (they are still image counts, just chosen from the data). Duplicate edges
    are collapsed, so a corpus that genuinely has few distinct frequencies
    yields fewer bins rather than empty ones.

    Edges are placed just *above* each quantile, so a bin is `m_c <= q` rather
    than `m_c < q` and a group of classes tied at the quantile stays whole.
    That is not cosmetic. Counts are heavily tied on a corpus whose classes are
    near-uniform, and a strict edge then drops the entire tie group into the bin
    above: CUB has 64 of its 200 classes at exactly 23 training images, which is
    its 25th percentile, so `m_c < 23` left a bottom bin of one class and three
    test images -- a "bottom quartile" that no longer measured anything. The
    alternative, cutting by rank and taking the lowest ceil(C/4) classes, would
    have to split that tie group on an arbitrary tie-break: 49 of CUB's 64 tied
    classes, and 342 of Fish-Vista's 561 classes tied at two training images,
    would land in the rare bin and the rest outside it for no reason a reader
    could defend. Keeping tie groups whole costs only that the bottom stratum is
    a quarter of the classes approximately rather than exactly (25.9% on
    BeeMachine, 32.5% on CUB, 37.5% on Fish-Vista) and is deterministic.
    """
    counts = sorted(train_counts.values())
    if not counts:
        return [5, 30, 90]
    qs = np.quantile(counts, [i / n_bins for i in range(1, n_bins)])
    edges: list[int] = []
    for q in qs:
        e = int(round(float(q))) + 1
        # An edge above the largest class opens a bin no class can enter. That
        # cannot happen with strict edges (an edge is always some quantile, so
        # some class sits at or above it) but can once edges sit one above the
        # quantile, and CUB's upper two quartiles both land on its maximum.
        if e > counts[-1]:
            continue
        if not edges or e > edges[-1]:
            edges.append(e)
    return edges or [max(1, counts[len(counts) // 2])]


def long_tail_bin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_counts: dict[int, int],
    bins: list[int] | tuple[int, ...] | str = (5, 30, 90),
) -> dict[str, dict]:
    """`bins` may be explicit integer edges, or the string ``"quantile"`` to
    derive them from `train_counts` via `quantile_long_tail_edges`."""
    if isinstance(bins, str):
        if bins != "quantile":
            raise ValueError(f"bins must be edges or 'quantile', got {bins!r}")
        bins = quantile_long_tail_edges(train_counts)
    edges = [0] + list(bins) + [10**9]
    names = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        names.append(f">={lo}" if hi >= 10**9 else f"{lo}-{hi - 1}")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, yt in enumerate(y_true):
        c = train_counts.get(int(yt), 0)
        for name, lo, hi in zip(names, edges[:-1], edges[1:]):
            if lo <= c < hi:
                groups[name].append(i)
                break

    results = {}
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        idxs = groups.get(name, [])
        if not idxs:
            results[name] = {"top1": float("nan"), "macro_f1": float("nan"), "n": 0}
            continue
        yt, yp = y_true[idxs], y_pred[idxs]
        bin_labels = sorted(k for k, count in train_counts.items() if lo <= count < hi)
        results[name] = {
            "top1": float(accuracy_score(yt, yp)),
            "macro_f1": float(
                f1_score(
                    yt,
                    yp,
                    labels=bin_labels or None,
                    average="macro",
                    zero_division=0,
                )
            ),
            "n": int(len(idxs)),
        }
    return results


def summarize_iou(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    tn: torch.Tensor,
    part_labels: Iterable[str],
) -> dict:
    labels = list(part_labels)
    dataset_iou = float(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"))
    per_class_t = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").nanmean(dim=0)
    miou = float(per_class_t.nanmean())
    per_class = per_class_t.tolist()
    return {
        "dataset_iou": dataset_iou,
        "miou": miou,
        "per_part_iou": {lab: float(v) for lab, v in zip(labels, per_class)},
    }


@torch.no_grad()
def evaluate_seg_loader(predict_fn, loader, device, num_classes: int, part_labels) -> dict:
    all_tp, all_fp, all_fn, all_tn = [], [], [], []
    for batch in loader:
        images, masks = batch[0].to(device), batch[1].to(device)
        pred = predict_fn(images).argmax(1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred, masks, mode="multiclass", num_classes=num_classes
        )
        all_tp.append(tp.cpu())
        all_fp.append(fp.cpu())
        all_fn.append(fn.cpu())
        all_tn.append(tn.cpu())
    return summarize_iou(
        torch.cat(all_tp), torch.cat(all_fp), torch.cat(all_fn), torch.cat(all_tn), part_labels
    )


class AugPlan:
    """One batch's sampled augmentation: who mixes with whom, and how much.

    Held as an object rather than recomputed per tensor because every input a
    batch carries -- the photograph, the K+1 part crops, the descriptor vector,
    the segmentation mask -- has to be mixed against the SAME partner with the
    SAME weight. Sampling `lam` twice for the image and the descriptors would
    train an arm on one specimen's pixels described by another specimen's
    measurements, which is not augmentation but label noise.
    """

    __slots__ = ("perm", "lam", "box")

    def __init__(self, perm, lam: float, box):
        self.perm = perm
        self.lam = float(lam)
        self.box = box  # (y1, y2, x1, x2) for CutMix, else None

    @property
    def is_cutmix(self) -> bool:
        return self.box is not None


class HeavyAug:
    """Mixup / CutMix / RandomErasing for every arm under the `heavy_aug` protocol.

    The single-arm heavy-augmentation control (`stage_b.cmd_heavy_aug`) keeps
    its own inlined copy of this recipe and is deliberately NOT refactored onto
    this class: its results are already published, and routing it through a
    different sequence of RNG draws would change them for no scientific reason.
    This class exists to extend the same recipe to the other six arms, whose
    inputs the original could not handle.

    What each input type gets, and why:

    * **Images** -- RandomErasing, then Mixup or CutMix. A ``(B, K+1, C, H, W)``
      crop stack is erased per crop and mixed along the batch axis, so all of
      sample *i*'s crops mix with all of sample ``perm[i]``'s and the part
      correspondence between streams survives.
    * **Descriptor vectors** -- convex-mixed at the same ``lam``. They are
      continuous measurements, so this is exact.
    * **Hard masks** (the gated arm's shape channel) -- CutMix pastes the same
      box, which is exact. Mixup has no exact analogue, because a 0.6/0.4 blend
      of two binary masks is not a mask; the dominant partner's mask is kept so
      the shape encoder always sees a real binary mask rather than a blur.
    * **Segmentation targets** (multi-task) -- never blended. The auxiliary
      Dice loss takes hard class indices, so the *loss* is mixed instead:
      ``lam * L(pred, m_a) + (1 - lam) * L(pred, m_b)``. That is exact for
      Mixup, and it is why the multi-task arm can run the full recipe rather
      than being restricted to CutMix.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.2,
        erase_p: float = 0.5,
        erase_scale=(0.02, 0.2),
        label_smoothing: float = 0.0,
    ):
        from torchvision import transforms as T

        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)
        self._erase = T.RandomErasing(p=erase_p, scale=tuple(erase_scale))

    def plan(self, batch_size: int, device, spatial=None) -> AugPlan:
        """Sample this batch's partner permutation, weight, and CutMix box."""
        lam = float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0 else 1.0
        perm = torch.randperm(batch_size, device=device)
        box = None
        if spatial is not None and np.random.rand() >= 0.5:
            h, w = spatial
            cut_w, cut_h = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
            cx, cy = np.random.randint(w), np.random.randint(h)
            x1, x2 = int(np.clip(cx - cut_w // 2, 0, w)), int(np.clip(cx + cut_w // 2, 0, w))
            y1, y2 = int(np.clip(cy - cut_h // 2, 0, h)), int(np.clip(cy + cut_h // 2, 0, h))
            box = (y1, y2, x1, x2)
            # The pasted area, not the sampled beta, is the true mixing weight.
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(w * h))
        return AugPlan(perm, lam, box)

    def plan_for(self, imgs: torch.Tensor) -> AugPlan:
        """Convenience: sample a plan sized and shaped for this image batch."""
        return self.plan(imgs.size(0), imgs.device, spatial=tuple(imgs.shape[-2:]))

    def erase(self, imgs: torch.Tensor) -> torch.Tensor:
        """RandomErasing, transparently handling a (B, S, C, H, W) crop stack."""
        if imgs.dim() == 5:
            b, s = imgs.shape[:2]
            flat = self._erase(imgs.reshape(b * s, *imgs.shape[2:]))
            return flat.reshape(b, s, *imgs.shape[2:])
        return self._erase(imgs)

    def mix_images(self, imgs: torch.Tensor, plan: AugPlan) -> torch.Tensor:
        """Mixup or CutMix along the batch axis, for 4D or 5D image tensors."""
        partner = imgs[plan.perm]
        if plan.is_cutmix:
            y1, y2, x1, x2 = plan.box
            out = imgs.clone()
            out[..., y1:y2, x1:x2] = partner[..., y1:y2, x1:x2]
            return out
        return plan.lam * imgs + (1.0 - plan.lam) * partner

    def transform_images(self, imgs: torch.Tensor, plan: AugPlan) -> torch.Tensor:
        return self.mix_images(self.erase(imgs), plan)

    def mix_vector(self, vec: torch.Tensor, plan: AugPlan) -> torch.Tensor:
        """Convex mix for continuous side-inputs (descriptor vectors)."""
        return plan.lam * vec + (1.0 - plan.lam) * vec[plan.perm]

    def mix_hard(self, hard: torch.Tensor, plan: AugPlan) -> torch.Tensor:
        """Mask-valued inputs: paste for CutMix, keep the dominant one for Mixup."""
        if plan.is_cutmix:
            y1, y2, x1, x2 = plan.box
            out = hard.clone()
            out[..., y1:y2, x1:x2] = hard[plan.perm][..., y1:y2, x1:x2]
            return out
        return hard if plan.lam >= 0.5 else hard[plan.perm]

    def soft_targets(self, y: torch.Tensor, plan: AugPlan) -> torch.Tensor:
        """One-hot targets mixed at `lam`, with label smoothing folded in.

        Smoothing is applied here rather than by the criterion because
        nn.CrossEntropyLoss only smooths hard targets; leaving it off would
        make this protocol's loss differ from every other arm's by more than
        the augmentation it is supposed to isolate.
        """
        onehot = torch.zeros(y.size(0), self.num_classes, device=y.device)
        onehot.scatter_(1, y.unsqueeze(1), 1.0)
        soft = plan.lam * onehot + (1.0 - plan.lam) * onehot[plan.perm]
        eps = self.label_smoothing
        if eps > 0.0:
            soft = soft * (1.0 - eps) + eps / self.num_classes
        return soft

    @staticmethod
    def soft_ce(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against a soft target distribution."""
        return -(torch.log_softmax(logits, 1) * soft).sum(1).mean()


def run_cls_epoch(
    model, loader, criterion, device, optimizer=None, batch_fn=None, desc=None,
    heavy_aug: "HeavyAug | None" = None,
):
    """
    Shared classification epoch.
    batch_fn(batch) -> (inputs_to_model, y)  if model needs nonstandard batch unpacking.

    `heavy_aug`, when given, applies the Mixup/CutMix/RandomErasing recipe to
    TRAINING batches only (never validation, so the early-stopping signal and
    the saved checkpoint are still chosen on clean data). Reported training
    accuracy is then measured against the pre-mix labels and is approximate by
    construction -- nothing reads it for a decision, and validation loss, which
    does, is untouched.
    Default: batch is (imgs, ..., y) with y last (or imgs, y).

    `desc` (e.g. "epoch 12/30 train") is put on the tqdm bar so a log tail
    mid-epoch shows which epoch is running, not just how far into it -- a bare
    percentage is ambiguous between epoch 1/30 and epoch 29/30.
    """
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    show = tqdm(loader, desc=desc, leave=False) if _is_rank0() else loader
    # `model.train(False)` switches BatchNorm/Dropout but does NOT stop autograd
    # from recording. Every validation pass -- one per epoch, per job, for all
    # ~145 Stage B jobs and both Stage C runs -- was building and discarding a
    # full backward graph, at training-level peak memory and roughly a third
    # more time than the forward alone needs.
    for batch in show:
        if batch_fn is not None:
            x, y = batch_fn(batch)
        else:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[-1]
        if isinstance(x, tuple):
            x = tuple(item.to(device) for item in x)
        else:
            x = x.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad()
        soft = None
        if train and heavy_aug is not None:
            # `x` is a plain image tensor for every arm that reaches this loop
            # (baseline, capacity-matched, body-masked, part-crop); the tuple
            # form belongs to the fusion arms, which run their own epoch
            # functions in stage_b and apply the augmenter there.
            if isinstance(x, tuple):
                raise TypeError(
                    "run_cls_epoch: heavy_aug does not handle tuple inputs; the "
                    "fusion arms apply it in their own epoch runners."
                )
            plan = heavy_aug.plan_for(x)
            x = heavy_aug.transform_images(x, plan)
            soft = heavy_aug.soft_targets(y, plan)
        with torch.set_grad_enabled(train):
            logits = model(x) if not isinstance(x, tuple) else model(*x)
            loss = criterion(logits, y) if soft is None else heavy_aug.soft_ce(logits, soft)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)

    totals = torch.tensor([total_loss, correct, float(n)], device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    total_loss, correct, n = totals.tolist()
    return total_loss / max(n, 1), correct / max(n, 1)


class EarlyStopper:
    """Stop when validation loss has not improved for `patience` epochs.

    One implementation for all three hand-rolled classification loops
    (`fit_cls_model`, Stage B's heavy_aug loop, Stage B's fusion loop), which
    were byte-for-byte identical in structure and each ran its full epoch
    budget unconditionally. Stage A's segmenters already stop early through
    Lightning's `EarlyStopping`; this is the same rule for everything else.

    Monitors validation **loss**, the same signal these loops already use for
    `ReduceLROnPlateau` and for best-checkpoint selection, so the epoch training
    stops on and the epoch that gets kept are judged by one metric. Validation
    accuracy is the noisier choice on the small validation folds here (CUB's is
    258 images, where one image is 0.4 %).

    DDP-safe without any extra collective: `run_cls_epoch` all-reduces its loss
    and count across ranks before returning, so every rank feeds this the same
    number and reaches the same decision on the same epoch. A rank-local metric
    would desync the ranks and hang at the next barrier.

    `min_delta` is 0.0, matching Lightning's default: any improvement, however
    small, resets the counter.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad_epochs = 0
        self.best_epoch = 0
        self.stopped_epoch: int | None = None

    @property
    def enabled(self) -> bool:
        """`patience <= 0` disables the rule and runs the full budget."""
        return self.patience > 0

    def step(self, value: float, epoch: int) -> bool:
        """Record one epoch's validation loss; return True to stop training."""
        if value < self.best - self.min_delta:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.enabled and self.bad_epochs >= self.patience:
            self.stopped_epoch = int(epoch)
            return True
        return False

    def summary(self, total_epochs: int) -> str:
        if self.stopped_epoch is None:
            return f"ran the full {total_epochs}-epoch budget (no plateau)"
        return (
            f"early-stopped at epoch {self.stopped_epoch + 1}/{total_epochs} "
            f"(best epoch {self.best_epoch + 1}, val_loss={self.best:.4f}, "
            f"patience={self.patience})"
        )


def _is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


@torch.no_grad()
def evaluate_cls_loader(model, loader, device, train_counts, bins, batch_fn=None) -> dict:
    """Test-fold evaluation. `no_grad` is not an optimization here, it is load-bearing.

    Without it this built a full autograd graph for every test batch while the
    trained model still held its gradients and AdamW its two moment buffers.
    Peak memory during *evaluation* therefore exceeded peak memory during
    training, and three of the nine configured backbones
    (efficientnetv2_rw_m, seresnext101_32x4d, swin_base) died with CUDA OOM on
    a 46 GB A40 at Beemachine's 320 px -- after their training epochs had
    already completed, so the whole run was lost at the last step.
    """
    from distributed_utils import unwrap_model

    model.eval()
    raw = unwrap_model(model)
    ys, preds, probas = [], [], []
    for batch in loader:
        if batch_fn is not None:
            x, y = batch_fn(batch)
        else:
            x, y = (batch[0], batch[-1]) if len(batch) > 2 else batch
        if isinstance(x, tuple):
            x = tuple(item.to(device) for item in x)
        else:
            x = x.to(device)
        logits = raw(x) if not isinstance(x, tuple) else raw(*x)
        ys.append(y.numpy() if hasattr(y, "numpy") else np.asarray(y))
        preds.append(logits.argmax(1).cpu().numpy())
        probas.append(torch.softmax(logits, 1).cpu().numpy())
    y_true, y_pred, y_proba = map(np.concatenate, (ys, preds, probas))
    report = classification_report_dict(
        y_true,
        y_pred,
        y_proba,
        labels=range(y_proba.shape[1]),
    )
    report["long_tail"] = long_tail_bin_metrics(y_true, y_pred, train_counts, bins)
    return report


def bootstrap_ci(
    values: np.ndarray | list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Nonparametric bootstrap mean + (1-alpha) CI over per-run values.

    Used to summarize a fusion arm's metric across seeds (n_boot resamples
    of the seed list) rather than reporting a single-seed point estimate.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "lo": float(lo),
        "hi": float(hi),
        "n": int(len(values)),
    }


def paired_bootstrap_test(
    values_a: np.ndarray | list[float],
    values_b: np.ndarray | list[float],
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Two-sided paired bootstrap test for 'arm A beats arm B' on matched seeds.

    Intended for comparing two fusion arms run under the same seed list
    (same seed => same split/init/order), so pairing is by seed index. If
    the arms were evaluated on the same held-out predictions instead, the
    same routine can be pointed at per-example correctness vectors.
    Returns the observed mean difference, its CI, and a two-sided p-value
    (fraction of bootstrap resamples on the wrong side of zero, doubled).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_side = min((boots <= 0).mean(), (boots >= 0).mean())
    p_value = float(min(1.0, 2 * p_side))
    return {
        "mean_diff": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05 and not (lo < 0 < hi)),
        "n_pairs": int(n),
    }


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """Expected Calibration Error with equal-width confidence bins.

    Shared implementation used by both Stage B/C evaluation and Stage D's
    `calibration` command, so every arm and every dataset compute ECE the
    same way.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    rows = []
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if not np.any(m):
            continue
        acc = correct[m].mean()
        conf = confidences[m].mean()
        ece_val += m.mean() * abs(acc - conf)
        rows.append({"bin": i, "acc": float(acc), "conf": float(conf), "n": int(m.sum())})
    return float(ece_val), rows


def selective_prediction_curve(
    confidences: np.ndarray, correct: np.ndarray, coverages: Iterable[float] | None = None
) -> list[dict]:
    """Risk-coverage curve for abstention: sort by confidence, keep top-c%.

    At each coverage level, accepted predictions are the c% with highest
    confidence; risk is the error rate among those accepted. Used to compare
    fusion arms on "how reliably can this arm abstain on what it doesn't
    know" rather than only on unconditional accuracy. AURC (area under the
    risk-coverage curve) summarizes the whole curve in one number — lower
    is better.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-confidences)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    if coverages is None:
        coverages = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    rows = []
    cum_correct = np.cumsum(correct_sorted)
    for c in coverages:
        k = max(1, int(round(c * n)))
        acc_at_k = cum_correct[k - 1] / k
        rows.append({"coverage": float(c), "risk": float(1 - acc_at_k), "n": int(k)})
    return rows


def aurc(rows: list[dict]) -> float:
    """Area under the risk-coverage curve produced by selective_prediction_curve."""
    rows = sorted(rows, key=lambda r: r["coverage"])
    cov = np.asarray([r["coverage"] for r in rows])
    risk = np.asarray([r["risk"] for r in rows])
    return float(np.trapz(risk, cov) / max(cov.max() - cov.min(), 1e-9))


def reliability_by_frequency_bin(
    confidences: np.ndarray,
    correct: np.ndarray,
    y_true: np.ndarray,
    train_counts: dict[int, int],
    bins: list[int] | tuple[int, ...] | str = (5, 30, 90),
    n_bins: int = 15,
) -> dict[str, dict]:
    """ECE and AURC stratified by training-frequency bin.

    Overall ECE can look acceptable while a model is badly miscalibrated on
    exactly the rare classes a reviewer most needs to defer on: the head of
    the distribution dominates the average. Reporting reliability per
    frequency bin exposes that, and is the form in which calibration is
    actionable for triage (a program sets its abstention policy per stratum,
    not globally).

    `bins` may be explicit integer edges, or the string ``"quantile"`` to
    derive them from `train_counts` via `quantile_long_tail_edges`.
    """
    if isinstance(bins, str):
        if bins != "quantile":
            raise ValueError(f"bins must be edges or 'quantile', got {bins!r}")
        bins = quantile_long_tail_edges(train_counts)
    edges = [0] + list(bins) + [10**9]
    names = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        names.append(f">={lo}" if hi >= 10**9 else f"{lo}-{hi - 1}")

    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    y_true = np.asarray(y_true)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, yt in enumerate(y_true):
        c = train_counts.get(int(yt), 0)
        for name, lo, hi in zip(names, edges[:-1], edges[1:]):
            if lo <= c < hi:
                groups[name].append(i)
                break

    out = {}
    for name in names:
        idx = groups.get(name, [])
        if not idx:
            out[name] = {"ece": float("nan"), "aurc": float("nan"), "n": 0}
            continue
        conf_b, corr_b = confidences[idx], correct[idx]
        ece_val, _ = expected_calibration_error(conf_b, corr_b, n_bins=n_bins)
        rc = selective_prediction_curve(conf_b, corr_b)
        out[name] = {
            "ece": ece_val,
            "aurc": aurc(rc),
            "accuracy": float(corr_b.mean()),
            "n": int(len(idx)),
        }
    return out


def corrupt_batch(imgs: torch.Tensor, kind: str, severity: int = 3) -> torch.Tensor:
    """Apply one synthetic corruption to a batch for OOD / robustness eval.

    Lightweight, dependency-free stand-ins for common ImageNet-C-style
    corruptions, used to check whether a fusion arm's ranking (and its
    reliance on part evidence) holds under distribution shift the training
    protocol never sees. `severity` in {1..5} scales the perturbation.
    """
    s = max(1, min(5, severity))
    x = imgs.clone()
    if kind == "gaussian_noise":
        sigma = [0.02, 0.04, 0.06, 0.09, 0.12][s - 1]
        x = (x + torch.randn_like(x) * sigma).clamp(0, 1)
    elif kind == "gaussian_blur":
        k = [3, 5, 7, 9, 11][s - 1]
        x = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    elif kind == "brightness":
        delta = [0.1, 0.2, 0.3, 0.4, 0.5][s - 1]
        x = (x + delta).clamp(0, 1)
    elif kind == "contrast":
        factor = [0.9, 0.75, 0.6, 0.45, 0.3][s - 1]
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = ((x - mean) * factor + mean).clamp(0, 1)
    elif kind == "jpeg_block":
        # Cheap blockiness proxy: downsample then upsample by a block factor.
        factor = [2, 3, 4, 6, 8][s - 1]
        h, w = x.shape[-2:]
        x = torch.nn.functional.interpolate(x, size=(h // factor, w // factor), mode="bilinear")
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="nearest")
    elif kind == "occlusion":
        frac = [0.05, 0.1, 0.15, 0.25, 0.35][s - 1]
        h, w = x.shape[-2:]
        ch, cw = int(h * frac**0.5), int(w * frac**0.5)
        y0 = np.random.randint(0, max(1, h - ch))
        x0 = np.random.randint(0, max(1, w - cw))
        x[:, :, y0 : y0 + ch, x0 : x0 + cw] = 0.0
    else:
        raise ValueError(f"Unknown corruption kind: {kind}")
    return x


CORRUPTION_KINDS = (
    "gaussian_noise",
    "gaussian_blur",
    "brightness",
    "contrast",
    "jpeg_block",
    "occlusion",
)


def make_cls_criterion(cfg) -> nn.Module:
    """The classification loss shared by every Stage B / Stage C arm.

    Exists so `classification.label_smoothing` reaches all five training paths
    from one place: `fit_cls_model` (baseline / capacity-matched / masked /
    partcrop / Stage C), the hand-rolled fusion loop, the heavy-augmentation
    loop, and the multitask Lightning module. Those grew separate
    `nn.CrossEntropyLoss()` constructions, so a loss-level protocol change had
    four independent places to be forgotten -- and was, for label smoothing,
    which the reference protocol uses at 0.1 and this repo used at 0.0.
    """
    from config import label_smoothing

    return nn.CrossEntropyLoss(label_smoothing=label_smoothing(cfg))


def fit_cls_model(
    model,
    train_loader,
    val_loader,
    device,
    lr: float,
    epochs: int,
    out_ckpt,
    batch_fn=None,
    criterion=None,
    train_sampler=None,
    patience: int = 0,
    history: dict | None = None,
    rebuild_loaders=None,
    batch_size: int | None = None,
    min_batch: int = 1,
    heavy_aug: "HeavyAug | None" = None,
):
    """Train with AdamW + ReduceLROnPlateau; save best-by-val-loss checkpoint (rank 0).

    `patience` > 0 stops once validation loss has not improved for that many
    epochs (see `EarlyStopper`); 0 runs the full `epochs` budget. Callers pass
    `classification.patience`.

    `history`, when given, is filled in place with what actually ran
    (`epochs_ran`, `early_stopped`, `best_epoch`, `best_val_loss`,
    `final_batch_size`) so the caller can record it in `run_meta.json` -- with
    early stopping enabled, the configured epoch count is a ceiling and no
    longer describes the run.

    `rebuild_loaders`, when given, is `callable(batch_size) -> (train_loader,
    val_loader)`. On a real CUDA OOM the up-front batch-size estimate (a flat
    per-sample memory multiplier, blind to this specific model/data/GPU
    state) is halved and the loaders rebuilt, then the same epoch is retried
    -- self-correcting on the real GPU instead of crashing the whole job.
    Without it (the DDP scaling caller, where ranks must not diverge on batch
    size), an OOM still raises as before.
    """
    from pathlib import Path

    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from distributed_utils import is_main_process, unwrap_model

    criterion = criterion or nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    stopper = EarlyStopper(patience)
    best = float("inf")
    epochs_ran = 0          # `epochs` may be 0; keep this defined for `history`
    out_ckpt = Path(out_ckpt)
    bs = batch_size
    epoch = 0
    while epoch < epochs:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        oom_hit = False
        try:
            tr_loss, tr_acc = run_cls_epoch(
                model, train_loader, criterion, device, opt, batch_fn,
                desc=f"epoch {epoch + 1}/{epochs} train",
                heavy_aug=heavy_aug,
            )
            va_loss, va_acc = run_cls_epoch(
                model, val_loader, criterion, device, None, batch_fn,
                desc=f"epoch {epoch + 1}/{epochs} val",
            )
        except torch.cuda.OutOfMemoryError:
            oom_hit = True
        if oom_hit:
            if rebuild_loaders is None or bs is None or bs <= min_batch:
                raise RuntimeError(
                    f"fit_cls_model: CUDA out of memory even at batch_size={min_batch}"
                )
            # `empty_cache()` only returns *unreferenced* blocks. Calling it
            # from inside the `except` clause is too early: Python keeps the
            # exception's traceback (and every frame/tensor it points to,
            # e.g. `logits` in `run_cls_epoch`) alive for as long as the
            # exception is "currently being handled", so nothing was actually
            # freed and every retry OOM'd again at a smaller and smaller free
            # amount -- confirmed by instrumenting a synthetic OOM: free
            # memory kept shrinking across retries instead of recovering
            # until this cleanup moved outside the `except` block (plus a
            # `gc.collect()`, since the autograd graph's node cycle also
            # needs a cyclic-GC pass, not just refcounting, to actually drop).
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            bs = max(min_batch, bs // 2)
            if is_main_process():
                print(f"[fit_cls_model] CUDA OOM; retrying epoch {epoch + 1} at batch_size={bs}", flush=True)
            train_loader, val_loader = rebuild_loaders(bs)
            continue
        sch.step(va_loss)
        if is_main_process():
            print(f"epoch {epoch + 1}: train={tr_acc:.4f} val={va_acc:.4f}")
            if va_loss < best:
                best = va_loss
                out_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(unwrap_model(model).state_dict(), out_ckpt)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        # `va_loss` is all-reduced inside run_cls_epoch, so every rank decides
        # identically here and they leave the loop on the same epoch.
        epochs_ran = epoch + 1
        if stopper.step(va_loss, epoch):
            if is_main_process():
                print(f"[early stop] {stopper.summary(epochs)}")
            break
        epoch += 1
    if history is not None:
        history.update(
            {
                "epochs_ran": epochs_ran,
                "early_stopped": stopper.stopped_epoch is not None,
                "best_epoch": stopper.best_epoch + 1,
                "best_val_loss": round(stopper.best, 6),
                "final_batch_size": bs,
            }
        )
    if out_ckpt.exists():
        unwrap_model(model).load_state_dict(
            torch.load(out_ckpt, map_location=device, weights_only=True)
        )
    return model
