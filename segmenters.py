"""SMP segmenters and Lightning train helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.optim import lr_scheduler


def run_name(backend: str, arch: str, encoder: str) -> str:
    enc = encoder.replace("/", "_").replace(":", "_")
    return f"{arch}_{enc}"


def resolve_model_spec(cfg: dict[str, Any], arch=None, encoder=None, backend=None):
    seg = cfg["segmentation"]
    return (
        backend or seg.get("backend", "smp"),
        arch or seg["arch"],
        encoder or seg["encoder"],
    )


def make_seg_trainer(
    out_dir: Path,
    epochs: int,
    patience: int,
    device: int = 0,
    monitor: str = "valid_miou",
    num_gpus: int | None = None,
    mode: str = "max",
    early_stop: bool = True,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """`mode="min"` / `monitor="valid_loss"` let a caller (the multitask
    comparison arm) select and stop on validation loss, matching the
    `metrics.EarlyStopper` rule the hand-rolled classification loops use,
    instead of Stage A's IoU-based criterion. `early_stop=False` disables the
    rule entirely and runs the full epoch budget.
    """
    out_dir = Path(out_dir)
    ckpt_cb = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_weights_only=True,
        filename=f"best-{{epoch:03d}}-{{{monitor}:.4f}}",
        dirpath=str(out_dir / "checkpoints"),
    )
    use_cuda = torch.cuda.is_available()
    if num_gpus is None:
        num_gpus = 1
    num_gpus = max(1, int(num_gpus))
    if use_cuda and num_gpus > 1:
        devices: int | list[int] = num_gpus
        strategy: str = "ddp_find_unused_parameters_true"
    elif use_cuda:
        devices = [int(device)]
        strategy = "auto"
    else:
        devices = 1
        strategy = "auto"
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="epoch")]
    # `patience: 0` means "run the full budget", matching how
    # `classification.patience` is documented. Passing 0 straight to
    # EarlyStopping would instead stop at the first epoch that fails to
    # improve, which is the opposite of what the config says.
    if early_stop and int(patience) > 0:
        callbacks.insert(1, EarlyStopping(monitor=monitor, mode=mode, patience=int(patience)))
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=devices,
        strategy=strategy,
        logger=CSVLogger(save_dir=str(out_dir), name="logs"),
        callbacks=callbacks,
        precision="16-mixed" if use_cuda else 32,
        sync_batchnorm=use_cuda and num_gpus > 1,
    )
    return trainer, ckpt_cb


class _SegLightningMixin:
    number_of_classes: int
    loss_fn: Any
    learning_rate: float
    t_max: int
    _train_stats: list
    _valid_stats: list

    def _shared_step(self, batch, stage: str):
        image, mask = batch[0], batch[1]
        logits = self.forward(image)
        loss = self.loss_fn(logits, mask.long())
        pred = logits.argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred, mask, mode="multiclass", num_classes=self.number_of_classes
        )
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss, torch.stack([tp, fp, fn, tn])

    def _epoch_end_iou(self, stats_list, stage: str):
        device = self.device
        if stats_list:
            stacked = torch.cat(stats_list, dim=1).to(device)
            totals = stacked.sum(dim=1)
            per_image = smp.metrics.iou_score(
                stacked[0], stacked[1], stacked[2], stacked[3], reduction="micro-imagewise"
            )
        else:
            totals = torch.zeros(4, self.number_of_classes, device=device)
            per_image = torch.tensor(0.0, device=device)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)

        tp, fp, fn, tn = totals
        dataset_iou = smp.metrics.iou_score(
            tp.unsqueeze(0),
            fp.unsqueeze(0),
            fn.unsqueeze(0),
            tn.unsqueeze(0),
            reduction="micro",
        )
        per_class_iou = smp.metrics.iou_score(
            tp.unsqueeze(0),
            fp.unsqueeze(0),
            fn.unsqueeze(0),
            tn.unsqueeze(0),
            reduction="none",
        )
        self.log_dict(
            {
                f"{stage}_per_image_iou": per_image,
                f"{stage}_dataset_iou": dataset_iou,
                f"{stage}_miou": per_class_iou.nanmean(),
            },
            prog_bar=True,
            sync_dist=False,
        )
        stats_list.clear()

    def training_step(self, batch, batch_idx):
        loss, stats = self._shared_step(batch, "train")
        self._train_stats.append(stats.detach())
        return loss

    def on_train_epoch_end(self):
        self._epoch_end_iou(self._train_stats, "train")

    def validation_step(self, batch, batch_idx):
        loss, stats = self._shared_step(batch, "valid")
        self._valid_stats.append(stats.detach())
        return loss

    def on_validation_epoch_end(self):
        self._epoch_end_iou(self._valid_stats, "valid")


class BeePartSegModel(_SegLightningMixin, pl.LightningModule):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int = 3,
        out_classes: int = 4,
        learning_rate: float = 1e-4,
        t_max: int = 200,
        backend: str = "smp",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.t_max = t_max
        self.number_of_classes = out_classes
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self._train_stats, self._valid_stats = [], []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model((image - self.mean) / self.std)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


def cosine_t_max(cfg: dict) -> int:
    """Cosine period for Stage A, in epochs.

    This is `segmentation.t_max`, NOT `segmentation.epochs`. Tying the two
    together is only correct when training actually runs the full budget, and
    Stage A stops on validation mIoU long before that: the 2026-08 sweep ended
    between epochs 22 and 94 against a 200-epoch ceiling, so a T_max=200 cosine
    was still at 0.65-0.9x the base learning rate when training stopped and the
    anneal never completed for any of the 27 runs. The reference protocol
    (Choton et al., 2026) uses T_max=50, which completes an anneal to eta_min
    every 50 epochs regardless of where the run ends.
    """
    seg = cfg["segmentation"]
    return int(seg.get("t_max") or seg["epochs"])


def build_segmenter(backend: str, arch: str, encoder: str, num_classes: int, cfg: dict):
    return BeePartSegModel(
        arch,
        encoder,
        out_classes=num_classes,
        learning_rate=cfg["segmentation"]["lr"],
        t_max=cosine_t_max(cfg),
    )


def load_segmenter(ckpt_path: str | Path, device: torch.device, backend: str | None = None):
    model = BeePartSegModel.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.to(device).eval()
    return model
