import math
import sys
import time
import datetime
from typing import Iterable, Optional, Dict

import torch
import torch.nn.functional as F

import utils.misc as misc
import utils.lr_sched as lr_sched

from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    iters_per_epoch: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
) -> Dict[str, float]:
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = getattr(args, "print_freq", 50)

    accum_iter = getattr(args, "accum_iter", 1)
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print(f"TensorBoard log dir: {log_writer.log_dir}")

    time_consume = 0.0
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start = time.time()

        if step % accum_iter == 0:
            lr_sched.lr_wd_annealing(
                optimizer,
                args.lr,
                args.weight_decay,
                args.weight_decay_end,
                step + epoch * iters_per_epoch,
                args.warmup_epochs * iters_per_epoch,
                args.epochs * iters_per_epoch,
            )

        if isinstance(batch, list):
            images = torch.cat([b["image"] for b in batch], dim=0).to(device, non_blocking=True)
            targets = torch.cat([b["label"] for b in batch], dim=0).to(device, non_blocking=True)
        else:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)

        if targets.ndim == 5 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        with torch.cuda.amp.autocast(enabled=getattr(args, "amp", True)):
            logits = model(images)
            loss = criterion(logits, targets)

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"Non-finite loss {loss_value}. Stopping.")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=getattr(args, "clip_grad", None),
            parameters=model.parameters(),
            update_grad=((step + 1) % accum_iter == 0),
        )

        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        time_consume += time.time() - start

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (step + 1) % accum_iter == 0:
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch_1000x)

    total_time_str = str(datetime.timedelta(seconds=int(time_consume)))
    print(f"Training time (true compute): {total_time_str}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    criterion,
    num_classes: int,
    roi_size=(96, 96, 96),
    sw_batch_size=8,
    overlap=0.25,
) -> Dict[str, float]:
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Validation:"
    model.eval()

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            logits = inferer(images, model)
            loss = criterion(logits, labels)

        # [B, C, D, H, W]
        pred = torch.argmax(logits, dim=1)  # [B, D, H, W]
        pred_oh = F.one_hot(pred, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        gt = labels.squeeze(1).long()
        gt_oh = F.one_hot(gt, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        dice_t = dice_metric(y_pred=pred_oh, y=gt_oh)
        dice = dice_t.mean().item()
        dice_metric.reset()

        metric_logger.update(loss=float(loss.item()))
        metric_logger.update(dice=dice)

    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}
