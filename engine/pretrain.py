import math
import sys
import time
import datetime
from typing import Iterable, Any

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched


def _extract_images_from_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, list):
        if len(batch) == 0:
            raise ValueError("Empty batch received.")
        if isinstance(batch[0], dict) and "image" in batch[0]:
            imgs = [x["image"] for x in batch]
            return torch.cat(imgs, dim=0)
        if torch.is_tensor(batch[0]):
            return torch.cat(batch, dim=0)

    if isinstance(batch, dict) and "image" in batch:
        return batch["image"]

    if torch.is_tensor(batch):
        return batch

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    iters_train: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Pretrain Epoch: [{epoch}]'
    print_freq = 50

    accum_iter = args.accum_iter
    optimizer.zero_grad(set_to_none=True)

    time_consume = 0.0

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start = time.time()

        if data_iter_step % accum_iter == 0:
            lr_sched.lr_wd_annealing(
                optimizer,
                args.lr,
                args.weight_decay,
                args.weight_decay_end,
                data_iter_step + epoch * iters_train,
                args.warmup_epochs * iters_train,
                args.epochs * iters_train
            )

        images = _extract_images_from_batch(batch).to(device, non_blocking=True)  # [B,C,D,H,W]

        with torch.cuda.amp.autocast():
            loss = model(images)

        loss_value = float(loss.item())

        if not math.isfinite(loss_value):
            print(f"[Error] Loss is {loss_value}, stopping training.")
            sys.exit(1)

        loss = loss / accum_iter

        loss_scaler(
            loss,
            optimizer,
            clip_grad=args.clip_grad,
            parameters=model.parameters(),
            update_grad=((data_iter_step + 1) % accum_iter == 0),
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        time_consume += time.time() - start

    total_time_str = str(datetime.timedelta(seconds=int(time_consume)))
    if misc.is_main_process():
        print(f'[Timing] True compute time: {total_time_str}')

    metric_logger.synchronize_between_processes()
    if misc.is_main_process():
        print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
