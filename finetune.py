import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.datasets import get_loader
from utils.loss import HybridSegLoss, DiceLoss3D

from models.network.gcsmim_model import build_gcsmim
from engine.finetune import train_one_epoch, evaluate


def get_args():
    parser = argparse.ArgumentParser(
        "GCSMIM fine-tuning for 3D medical image segmentation", add_help=True
    )

    # Training
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size per GPU (effective bs = batch_size * accum_iter * world_size)")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--no_amp", action="store_true", help="Disable AMP (default: enabled)")
    parser.add_argument("--print_freq", default=50, type=int)

    # Data
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--input_size", default=96, type=int)
    parser.add_argument("--num_classes", default=3, type=int,
                        help="Number of segmentation classes (including background)")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--val_split", default=0.2, type=float,
                        help="Train/val split for finetune datalist")

    # Checkpoint / logging
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Directory to save checkpoints/logs")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="TensorBoard log dir (default: output_dir/tb)")
    parser.add_argument("--resume", default="", type=str,
                        help="Resume from a fine-tune checkpoint (full state)")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Auto-resume from latest checkpoint in output_dir")
    parser.set_defaults(auto_resume=True)

    # Optional
    parser.add_argument("--pretrained_ckpt", default="", type=str,
                        help="Path to pretrain checkpoint; will load encoder weights (prefix: sparse_encoder.)")

    # Loss
    parser.add_argument("--loss_type", default="hybrid", type=str,
                        choices=["hybrid", "dice"])
    parser.add_argument("--clip_grad", default=5.0, type=float)

    # Optim
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--weight_decay_end", default=0.2, type=float)
    parser.add_argument("--warmup_epochs", default=40, type=int)

    # Sliding-window val
    parser.add_argument("--roi_size", default=(96, 96, 96), type=int, nargs=3,
                    help="roi_size for SlidingWindowInferer, e.g. --roi_size 96 96 96")
    parser.add_argument("--sw_batch_size", default=2, type=int)
    parser.add_argument("--overlap", default=0.25, type=float)

    # DDP
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)

    return parser.parse_args()


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_loaders(args):
    dataset_train, dataset_val = get_loader(
        data_dir=args.data_path,
        size=args.input_size,
        mode="finetune",
        split=args.val_split,
        seed=args.seed,
    )

    if args.distributed:
        sampler_train = DistributedSampler(
            dataset_train,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        sampler_val = DistributedSampler(
            dataset_val,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler_train = SequentialSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    return data_loader_train, data_loader_val


def _build_model(args, device):
    model = build_gcsmim(img_size=args.input_size, in_channel=1, n_classes=args.num_classes)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    return model, model_without_ddp


def _load_pretrained_encoder(args, model_without_ddp):
    if not args.pretrained_ckpt:
        return

    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)

    enc_state = {}
    for k, v in state.items():
        if k.startswith("sparse_encoder."):
            enc_state[k.replace("sparse_encoder.", "", 1)] = v

    if len(enc_state) == 0:
        print(f"[warn] No keys with prefix 'sparse_encoder.' found in {args.pretrained_ckpt}. Skip.")
        return

    msg = model_without_ddp.load_state_dict(enc_state, strict=False)
    print(f"[pretrained] Loaded encoder weights from: {args.pretrained_ckpt}")
    print(f"[pretrained] load_state_dict msg: {msg}")


def _build_optimizer(args, model_without_ddp):
    return torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )


def _build_criterion(args):
    if args.loss_type == "dice":
        return DiceLoss3D(num_classes=args.num_classes)
    if args.loss_type == "hybrid":
        return HybridSegLoss(num_classes=args.num_classes)
    return nn.CrossEntropyLoss()


def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    _seed_everything(seed)
    cudnn.benchmark = True

    if misc.is_main_process():
        print("Args:", args)

    log_writer = None
    if misc.is_main_process():
        tb_dir = args.log_dir if args.log_dir is not None else os.path.join(args.output_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=tb_dir)

    data_loader_train, data_loader_val = _build_loaders(args)
    iters_train = len(data_loader_train)

    model, model_without_ddp = _build_model(args, device)

    _load_pretrained_encoder(args, model_without_ddp)

    optimizer = _build_optimizer(args, model_without_ddp)
    criterion = _build_criterion(args)
    loss_scaler = NativeScaler()

    args.amp = (not args.no_amp)

    misc.auto_load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        mode="finetune",
    )

    best_dice = -1.0
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and isinstance(data_loader_train.sampler, DistributedSampler):
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            iters_per_epoch=iters_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        val_stats = evaluate(
            data_loader=data_loader_val,
            model=model,
            device=device,
            criterion=criterion,
            num_classes=args.num_classes,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
        )

        cur_dice = float(val_stats.get("dice", -1.0))
        is_best = cur_dice > best_dice
        best_dice = max(best_dice, cur_dice)

        if misc.is_main_process():
            if (epoch % 10 == 0) or (epoch + 1 == args.epochs):
                misc.save_model(
                    args=args,
                    epoch=epoch,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    best=False,
                    mode="finetune",
                )

            if is_best:
                misc.save_model(
                    args=args,
                    epoch=epoch,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    best=True,
                    mode="finetune",
                )

            log_stats = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "best_val_dice": best_dice,
            }
            with open(os.path.join(args.output_dir, "log_finetune.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if log_writer is not None:
                log_writer.add_scalar("val/dice", cur_dice, epoch)
                log_writer.add_scalar("val/loss", float(val_stats.get("loss", 0.0)), epoch)

            print(f"[epoch {epoch}] val_dice={cur_dice:.4f} best={best_dice:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print(f"Total training time: {total_time_str}")


if __name__ == "__main__":
    main()
