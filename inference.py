import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

import utils.misc as misc
from utils.datasets import get_loader
from utils.loss import HybridSegLoss, DiceLoss3D

from engine.finetune import evaluate
from models.network.gcsmim_model import build_gcsmim


def get_args():
    parser = argparse.ArgumentParser("GCSMIM testing for 3D medical image segmentation", add_help=True)

    # Data
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--input_size", default=96, type=int)
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=True)

    # Model / ckpt
    parser.add_argument("--ckpt", required=True, type=str, help="Path to finetune checkpoint (.pth)")
    parser.add_argument("--device", default="cuda", type=str)

    # Repro
    parser.add_argument("--seed", default=0, type=int)

    # Loss
    parser.add_argument("--loss_type", default="hybrid", type=str, choices=["hybrid", "dice"])

    # Sliding window
    parser.add_argument("--roi_size", default=(96, 96, 96), type=int, nargs=3,
                    help="roi_size for SlidingWindowInferer, e.g. --roi_size 96 96 96")
    parser.add_argument("--sw_batch_size", default=2, type=int)
    parser.add_argument("--overlap", default=0.25, type=float)

    # Save
    parser.add_argument("--output_dir", default="", type=str, help="Optional: save test log jsonl here")

    # DDP
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)

    return parser.parse_args()


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_criterion(args):
    if args.loss_type == "dice":
        return DiceLoss3D(num_classes=args.num_classes)
    if args.loss_type == "hybrid":
        return HybridSegLoss(num_classes=args.num_classes)
    return nn.CrossEntropyLoss()


def main():
    args = get_args()
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    _seed_everything(seed)
    cudnn.benchmark = True

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # dataset (test)
    dataset_test, _ = get_loader(
        data_dir=args.data_path,
        size=args.input_size,
        mode="test",
    )

    if args.distributed:
        sampler_test = DistributedSampler(
            dataset_test,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler_test = SequentialSampler(dataset_test)

    data_loader_test = DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # model
    model = build_gcsmim(img_size=args.input_size, in_channel=1, n_classes=args.num_classes).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    msg = model_without_ddp.load_state_dict(state, strict=True)

    if misc.is_main_process():
        print(f"[test] Loaded checkpoint: {args.ckpt}")
        print(f"[test] load_state_dict msg: {msg}")

    criterion = _build_criterion(args)

    test_stats = evaluate(
        data_loader=data_loader_test,
        model=model,
        device=device,
        criterion=criterion,
        num_classes=args.num_classes,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
    )

    if misc.is_main_process():
        print("Test stats:", test_stats)
        if args.output_dir:
            out_path = os.path.join(args.output_dir, "log_test.jsonl")
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(test_stats) + "\n")
            print(f"[test] Saved: {out_path}")


if __name__ == "__main__":
    main()
