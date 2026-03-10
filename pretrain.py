import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.datasets import get_loader

from models import build_sparse_encoder
from models.decoder import Light_Decoder
from models.gcsmim import GCSMIM

from engine.pretrain import train_one_epoch


def get_args():
    parser = argparse.ArgumentParser('GCSMIM Pretraining (3D)', add_help=False)

    # training
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch = batch_size * accum_iter * world_size)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    # resume / checkpoint
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=False)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--output_dir', default='', type=str)

    # data
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--input_size', type=int, default=96, help='Input crop size (cube).')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # model
    parser.add_argument('--model', default='gcsmim', type=str, help='Model name for build_sparse_encoder.')
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--dp', default=0.0, type=float, help='Drop path rate for sparse encoder.')
    parser.add_argument('--sbn', type=bool, default=False, help='Use SyncBN for sparse encoder.')
    parser.add_argument('--densify_norm', type=str, default='', help='Kept for compatibility; currently Identity in code.')

    # optim
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=0.2)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--clip_grad', type=float, default=5.0)

    # distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        self._setup_distributed()
        self._build_dataset()
        self._build_model()
        self._configure_optimizer()
        self._auto_load_model()

    def _setup_distributed(self):
        misc.init_distributed_mode(self.args)

        if misc.is_main_process():
            print(f'[Init] args = {self.args}')

        seed = self.args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

    def _build_dataset(self):
        dataset_train, _ = get_loader(self.args.data_path, self.args.input_size, mode="pretrain")

        if self.args.distributed:
            sampler_train = DistributedSampler(
                dataset_train,
                num_replicas=misc.get_world_size(),
                rank=misc.get_rank(),
                shuffle=True,
            )
        else:
            sampler_train = SequentialSampler(dataset_train)

        self.data_loader_train = DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
        )

        self.iters_train = len(self.data_loader_train)

        if misc.is_main_process():
            print(f'[Data] iters_train={self.iters_train} | sampler={type(sampler_train).__name__}')

    def _build_model(self):
        enc = build_sparse_encoder(
            name=self.args.model,
            input_size=self.args.input_size,
            sbn=self.args.sbn,
            drop_path_rate=self.args.dp,
            verbose=False,
        )

        dec = Light_Decoder(up_sample_ratio=enc.downsample_ratio, width=768, sbn=self.args.sbn)

        self.model = GCSMIM(
            sparse_encoder=enc,
            dense_decoder=dec,
            mask_ratio=self.args.mask_ratio,
            densify_norm=self.args.densify_norm,
            sbn=self.args.sbn,
        ).to(self.device)

        if misc.is_main_process():
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
            print(f'[Model] total={total_params:.2f}M | trainable={trainable_params:.2f}M')
            print(f'[Model] {self.model}')

        self.model_without_ddp = self.model

        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.gpu],
                find_unused_parameters=True
            )
            self.model_without_ddp = self.model.module

    def _configure_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model_without_ddp.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )
        self.loss_scaler = NativeScaler()

        if misc.is_main_process():
            print(f'[Optim] {self.optimizer}')

    def _auto_load_model(self):
        if not self.args.output_dir:
            return
        misc.auto_load_model(
            args=self.args,
            model_without_ddp=self.model_without_ddp,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler,
            mode='pretrain',
        )

    def run(self):
        if misc.is_main_process():
            print(f'[Train] start for {self.args.epochs} epochs')

        start_time = time.time()
        best_loss = float('inf')

        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.data_loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model=self.model,
                data_loader=self.data_loader_train,
                iters_train=self.iters_train,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                loss_scaler=self.loss_scaler,
                args=self.args,
            )

            train_loss = float(train_stats.get('loss', 0.0))

            if train_loss <= best_loss:
                best_loss = min(best_loss, train_loss)
                self._save_checkpoint(epoch, best=True)

            self._save_checkpoint(epoch, best=False)

            self._log(epoch, train_stats)

        total_time = time.time() - start_time
        if misc.is_main_process():
            print(f'[Train] total time {str(datetime.timedelta(seconds=int(total_time)))}')

    def _save_checkpoint(self, epoch: int, best: bool):
        if not self.args.output_dir:
            return
        if not misc.is_main_process():
            return

        if (not best) and not (epoch % 10 == 0 or epoch + 1 == self.args.epochs):
            return

        misc.save_model(
            args=self.args,
            model=self.model,
            model_without_ddp=self.model_without_ddp,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler,
            epoch=epoch,
            best=best,
            mode='pretrain',
        )

    def _log(self, epoch: int, train_stats: dict):
        if not self.args.output_dir or not misc.is_main_process():
            return
        log_stats = {f'train_{k}': v for k, v in train_stats.items()}
        log_stats['epoch'] = epoch
        with open(os.path.join(self.args.output_dir, "log_pretrain.jsonl"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Trainer(args).run()
