# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # print(f"epoch: {epoch}")
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def lr_wd_annealing(optimizer, peak_lr, wd, wd_end, cur_it, wp_it, max_it):
    wp_it = round(wp_it)
    if cur_it < wp_it:
        cur_lr = 0.005 * peak_lr + 0.995 * peak_lr * cur_it / wp_it
    else:
        ratio = (cur_it - wp_it) / (max_it - 1 - wp_it)
        cur_lr = 0.001 * peak_lr + 0.999 * peak_lr * (0.5 + 0.5 * math.cos(math.pi * ratio))
        
    ratio = cur_it / (max_it - 1)
    cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * ratio))
    
    min_lr, max_lr = cur_lr, cur_lr
    min_wd, max_wd = cur_wd, cur_wd
    for param_group in optimizer.param_groups:
        scaled_lr = param_group['lr'] = cur_lr * param_group.get('lr_scale', 1)     # 'lr_scale' could be assigned
        min_lr, max_lr = min(min_lr, scaled_lr), max(max_lr, scaled_lr)
        scaled_wd = param_group['weight_decay'] = cur_wd * param_group.get('weight_decay_scale', 1) # 'weight_decay_scale' could be assigned
        min_wd, max_wd = min(min_wd, scaled_wd), max(max_wd, scaled_wd)
    return min_lr, max_lr, min_wd, max_wd