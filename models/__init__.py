from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
import torch.nn as nn

from timm.loss import SoftTargetCrossEntropy
from timm.layers import DropPath  

from models.encoder import SparseEncoder
from models.network.gcsmim_model import GCSMIMEncoder


def _ex_repr(self):
    return ", ".join(
        f"{k}=" + (f"{v:g}" if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith("_")
        and k != "training"
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )


for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, DropPath):
    if hasattr(clz, "extra_repr"):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f"{type(self).__name__}({_ex_repr(self)})"


pretrain_default_model_kwargs = {
    "gcsmim": dict(sparse=True, drop_path_rate=0.1),
}

for kw in pretrain_default_model_kwargs.values():
    kw["pretrained"] = False
    kw["num_classes"] = 0
    kw["global_pool"] = ""


def build_sparse_encoder(
    name: str,
    input_size: int | Tuple[int, int, int],
    sbn: bool = False,
    drop_path_rate: float = 0.0,
    verbose: bool = False,
) -> SparseEncoder:

    if name not in pretrain_default_model_kwargs:
        raise ValueError(
            f"Unknown encoder name='{name}'. Available: {list(pretrain_default_model_kwargs.keys())}"
        )

    kwargs = dict(pretrain_default_model_kwargs[name])
    if drop_path_rate != 0:
        kwargs["drop_path_rate"] = drop_path_rate

    print(f"[build_sparse_encoder] name={name}, kwargs={kwargs}")

    enc = GCSMIMEncoder(
        img_size=96,
        in_chans=1,
        embed_dims=(32, 64, 128, 256, 384),
        depth=[[1], [1], [1], [1, 2], [1, 2]],
        kernels=(3, 3, 3, 3, 3),
        exp_r=(2, 2, 2, 2, 2),
        down_ratio=(1, 2, 4, 8, 16),
        drop_rate=0.0,
        drop_path_rate=kwargs.get("drop_path_rate", 0.0),
        act_layer=partial(nn.GELU),
        sparse=True,
    )

    return SparseEncoder(encoder=enc, input_size=input_size, sbn=sbn, verbose=verbose)
