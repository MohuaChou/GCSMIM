import math
from typing import List, Sequence, Union

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from utils.misc import is_pow2n



class UNetUpBlock(nn.Module):
    """
    Lightweight UNet-style upsample block: 2x upsample + 2 convs.
    """
    def __init__(self, cin: int, cout: int, bn3d):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cin, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, bias=True),
            bn3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=True),
            bn3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class FusionBlock(nn.Module):
    """
    Fuse skip + decoder features after concat: 2 convs.
    """
    def __init__(self, cin: int, cout: int, bn3d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, bias=True),
            bn3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=True),
            bn3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Light_Decoder(nn.Module):
    """
    Decoder used in sparse MIM:
    - input: `to_dec` list of dense feature maps
    - output: ONE reconstruction volume at the final stage only (B, 1, D, H, W)
    """
    def __init__(
        self,
        up_sample_ratio: Union[int, Sequence[int]],
        width: int = 768,
        sbn: bool = True,
    ):
        super().__init__()
        self.width = width

        if isinstance(up_sample_ratio, (list, tuple)):
            for r in up_sample_ratio:
                assert is_pow2n(r)
            max_ratio = max(up_sample_ratio)
            assert is_pow2n(max_ratio)
            n = round(math.log2(max_ratio))
        else:
            assert is_pow2n(up_sample_ratio)
            n = round(math.log2(up_sample_ratio))

        channels = [self.width // (2 ** i) for i in range(n + 1)]
        bn3d = nn.BatchNorm3d

        self.dec = nn.ModuleList([UNetUpBlock(cin, cout, bn3d) for cin, cout in zip(channels[:-1], channels[1:])])
        self.fuse = nn.ModuleList([FusionBlock(cin * 2, cin, bn3d) for cin, _ in zip(channels[:-1], channels[1:])])

        self.out = nn.Conv3d(channels[-1], 1, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]) -> torch.Tensor:
        x = 0
        for i, up in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                if isinstance(x, int):
                    x = x + to_dec[i]
                else:
                    x = torch.cat((x, to_dec[i]), dim=1)
                    x = self.fuse[i](x)

            x = up(x)

        return self.out(x)

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
