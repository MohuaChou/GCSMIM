from pprint import pformat
from typing import List, Optional
import sys

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

import models.encoder as encoder
from models.decoder import Light_Decoder


class GCSMIM(nn.Module):
    """
    Sparse masked image modeling wrapper for GCSMIM.

    Pipeline:
      1) Random mask (active tokens)
      2) Sparse encode on masked input
      3) Densify (fill masked positions with mask tokens + project)
      4) Decode to reconstruction
      5) Compute loss
    """

    def __init__(
        self,
        sparse_encoder: encoder.SparseEncoder,
        dense_decoder: Light_Decoder,
        mask_ratio: float = 0.5,
        densify_norm: str = 'ln',
        sbn: bool = True,
    ):
        super().__init__()

        input_size, downsample_ratio = sparse_encoder.input_size, sparse_encoder.downsample_ratio
        self.downsample_ratio = downsample_ratio

        deepest_ratio = downsample_ratio[-1]
        self.fmap_d = input_size // deepest_ratio
        self.fmap_h = input_size // deepest_ratio
        self.fmap_w = input_size // deepest_ratio

        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_d * self.fmap_h * self.fmap_w * (1 - mask_ratio))

        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder

        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()

        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        # build densify modules
        e_widths = list(self.sparse_encoder.enc_feat_map_chs)
        d_width = self.dense_decoder.width

        for i in range(self.hierarchy):
            e_width = e_widths.pop()

            # mask token at this scale
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)

            # densify norm
            self.densify_norms.append(nn.Identity())

            # densify projection
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv3d(
                    e_width, d_width,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=True
                )
            self.densify_projs.append(densify_proj)

            # decoder width halving rule
            d_width //= 2

    def mask(self, B: int, device, generator=None) -> torch.BoolTensor:
        d, h, w = self.fmap_d, self.fmap_h, self.fmap_w
        idx = torch.rand(B, d * h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)
        return torch.zeros(B, d * h * w, dtype=torch.bool, device=device) \
            .scatter_(dim=1, index=idx, value=True).view(B, 1, d, h, w)

    def forward(self, inp_bcdhw: torch.Tensor, active_b1fff: Optional[torch.Tensor] = None, vis: bool = False):
        # ---------------- Mask ----------------
        if active_b1fff is None:
            active_b1fff = self.mask(inp_bcdhw.shape[0], inp_bcdhw.device)  # (B,1,f,f,f)

        encoder._cur_active = active_b1fff

        active_masks = []
        for ratio in self.downsample_ratio:
            expanded_mask = active_b1fff.repeat_interleave(ratio, dim=2)\
                                        .repeat_interleave(ratio, dim=3)\
                                        .repeat_interleave(ratio, dim=4)
            active_masks.append(expanded_mask)

        masked_bcdhw = inp_bcdhw * active_masks[-1]

        # ---------------- Sparse Encode ----------------
        fea_bcfffs: List[torch.Tensor] = self.sparse_encoder(masked_bcdhw, active_masks)
        fea_bcfffs.reverse()  # smallest -> largest for densify

        # ---------------- Densify ----------------
        cur_active = active_b1fff
        to_dec = []
        for i, bcfff in enumerate(fea_bcfffs):
            if bcfff is None:
                continue

            bcfff = self.densify_norms[i](bcfff)
            mask_tokens = self.mask_tokens[i].expand_as(bcfff).to(bcfff.dtype)
            bcfff = torch.where(cur_active.expand_as(bcfff), bcfff, mask_tokens)
            bcfff = self.densify_projs[i](bcfff)
            to_dec.append(bcfff)

            # upsample mask for next stage
            cur_active = cur_active.repeat_interleave(2, dim=2)\
                                   .repeat_interleave(2, dim=3)\
                                   .repeat_interleave(2, dim=4)

        # ---------------- Decode ----------------
        rec_b1dhw = self.dense_decoder(to_dec)  # (B,1,D,H,W)

        # ---------------- Loss ----------------
        # Use active_b1fff to compute loss on masked tokens only.
        # Patchify both rec and inp at deepest scale.
        reg = self.patchify_top(rec_b1dhw)
        inp = self.patchify_top(inp_bcdhw)
        
        if inp.size(-1) == 1:
            mean = inp.mean(dim=1, keepdim=True)
            var = (inp.var(dim=1, keepdim=True) + 1e-6) ** 0.5
        else:
            mean = inp.mean(dim=-1, keepdim=True)
            var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
        inp_norm = (inp - mean) / var

        non_active = active_b1fff.logical_not().int().view(active_b1fff.shape[0], -1)  # (B, N)

        loss_map = ((reg - inp_norm) ** 2).mean(dim=2, keepdim=False)  # (B, N)
        total_loss = (loss_map.mul(non_active).sum() / (non_active.sum() + 1e-8))

        if not vis:
            return total_loss

        # ---------------- Visualization ----------------
        rec_unpa = self.unpatchify_top(reg * var + mean)
        rec_or_inp = torch.where(active_masks[-1], inp_bcdhw, rec_unpa)
        return inp_bcdhw, masked_bcdhw, rec_or_inp, rec_unpa


    def patchify_top(self, bcdhw: torch.Tensor) -> torch.Tensor:
        """
        Output: (B, N, patch_dim)
        """
        p = self.downsample_ratio[-1]
        d, h, w = self.fmap_d, self.fmap_h, self.fmap_w
        B, C = bcdhw.shape[:2]

        x = bcdhw.reshape(B, C, d, p, h, p, w, p)
        x = torch.einsum('bcdphqwr->bdhwpqrc', x)  # -> (B,d,h,w,p,p,p,C)
        bln = x.reshape(B, d * h * w, C * (p ** 3))
        return bln

    def unpatchify_top(self, bln: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, N, C*p^3) where C is 1 in your reconstruction.
        Output: (B, 1, D, H, W)
        """
        p = self.downsample_ratio[-1]
        d, h, w = self.fmap_d, self.fmap_h, self.fmap_w

        B = bln.shape[0]
        C = 1  # reconstruction is single-channel

        x = bln.reshape(B, d, h, w, p, p, p, C)
        x = torch.einsum('bdhwpqrc->bcdphqwr', x)
        bcdhw = x.reshape(B, C, d * p, h * p, w * p)
        return bcdhw


    def get_config(self):
        return {
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn,
            'hierarchy': self.hierarchy,
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            'dense_decoder.width': self.dense_decoder.width,
        }

    def __repr__(self):
        return (
            f'\n'
            f'[GCSMIM.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[GCSMIM.structure]: {super(GCSMIM, self).__repr__().replace(GCSMIM.__name__, "")}'
        )

    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state

    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super().load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[GCSMIM.load_state_dict] config mismatch: this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    print(err, file=sys.stderr)
        return incompatible_keys
