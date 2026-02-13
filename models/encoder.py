import torch
import torch.nn as nn
from timm.layers import DropPath


_cur_active: torch.Tensor = None  # (B,1,f,f,f)


def _get_active_ex_or_ii(D, H, W, returning_active_ex=True):
    """
    Expand `_cur_active` from (B,1,f,f,f) to (B,1,D,H,W) or return indices.
    """
    h_repeat = H // _cur_active.shape[-2]
    w_repeat = W // _cur_active.shape[-1]
    d_repeat = D // _cur_active.shape[-3]
    active_ex = _cur_active.repeat_interleave(d_repeat, dim=2)\
                          .repeat_interleave(h_repeat, dim=3)\
                          .repeat_interleave(w_repeat, dim=4)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)


def get_2d_mask_from_3d(D, H, W) -> torch.Tensor:
    """
    Convert expanded 3D mask to token mask (B, N) where N = D*H*W.
    """
    B = _cur_active.shape[0]
    mask = _get_active_ex_or_ii(D, H, W, returning_active_ex=True)
    return mask.view(B, 1, -1).squeeze(1).bool()


def _get_active_ex_or_ii_2d(D, H, W, returning_active_ex=True):
    _cur_active_2d = get_2d_mask_from_3d(D=D, H=H, W=W)
    if returning_active_ex:
        return _cur_active_2d.unsqueeze(-1)  # (B, N, 1)
    return _cur_active_2d.nonzero(as_tuple=True)  # (bi, ni)


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    mask = _get_active_ex_or_ii(D=x.shape[2], H=x.shape[3], W=x.shape[4], returning_active_ex=True)
    x *= mask
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(D=x.shape[2], H=x.shape[3], W=x.shape[4], returning_active_ex=False)
    bdhwc = x.permute(0, 2, 3, 4, 1)
    nc = bdhwc[ii]  # (num_active, C)
    nc = super(type(self), self).forward(nc)
    bdhwc_out = torch.zeros_like(bdhwc)
    bdhwc_out[ii] = nc
    return bdhwc_out.permute(0, 4, 1, 2, 3)


def sp_in_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(D=x.shape[2], H=x.shape[3], W=x.shape[4], returning_active_ex=False)
    bdhwc = x.permute(0, 2, 3, 4, 1)
    cn = bdhwc[ii].permute(1, 0)  # (C, num_active)
    C, N = cn.shape
    bc1 = cn.reshape(C, -1, x.shape[0]).permute(2, 0, 1)
    bc1 = super(type(self), self).forward(bc1)
    nc = bc1.permute(1, 2, 0).reshape(C, -1).permute(1, 0)
    bdhwc_out = torch.zeros_like(bdhwc)
    bdhwc_out[ii] = nc
    return bdhwc_out.permute(0, 4, 1, 2, 3)


class SparseConv3d(nn.Conv3d):
    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool3d):
    forward = sp_conv_forward


class SparseAvgPooling(nn.AvgPool3d):
    forward = sp_conv_forward


class SparseBatchNorm3d(nn.BatchNorm1d):
    forward = sp_bn_forward


class SparseSyncBatchNorm3d(nn.SyncBatchNorm):
    forward = sp_bn_forward


class SparseInstanceNorm3d(nn.InstanceNorm1d):
    forward = sp_in_forward


class SparseLinear(nn.Linear):
    """
    Sparse token-wise Linear:
    - if sparse_mode=False: normal Linear on all tokens
    - if sparse_mode=True: only apply Linear on active tokens determined by `_cur_active`
    """
    def __init__(self, in_features, out_features, sparse=False):
        super().__init__(in_features, out_features)
        self.sparse_mode = sparse

    def forward(self, x):
        if not self.sparse_mode:
            return super().forward(x)

        B, N, _ = x.shape
        D = H = W = int(round(N ** (1 / 3)))
        ii = _get_active_ex_or_ii_2d(D=D, H=H, W=W, returning_active_ex=False)
        active_features = x[ii]                   # (num_active, C_in)
        transformed = super().forward(active_features)  # (num_active, C_out)
        transformed = transformed.to(dtype=x.dtype)

        output = torch.zeros(B, N, self.out_features, device=x.device, dtype=x.dtype)
        output[ii] = transformed
        return output


class SparseBatchNorm3dSP(nn.BatchNorm3d):
    def forward(self, x):
        return super().forward(x)


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    """
    LayerNorm supporting channels_last and channels_first, and sparse token LN.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 5:
            # 5D: BHWDC or BCHWD
            if self.data_format == "channels_last":
                if self.sparse:
                    ii = _get_active_ex_or_ii(D=x.shape[1], H=x.shape[2], W=x.shape[3], returning_active_ex=False)
                    nc = x[ii]
                    nc = super().forward(nc)
                    nc = nc.to(dtype=x.dtype)
                    out = torch.zeros_like(x)
                    out[ii] = nc
                    return out
                return super().forward(x)
            else:
                # channels_first: BCHWD
                if self.sparse:
                    ii = _get_active_ex_or_ii(D=x.shape[2], H=x.shape[3], W=x.shape[4], returning_active_ex=False)
                    bdhwc = x.permute(0, 2, 3, 4, 1)
                    nc = bdhwc[ii]
                    nc = super().forward(nc)
                    nc = nc.to(dtype=x.dtype)
                    out = torch.zeros_like(bdhwc)
                    out[ii] = nc
                    return out.permute(0, 4, 1, 2, 3)
                # dense channels_first LN
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
                return x

        # 3D tokens: (B, N, C) or (B, C)
        if self.sparse:
            B, N, _ = x.shape
            D = H = W = int(round(N ** (1 / 3)))
            ii = _get_active_ex_or_ii_2d(D=D, H=H, W=W, returning_active_ex=False)
            nc = x[ii]
            nc = super().forward(nc)
            nc = nc.to(dtype=x.dtype)
            out = torch.zeros_like(x)
            out[ii] = nc
            return out
        return super().forward(x)

    def __repr__(self):
        return super().__repr__()[:-1] + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'


class SparseConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, exp_r=4, do_res=False,
                 drop_path=0., layer_scale_init_value=1e-6, sparse=True):
        super().__init__()
        self.do_res = do_res
        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=in_channels)
        self.norm = SparseConvNeXtLayerNorm(in_channels, eps=1e-6, sparse=sparse)
        self.pwconv1 = nn.Linear(in_channels, exp_r * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(exp_r * in_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sparse = sparse

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B,C,D,H,W)->(B,D,H,W,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # back to (B,C,D,H,W)

        if self.sparse:
            x *= _get_active_ex_or_ii(D=x.shape[2], H=x.shape[3], W=x.shape[4], returning_active_ex=True)

        if self.do_res:
            x = inp + self.drop_path(x)
        return x

    def __repr__(self):
        return super().__repr__()[:-1] + f', sp={self.sparse})'


class SparseEncoder(nn.Module):
    def __init__(self, encoder, input_size, sbn=False, verbose=False):
        super().__init__()
        self.embeddings = SparseEncoder.dense_model_to_sparse(m=encoder.embeddings, verbose=verbose, sbn=sbn)
        self.input_size = input_size
        self.downsample_ratio = encoder.get_downsample_ratio()
        self.enc_feat_map_chs = encoder.get_feature_map_channels()

    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        oup = m

        if isinstance(m, nn.Conv3d):
            bias = m.bias is not None
            oup = SparseConv3d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)

        elif isinstance(m, nn.MaxPool3d):
            oup = SparseMaxPooling(
                m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation,
                return_indices=m.return_indices, ceil_mode=m.ceil_mode
            )

        elif isinstance(m, nn.AvgPool3d):
            oup = SparseAvgPooling(
                m.kernel_size, m.stride, m.padding,
                ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override
            )

        elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm)) and not isinstance(m, SparseBatchNorm3dSP):
            if sbn:
                oup = SparseSyncBatchNorm3d(
                    m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine,
                    track_running_stats=m.track_running_stats
                )
            else:
                oup = SparseBatchNorm3d(
                    m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine,
                    track_running_stats=m.track_running_stats
                )
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig

        elif isinstance(m, nn.InstanceNorm3d):
            oup = SparseInstanceNorm3d(
                m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine,
                track_running_stats=m.track_running_stats
            )
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig

        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            oup = SparseConvNeXtLayerNorm(m.normalized_shape[0], eps=m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)

        elif isinstance(m, (nn.Conv1d,)):
            raise NotImplementedError

        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
        del m
        return oup

    def forward(self, x, active_masks):
        return self.embeddings(x, active_masks)
