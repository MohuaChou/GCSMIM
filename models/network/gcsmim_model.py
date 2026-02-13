import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union, Sequence, Optional
from functools import partial

from timm.layers import DropPath, to_3tuple

from models.encoder import (
    SparseConvNeXtLayerNorm,
    SparseLinear,
    _get_active_ex_or_ii,
    _get_active_ex_or_ii_2d,
)


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        n_groups: Optional[int] = None,
        sparse: bool = False,
    ):

        super().__init__()
        self.do_res = do_res
        self.sparse = sparse
        conv = nn.Conv3d

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        self.norm = SparseConvNeXtLayerNorm(
            normalized_shape=in_channels,
            data_format="channels_first",
            sparse=sparse,
        )

        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act = nn.GELU()

        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)

        if self.sparse:
            x1 *= _get_active_ex_or_ii(
                D=x1.shape[2], H=x1.shape[3], W=x1.shape[4],
                returning_active_ex=True,
            )
        if self.do_res:
            x1 = x + x1
        return x1


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FeatureIncentiveBlock(nn.Module):
    def __init__(self, img_size=96, patch_size=7, stride=4, in_chans=3, embed_dim=768, sparse=False):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.sparse = sparse

        self.img_size = img_size
        self.patch_size = patch_size
        self.D = img_size[0] // patch_size[0]
        self.H = img_size[1] // patch_size[1]
        self.W = img_size[2] // patch_size[2]
        self.num_patches = self.D * self.H * self.W

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2),
        )
        self.norm = SparseConvNeXtLayerNorm(normalized_shape=embed_dim, sparse=sparse)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        if self.sparse:
            mask = _get_active_ex_or_ii(D=x.shape[2], H=x.shape[3], W=x.shape[4], returning_active_ex=True)
            x *= mask

        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.act(x)
        x = self.norm(x)
        return x, D, H, W


class GroupedCyclicShift3D:
    @staticmethod
    def get_offsets(x_bcdhw: torch.Tensor, D: int, H: int, W: int):
        _, C, _, _, _ = x_bcdhw.shape
        device = x_bcdhw.device

        base_group_size = min(D, H, W)
        num_groups = (C + base_group_size - 1) // base_group_size

        ch_indices = torch.arange(C, device=device)
        group_numbers = ch_indices // base_group_size
        idx_in_group = ch_indices % base_group_size

        group_base_d = torch.arange(num_groups, device=device)
        group_base_h = torch.arange(num_groups, device=device)
        group_base_w = torch.arange(num_groups, device=device)

        depth_base = group_base_d[group_numbers]
        height_base = group_base_h[group_numbers]
        width_base = group_base_w[group_numbers]

        depth_shifts = (depth_base + idx_in_group) % D
        height_shifts = (height_base + idx_in_group) % H
        width_shifts = (width_base + idx_in_group) % W

        return depth_shifts, height_shifts, width_shifts

    @staticmethod
    def apply_shift(
        x_bcdhw: torch.Tensor,
        shifts_c: torch.Tensor,
        axis: int,
        reverse: bool,
        D: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        B, C, D, H, W = x_bcdhw.shape
        device = x_bcdhw.device
        direction = -1 if reverse else 1

        indices = torch.arange(D * H * W, device=device).view(1, 1, D, H, W).expand(B, C, -1, -1, -1)

        if axis == 2:  # Depth
            depth_idx = indices // (H * W)
            shifted_depth = (depth_idx - direction * shifts_c.view(1, C, 1, 1, 1)) % D
            new_indices = shifted_depth * (H * W) + (indices % (H * W))
        elif axis == 3:  # Height
            plane_idx = indices % (H * W)
            height_idx = plane_idx // W
            shifted_height = (height_idx - direction * shifts_c.view(1, C, 1, 1, 1)) % H
            new_plane_idx = shifted_height * W + (plane_idx % W)
            new_indices = (indices // (H * W)) * (H * W) + new_plane_idx
        elif axis == 4:  # Width
            plane_idx = indices % (H * W)
            width_idx = plane_idx % W
            shifted_width = (width_idx - direction * shifts_c.view(1, C, 1, 1, 1)) % W
            new_plane_idx = (plane_idx // W) * W + shifted_width
            new_indices = (indices // (H * W)) * (H * W) + new_plane_idx
        else:
            raise ValueError(f"Invalid axis={axis}, expected 2/3/4")

        x_flat = x_bcdhw.view(B, C, -1)
        indices_flat = new_indices.view(B, C, -1)
        shifted_x = torch.gather(x_flat, dim=2, index=indices_flat.long())
        return shifted_x.view(B, C, D, H, W)


class MGFR:
    @staticmethod
    def restore(shifted_x: torch.Tensor, original_x: torch.Tensor, mask_shifted: torch.Tensor) -> torch.Tensor:
        return shifted_x * mask_shifted + original_x * (1 - mask_shifted)


class DWConv3D(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv3d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x


class GCSMixer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, sparse=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.sparse = sparse

        # ---- Path-1 (Forward GCS): D -> H -> W ----
        self.path1_fc_d = SparseLinear(in_features, hidden_features, sparse=sparse)
        self.path1_fc_h = SparseLinear(in_features, hidden_features, sparse=sparse)
        self.path1_fc_w = SparseLinear(in_features, hidden_features, sparse=sparse)

        # ---- Path-2 (Reverse GCS): W -> H -> D ----
        self.path2_fc_w = SparseLinear(in_features, hidden_features, sparse=sparse)
        self.path2_fc_h = SparseLinear(in_features, hidden_features, sparse=sparse)
        self.path2_fc_d = SparseLinear(in_features, hidden_features, sparse=sparse)

        # ---- Fusion + Output projection ----
        self.fusion_fc = SparseLinear(in_features * 2, hidden_features, sparse=sparse)
        self.out_fc = SparseLinear(in_features * 2, out_features, sparse=sparse)

        self.drop = nn.Dropout(drop)
        self.dwconv = DWConv3D(hidden_features)
        self.act1 = act_layer()
        self.act2 = nn.ReLU()
        self.norm_fusion = SparseConvNeXtLayerNorm(normalized_shape=hidden_features * 2, sparse=sparse)
        self.norm_dw = nn.BatchNorm3d(hidden_features)

    def forward(self, x, D, H, W, mask=None):
        B, N, C = x.size(0), x.size(1), x.size(2)

        xn = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        original_x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        depth_shifts, height_shifts, width_shifts = GroupedCyclicShift3D.get_offsets(xn, D, H, W)

        if self.sparse:
            if mask.dtype == torch.bool:
                mask = mask.float()
            mpd = mask.expand(-1, C, -1, -1, -1)

            m_d_fwd = GroupedCyclicShift3D.apply_shift(mpd, depth_shifts, axis=2, reverse=False, D=D, H=H, W=W)
            m_h_fwd = GroupedCyclicShift3D.apply_shift(m_d_fwd, height_shifts, axis=3, reverse=False, D=D, H=H, W=W)
            m_w_fwd = GroupedCyclicShift3D.apply_shift(m_h_fwd, width_shifts, axis=4, reverse=False, D=D, H=H, W=W)

            m_w_rev = GroupedCyclicShift3D.apply_shift(mpd, width_shifts, axis=4, reverse=True, D=D, H=H, W=W)
            m_h_rev = GroupedCyclicShift3D.apply_shift(m_w_rev, height_shifts, axis=3, reverse=True, D=D, H=H, W=W)
            m_d_rev = GroupedCyclicShift3D.apply_shift(m_h_rev, depth_shifts, axis=2, reverse=True, D=D, H=H, W=W)

        # ===================== Path 1: Forward GCS (D -> H -> W) =====================
        x_cat = GroupedCyclicShift3D.apply_shift(xn, depth_shifts, axis=2, reverse=False, D=D, H=H, W=W)
        if self.sparse:
            x_cat = MGFR.restore(x_cat, original_x, m_d_fwd)
        x_cat = self.path1_fc_d(x_cat.permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, C))
        x_cat = self.act1(x_cat)
        x_cat = self.drop(x_cat)

        x_cat = GroupedCyclicShift3D.apply_shift(
            x_cat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous(),
            height_shifts, axis=3, reverse=False, D=D, H=H, W=W
        )
        if self.sparse:
            x_cat = MGFR.restore(x_cat, original_x, m_h_fwd)
        x_cat = self.path1_fc_h(x_cat.permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, C))
        x_cat = self.act1(x_cat)
        x_cat = self.drop(x_cat)

        x_cat = GroupedCyclicShift3D.apply_shift(
            x_cat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous(),
            width_shifts, axis=4, reverse=False, D=D, H=H, W=W
        )
        if self.sparse:
            x_cat = MGFR.restore(x_cat, original_x, m_w_fwd)
        x_cat = self.path1_fc_w(x_cat.permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, C))
        x_1 = self.drop(x_cat)

        # ===================== Path 2: Reverse GCS (W -> H -> D) =====================
        xn = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        x_cat = GroupedCyclicShift3D.apply_shift(xn, width_shifts, axis=4, reverse=True, D=D, H=H, W=W)
        if self.sparse:
            x_cat = MGFR.restore(x_cat, original_x, m_w_rev)
        x_cat = self.path2_fc_w(x_cat.permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, C))
        x_cat = self.act1(x_cat)
        x_cat = self.drop(x_cat)

        x_cat = GroupedCyclicShift3D.apply_shift(
            x_cat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous(),
            height_shifts, axis=3, reverse=True, D=D, H=H, W=W
        )
        if self.sparse:
            x_cat = MGFR.restore(x_cat, original_x, m_h_rev)
        x_cat = self.path2_fc_h(x_cat.permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, C))
        x_cat = self.act1(x_cat)
        x_cat = self.drop(x_cat)

        x_cat = GroupedCyclicShift3D.apply_shift(
            x_cat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous(),
            depth_shifts, axis=2, reverse=True, D=D, H=H, W=W
        )
        if self.sparse:
            x_cat = MGFR.restore(x_cat, original_x, m_d_rev)
        x_cat = self.path2_fc_d(x_cat.permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, C))
        x_2 = self.drop(x_cat)

        # ===================== Fusion & Output (unchanged) =====================
        x_1 = torch.add(x_1, x)
        x_2 = torch.add(x_2, x)

        x_fuse = torch.cat([x_1, x_2], dim=2)
        x_fuse = self.norm_fusion(x_fuse)
        x_fuse = self.fusion_fc(x_fuse)
        x_fuse = self.drop(x_fuse)
        x_fuse = torch.add(x_fuse, x)

        x_local = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        x_local = self.dwconv(x_local)
        x_local = self.act2(x_local)
        x_local = self.norm_dw(x_local)
        x_local = x_local.flatten(2).transpose(1, 2).contiguous()

        x_out = torch.cat([x_fuse, x_local], dim=2)
        x_out = self.out_fc(x_out)
        if self.sparse:
            x_out *= _get_active_ex_or_ii_2d(D=D, H=H, W=W, returning_active_ex=True)
        x_out = self.drop(x_out)
        return x_out



class GCSBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, sparse=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = SparseConvNeXtLayerNorm(normalized_shape=dim, sparse=sparse)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mixer = GCSMixer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            sparse=sparse,
        )

    def forward(self, x, D, H, W, mask=None):
        x = self.drop_path(self.mixer(x, D, H, W, mask))
        return x


class GCSMIMEmbedding(nn.Module):
    def __init__(
        self,
        img_size=96,
        in_chans=1,
        embed_dims: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 384),
        depth: List[List[int]] = [[1], [1], [1], [2], [1, 2]],
        kernels: Tuple[int, int, int, int, int] = (3, 3, 3, 3, 3),
        exp_r: Tuple[int, int, int, int, int] = (2, 2, 2, 2, 2),
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=partial(nn.GELU),
        sparse=False,
    ):
        super().__init__()
        self.sparse = sparse

        self.stem = DoubleConv(in_chans, embed_dims[0])

        self.layer1 = nn.Sequential(
            *[
                MedNeXtBlock(embed_dims[0], embed_dims[0], exp_r=exp_r[0], kernel_size=kernels[0], do_res=True, sparse=sparse)
                for _ in range(depth[0][0])
            ]
        )
        self.layer2 = nn.Sequential(
            *[
                MedNeXtBlock(embed_dims[1], embed_dims[1], exp_r=exp_r[1], kernel_size=kernels[1], do_res=True, sparse=sparse)
                for _ in range(depth[1][0])
            ]
        )

        self.expand1 = nn.Conv3d(embed_dims[0], embed_dims[1], 3, 1, 1)
        self.expand2 = nn.Conv3d(embed_dims[1], embed_dims[2], 3, 1, 1)
        self.expand3 = nn.Conv3d(embed_dims[2], embed_dims[3], 3, 1, 1)
        self.expand4 = nn.Conv3d(embed_dims[3], embed_dims[4], 3, 1, 1)

        self.fib1 = FeatureIncentiveBlock(img_size=img_size // 4, patch_size=7, stride=1, in_chans=embed_dims[2], embed_dim=embed_dims[3], sparse=sparse)
        self.fib2 = FeatureIncentiveBlock(img_size=img_size // 8, patch_size=5, stride=1, in_chans=embed_dims[3], embed_dim=embed_dims[4], sparse=sparse)
        self.fib3 = FeatureIncentiveBlock(img_size=img_size // 8, patch_size=5, stride=1, in_chans=embed_dims[4], embed_dim=embed_dims[4], sparse=sparse)

        dpdepths = depth[3][1] + depth[4][1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, dpdepths)]

        self.stage4 = nn.ModuleList(
            [
                GCSBlock(
                    dim=embed_dims[3],
                    mlp_ratio=1,
                    drop=drop_rate + 0.2 if not self.sparse else drop_rate,
                    drop_path=dpr[i],
                    sparse=sparse,
                )
                for i in range(depth[3][1])
            ]
        )

        self.stage5 = nn.ModuleList(
            [
                GCSBlock(
                    dim=embed_dims[4],
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=dpr[i + depth[3][1]],
                    sparse=sparse,
                )
                for i in range(depth[4][1])
            ]
        )

        self.downpool1 = nn.MaxPool3d((2, 2, 2))
        self.downpool2 = nn.MaxPool3d((2, 2, 2))
        self.downpool3 = nn.MaxPool3d((2, 2, 2))
        self.downpool4 = nn.MaxPool3d((2, 2, 2))

        self.norm_stage4 = SparseConvNeXtLayerNorm(normalized_shape=embed_dims[3], sparse=sparse)
        self.norm_stage5 = SparseConvNeXtLayerNorm(normalized_shape=embed_dims[4], sparse=sparse)

    def forward(self, x, active_masks=None):
        B = x.shape[0]

        out = self.stem(x)
        m1 = out

        out = self.downpool1(out)
        out = self.layer1(out)
        out = self.expand1(out)
        m2 = out

        out = self.downpool2(out)
        out = self.layer2(out)
        out = self.expand2(out)
        m3 = out

        out = self.downpool3(out)

        out, D, H, W = self.fib1(out)
        for blk in self.stage4:
            out = blk(out, D, H, W, active_masks[1] if self.sparse else None)
        out = self.norm_stage4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        m4 = out

        out = self.downpool4(out)

        out, D, H, W = self.fib2(out)
        for blk in self.stage5:
            out = blk(out, D, H, W, active_masks[0] if self.sparse else None)
        out = self.norm_stage5(out)

        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out, D, H, W = self.fib3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        m5 = out

        return [m1, m2, m3, m4, m5]


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3, do_res=True):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False)
        self.resample_do_res = do_res

        conv = nn.ConvTranspose3d
        if do_res:
            self.res_conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))
        if self.resample_do_res:
            res = self.res_conv(x)
            res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res
        return x1


class UnetResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

        if self.downsample:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.norm3 = nn.InstanceNorm3d(out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)

        out += residual
        out = self.lrelu(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, exp_r=4):
        super().__init__()
        self.layer = MedNeXtBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, exp_r=exp_r)
        self.up_block = MedNeXtUpBlock(in_channels=in_channels, out_channels=out_channels)
        self.fusion = UnetResBlock(in_channels=out_channels + out_channels, out_channels=out_channels, kernel_size=3, stride=1)

    def forward(self, d, e):
        d = self.up_block(d)
        e = self.layer(e)
        return self.fusion(torch.cat((e, d), dim=1))


class UnetOutBlock(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, n_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class GCSMIMDecoder(nn.Module):
    def __init__(self, n_classes: int = 3, embed_dims: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 384), exp_r: Tuple[int, ...] = (2, 4, 4, 4, 2)):
        super().__init__()
        self.decoder4 = FusionBlock(in_channels=embed_dims[4], out_channels=embed_dims[3], kernel_size=3, exp_r=exp_r[1])
        self.decoder3 = FusionBlock(in_channels=embed_dims[3], out_channels=embed_dims[2], kernel_size=3, exp_r=exp_r[2])
        self.decoder2 = FusionBlock(in_channels=embed_dims[2], out_channels=embed_dims[1], kernel_size=3, exp_r=exp_r[3])
        self.decoder1 = FusionBlock(in_channels=embed_dims[1], out_channels=embed_dims[0], kernel_size=3, exp_r=exp_r[4])
        self.out = UnetOutBlock(in_channels=embed_dims[0], n_classes=n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)
        return self.out(d1)


class GCSMIMEncoder(nn.Module):
    def __init__(
        self,
        img_size=96,
        in_chans=1,
        embed_dims=(32, 64, 128, 256, 384),
        depth=[[1], [1], [1], [1, 1], [1, 1]],
        kernels=(3, 3, 3, 3, 3),
        exp_r=(2, 2, 2, 2, 2),
        down_ratio=(1, 2, 4, 8, 16),
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=partial(nn.GELU),
        sparse=False,
    ):
        super().__init__()
        self.dim = [embed_dims[0], embed_dims[1], embed_dims[2], embed_dims[3], embed_dims[4]]
        self.downratio = down_ratio
        self.embeddings = GCSMIMEmbedding(
            img_size=img_size,
            in_chans=in_chans,
            embed_dims=embed_dims,
            depth=depth,
            kernels=kernels,
            exp_r=exp_r,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            act_layer=act_layer,
            sparse=sparse,
        )

    def get_downsample_ratio(self):
        return self.downratio

    def get_feature_map_channels(self) -> List[int]:
        return self.dim

    def forward(self, x, active_masks=None):
        return self.embeddings(x, active_masks)


class GCSMIM(nn.Module):
    def __init__(
        self,
        img_size=96,
        in_chans=1,
        n_classes=3,
        embed_dims=(32, 64, 128, 256, 384),
        depth=[[1], [1], [1], [1, 2], [1, 2]],
        kernels=(3, 3, 3, 3, 3),
        exp_r=(2, 2, 2, 2, 2),
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=partial(nn.GELU),
        sparse=False,
    ):
        super().__init__()
        self.embeddings = GCSMIMEmbedding(
            img_size=img_size,
            in_chans=in_chans,
            embed_dims=embed_dims,
            depth=depth,
            kernels=kernels,
            exp_r=exp_r,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            act_layer=act_layer,
            sparse=sparse,
        )
        self.decoder = GCSMIMDecoder(n_classes=n_classes, embed_dims=embed_dims)

    def forward(self, x, active_masks=None):
        feats = self.embeddings(x, active_masks)
        return self.decoder(feats[0], feats[1], feats[2], feats[3], feats[4])


def build_gcsmim(img_size=96, in_channel=1, n_classes=3):
    return GCSMIM(
        img_size=img_size,
        in_chans=in_channel,
        n_classes=n_classes,
        embed_dims=(32, 64, 128, 256, 384),
        depth=[[1], [1], [1], [1, 2], [1, 2]],
        kernels=(3, 3, 3, 3, 3),
        exp_r=(2, 2, 2, 2, 2),
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=partial(nn.GELU),
        sparse=False,
    )
