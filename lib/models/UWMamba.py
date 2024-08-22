import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# -----------------------------Enhance---------------------------------
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 8

        x_hwwh = torch.stack([x.view(B, -1, L),
                              torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
                              torch.rot90(x, 1, dims=[2, 3]).contiguous().view(B, -1, L),
                              torch.rot90(x, -1, dims=[2, 3]).contiguous().view(B, -1, L)
                              ],dim=1).view(B, 4, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # First 4 original ways
        y1 = out_y[:, 0]  # x.view(B, -1, L)
        y2 = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)  # transpose
        y3 = torch.rot90(out_y[:, 2].view(B, -1, W, H), -1, dims=[2, 3]).contiguous().view(B, -1, L)  # rotate 90 clockwise
        y4 = torch.rot90(out_y[:, 3].view(B, -1, W, H), 1, dims=[2, 3]).contiguous().view(B, -1,L)  # rotate 90 counterclockwise

        # Flipped ways
        y5 = torch.flip(out_y[:, 4], dims=[-1])
        y6 = torch.flip(out_y[:, 5].view(B, -1, W, H), dims=[2, 3]).contiguous().view(B, -1, L)
        y7 = torch.rot90(out_y[:, 6].view(B, -1, W, H), -1, dims=[2, 3]).flip(dims=[-1]).contiguous().view(B, -1, L)
        y8 = torch.rot90(out_y[:, 7].view(B, -1, W, H), 1, dims=[2, 3]).flip(dims=[-1]).contiguous().view(B, -1, L)

        return y1, y2, y3, y4, y5, y6, y7, y8

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            input_dim: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.reduction = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, input: torch.Tensor):
        input = self.reduction(input)
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

class VSSBlock_(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2):
        super(UNetConvBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_size, in_size, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm2d(in_size, affine=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_2 = nn.Conv2d(in_size * 2, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.vssb_1 = VSSBlock(input_dim=in_size, hidden_dim=in_size, norm_layer=nn.LayerNorm)
        self.vssb_2 = VSSBlock(input_dim=in_size * 2, hidden_dim=out_size, norm_layer=nn.LayerNorm)

        self.conv_1_1_1 = nn.Conv2d(in_size, in_size, 1, 1, 0)
        self.conv_1_1 = nn.Conv2d(in_size * 2, out_size, 1, 1, 0)

    def forward(self, x, idx=1):
        out = self.conv_1(x)
        out = self.norm1(out)
        out = self.relu_1(out)

        if idx == 0:
            out1 = self.conv_1_1_1(x)
        else:
            out1 = self.vssb_1(x.permute(0,2,3,1)).permute(0,3,1,2)

        out2 = torch.cat([out, out1], dim=1)

        out = self.conv_2(out2)
        out = self.relu_2(out)

        if idx == 0:
            res = self.conv_1_1(out2)
        else:
            res = self.vssb_2(out2.permute(0,2,3,1)).permute(0,3,1,2)

        out += res

        return out

class MAFM(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2):
        super(MAFM, self).__init__()
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)

        self.vssb_1 = VSSBlock(input_dim=in_size, hidden_dim=out_size, norm_layer=nn.LayerNorm)

        self.conv_trans = nn.Conv2d(in_size, out_size, kernel_size=1)
        self.sp_query_conv = nn.Conv2d(in_channels=out_size, out_channels=out_size//8, kernel_size=1)
        self.sp_key_conv = nn.Conv2d(in_channels=out_size, out_channels=out_size//8, kernel_size=1)
        self.sp_value_conv = nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=1)
        self.sp_gamma = nn.Parameter(torch.zeros(1))
        self.ch_gamma = nn.Parameter(torch.zeros(1))
        self.cov_gamma = nn.Parameter(torch.zeros(1))
        self.vssb_gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        cov_out = self.conv_2(self.conv_1(x))
        vssb_out = self.vssb_1(x.permute(0,2,3,1)).permute(0,3,1,2)

        x_trans = self.conv_trans(x)
        m_batchsize, C, height, width = x_trans.size()

        # sp att
        sp_proj_query = self.sp_query_conv(x_trans).view(m_batchsize, C//8, width*height).permute(0, 2, 1)
        sp_proj_key = self.sp_key_conv(x_trans).view(m_batchsize, C//8, width*height)
        sp_energy = torch.bmm(sp_proj_query, sp_proj_key)
        sp_attention = self.softmax(sp_energy)
        sp_proj_value = self.sp_value_conv(x_trans).view(m_batchsize, C, width*height)
        sp_out = torch.bmm(sp_proj_value, sp_attention.permute(0, 2, 1)).view(m_batchsize, C, height, width)

        # ch att
        ch_proj_query = x_trans.view(m_batchsize, C, -1)
        ch_proj_key = x_trans.view(m_batchsize, C, -1).permute(0, 2, 1)
        ch_energy = torch.bmm(ch_proj_query, ch_proj_key)
        ch_energy_new = torch.max(ch_energy, -1, keepdim=True)[0].expand_as(ch_energy)-ch_energy
        ch_attention = self.softmax(ch_energy_new)
        ch_proj_value = x_trans.view(m_batchsize, C, -1)
        ch_out = torch.bmm(ch_attention, ch_proj_value).view(m_batchsize, C, height, width)

        out = self.sp_gamma*sp_out + self.ch_gamma*ch_out + self.vssb_gamma*vssb_out + self.cov_gamma*cov_out
        return out

class Unet(nn.Module):

    def __init__(self, input_channels=3, out_channels=3):
        super(Unet, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = UNetConvBlock(input_channels, nb_filter[0])
        self.conv1_0 = UNetConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = UNetConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = UNetConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = MAFM(nb_filter[3], nb_filter[4])

        self.conv3_1 = UNetConvBlock(nb_filter[4], nb_filter[3])
        self.conv2_2 = UNetConvBlock(nb_filter[3], nb_filter[2])
        self.conv1_3 = UNetConvBlock(nb_filter[2], nb_filter[1])
        self.conv0_4 = UNetConvBlock(nb_filter[1], nb_filter[0])

        self.vssb_2 = VSSBlock_(hidden_dim=nb_filter[1], norm_layer=nn.LayerNorm)
        self.vssb_3 = VSSBlock_(hidden_dim=nb_filter[2], norm_layer=nn.LayerNorm)
        self.vssb_4 = VSSBlock_(hidden_dim=nb_filter[3], norm_layer=nn.LayerNorm)

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, input):

        x0_0 = self.conv0_0(input, idx=0)

        x0_0_size = x0_0.size()[2:]
        up0_0 = nn.Upsample(size=x0_0_size)
        x1_0 = self.conv1_0(self.pool(x0_0))

        x1_0_size = x1_0.size()[2:]
        up1_0 = nn.Upsample(size=x1_0_size)
        x2_0 = self.conv2_0(self.pool(x1_0))

        x2_0_size = x2_0.size()[2:]
        up2_0 = nn.Upsample(size=x2_0_size)
        x3_0 = self.conv3_0(self.pool(x2_0))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_0_size = x3_0.size()[2:]
        up3_0 = nn.Upsample(size=x3_0_size)
        x3_1 = self.vssb_4(x3_0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + up3_0(self.conv3_1(x4_0))
        x2_2 = self.vssb_3(x2_0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + up2_0(self.conv2_2(x3_1))
        x1_3 = self.vssb_2(x1_0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + up1_0(self.conv1_3(x2_2))
        x0_4 = x0_0 + up0_0(self.conv0_4(x1_3))

        output = self.final(x0_4)+input

        return torch.clamp(output, 0, 1)
# ---------------------------------------------------------------------

def get_Unet(cfg, pretrained=False):
    model = Unet()
    if pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}
        # pretrained_state = {k[8:]: v for k, v in pretrained_state.items()
        #                    if k[8:] in model_dict.keys()}

        for k, _ in pretrained_state.items():
            print('=> loading {} from pretrained model'.format(k))
        model_dict.update(pretrained_state)

        model.load_state_dict(model_dict, strict=False)
    return model


def get_seg_model(cfg, **kwargs):
    model = get_Unet(cfg, pretrained=False)
    return model

if __name__ == '__main__':
    x = torch.rand(4, 3, 200, 200).cuda()
    model = Unet().cuda()
    pred = model(x)
    print(pred.shape)