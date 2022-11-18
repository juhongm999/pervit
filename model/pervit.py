import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .utils import OffsetGenerator


class PeripheralPositionEncoding(nn.Module):
    def __init__(self, num_heads, norm_init, kernel_size=3):
        super().__init__()
        in_channel = hid_channel = num_heads * 4
        out_channel = num_heads
        self.remove_pad = (kernel_size // 2) * 2
        self.norm_init = norm_init

        self.pad_size = 0
        self.conv1 = nn.Conv2d(in_channel, hid_channel, kernel_size=kernel_size, stride=1, padding=self.pad_size, bias=True)
        self.conv2 = nn.Conv2d(hid_channel, out_channel, kernel_size=kernel_size, stride=1, padding=self.pad_size, bias=True)
        self.gn1  = nn.GroupNorm(hid_channel, hid_channel)
        self.gn2  = nn.GroupNorm(out_channel, out_channel)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.Sigmoid()
        self.apply(self._init_weights)
        self.gn2.apply(self._peripheral_init)

    def _peripheral_init(self, m):
        if isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, self.norm_init[0])
            nn.init.constant_(m.weight, self.norm_init[1])

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        n_token = x.size(1)
        side = int(math.sqrt(n_token))

        x = rearrange(x, '(i j) (k l) d -> d i j k l', i=side, j=side, k=side, l=side)
        x = x[:, self.remove_pad:-self.remove_pad, self.remove_pad:-self.remove_pad, ...]
        x = rearrange(x, 'd i j k l -> (i j) d k l')

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)

        y = rearrange(x, '(i j) h k l -> () h (i j) (k l)', i=side-(self.remove_pad*2), j=side-(self.remove_pad*2))
        y = self.act2(y)
        return y


class MPA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_init=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.activation = PeripheralPositionEncoding(num_heads, norm_init)
        self.exp = lambda x: torch.exp(x - torch.max(x, -1, keepdim=True)[0])
        self.weight = None

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_rpe(self, rpe):
        self.weight = self.get_weight(rpe)

    def get_weight(self, rpe):
        return self.activation(rpe)

    def forward(self, x, rpe):
        n_patch = int(math.sqrt(x.size(1)))


        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        weight = self.get_weight(rpe) if self.weight is None else self.weight
        dots = self.exp(dots)

        attn = weight * dots
        attn = F.normalize(attn, p=1, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CPE(nn.Module):
    def __init__(self, dim, k=3):
        super(CPE, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        side = int(math.sqrt(N))

        x = rearrange(x, 'b (i j) d -> b d i j', i=side, j=side)
        x = self.proj(x)
        x = rearrange(x, 'b d i j -> b (i j) d', i=side, j=side)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_init=None, next_dim=None, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.stage_change = dim != next_dim
        self.res_path = nn.Linear(dim, next_dim) if self.stage_change else nn.Identity()


        self.attn = MPA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, norm_init=norm_init, **kwargs)
        self.conv = CPE(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=next_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rel_pos_embed):
        x = x + self.drop_path(self.attn(self.norm1(self.conv(x)), rel_pos_embed))
        x = self.res_path(x) + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_pos_embed=False, emb_dims=None, convstem_dims=None):
        super().__init__()

        # Convolutional stem initialization
        self.conv_layers = ConvolutionalStem(convstem_dims)

        # Metadata initialization
        stages = [2, 2, 6, 2]
        ch_dims = []
        for dim, stage in zip(emb_dims, stages):
            ch_dims += [dim] * stage
        ch_dims += [ch_dims[-1]]
        self.num_classes = num_classes
        depth = len(ch_dims) - 1

        # Patch embedding module
        num_patches = 196
        self.num_patches = num_patches

        # Build relative position encoding (Euclidean distance)
        self.rpe_initialized = False
        def _build_rpe(num_patches, num_heads, ksz=3):
            D_r = num_heads * 4
            pad_size = (ksz // 2) * 2

            OffsetGenerator.initialize(int(math.sqrt(num_patches)), pad_size=pad_size)
            rpe = OffsetGenerator.get_qk_vec().norm(p=2, dim=-1, keepdim=True)
            self.rpe_proj = nn.Linear(1, D_r)

            return rpe

        self.rpe = _build_rpe(num_patches, num_heads)
        self.rpe_proj.apply(self._peripheral_init)

        # Peripheral initialization
        norm_bias = torch.linspace(-5.0, 4.0, steps=depth).cuda()  # controls locality size
        norm_weight = torch.linspace(3.0, 0.01, steps=depth).cuda()  # controls locality strength
        norm_init = torch.stack([norm_bias, norm_weight]).t().contiguous()

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=ch_dims[i], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                norm_init=norm_init[i], next_dim=ch_dims[i+1])
            for i in range(depth)])
        self.norm = norm_layer(ch_dims[-1])

        # Classifier head & class token
        self.head = nn.Linear(ch_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(self._init_weights)

    def init_rpe(self):
        self.rpe_initialized = True
        rpe = self.get_rpe()
        for u, blk in enumerate(self.blocks):
            blk.attn.init_rpe(rpe)

    def _peripheral_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, -0.02)
            nn.init.constant_(m.bias, 0.0)

    def initialize(self):
        for u, blk in enumerate(self.blocks):
            blk.attn.dots_sum = 0
            blk.attn.num_sample = 0

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def get_rpe(self):
        return self.rpe_proj(self.rpe)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.conv_layers(x)

        rpe = self.get_rpe() if not self.rpe_initialized else None

        for u, blk in enumerate(self.blocks):
            x = blk(x, rpe)

        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvolutionalStem(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage """
    def __init__(self, n_filter_list, kernel_sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2]):
        super().__init__()
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                          out_channels=n_filter_list[i + 1],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          padding=kernel_sizes[i] // 2),
                nn.BatchNorm2d(n_filter_list[i + 1]),
                nn.ReLU(inplace=True),
            )
                for i in range(len(n_filter_list)-1)
            ])

        self.conv1x1 = nn.Conv2d(in_channels=n_filter_list[-1], out_channels=n_filter_list[-1], stride=1, kernel_size=1, padding=0)
        self.flatten = Rearrange('b c h w -> b (h w) c')

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv1x1(x)
        x = self.flatten(x)

        return x

