import torch
import torch.nn as nn
from functools import partial
from timm.models.efficientnet import EfficientNet
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

from model.pervit import VisionTransformer


@register_model
def pervit_tiny(pretrained=False, **kwargs):
    num_heads = 4
    kwargs['emb_dims'] = [128, 192, 224, 280]
    kwargs['convstem_dims'] = [3, 48, 64, 96, 128]

    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def pervit_small(pretrained=False, **kwargs):
    num_heads = 8
    kwargs['emb_dims'] = [272, 320, 368, 464]
    kwargs['convstem_dims'] = [3, 64, 128, 192, 272]

    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def pervit_medium(pretrained=False, **kwargs):
    num_heads = 12
    kwargs['emb_dims'] = [312, 468, 540, 684]
    kwargs['convstem_dims'] = [3, 64, 192, 256, 312]

    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model

