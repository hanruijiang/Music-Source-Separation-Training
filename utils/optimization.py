"""Opt-in, inference-only transforms after strict checkpoint loading.

Packed weights replace (not duplicate) the original modules. Optimized models
are runtime objects, not interchangeable training checkpoint formats.
"""
from collections import defaultdict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


# Bound temporary activations: a full stack of all frequency-band MLPs can
# consume hundreds of MB more than the original serial implementation.
_MAX_BANDS = 4


def _groups(keys):
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    for indices in groups.values():
        for start in range(0, len(indices), _MAX_BANDS):
            yield tuple(indices[start:start + _MAX_BANDS])


class PackedLinear(nn.Module):
    """Independent per-band affine transforms, evaluated by batched GEMM."""

    def __init__(self, linears):
        super().__init__()
        self.register_buffer('weight', torch.stack([
            layer.weight.detach().transpose(0, 1) for layer in linears
        ]).contiguous())
        self.register_buffer('bias', torch.stack([
            layer.bias.detach() for layer in linears
        ]).unsqueeze(1) if linears[0].bias is not None else None)

    def forward(self, x):
        # [bands, ..., input_features] -> [bands, tokens, input_features]
        flat = x.reshape(x.shape[0], -1, x.shape[-1])
        if self.bias is None:
            out = torch.bmm(flat, self.weight)
        else:
            out = torch.baddbmm(self.bias, flat, self.weight)
        return out.reshape(*x.shape[:-1], self.weight.shape[-1])


class _BandGroup(nn.Module):
    def __init__(self, indices, nets):
        super().__init__()
        self.indices = indices
        self.scale = nets[0][0].scale
        self.register_buffer('gamma', torch.stack([
            net[0].gamma.detach() for net in nets
        ]))
        self.linear = PackedLinear([net[1] for net in nets])

    def forward(self, splits):
        x = torch.stack([splits[i] for i in self.indices])
        # Retain the exact F.normalize epsilon, per-band statistics, and
        # multiplication order, including under autocast.
        gamma = self.gamma.reshape(self.gamma.shape[0], *([1] * (x.ndim - 2)), self.gamma.shape[1])
        x = F.normalize(x, dim=-1) * self.scale * gamma
        return self.linear(x)


class PackedBandSplit(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.dim_inputs = original.dim_inputs
        self.groups = nn.ModuleList([
            _BandGroup(indices, [original.to_features[i] for i in indices])
            for indices in _groups(self.dim_inputs)
        ])

    def forward(self, x):
        splits = x.split(self.dim_inputs, dim=-1)
        outs = [None] * len(self.dim_inputs)
        for group in self.groups:
            for index, value in zip(group.indices, group(splits).unbind(0)):
                outs[index] = value
        return torch.stack(outs, dim=-2)


class _MaskGroup(nn.Module):
    def __init__(self, indices, nets):
        super().__init__()
        self.indices = indices
        # Both retained implementations have Linear/Tanh alternating,
        # but their MLP depth conventions differ. Read actual layers.
        layers = []
        for position, layer in enumerate(nets[0][0]):
            if type(layer) is nn.Linear:
                layers.append(PackedLinear([net[0][position] for net in nets]))
            elif type(layer) is nn.Tanh:
                layers.append(nn.Tanh())
            else:
                raise ValueError(f'Unexpected mask layer: {type(layer).__name__}')
        layers.append(nn.GLU(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, bands):
        return self.net(torch.stack([bands[i] for i in self.indices]))


def _mask_signature(net):
    return tuple((type(layer), layer.in_features, layer.out_features, layer.bias is not None)
                 if type(layer) is nn.Linear else (type(layer),)
                 for layer in net[0])


class PackedMaskEstimator(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.num_bands = len(original.to_freqs)
        self.groups = nn.ModuleList([
            _MaskGroup(indices, [original.to_freqs[i] for i in indices])
            for indices in _groups([_mask_signature(net) for net in original.to_freqs])
        ])

    def forward(self, x):
        bands = x.unbind(dim=-2)
        outs = [None] * self.num_bands
        for group in self.groups:
            for index, value in zip(group.indices, group(bands).unbind(0)):
                outs[index] = value
        return torch.cat(outs, dim=-1)


def _fuse_qkv_gate(attn):
    qkv, gate = attn.to_qkv, attn.to_gates
    fused = nn.Linear(qkv.in_features, qkv.out_features + gate.out_features,
                      bias=True, device=qkv.weight.device, dtype=qkv.weight.dtype)
    fused.weight.copy_(torch.cat((qkv.weight, gate.weight), dim=0))
    fused.bias.zero_()
    fused.bias[qkv.out_features:].copy_(gate.bias)
    attn.to_qkv_gate = fused.requires_grad_(False).eval()
    del attn.to_qkv
    del attn.to_gates


def _fuse_sequential_bn(model):
    """Only fuse proven adjacent pairs in standard Sequential forward paths.

    In particular, do not infer adjacency from registration order in arbitrary
    MaxViT blocks, or fold normalization subclasses containing activations.
    """
    counts = defaultdict(int)
    for _, module in model.named_modules(remove_duplicate=False):
        counts[id(module)] += 1
    total = 0
    for parent in list(model.modules()):
        if not isinstance(parent, nn.Sequential) or type(parent).forward is not nn.Sequential.forward:
            continue
        keys = list(parent._modules)
        for a, b in zip(keys, keys[1:]):
            first, bn = parent._modules[a], parent._modules[b]
            if counts[id(first)] != 1 or counts[id(bn)] != 1:
                continue
            if any(m._forward_hooks or m._forward_pre_hooks for m in (first, bn)):
                continue
            if first.training or bn.training or not getattr(bn, 'track_running_stats', False):
                continue
            if getattr(bn, 'running_mean', None) is None or getattr(bn, 'running_var', None) is None:
                continue
            if type(first) is nn.Conv2d and type(bn) is nn.BatchNorm2d and first.out_channels == bn.num_features:
                fused = fuse_conv_bn_eval(first, bn)
            else:
                # Linear+BN1d needs proof of the normalization axis;
                # ConvTranspose groups need separate handling. Skip both
                # rather than guess from matching feature counts.
                continue
            parent._modules[a] = fused
            parent._modules[b] = nn.Identity()
            total += 1
    return total


@torch.no_grad()
def optimize_model(model, model_type, *, mode='safe'):
    """Return (model, applied transformations). Call before device/wrapping/compile."""
    if mode not in ('off', 'safe', 'advanced'):
        raise ValueError(f'Unsupported optimization mode: {mode}')
    if mode == 'off' or getattr(model, '_inference_optimized', False):
        return model, []
    if any(module.training for module in model.modules()):
        raise ValueError('Inference optimization requires model.eval()')

    report = []
    if model_type in ('bs_roformer', 'mel_band_roformer', 'bs_polarformer'):
        # Import lazily, retaining model-specific optional dependencies.
        if model_type == 'mel_band_roformer':
            from models.bs_roformer.mel_band_roformer import Attention, BandSplit, MaskEstimator
        else:
            from models.bs_roformer.bs_roformer import Attention, BandSplit, MaskEstimator

        count = 0
        for module in list(model.modules()):
            if type(module) is Attention and not hasattr(module, 'to_qkv_gate'):
                _fuse_qkv_gate(module)
                count += 1
        if count:
            report.append(f'QKV+Gate projections: {count}')
        if type(model.band_split) is BandSplit:
            model.band_split = PackedBandSplit(model.band_split).eval()
            report.append(f'BandSplit packed groups: {len(model.band_split.groups)}')
        count = 0
        for index, estimator in enumerate(model.mask_estimators):
            if type(estimator) is MaskEstimator:
                model.mask_estimators[index] = PackedMaskEstimator(estimator).eval()
                count += 1
        if count:
            report.append(f'MaskEstimator packed heads: {count}')

        # Only cache the known deterministic default window. Custom window
        # factories remain untouched; the buffer follows model device moves.
        factory = model.stft_window_fn
        if isinstance(factory, partial) and factory.func is torch.hann_window:
            model.register_buffer('_inference_stft_window', factory(device=next(model.parameters()).device),
                                  persistent=False)
            report.append('cached Hann window')

    if model_type == 'segm_models':
        report.append(f'Conv-BN pairs fused: {_fuse_sequential_bn(model)}')

    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.Dropout:
                setattr(parent, name, nn.Identity())
                count += 1
    if count:
        report.append(f'Dropout identities: {count}')
    model._inference_optimized = True
    return model.eval(), report


@torch.no_grad()
def prepare_device(model, device, mode, multi_gpu=False):
    """Apply optional layout and non-module STFT cache after model.to(device)."""
    report = []
    if mode == 'off':
        return report
    if not multi_gpu and hasattr(model, 'stft') and hasattr(model.stft, 'window'):
        model.stft.window = model.stft.window.to(device)
        report.append('cached STFT window on device')
    if (mode == 'advanced' and not multi_gpu and torch.device(device).type == 'cuda'
            and hasattr(model, 'first_conv')):
        model.to(memory_format=torch.channels_last)
        report.append('CUDA channels-last convolution weights')
    return report
